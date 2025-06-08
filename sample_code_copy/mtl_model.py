import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings

from backbone_models import BackboneModel
from mtl_config import MTLConfig

class MTLModel(nn.Module):
    """Multi-task learning model with configurable backbone"""
    def __init__(
        self,
        config: MTLConfig,
        use_asr: bool = True,
        use_prosody: bool = True,
        use_ser: bool = True,
        tokenizer = None
    ):
        super().__init__()
        self.config = config
        self.use_asr = use_asr
        self.use_prosody = use_prosody
        self.use_ser = use_ser
        self.tokenizer = tokenizer

        # Initialize backbone
        self.backbone = BackboneModel(config.backbone_config)
        
        # Get actual hidden size from backbone
        self.hidden_size = self.backbone.get_hidden_size()
        
        # Dropout layer
        self.dropout = nn.Dropout(config.final_dropout)
        
        # Initialize heads
        self._initialize_heads()
        
        # Initialize loss functions
        self._initialize_loss_functions()

    def _initialize_heads(self):
        """Initialize task-specific heads based on configuration"""
        # ASR head
        if self.use_asr:
            vocab_size = self.tokenizer.get_vocab_size() if self.tokenizer else self.config.vocab_size
            self.asr_head = nn.Linear(self.hidden_size, vocab_size)
            print(f"ASR head initialized with vocab size: {vocab_size}")
        
        # Prosody head
        if self.use_prosody:
            self.prosody_bilstm = nn.LSTM(
                self.hidden_size,
                self.config.prosody_lstm_hidden,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )
            # Binary classification output
            prosody_out_features = 1 if self.config.prosody_classes == 2 else self.config.prosody_classes
            self.prosody_head = nn.Linear(
                self.config.prosody_lstm_hidden * 2,
                prosody_out_features
            )
            print(f"Prosody head initialized with {self.config.prosody_classes} classes")
        
        # SER head
        if self.use_ser:
            self.ser_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.config.final_dropout),
                nn.Linear(self.hidden_size // 2, self.config.emotion_classes)
            )
            print(f"SER head initialized with {self.config.emotion_classes} classes")

    def _initialize_loss_functions(self):
        """Initialize loss functions for each task"""
        if self.use_asr:
            blank_id = self.tokenizer.blank_id if self.tokenizer else 0
            self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        
        if self.use_prosody:
            if self.config.prosody_classes == 2:
                self.prosody_loss = nn.BCEWithLogitsLoss()
            else:
                self.prosody_loss = nn.CrossEntropyLoss()
        
        if self.use_ser:
            self.ser_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_features: torch.Tensor,
        asr_targets: Optional[torch.Tensor] = None,
        asr_lengths: Optional[torch.Tensor] = None,
        prosody_targets: Optional[torch.Tensor] = None,
        emotion_targets: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        # Get backbone features
        backbone_features = self.backbone(input_features)
        backbone_features = self.dropout(backbone_features)
        
        outputs = {}
        total_loss = 0.0

        # ASR forward pass
        if self.use_asr and self.asr_head is not None:
            asr_logits = self.asr_head(backbone_features)
            asr_log_probs = F.log_softmax(asr_logits, dim=-1)
            outputs['asr_logits'] = asr_logits
            outputs['asr_log_probs'] = asr_log_probs
            
            if return_loss and asr_targets is not None:
                asr_loss = self._compute_asr_loss(asr_log_probs, asr_targets, asr_lengths)
                outputs['asr_loss'] = asr_loss
                total_loss += self.config.loss_weights['asr'] * asr_loss

        # Prosody forward pass
        if self.use_prosody and self.prosody_head is not None:
            prosody_target_len = prosody_targets.shape[1] if prosody_targets is not None else None
            
            B, T, D = backbone_features.shape
            
            if T < 1:
                prosody_logits = torch.zeros(B, prosody_target_len, device=backbone_features.device)
            else:
                # Transpose and pool features
                features_t = backbone_features.transpose(1, 2)
                pooled_features = F.adaptive_avg_pool1d(features_t, prosody_target_len)
                pooled_features = pooled_features.transpose(1, 2)
                
                # BiLSTM processing
                prosody_bilstm_out, _ = self.prosody_bilstm(pooled_features)
                
                # Apply prosody head
                batch_size, seq_len, bilstm_hidden_dim = prosody_bilstm_out.shape
                prosody_bilstm_flat = prosody_bilstm_out.reshape(-1, bilstm_hidden_dim)
                prosody_logits_flat = self.prosody_head(prosody_bilstm_flat)
                
                # Reshape based on output type
                if len(prosody_logits_flat.shape) > 1 and prosody_logits_flat.shape[-1] > 1:
                    prosody_logits = prosody_logits_flat.view(batch_size, seq_len, -1)
                else:
                    prosody_logits = prosody_logits_flat.view(batch_size, seq_len)
            
            outputs['prosody_logits'] = prosody_logits
            
            if return_loss and prosody_targets is not None:
                prosody_loss = self._compute_prosody_loss(prosody_logits, prosody_targets)
                outputs['prosody_loss'] = prosody_loss
                total_loss += self.config.loss_weights['prosody'] * prosody_loss

        # SER forward pass
        if self.use_ser and self.ser_head is not None:
            # Global average pooling
            ser_features = torch.mean(backbone_features, dim=1)
            emotion_logits = self.ser_head(ser_features)
            outputs['emotion_logits'] = emotion_logits
            
            if return_loss and emotion_targets is not None:
                emotion_loss = self._compute_emotion_loss(emotion_logits, emotion_targets)
                outputs['emotion_loss'] = emotion_loss
                total_loss += self.config.loss_weights['ser'] * emotion_loss

        if return_loss:
            outputs['total_loss'] = total_loss

        return outputs

    def _compute_asr_loss(self, log_probs, targets, lengths):
        """Compute CTC loss for ASR"""
        max_target_value = self.tokenizer.get_vocab_size() - 1 if self.tokenizer else self.config.vocab_size - 1
        if targets.max() > max_target_value or targets.min() < 0:
            raise ValueError(f"ASR target values are out of bounds. Max target: {targets.max()}, Min target: {targets.min()}, Max allowed: {max_target_value}")

        log_probs = log_probs.transpose(0, 1)
        input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long, device=log_probs.device)

        if targets.dim() == 2:
            target_lengths = lengths
            targets_flat = []
            for i, length in enumerate(lengths):
                targets_flat.extend(targets[i, :length].tolist())
            targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=targets.device)
        else:
            targets_flat = targets
            target_lengths = lengths

        if target_lengths.sum() != targets_flat.shape[0]:
            warnings.warn("Sum of ASR target lengths does not match the flattened target tensor size.")

        return self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)

    def _compute_prosody_loss(self, logits, targets):
        """Compute loss for prosodic sequence labeling"""
        if self.config.prosody_classes == 2:
            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError(f"Prosody target values for binary classification must be 0 or 1. Found values: {torch.unique(targets)}")
            return self.prosody_loss(logits, targets.float())
        else:
            if targets.max() >= self.config.prosody_classes or targets.min() < 0:
                raise ValueError(f"Prosody target values for multi-class classification are out of bounds. Max target: {targets.max()}, Min target: {targets.min()}, Max allowed: {self.config.prosody_classes - 1}")
            return self.prosody_loss(logits.view(-1, self.config.prosody_classes), targets.view(-1))

    def _compute_emotion_loss(self, logits, targets):
        """Compute cross-entropy loss for emotion classification"""
        if targets.max() >= self.config.emotion_classes or targets.min() < 0:
            raise ValueError(f"Emotion target values are out of bounds. Max target: {targets.max()}, Min target: {targets.min()}, Max allowed: {self.config.emotion_classes - 1}")
        return self.ser_loss(logits, targets)

    def get_active_heads(self) -> Dict[str, bool]:
        """Get dictionary of active task heads"""
        return {
            'asr': self.use_asr,
            'prosody': self.use_prosody,
            'ser': self.use_ser
        } 