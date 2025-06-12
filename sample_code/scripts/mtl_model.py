import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings

from sample_code.scripts.backbone_models import BackboneModel
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.ctc_decoder import CTCLossWithRegularization


class MTLModel(nn.Module):
    """Multi-task learning model with configurable backbone and improved CTC handling"""

    def __init__(
        self,
        config: MTLConfig,
        use_asr: bool = True,
        use_prosody: bool = True,
        use_ser: bool = True,
        tokenizer=None
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
        # ASR head with layer normalization for stability
        if self.use_asr:
            vocab_size = self.tokenizer.get_vocab_size(
            ) if self.tokenizer else self.config.vocab_size
            self.asr_layer_norm = nn.LayerNorm(self.hidden_size)
            self.asr_head = nn.Linear(self.hidden_size, vocab_size)

            # Initialize ASR head with smaller weights to prevent collapse
            nn.init.xavier_uniform_(self.asr_head.weight, gain=0.1)
            nn.init.constant_(self.asr_head.bias, 0)

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
            print(
                f"Prosody head initialized with {self.config.prosody_classes} classes")

        # SER head
        if self.use_ser:
            self.ser_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.config.final_dropout),
                nn.Linear(self.hidden_size // 2, self.config.emotion_classes)
            )
            print(
                f"SER head initialized with {self.config.emotion_classes} classes")

    def _initialize_loss_functions(self):
        """Initialize loss functions for each task"""
        if self.use_asr:
            # Use improved CTC loss with regularization
            blank_id = 0  # Standard CTC blank token
            self.ctc_loss = CTCLossWithRegularization(
                blank_id=blank_id,
                zero_infinity=True,
                entropy_weight=0.01,  # Prevent overconfidence
                blank_weight=0.95     # Prevent excessive blank predictions
            )
            print(
                f"CTC Loss initialized with blank_id={blank_id} and regularization")

        if self.use_prosody:
            if self.config.prosody_classes == 2:
                self.prosody_loss = nn.BCEWithLogitsLoss()
            else:
                self.prosody_loss = nn.CrossEntropyLoss()

        if self.use_ser:
            self.ser_loss = nn.CrossEntropyLoss()

    def _adaptive_pool_workaround(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Workaround for adaptive pooling CUDA error.
        Uses fixed-size average pooling instead of adaptive pooling.
        """
        B, T, D = features.shape

        if T == target_length:
            return features

        # Transpose for pooling: (B, D, T)
        features_t = features.transpose(1, 2)

        if T < target_length:
            # Upsample if needed
            pooled = F.interpolate(
                features_t,
                size=target_length,
                mode='linear',
                align_corners=False
            )
        else:
            # Use fixed avg_pool1d instead of adaptive
            kernel_size = T // target_length
            stride = kernel_size

            # Handle edge cases where division isn't perfect
            padding = 0
            expected_out = (T - kernel_size + 2 * padding) // stride + 1

            if expected_out < target_length:
                kernel_size = max(1, kernel_size - 1)
                stride = kernel_size

            # Apply average pooling
            pooled = F.avg_pool1d(
                features_t, kernel_size=kernel_size, stride=stride)

            # If we still don't have exact target length, interpolate
            if pooled.size(2) != target_length:
                pooled = F.interpolate(
                    pooled,
                    size=target_length,
                    mode='linear',
                    align_corners=False
                )

        # Transpose back: (B, T, D)
        return pooled.transpose(1, 2)

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
        loss_details = {}

        # ASR forward pass with layer norm for stability
        if self.use_asr and hasattr(self, 'asr_head'):
            # Apply layer normalization before ASR head
            asr_features = self.asr_layer_norm(backbone_features)
            asr_logits = self.asr_head(asr_features)

            # Don't apply log_softmax here - let the loss function handle it
            outputs['asr_logits'] = asr_logits

            if return_loss and asr_targets is not None:
                asr_loss, asr_loss_details = self._compute_asr_loss(
                    asr_logits, asr_targets, asr_lengths)
                outputs['asr_loss'] = asr_loss
                loss_details['asr'] = asr_loss_details
                total_loss += self.config.loss_weights['asr'] * asr_loss

        # Prosody forward pass with CUDA fix
        if self.use_prosody and hasattr(self, 'prosody_head'):
            B, T, D = backbone_features.shape
            prosody_target_len = prosody_targets.shape[1] if prosody_targets is not None else 0

            # Safeguard against zero-length inputs or targets
            if T <= 1 or prosody_target_len <= 1:
                if self.config.prosody_classes == 2:
                    prosody_logits = torch.zeros(
                        B, prosody_target_len, device=backbone_features.device)
                else:
                    prosody_logits = torch.zeros(
                        B, prosody_target_len, self.config.prosody_classes, device=backbone_features.device)
            else:
                # Always use the safe pooling method
                pooled_features = self._adaptive_pool_workaround(
                    backbone_features, prosody_target_len)

                # BiLSTM processing
                prosody_bilstm_out, _ = self.prosody_bilstm(pooled_features)

                # Apply prosody head
                batch_size, seq_len, bilstm_hidden_dim = prosody_bilstm_out.shape
                prosody_bilstm_flat = prosody_bilstm_out.reshape(
                    -1, bilstm_hidden_dim)
                prosody_logits_flat = self.prosody_head(prosody_bilstm_flat)

                # Reshape based on output type
                if len(prosody_logits_flat.shape) > 1 and prosody_logits_flat.shape[-1] > 1:
                    prosody_logits = prosody_logits_flat.view(
                        batch_size, seq_len, -1)
                else:
                    prosody_logits = prosody_logits_flat.view(
                        batch_size, seq_len)

            outputs['prosody_logits'] = prosody_logits

            if return_loss and prosody_targets is not None:
                prosody_loss = self._compute_prosody_loss(
                    prosody_logits, prosody_targets)
                outputs['prosody_loss'] = prosody_loss
                total_loss += self.config.loss_weights['prosody'] * \
                    prosody_loss

        # SER forward pass
        if self.use_ser and hasattr(self, 'ser_head'):
            # Global average pooling
            ser_features = torch.mean(backbone_features, dim=1)
            emotion_logits = self.ser_head(ser_features)
            outputs['emotion_logits'] = emotion_logits

            if return_loss and emotion_targets is not None:
                emotion_loss = self._compute_emotion_loss(
                    emotion_logits, emotion_targets)
                outputs['emotion_loss'] = emotion_loss
                total_loss += self.config.loss_weights['ser'] * emotion_loss

        if return_loss:
            outputs['total_loss'] = total_loss
            outputs['loss_details'] = loss_details

        return outputs

    def _compute_asr_loss(self, logits, targets, lengths):
        """Compute CTC loss for ASR with improved handling"""
        max_target_value = self.tokenizer.get_vocab_size(
        ) - 1 if self.tokenizer else self.config.vocab_size - 1
        if targets.max() > max_target_value or targets.min() < 0:
            raise ValueError(
                f"ASR target values are out of bounds. Max target: {targets.max()}, "
                f"Min target: {targets.min()}, Max allowed: {max_target_value}")

        # Apply log_softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose for CTC loss: (T, N, C)
        log_probs = log_probs.transpose(0, 1)
        input_lengths = torch.full(
            (log_probs.size(1),), log_probs.size(0),
            dtype=torch.long, device=log_probs.device
        )

        # Prepare targets
        if targets.dim() == 2:
            target_lengths = lengths
            targets_flat = []
            for i, length in enumerate(lengths):
                targets_flat.extend(targets[i, :length].tolist())
            targets_flat = torch.tensor(
                targets_flat, dtype=torch.long, device=targets.device)
        else:
            targets_flat = targets
            target_lengths = lengths

        if target_lengths.sum() != targets_flat.shape[0]:
            warnings.warn(
                "Sum of ASR target lengths does not match the flattened target tensor size.")

        # Compute loss with regularization
        loss, loss_details = self.ctc_loss(
            log_probs, targets_flat, input_lengths, target_lengths)
        return loss, loss_details

    def _compute_prosody_loss(self, logits, targets):
        """Compute loss for prosodic sequence labeling"""
        if self.config.prosody_classes == 2:
            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError(
                    f"Prosody target values for binary classification must be 0 or 1. "
                    f"Found values: {torch.unique(targets)}")
            return self.prosody_loss(logits, targets.float())
        else:
            if targets.max() >= self.config.prosody_classes or targets.min() < 0:
                raise ValueError(
                    f"Prosody target values for multi-class classification are out of bounds. "
                    f"Max target: {targets.max()}, Min target: {targets.min()}, "
                    f"Max allowed: {self.config.prosody_classes - 1}")
            return self.prosody_loss(logits.view(-1, self.config.prosody_classes), targets.view(-1))

    def _compute_emotion_loss(self, logits, targets):
        """Compute cross-entropy loss for emotion classification"""
        if targets.max() >= self.config.emotion_classes or targets.min() < 0:
            raise ValueError(
                f"Emotion target values are out of bounds. Max target: {targets.max()}, "
                f"Min target: {targets.min()}, Max allowed: {self.config.emotion_classes - 1}")
        return self.ser_loss(logits, targets)

    def get_active_heads(self) -> Dict[str, bool]:
        """Get dictionary of active task heads"""
        return {
            'asr': self.use_asr,
            'prosody': self.use_prosody,
            'ser': self.use_ser
        }
