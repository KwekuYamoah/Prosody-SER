"""
MTL Model Module
Paper-style Multi-task Learning Model following:
"Speech Emotion Recognition with Multi-task Learning"

Key principles:
1. SER is the main task (weight = 1.0)
2. ASR and Prosody are auxiliary tasks (weighted by alpha)
3. Shared wav2vec/whisper backbone with task-specific heads
4. Loss: L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings

from sample_code.scripts.backbone_models import BackboneModel
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.ctc_decoder import CTCLoss


class MTLModel(nn.Module):
    """
    Paper-style Multi-task Learning Model following:
    "Speech Emotion Recognition with Multi-task Learning"
    """

    def __init__(self, config: MTLConfig, use_asr: bool = True,
                 use_prosody: bool = True, use_ser: bool = True, tokenizer=None):
        super().__init__()
        self.config = config
        self.use_asr = use_asr
        self.use_prosody = use_prosody
        self.use_ser = use_ser
        self.tokenizer = tokenizer

        # Initialize shared backbone (following paper's approach)
        self.backbone = BackboneModel(config.backbone_config)
        self.hidden_size = self.backbone.get_hidden_size()

        # Dropout layer
        self.dropout = nn.Dropout(config.final_dropout)

        # Initialize task-specific heads
        self._initialize_task_heads()

        # Initialize loss functions with paper-style regularization
        self._initialize_loss_functions()

        print(f"Paper-style MTL Model initialized:")
        print(f"  Main task: SER (weight=1.0)")
        print(
            f"  Auxiliary tasks: ASR (α={config.alpha_asr}), Prosody (α={config.alpha_prosody})")
        print(f"  Active heads: {self.get_active_heads()}")

    def _initialize_task_heads(self):
        """Initialize task-specific heads following paper architecture"""

        # SER head (main task) - simple classifier with pooling
        if self.use_ser:
            self.ser_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.config.final_dropout),
                nn.Linear(self.hidden_size // 2, self.config.emotion_classes)
            )
            print(
                f"SER head initialized: {self.config.emotion_classes} emotion classes")

        # ASR head (auxiliary task) - CTC-based sequence labeling
        if self.use_asr:
            vocab_size = self.tokenizer.get_vocab_size(
            ) if self.tokenizer else self.config.vocab_size

            # Layer normalization for stability (helps with blank prediction issue)
            self.asr_layer_norm = nn.LayerNorm(self.hidden_size)
            self.asr_head = nn.Linear(self.hidden_size, vocab_size)

            # Initialize with smaller weights to prevent early collapse to blanks
            nn.init.xavier_uniform_(self.asr_head.weight, gain=0.1)
            nn.init.constant_(self.asr_head.bias, 0)

            print(
                f"ASR head initialized: vocab_size={vocab_size} (auxiliary task)")

        # Prosody head (auxiliary task) - sequence labeling with BiLSTM
        if self.use_prosody:
            self.prosody_bilstm = nn.LSTM(
                self.hidden_size,
                self.config.prosody_lstm_hidden,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )

            # Binary or multi-class output
            prosody_out_features = 1 if self.config.prosody_classes == 2 else self.config.prosody_classes
            self.prosody_head = nn.Linear(
                self.config.prosody_lstm_hidden * 2,  # Bidirectional
                prosody_out_features
            )
            print(
                f"Prosody head initialized: {self.config.prosody_classes} classes (auxiliary task)")

    def _initialize_loss_functions(self):
        """Initialize loss functions with paper-style regularization"""

        # SER loss (main task)
        if self.use_ser:
            self.ser_loss = nn.CrossEntropyLoss()

        # ASR loss (auxiliary task) with enhanced CTC to fix blank prediction
        if self.use_asr:
            self.asr_loss = CTCLoss(
                blank_id=0,  # Standard CTC blank
                zero_infinity=True,
                entropy_weight=self.config.ctc_entropy_weight,
                blank_penalty=self.config.ctc_blank_penalty,
                blank_threshold=self.config.ctc_blank_threshold
            )
            print(f"ASR CTC Loss initialized with regularization:")
            print(f"  Entropy weight: {self.config.ctc_entropy_weight}")
            print(f"  Blank penalty: {self.config.ctc_blank_penalty}")
            print(f"  Blank threshold: {self.config.ctc_blank_threshold}")

        # Prosody loss (auxiliary task)
        if self.use_prosody:
            if self.config.prosody_classes == 2:
                self.prosody_loss = nn.BCEWithLogitsLoss()
            else:
                self.prosody_loss = nn.CrossEntropyLoss()

    def forward(self, input_features: torch.Tensor, asr_targets: Optional[torch.Tensor] = None,
                asr_lengths: Optional[torch.Tensor] = None, prosody_targets: Optional[torch.Tensor] = None,
                emotion_targets: Optional[torch.Tensor] = None, return_loss: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass following paper's architecture and loss computation.

        Paper's loss formula: L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody
        """

        # Shared backbone feature extraction
        backbone_features = self.backbone(
            input_features)  # (batch, time, hidden)
        backbone_features = self.dropout(backbone_features)

        outputs = {}
        task_losses = {}
        loss_details = {}

        # === SER Forward Pass (Main Task) ===
        if self.use_ser and hasattr(self, 'ser_head'):
            # Global average pooling for utterance-level classification
            ser_features = torch.mean(
                backbone_features, dim=1)  # (batch, hidden)
            emotion_logits = self.ser_head(ser_features)
            outputs['emotion_logits'] = emotion_logits

            if return_loss and emotion_targets is not None:
                ser_loss = self._compute_ser_loss(
                    emotion_logits, emotion_targets)
                task_losses['ser'] = ser_loss
                outputs['emotion_loss'] = ser_loss

        # === ASR Forward Pass (Auxiliary Task) ===
        if self.use_asr and hasattr(self, 'asr_head'):
            # Apply layer normalization for stability
            asr_features = self.asr_layer_norm(backbone_features)
            asr_logits = self.asr_head(asr_features)  # (batch, time, vocab)
            outputs['asr_logits'] = asr_logits

            if return_loss and asr_targets is not None:
                asr_loss, asr_loss_details = self._compute_asr_loss(
                    asr_logits, asr_targets, asr_lengths)
                task_losses['asr'] = asr_loss
                outputs['asr_loss'] = asr_loss
                loss_details['asr'] = asr_loss_details

        # === Prosody Forward Pass (Auxiliary Task) ===
        if self.use_prosody and hasattr(self, 'prosody_head'):
            batch_size, seq_len, hidden_dim = backbone_features.shape
            prosody_target_len = prosody_targets.shape[1] if prosody_targets is not None else seq_len

            # Handle sequence length alignment
            if seq_len != prosody_target_len and prosody_target_len > 0:
                # Use interpolation to align sequence lengths
                backbone_features_t = backbone_features.transpose(
                    1, 2)  # (batch, hidden, time)
                aligned_features = F.interpolate(
                    backbone_features_t, size=prosody_target_len,
                    mode='linear', align_corners=False
                ).transpose(1, 2)  # (batch, target_len, hidden)
            else:
                aligned_features = backbone_features

            # BiLSTM processing for sequence labeling
            prosody_lstm_out, _ = self.prosody_bilstm(aligned_features)

            # Apply prosody head
            batch_size, target_len, lstm_hidden = prosody_lstm_out.shape
            prosody_lstm_flat = prosody_lstm_out.reshape(-1, lstm_hidden)
            prosody_logits_flat = self.prosody_head(prosody_lstm_flat)

            # Reshape based on output type
            if prosody_logits_flat.shape[-1] > 1:
                prosody_logits = prosody_logits_flat.view(
                    batch_size, target_len, -1)
            else:
                prosody_logits = prosody_logits_flat.view(
                    batch_size, target_len)

            outputs['prosody_logits'] = prosody_logits

            if return_loss and prosody_targets is not None:
                prosody_loss = self._compute_prosody_loss(
                    prosody_logits, prosody_targets)
                task_losses['prosody'] = prosody_loss
                outputs['prosody_loss'] = prosody_loss

        # === Paper-Style Loss Computation ===
        if return_loss:
            total_loss, alpha_info = self._compute_total_loss_paper_style(
                task_losses)
            outputs['total_loss'] = total_loss
            outputs['loss_details'] = loss_details
            outputs['alpha_values'] = alpha_info

        return outputs

    def _compute_total_loss_paper_style(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss following paper's formula:
        L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody

        Where:
        - SER is the main task (weight = 1.0)
        - ASR and Prosody are auxiliary tasks (weighted by alpha)
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        alpha_info = self.config.get_alpha_values()

        # Main task: SER (always weight = 1.0)
        if 'ser' in task_losses:
            total_loss += 1.0 * task_losses['ser']

        # Auxiliary task: ASR (weighted by α_ASR)
        if 'asr' in task_losses:
            total_loss += self.config.alpha_asr * task_losses['asr']

        # Auxiliary task: Prosody (weighted by α_Prosody)
        if 'prosody' in task_losses:
            total_loss += self.config.alpha_prosody * task_losses['prosody']

        return total_loss, alpha_info

    def _compute_ser_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for SER (main task)"""
        if targets.max() >= self.config.emotion_classes or targets.min() < 0:
            raise ValueError(
                f"SER target values out of bounds. Max: {targets.max()}, "
                f"Min: {targets.min()}, Expected: 0-{self.config.emotion_classes-1}")
        return self.ser_loss(logits, targets)

    def _compute_asr_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                          lengths: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute enhanced CTC loss for ASR (auxiliary task)"""
        max_target_value = self.tokenizer.get_vocab_size(
        ) - 1 if self.tokenizer else self.config.vocab_size - 1
        if targets.max() > max_target_value or targets.min() < 0:
            raise ValueError(
                f"ASR target values out of bounds. Max: {targets.max()}, "
                f"Min: {targets.min()}, Expected: 0-{max_target_value}")

        # Apply log_softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose for CTC: (time, batch, vocab)
        log_probs = log_probs.transpose(0, 1)
        input_lengths = torch.full(
            (log_probs.size(1),), log_probs.size(0),
            dtype=torch.long, device=log_probs.device
        )

        # Prepare targets for CTC
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

        # Compute enhanced CTC loss with regularization
        loss, loss_details = self.asr_loss(
            log_probs, targets_flat, input_lengths, target_lengths)

        return loss, loss_details

    def _compute_prosody_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for prosody sequence labeling (auxiliary task)"""
        if self.config.prosody_classes == 2:
            # Binary classification
            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError(
                    f"Binary prosody targets must be 0 or 1. Found: {torch.unique(targets)}")
            return self.prosody_loss(logits, targets.float())
        else:
            # Multi-class classification
            if targets.max() >= self.config.prosody_classes or targets.min() < 0:
                raise ValueError(
                    f"Prosody targets out of bounds. Max: {targets.max()}, "
                    f"Min: {targets.min()}, Expected: 0-{self.config.prosody_classes-1}")
            return self.prosody_loss(logits.view(-1, self.config.prosody_classes), targets.view(-1))

    def get_active_heads(self) -> Dict[str, bool]:
        """Get dictionary of active task heads"""
        return {
            'ser': self.use_ser,
            'asr': self.use_asr,
            'prosody': self.use_prosody
        }

    def get_paper_training_info(self) -> Dict:
        """Get training information following paper's methodology"""
        return {
            'model_type': 'Paper-Style Multi-Task Learning',
            'main_task': 'SER (Speech Emotion Recognition)',
            'auxiliary_tasks': ['ASR (Automatic Speech Recognition)', 'Prosody Classification'],
            'loss_formula': 'L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody',
            'alpha_values': self.config.get_alpha_values(),
            'backbone': {
                'model': self.config.backbone_name,
                'hidden_size': self.hidden_size,
                'trainable_params': self.backbone.get_num_trainable_params(),
                'total_params': self.backbone.get_num_total_params()
            },
            'regularization': {
                'ctc_entropy_weight': self.config.ctc_entropy_weight,
                'ctc_blank_penalty': self.config.ctc_blank_penalty,
                'dropout': self.config.final_dropout
            }
        }

    def update_alpha_values(self, alpha_asr: float, alpha_prosody: float):
        """Update alpha values for auxiliary tasks (useful for experiments)"""
        self.config.update_alpha_weights(alpha_asr, alpha_prosody)
        print(
            f"Updated alpha values: ASR={alpha_asr}, Prosody={alpha_prosody}")

    def set_paper_optimal_alpha(self):
        """Set alpha values to paper's optimal configuration (α=0.1 for both)"""
        self.update_alpha_values(0.1, 0.1)
        print("Set to paper's optimal alpha configuration (both auxiliary tasks = 0.1)")

    def disable_auxiliary_tasks(self):
        """Disable auxiliary tasks (α=0) for SER-only training"""
        self.update_alpha_values(0.0, 0.0)
        print("Disabled auxiliary tasks (SER-only mode)")

    def enable_asr_only(self, alpha_asr: float = 0.1):
        """Enable only ASR auxiliary task"""
        self.update_alpha_values(alpha_asr, 0.0)
        print(f"Enabled ASR auxiliary task only (α_ASR={alpha_asr})")

    def enable_prosody_only(self, alpha_prosody: float = 0.1):
        """Enable only Prosody auxiliary task"""
        self.update_alpha_values(0.0, alpha_prosody)
        print(
            f"Enabled Prosody auxiliary task only (α_Prosody={alpha_prosody})")
