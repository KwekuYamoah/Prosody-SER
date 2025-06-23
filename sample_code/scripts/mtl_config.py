from dataclasses import dataclass
from typing import Dict, Optional, Any
from sample_code.scripts.backbone_models import BackboneConfig, BACKBONE_CONFIGS


@dataclass
class MTLConfig:
    """
    MTL Configuration with proper alpha control.

    Following "Speech Emotion Recognition with Multi-task Learning" paper:
    - SER is the main task (weight = 1.0)
    - ASR and Prosody are auxiliary tasks (weighted by alpha values)
    - Loss formula: L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody
    """
    backbone_name: str = "whisper"
    vocab_size: int = 51865
    emotion_classes: int = 9
    prosody_classes: int = 2
    prosody_lstm_hidden: int = 256
    final_dropout: float = 0.1

    # alpha control (key innovation)
    alpha_ser: float = 1.0          # main task (SER) weight
    alpha_asr: float = 0.1          # optimal value for ASR auxiliary task
    alpha_prosody: float = 0.1      # optimal value for Prosody auxiliary task

    # Enhanced CTC regularization parameters
    # Entropy regularization to prevent overconfidence
    ctc_entropy_weight: float = 0.01
    # Very strong penalty for excessive blank predictions (increased to 50.0)
    ctc_blank_penalty: float = 50.0
    # Threshold for blank penalty (much more reasonable than 0.8)
    ctc_blank_threshold: float = 0.3
    ctc_label_smoothing: float = 0.0    # Label smoothing for CTC loss
    ctc_confidence_penalty: float = 0.0  # Confidence penalty for CTC loss

    # Backbone fine-tuning strategy (paper-style)
    backbone_learning_rate: float = 1e-5     # Lower LR for backbone
    task_head_learning_rate: float = 5e-5    # Higher LR for task heads
    freeze_backbone_initially: bool = False

    # Loss computation settings
    loss_weights: Optional[Dict[str, float]] = None
    custom_backbone_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize configuration following paper's methodology"""
        # Validate alpha values (paper tested 0.0, 0.001, 0.01, 0.1, 1.0)
        if not 0.0 <= self.alpha_asr <= 1.0:
            raise ValueError("alpha_asr must be between 0.0 and 1.0")
        if not 0.0 <= self.alpha_prosody <= 1.0:
            raise ValueError("alpha_prosody must be between 0.0 and 1.0")
        if not 0.0 <= self.alpha_ser <= 1.0:
            raise ValueError("alpha_ser must be between 0.0 and 1.0")

        # Validate other parameters
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.emotion_classes <= 0:
            raise ValueError("emotion_classes must be positive")
        if self.prosody_classes <= 0:
            raise ValueError("prosody_classes must be positive")

        # Setup backbone configuration
        if self.custom_backbone_config is not None:
            self.backbone_config = BackboneConfig(
                **self.custom_backbone_config)
        else:
            if self.backbone_name not in BACKBONE_CONFIGS:
                raise ValueError(f"Unsupported backbone: {self.backbone_name}")
            self.backbone_config = BACKBONE_CONFIGS[self.backbone_name]

        # Paper-style loss weights: SER=1.0 (main), others weighted by alpha
        if self.loss_weights is None:
            self.loss_weights = {
                'ser': self.alpha_ser,         # Main task (mostly 1.0)
                'asr': self.alpha_asr,         # Auxiliary task weighted by alpha
                'prosody': self.alpha_prosody  # Auxiliary task weighted by alpha
            }

        # Validate loss weights follow paper's structure
        if self.loss_weights.get('ser', 0) != 1.0:
            print("Warning: SER should be main task with weight=1.0 (paper methodology)")

    def update_alpha_weights(self, alpha_ser: float, alpha_asr: float, alpha_prosody: float):
        """
        Update alpha values and recalculate loss weights.

        This follows the paper's experimental setup where different alpha values
        are tested to find optimal auxiliary task contribution.
        """
        if not 0.0 <= alpha_asr <= 1.0:
            raise ValueError("alpha_asr must be between 0.0 and 1.0")
        if not 0.0 <= alpha_prosody <= 1.0:
            raise ValueError("alpha_prosody must be between 0.0 and 1.0")
        if not 0.0 <= self.alpha_ser <= 1.0:
            raise ValueError("alpha_ser must be between 0.0 and 1.0")

        self.alpha_asr = alpha_asr
        self.alpha_prosody = alpha_prosody
        self.alpha_ser = alpha_ser

        # Update loss weights following paper's formula
        self.loss_weights = {
            'ser': self.alpha_ser,         # Main task weight (mostly 1.0)
            'asr': self.alpha_asr,         # α_ASR from paper
            'prosody': self.alpha_prosody  # α_Prosody from paper
        }

    def get_alpha_values(self) -> Dict[str, float]:
        """Get current alpha values for logging/monitoring"""
        return {
            'alpha_asr': self.alpha_asr,
            'alpha_prosody': self.alpha_prosody,
            'main_task_weight': self.alpha_ser
        }

    def get_paper_summary(self) -> Dict[str, Any]:
        """Get a summary following paper's experimental setup"""
        return {
            'methodology': 'Multi-task Learning for Speech Emotion Recognition',
            'main_task': 'SER (Speech Emotion Recognition)',
            'auxiliary_tasks': ['ASR (Automatic Speech Recognition)', 'Prosody Classification'],
            'loss_formula': 'L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody',
            'backbone': {
                'model': self.backbone_name,
                'hidden_size': self.backbone_config.hidden_size,
                'fine_tuning': 'paper-style (no freeze/unfreeze)',
                'backbone_lr': self.backbone_learning_rate,
                'head_lr': self.task_head_learning_rate
            },
            'alpha_values': self.get_alpha_values(),
            'loss_weights': self.loss_weights,
            'ctc_regularization': {
                'entropy_weight': self.ctc_entropy_weight,
                'blank_penalty': self.ctc_blank_penalty,
                'label_smoothing': self.ctc_label_smoothing
            },
            'tasks': {
                'ser': {
                    'type': 'classification',
                    'classes': self.emotion_classes,
                    'weight': self.loss_weights['ser'],
                    'role': 'main_task'
                },
                'asr': {
                    'type': 'sequence_labeling_ctc',
                    'vocab_size': self.vocab_size,
                    'weight': self.loss_weights['asr'],
                    'role': 'auxiliary_task',
                    'alpha': self.alpha_asr
                },
                'prosody': {
                    'type': 'sequence_labeling',
                    'classes': self.prosody_classes,
                    'lstm_hidden': self.prosody_lstm_hidden,
                    'weight': self.loss_weights['prosody'],
                    'role': 'auxiliary_task',
                    'alpha': self.alpha_prosody
                }
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for saving"""
        return {
            'backbone_name': self.backbone_name,
            'vocab_size': self.vocab_size,
            'emotion_classes': self.emotion_classes,
            'prosody_classes': self.prosody_classes,
            'prosody_lstm_hidden': self.prosody_lstm_hidden,
            'final_dropout': self.final_dropout,
            'alpha_ser': self.alpha_ser,
            'alpha_asr': self.alpha_asr,
            'alpha_prosody': self.alpha_prosody,
            'ctc_entropy_weight': self.ctc_entropy_weight,
            'ctc_blank_penalty': self.ctc_blank_penalty,
            'ctc_blank_threshold': self.ctc_blank_threshold,
            'ctc_label_smoothing': self.ctc_label_smoothing,
            'ctc_confidence_penalty': self.ctc_confidence_penalty,
            'backbone_learning_rate': self.backbone_learning_rate,
            'task_head_learning_rate': self.task_head_learning_rate,
            'freeze_backbone_initially': self.freeze_backbone_initially,
            'loss_weights': self.loss_weights,
            'custom_backbone_config': self.custom_backbone_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MTLConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)

    @classmethod
    def create_paper_config(cls, backbone_name: str = "whisper", alpha_ser: float = 1.0,
                            alpha_asr: float = 0.1, alpha_prosody: float = 0.1) -> 'MTLConfig':
        """
        Create configuration matching paper's optimal setup.

        Paper found alpha=0.1 to be optimal for both auxiliary tasks.
        """
        return cls(
            backbone_name=backbone_name,
            vocab_size=51865,  # Default for Whisper
            alpha_ser=alpha_ser,
            alpha_asr=alpha_asr,
            alpha_prosody=alpha_prosody,
            ctc_entropy_weight=0.01,
            ctc_blank_penalty=50.0,  # Very strong penalty for lower threshold
            ctc_blank_threshold=0.3,
            ctc_label_smoothing=0.0,
            ctc_confidence_penalty=0.0
        )

    def get_ctc_config(self) -> Dict:
        """Get CTC loss configuration for debugging"""
        return {
            'entropy_weight': self.ctc_entropy_weight,
            'blank_penalty': self.ctc_blank_penalty,
            'blank_threshold': self.ctc_blank_threshold,
            'label_smoothing': self.ctc_label_smoothing,
            'confidence_penalty': self.ctc_confidence_penalty
        }
