from dataclasses import dataclass
from typing import Dict, Optional, Any
from backbone_models import BackboneConfig, BACKBONE_CONFIGS

@dataclass
class MTLConfig:
    """Configuration class for MTL system"""
    backbone_name: str = "whisper"  # Default to whisper
    vocab_size: int = 51865
    emotion_classes: int = 9
    prosody_classes: int = 2
    prosody_lstm_hidden: int = 256
    final_dropout: float = 0.1
    freeze_encoder: bool = True
    loss_weights: Optional[Dict[str, float]] = None
    custom_backbone_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Validate numeric parameters
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.emotion_classes <= 0:
            raise ValueError("emotion_classes must be positive")
        if self.prosody_classes <= 0:
            raise ValueError("prosody_classes must be positive")
        if self.prosody_lstm_hidden <= 0:
            raise ValueError("prosody_lstm_hidden must be positive")
        if not 0 <= self.final_dropout <= 1:
            raise ValueError("final_dropout must be between 0 and 1")

        # Handle custom backbone configuration
        if self.custom_backbone_config is not None:
            self.backbone_config = BackboneConfig(**self.custom_backbone_config)
        else:
            # Get backbone config from predefined configs
            if self.backbone_name not in BACKBONE_CONFIGS:
                raise ValueError(f"Unsupported backbone: {self.backbone_name}")
            self.backbone_config = BACKBONE_CONFIGS[self.backbone_name]
        
        # Set loss weights if not provided
        if self.loss_weights is None:
            self.loss_weights = {
                'asr': 1.0,
                'prosody': 1.0,
                'ser': 1.0
            }
        else:
            # Validate loss weights
            required_tasks = {'asr', 'prosody', 'ser'}
            if not all(task in self.loss_weights for task in required_tasks):
                raise ValueError(f"Loss weights must include all tasks: {required_tasks}")
            if not all(weight >= 0 for weight in self.loss_weights.values()):
                raise ValueError("All loss weights must be non-negative")

    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights with validation"""
        # Validate new weights
        required_tasks = {'asr', 'prosody', 'ser'}
        if not all(task in new_weights for task in required_tasks):
            raise ValueError(f"Loss weights must include all tasks: {required_tasks}")
        if not all(weight >= 0 for weight in new_weights.values()):
            raise ValueError("All loss weights must be non-negative")
        
        self.loss_weights = new_weights

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        return {
            'backbone': {
                'name': self.backbone_name,
                'hidden_size': self.backbone_config.hidden_size,
                'freeze_encoder': self.freeze_encoder
            },
            'tasks': {
                'asr': {
                    'vocab_size': self.vocab_size,
                    'loss_weight': self.loss_weights['asr']
                },
                'prosody': {
                    'num_classes': self.prosody_classes,
                    'lstm_hidden': self.prosody_lstm_hidden,
                    'loss_weight': self.loss_weights['prosody']
                },
                'ser': {
                    'num_classes': self.emotion_classes,
                    'loss_weight': self.loss_weights['ser']
                }
            },
            'training': {
                'final_dropout': self.final_dropout
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
            'freeze_encoder': self.freeze_encoder,
            'loss_weights': self.loss_weights,
            'custom_backbone_config': self.custom_backbone_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MTLConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict) 