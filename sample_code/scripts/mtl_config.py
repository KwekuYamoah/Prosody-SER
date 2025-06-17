from dataclasses import dataclass
from typing import Dict, Optional, Any
from transformers import AutoConfig, PretrainedConfig
from sample_code.scripts.backbone_models import BackboneConfig, BACKBONE_CONFIGS


class MTLConfig(PretrainedConfig):
    """Configuration class for MTL system that extends pretrained model configs"""
    model_type = "mtl"

    def __init__(
        self,
        backbone_name: str = "whisper",  # Default to whisper
        vocab_size: int = 51865,
        emotion_classes: int = 9,
        prosody_classes: int = 2,
        prosody_lstm_hidden: int = 256,
        final_dropout: float = 0.1,
        freeze_encoder: bool = True,
        alpha_asr: float = 0.1,      # Weight for ASR auxiliary task
        alpha_prosody: float = 0.1,  # Weight for Prosody auxiliary task
        custom_backbone_config: Optional[Dict[str, Any]] = None,
        ctc_entropy_weight: float = 0.01,  # Entropy regularization weight
        ctc_blank_weight: float = 0.95,    # Maximum allowed blank probability
        warmup_steps: int = 1000,          # Warmup steps for ASR head
        asr_lr_multiplier: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store all parameters
        self.backbone_name = backbone_name
        self.vocab_size = vocab_size
        self.emotion_classes = emotion_classes
        self.prosody_classes = prosody_classes
        self.prosody_lstm_hidden = prosody_lstm_hidden
        self.final_dropout = final_dropout
        self.freeze_encoder = freeze_encoder
        self.alpha_asr = alpha_asr
        self.alpha_prosody = alpha_prosody
        self.custom_backbone_config = custom_backbone_config
        self.ctc_entropy_weight = ctc_entropy_weight
        self.ctc_blank_weight = ctc_blank_weight
        self.warmup_steps = warmup_steps
        self.asr_lr_multiplier = asr_lr_multiplier

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
        if not 0 <= self.ctc_entropy_weight <= 1:
            raise ValueError("ctc_entropy_weight must be between 0 and 1")
        if not 0 <= self.ctc_blank_weight <= 1:
            raise ValueError("ctc_blank_weight must be between 0 and 1")
        if self.alpha_asr < 0 or self.alpha_prosody < 0:
            raise ValueError("Alpha values must be non-negative")

        # Load and extend pretrained model configuration
        self._setup_backbone_config()

        # Set loss weights following paper's approach: SER=1.0, others controlled by alpha
        self.loss_weights = {
            # Main task (emotion) always has weight 1
            'ser': 1.0,
            'asr': self.alpha_asr,         # Auxiliary task controlled by alpha
            'prosody': self.alpha_prosody  # Auxiliary task controlled by alpha
        }

    def _setup_backbone_config(self):
        """Setup backbone configuration by extending pretrained model config"""
        if self.custom_backbone_config is not None:
            self.backbone_config = BackboneConfig(
                **self.custom_backbone_config)
        else:
            # Get backbone config from predefined configs
            if self.backbone_name not in BACKBONE_CONFIGS:
                raise ValueError(f"Unsupported backbone: {self.backbone_name}")

            base_config = BACKBONE_CONFIGS[self.backbone_name]

            # Load the pretrained model's configuration
            try:
                pretrained_config = AutoConfig.from_pretrained(
                    base_config.pretrained_model_name,
                    trust_remote_code=False
                )

                # Extract relevant parameters from pretrained config
                pretrained_params = self._extract_pretrained_params(
                    pretrained_config)

                # Create enhanced backbone config with pretrained parameters
                # Fix: Only pass parameters that aren't already in base_config to avoid duplicates
                config_params = {
                    'model_name': base_config.model_name,
                    'pretrained_model_name': base_config.pretrained_model_name,
                    'freeze_encoder': self.freeze_encoder,
                }
                
                # Add pretrained parameters, avoiding conflicts with base config
                for key, value in pretrained_params.items():
                    if not hasattr(base_config, key) or key == 'hidden_size':
                        # Always override hidden_size with pretrained value
                        config_params[key] = value

                self.backbone_config = BackboneConfig(**config_params)

                # Store the full pretrained config for reference
                self.pretrained_config = pretrained_config

            except Exception as e:
                print(
                    f"Warning: Could not load pretrained config for {base_config.pretrained_model_name}: {e}")
                print("Falling back to default backbone config")
                self.backbone_config = base_config
                self.pretrained_config = None

    def _extract_pretrained_params(self, pretrained_config) -> Dict[str, Any]:
        """Extract relevant parameters from pretrained model configuration"""
        params = {}

        # Model-specific parameter extraction based on backbone type
        if self.backbone_name == "whisper":
            # Whisper uses d_model instead of hidden_size
            if hasattr(pretrained_config, 'd_model'):
                params['hidden_size'] = pretrained_config.d_model
            
            # Extract Whisper-specific parameters
            whisper_params = {
                'encoder_layers': 'encoder_layers',
                'decoder_layers': 'decoder_layers', 
                'encoder_attention_heads': 'encoder_attention_heads',
                'decoder_attention_heads': 'decoder_attention_heads',
                'encoder_ffn_dim': 'encoder_ffn_dim',
                'decoder_ffn_dim': 'decoder_ffn_dim',
                'dropout': 'dropout',
                'attention_dropout': 'attention_dropout',
                'activation_dropout': 'activation_dropout',
                'max_source_positions': 'max_source_positions',
                'vocab_size': 'vocab_size'
            }
            
            for param_key, config_key in whisper_params.items():
                if hasattr(pretrained_config, config_key):
                    params[param_key] = getattr(pretrained_config, config_key)

        elif self.backbone_name in ["wav2vec2", "xlsr", "mms"]:
            # These models use hidden_size directly
            if hasattr(pretrained_config, 'hidden_size'):
                params['hidden_size'] = pretrained_config.hidden_size
            
            # Extract Wav2Vec2-specific parameters
            wav2vec_params = {
                'num_hidden_layers': 'num_hidden_layers',
                'num_attention_heads': 'num_attention_heads',
                'intermediate_size': 'intermediate_size',
                'hidden_dropout': 'hidden_dropout',
                'attention_dropout': 'attention_dropout',
                'final_dropout': 'final_dropout',
                'layer_norm_eps': 'layer_norm_eps',
                'conv_dim': 'conv_dim',
                'conv_stride': 'conv_stride',
                'conv_kernel': 'conv_kernel',
                'conv_bias': 'conv_bias',
                'num_conv_pos_embeddings': 'num_conv_pos_embeddings',
                'num_conv_pos_embedding_groups': 'num_conv_pos_embedding_groups',
                'vocab_size': 'vocab_size'
            }
            
            for param_key, config_key in wav2vec_params.items():
                if hasattr(pretrained_config, config_key):
                    params[param_key] = getattr(pretrained_config, config_key)

        elif self.backbone_name == "wav2vec2-bert":
            # Wav2Vec2-BERT uses hidden_size
            if hasattr(pretrained_config, 'hidden_size'):
                params['hidden_size'] = pretrained_config.hidden_size
            
            # Extract Wav2Vec2-BERT-specific parameters
            bert_params = {
                'num_hidden_layers': 'num_hidden_layers',
                'num_attention_heads': 'num_attention_heads', 
                'intermediate_size': 'intermediate_size',
                'hidden_dropout': 'hidden_dropout',
                'attention_dropout': 'attention_dropout',
                'final_dropout': 'final_dropout',
                'layer_norm_eps': 'layer_norm_eps',
                'feature_projection_input_dim': 'feature_projection_input_dim',
                'position_embeddings_type': 'position_embeddings_type',
                'max_source_positions': 'max_source_positions',
                'vocab_size': 'vocab_size'
            }
            
            for param_key, config_key in bert_params.items():
                if hasattr(pretrained_config, config_key):
                    params[param_key] = getattr(pretrained_config, config_key)

        return params

    def get_pretrained_config_summary(self) -> Dict[str, Any]:
        """Get summary of the extended pretrained configuration"""
        if self.pretrained_config is None:
            return {"status": "No pretrained config loaded"}

        summary = {
            "model_type": getattr(self.pretrained_config, 'model_type', 'unknown'),
            "architectures": getattr(self.pretrained_config, 'architectures', []),
        }
        
        # Get hidden size based on model type
        if self.backbone_name == "whisper":
            summary["hidden_size"] = getattr(self.pretrained_config, 'd_model', 'unknown')
        else:
            summary["hidden_size"] = getattr(self.pretrained_config, 'hidden_size', 'unknown')
            
        summary["vocab_size"] = getattr(self.pretrained_config, 'vocab_size', 'unknown')

        return summary

    def update_alpha_weights(self, alpha_asr: float, alpha_prosody: float):
        """Update alpha values and corresponding loss weights"""
        if alpha_asr < 0 or alpha_prosody < 0:
            raise ValueError("Alpha values must be non-negative")

        self.alpha_asr = alpha_asr
        self.alpha_prosody = alpha_prosody

        # Update loss weights
        self.loss_weights = {
            'ser': 1.0,                    # Main task always 1.0
            'asr': self.alpha_asr,         # Auxiliary task
            'prosody': self.alpha_prosody  # Auxiliary task
        }

    def get_total_auxiliary_weight(self) -> float:
        """Get combined weight of auxiliary tasks"""
        return self.alpha_asr + self.alpha_prosody

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration"""
        summary = {
            'backbone': {
                'name': self.backbone_name,
                'pretrained_model': self.backbone_config.pretrained_model_name,
                'hidden_size': getattr(self.backbone_config, 'hidden_size', 'unknown'),
                'freeze_encoder': self.freeze_encoder,
                'pretrained_config_summary': self.get_pretrained_config_summary()
            },
            'tasks': {
                'ser': {
                    'num_classes': self.emotion_classes,
                    'loss_weight': 1.0,  # Always main task
                    'role': 'main_task'
                },
                'asr': {
                    'vocab_size': self.vocab_size,
                    'alpha': self.alpha_asr,
                    'loss_weight': self.alpha_asr,
                    'entropy_weight': self.ctc_entropy_weight,
                    'blank_weight': self.ctc_blank_weight,
                    'role': 'auxiliary_task'
                },
                'prosody': {
                    'num_classes': self.prosody_classes,
                    'lstm_hidden': self.prosody_lstm_hidden,
                    'alpha': self.alpha_prosody,
                    'loss_weight': self.alpha_prosody,
                    'role': 'auxiliary_task'
                }
            },
            'training': {
                'final_dropout': self.final_dropout,
                'warmup_steps': self.warmup_steps,
                'asr_lr_multiplier': self.asr_lr_multiplier,
                'total_auxiliary_weight': self.get_total_auxiliary_weight()
            }
        }

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for saving"""
        config_dict = {
            'backbone_name': self.backbone_name,
            'vocab_size': self.vocab_size,
            'emotion_classes': self.emotion_classes,
            'prosody_classes': self.prosody_classes,
            'prosody_lstm_hidden': self.prosody_lstm_hidden,
            'final_dropout': self.final_dropout,
            'freeze_encoder': self.freeze_encoder,
            'alpha_asr': self.alpha_asr,
            'alpha_prosody': self.alpha_prosody,
            'custom_backbone_config': self.custom_backbone_config,
            'ctc_entropy_weight': self.ctc_entropy_weight,
            'ctc_blank_weight': self.ctc_blank_weight,
            'warmup_steps': self.warmup_steps,
            'asr_lr_multiplier': self.asr_lr_multiplier
        }

        # Add pretrained config if available
        if hasattr(self, 'pretrained_config') and self.pretrained_config is not None:
            config_dict['pretrained_config_summary'] = self.get_pretrained_config_summary()

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MTLConfig':
        """Create configuration from dictionary"""
        # Remove pretrained_config_summary if present (it's derived)
        config_dict = config_dict.copy()
        config_dict.pop('pretrained_config_summary', None)
        return cls(**config_dict)