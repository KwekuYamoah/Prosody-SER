from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoConfig,
    AutoProcessor
)
from typing import Dict, Optional, Union
import torch
import torch.nn as nn


class BackboneConfig:
    """Configuration class for backbone models"""

    def __init__(
        self,
        model_name: str,
        pretrained_model_name: str,
        freeze_encoder: bool = True,
        hidden_size: Optional[int] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.pretrained_model_name = pretrained_model_name
        self.freeze_encoder = freeze_encoder
        self.hidden_size = hidden_size

        # Store any additional kwargs separately
        self.additional_config = kwargs


# Define available backbone configurations
BACKBONE_CONFIGS = {
    "whisper": BackboneConfig(
        model_name="whisper",
        pretrained_model_name="openai/whisper-large-v3",
        # Don't set hidden_size here, let it be determined from the model
    ),
    "xlsr": BackboneConfig(
        model_name="xlsr",
        pretrained_model_name="facebook/wav2vec2-xls-r-300m",
    ),
    "mms": BackboneConfig(
        model_name="mms",
        pretrained_model_name="facebook/mms-300m",
    ),
    "wav2vec2-bert": BackboneConfig(
        model_name="wav2vec2-bert",
        pretrained_model_name="facebook/w2v-bert-2.0",
    ),
}


class BackboneModel(nn.Module):
    """Wrapper class for different backbone models using AutoClasses"""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        # Load configuration first to get actual hidden size
        try:
            self.model_config = AutoConfig.from_pretrained(
                config.pretrained_model_name,
                trust_remote_code=False
            )

            # Update hidden size from actual config
            self._update_hidden_size_from_config()
        except Exception as e:
            print(
                f"Warning: Could not load pretrained config for {config.pretrained_model_name}: {e}")
            print("Falling back to default backbone config")
            self.model_config = None

        # Load model and feature extractor using AutoClasses
        self.model = self._load_model()
        self.feature_extractor = self._load_feature_extractor()

        # Track freeze state
        self._is_frozen = False

        if config.freeze_encoder:
            self.freeze_encoder()

    def _update_hidden_size_from_config(self):
        """Update hidden size from the actual model configuration"""
        if self.model_config is None:
            return

        # Try to get hidden size from config
        if hasattr(self.model_config, 'hidden_size'):
            self.config.hidden_size = self.model_config.hidden_size
        elif hasattr(self.model_config, 'd_model'):
            self.config.hidden_size = self.model_config.d_model
        elif hasattr(self.model_config, 'encoder_embed_dim'):
            self.config.hidden_size = self.model_config.encoder_embed_dim

        # If we still don't have a hidden size, use the default from BACKBONE_CONFIGS
        if self.config.hidden_size is None:
            print(
                f"Warning: Could not determine hidden size from config for {self.config.model_name}")
            print("Using default hidden size from BACKBONE_CONFIGS")

    def get_hidden_size(self) -> int:
        """Get the hidden size of the backbone model"""
        return self.config.hidden_size

    def _load_model(self) -> nn.Module:
        """Load model using AutoModel"""
        load_kwargs = {
            "use_safetensors": True,
            "trust_remote_code": False,
        }

        try:
            return AutoModel.from_pretrained(
                self.config.pretrained_model_name,
                **load_kwargs
            )
        except Exception as e:
            if "safetensors" in str(e):
                print(f"Warning: SafeTensors format not available for {self.config.pretrained_model_name}. "
                      f"Falling back to standard format.")
                load_kwargs.pop("use_safetensors", None)
                return AutoModel.from_pretrained(
                    self.config.pretrained_model_name,
                    **load_kwargs
                )
            else:
                raise e

    def _load_feature_extractor(self):
        """Load feature extractor using AutoFeatureExtractor"""
        return AutoFeatureExtractor.from_pretrained(
            self.config.pretrained_model_name,
            trust_remote_code=False
        )

    def freeze_encoder(self):
        """Freeze the encoder parameters"""
        print("Freezing encoder parameters...")
        self._is_frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        print("Unfreezing encoder parameters...")
        self._is_frozen = False
        for param in self.model.parameters():
            param.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if encoder is frozen"""
        return self._is_frozen

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone model"""
        # Use AutoModel's forward method - it handles different architectures automatically
        outputs = self.model(input_features, output_hidden_states=True)

        # Extract last hidden state based on model type
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            return outputs.hidden_states[-1]
        else:
            # Fallback for older model formats
            if self.config.model_name == "whisper" and hasattr(self.model, 'encoder'):
                encoder_outputs = self.model.encoder(input_features)
                return encoder_outputs.last_hidden_state
            else:
                raise ValueError(
                    f"Could not extract features from {self.config.model_name}")

    def extract_features(self, audio_array: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
        """Extract features from audio using the feature extractor"""
        # This method is now primarily for compatibility
        # Feature extraction should be handled in collate_fn as requested
        features = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        # Return appropriate feature type based on model
        if hasattr(features, 'input_features'):
            return features.input_features.squeeze(0)
        elif hasattr(features, 'input_values'):
            return features.input_values.squeeze(0)
        else:
            raise ValueError(
                f"Unsupported feature format for {self.config.model_name}")
