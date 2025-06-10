from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    WhisperModel,
    Wav2Vec2Model,
    Wav2Vec2BertModel
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
        hidden_size: int,
        freeze_encoder: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.pretrained_model_name = pretrained_model_name
        self.hidden_size = hidden_size
        self.freeze_encoder = freeze_encoder
        self.kwargs = kwargs


# Define available backbone configurations
BACKBONE_CONFIGS = {
    "whisper": BackboneConfig(
        model_name="whisper",
        pretrained_model_name="openai/whisper-large-v3",
        hidden_size=1280,
    ),
    "xlsr": BackboneConfig(
        model_name="xlsr",
        pretrained_model_name="facebook/wav2vec2-xls-r-300m",
        hidden_size=1024,
    ),
    "mms": BackboneConfig(
        model_name="mms",
        pretrained_model_name="facebook/mms-300m",
        hidden_size=1024,
    ),
    "wav2vec2-bert": BackboneConfig(
        model_name="wav2vec2-bert",
        pretrained_model_name="facebook/w2v-bert-2.0",
        hidden_size=1024,
    ),
}


class BackboneModel(nn.Module):
    """Wrapper class for different backbone models"""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.model = self._load_model()
        self.feature_extractor = self._load_feature_extractor()

        if config.freeze_encoder:
            self._freeze_encoder()

    def get_hidden_size(self) -> int:
        """Get the hidden size of the backbone model"""
        return self.config.hidden_size

    def _load_model(self) -> nn.Module:
        """Load the appropriate model based on configuration"""
        # Common kwargs to prefer safetensors format
        load_kwargs = {
            "use_safetensors": True,  # Prefer safetensors format
            "trust_remote_code": False,  # Security best practice
        }

        try:
            if self.config.model_name == "whisper":
                return WhisperModel.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
            elif self.config.model_name == "xlsr":
                return Wav2Vec2Model.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
            elif self.config.model_name == "mms":
                return Wav2Vec2Model.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
            elif self.config.model_name == "wav2vec2-bert":
                return Wav2Vec2BertModel.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
            else:
                raise ValueError(
                    f"Unsupported model name: {self.config.model_name}")
        except Exception as e:
            if "safetensors" in str(e):
                # If safetensors not available, try without it but warn the user
                print(f"Warning: SafeTensors format not available for {self.config.pretrained_model_name}. "
                      f"Consider upgrading torch to >=2.6 or using a different model.")
                # Remove use_safetensors and retry
                load_kwargs.pop("use_safetensors", None)

                if self.config.model_name == "whisper":
                    return WhisperModel.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
                elif self.config.model_name == "xlsr":
                    return Wav2Vec2Model.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
                elif self.config.model_name == "mms":
                    return Wav2Vec2Model.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
                elif self.config.model_name == "wav2vec2-bert":
                    return Wav2Vec2BertModel.from_pretrained(self.config.pretrained_model_name, **load_kwargs)
            else:
                raise e

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor"""
        return AutoFeatureExtractor.from_pretrained(self.config.pretrained_model_name)

    def _freeze_encoder(self):
        """Freeze the encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone model"""
        if self.config.model_name == "whisper":
            outputs = self.model.encoder(input_features)
            return outputs.last_hidden_state
        elif self.config.model_name in ["xlsr", "mms", "wav2vec2-bert"]:
            outputs = self.model(input_features)
            return outputs.last_hidden_state
        else:
            raise ValueError(
                f"Unsupported model name: {self.config.model_name}")

    def extract_features(self, audio_array: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
        """Extract features from audio using the feature extractor"""
        features = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        if self.config.model_name in ["whisper", "wav2vec2-bert"]:
            return features.input_features.squeeze(0)
        elif self.config.model_name in ["xlsr", "mms"]:
            return features.input_values.squeeze(0)
        else:
            raise ValueError(
                f"Unsupported model name: {self.config.model_name}")
