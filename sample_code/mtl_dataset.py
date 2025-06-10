import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Union
from backbone_models import BackboneModel
from mtl_config import MTLConfig


class MTLDataset(Dataset):
    """Enhanced dataset class with lazy feature extraction"""

    def __init__(
        self,
        dataset_dict: Dict,
        split: str = "train",
        config: Optional[MTLConfig] = None,
        max_length: int = 512,
        sampling_rate: int = 16000,
        feature_extractor=None  # Pass feature extractor instead of full model
    ):
        self.dataset = dataset_dict[split]
        self.config = config
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.split = split

        # Store only the feature extractor, not the full backbone model
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        elif config is not None:
            # Create only the feature extractor, not the full model
            from backbone_models import BackboneModel
            backbone = BackboneModel(config.backbone_config)
            self.feature_extractor = backbone.feature_extractor
            # Delete the backbone model to save memory
            del backbone
        else:
            raise ValueError(
                "Either feature_extractor or MTLConfig must be provided")

        # Cache for storing extracted features (optional, can be disabled)
        self.use_cache = False  # Set to True only if you have enough RAM
        self.feature_cache = {} if self.use_cache else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Check cache first if enabled
        if self.use_cache and idx in self.feature_cache:
            input_features = self.feature_cache[idx]
        else:
            # Extract audio features on-the-fly
            audio_array = sample['audio_filepath']['array']

            # Use feature extractor directly (no model forward pass)
            with torch.no_grad():
                features = self.feature_extractor(
                    audio_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                )

                # Extract the appropriate features based on model type
                if hasattr(features, 'input_features'):
                    input_features = features.input_features.squeeze(0)
                else:
                    input_features = features.input_values.squeeze(0)

            # Cache if enabled
            if self.use_cache:
                self.feature_cache[idx] = input_features

        # Process words for ASR
        words = sample['words']

        # For ASR targets, we'll handle tokenization in the collate function
        # to avoid storing tokenizer in dataset
        asr_target = words  # Keep as words, tokenize later
        asr_length = torch.tensor(len(words), dtype=torch.long)

        # Other targets
        prosody_annotations = torch.tensor(
            sample['prosody_annotations'], dtype=torch.float32)
        emotion = torch.tensor(sample['emotion'], dtype=torch.long)

        return {
            'input_features': input_features,
            'words': words,
            'asr_target': asr_target,
            'asr_length': asr_length,
            'prosody_annotations': prosody_annotations,
            'emotion': emotion
        }

    def clear_cache(self):
        """Clear the feature cache to free memory"""
        if self.feature_cache is not None:
            self.feature_cache.clear()
