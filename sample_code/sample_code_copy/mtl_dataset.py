import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from backbone_models import BackboneModel
from mtl_config import MTLConfig

class MTLDataset(Dataset):
    """Enhanced dataset class with dynamic backbone feature extraction"""

    def __init__(
        self,
        dataset_dict: Dict,
        split: str = "train",
        config: Optional[MTLConfig] = None,
        max_length: int = 512,
        sampling_rate: int = 16000
    ):
        self.dataset = dataset_dict[split]
        self.config = config
        self.max_length = max_length
        self.sampling_rate = sampling_rate

        # Initialize backbone model for feature extraction
        if config is not None:
            self.backbone = BackboneModel(config.backbone_config)
        else:
            raise ValueError("MTLConfig must be provided")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Extract audio features using backbone
        audio_array = sample['audio_filepath']['array']
        input_features = self.backbone.extract_features(
            audio_array,
            sampling_rate=self.sampling_rate
        )

        # Process words for ASR
        words = sample['words']
        if hasattr(self.backbone, 'tokenizer'):
            token_ids = self.backbone.tokenizer.encode_words(words, add_special_tokens=True)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            asr_target = torch.tensor(token_ids, dtype=torch.long)
            asr_length = torch.tensor(len(token_ids), dtype=torch.long)
        else:
            asr_target = words
            asr_length = torch.tensor(len(words), dtype=torch.long)

        # Other targets
        prosody_annotations = torch.tensor(sample['prosody_annotations'], dtype=torch.float32)
        emotion = torch.tensor(sample['emotion'], dtype=torch.long)

        return {
            'input_features': input_features,
            'words': words,
            'asr_target': asr_target,
            'asr_length': asr_length,
            'prosody_annotations': prosody_annotations,
            'emotion': emotion
        } 