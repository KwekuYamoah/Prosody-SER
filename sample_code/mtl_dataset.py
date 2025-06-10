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


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Return raw audio data - feature extraction will happen in the model
        audio_array = sample['audio_filepath']['array']

        # Convert to tensor if needed
        if not isinstance(audio_array, torch.Tensor):
            audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # Process words for ASR (simplified)
        words = sample['words']

        # Other targets
        prosody_annotations = torch.tensor(
            sample['prosody_annotations'], dtype=torch.float32)
        emotion = torch.tensor(sample['emotion'], dtype=torch.long)

        return {
            'audio_array': audio_array,  # Raw audio, not processed features
            'words': words,
            'prosody_annotations': prosody_annotations,
            'emotion': emotion
        }