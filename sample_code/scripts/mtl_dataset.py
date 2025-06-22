import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import numpy as np

from sample_code.scripts.mtl_config import MTLConfig


class MTLDataset(Dataset):
    """
    Dataset for Multi-Task Learning with proper feature extraction.
    This version does NOT truncate audio to preserve ASR quality.
    """

    def __init__(
        self,
        dataset,
        config: MTLConfig,
        feature_extractor,
        sampling_rate: int = 16000,
    ):
        """
        Initializes the dataset.

        Args:
            dataset: A Hugging Face dataset split object.
            config: The MTLConfig object for task configuration.
            feature_extractor: The pre-initialized feature extractor from the backbone model.
            sampling_rate: The target sampling rate for audio processing.
        """
        self.dataset = dataset
        self.config = config
        self.sampling_rate = sampling_rate
        self.feature_extractor = feature_extractor
        self.backbone_name = config.backbone_name

        if not self.feature_extractor:
            raise ValueError(
                "A pre-initialized feature_extractor must be provided to MTLDataset."
            )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves and processes a single data sample.
        Fixed to handle wav2vec2 models correctly without truncating audio.
        """
        # Get sample
        sample = self.dataset[idx]

        # Get audio array
        audio_info = sample["audio_filepath"]
        if isinstance(audio_info, dict) and "array" in audio_info:
            audio_array = audio_info["array"]
        else:
            # Handle case where audio might be stored differently
            audio_array = audio_info

        # Ensure audio is numpy array and 1D
        if torch.is_tensor(audio_array):
            audio_array = audio_array.numpy()

        if isinstance(audio_array, np.ndarray):
            audio_array = audio_array.astype(np.float32)
            # CRITICAL: Ensure audio is 1D
            while audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

        # Normalize audio for wav2vec2 models
        if self.backbone_name in ["xlsr", "mms", "wav2vec2-bert"]:
            # Simple normalization to prevent overflow
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val

        # Extract features - DO NOT return tensors yet
        with torch.no_grad():
            if self.backbone_name in ["whisper", "wav2vec2-bert"]:
                # Whisper, Wave2Vec-Bert feature extractor returns log-mel spectrograms
                features = self.feature_extractor(
                    audio_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                )
                if self.backbone_name == "wav2vec2-bert":
                    input_features = features.input_features.squeeze(0)

                    # Double-check it's 1D
                    if input_features.ndim != 1:
                        input_features = input_features.squeeze()
                else:
                    # Shape: (1, n_mels, time) -> (n_mels, time)
                    input_features = features.input_features.squeeze(0)

            elif self.backbone_name in ["xlsr", "mms"]:
                # Wav2Vec2 feature extractor returns normalized waveforms
                # IMPORTANT: Do not return tensors here, process raw audio
                features = self.feature_extractor(
                    audio_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=False  # No padding in dataset
                )
                # Shape: (1, time) -> (time,)
                input_features = features.input_values.squeeze(0)

                # Double-check it's 1D
                if input_features.ndim != 1:
                    input_features = input_features.squeeze()
            else:
                raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # Prepare targets
        words = sample["words"]
        asr_target = words  # List of words for later tokenization

        # Handle prosody targets
        prosody_annotations = sample["prosody_annotations"]
        if isinstance(prosody_annotations, list):
            prosody_targets = torch.tensor(
                prosody_annotations, dtype=torch.float32)
        else:
            prosody_targets = torch.tensor(
                [prosody_annotations], dtype=torch.float32)

        # Emotion target
        emotion_targets = torch.tensor(sample["emotion"], dtype=torch.long)

        return {
            "input_features": input_features,
            "words": words,
            "asr_target": asr_target,
            "prosody_targets": prosody_targets,
            "emotion_targets": emotion_targets,
        }