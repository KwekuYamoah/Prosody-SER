import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any
import numpy as np
from transformers import AutoProcessor, AutoFeatureExtractor

from contextlib import nullcontext


class MTLDataset(Dataset):
    """
    Improved dataset that works with preprocessed audio data.
    Much more memory efficient as audio is already loaded and processed.
    """

    def __init__(
        self,
        data_list: List[Dict],
        config,
        feature_extractor: Union[AutoProcessor, AutoFeatureExtractor],
        emotion_label_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_list: List of data samples with preprocessed audio
            config: MTLConfig object
            feature_extractor: HuggingFace processor/feature extractor
            emotion_label_map: Mapping from emotion strings to integers
        """
        self.data_list = data_list
        self.config = config
        self.feature_extractor = feature_extractor

        # Default emotion mapping if not provided
        if emotion_label_map is None:
            self.emotion_label_map = {
                'Neutral': 0, 'Confusion': 1, 'Anger': 2, 'Disgust': 3,
                'Frustration': 4, 'Sadness': 5, 'Surprise': 6, 'Joy': 7, 'Fear': 8
            }
        else:
            self.emotion_label_map = emotion_label_map

        print(f"Created dataset with {len(self.data_list)} samples")

        # Verify data structure
        print("\nVerifying data structure...")
        sample = self.data_list[0]
        print("Available keys in data:", list(sample.keys()))

        # Verify emotion labels
        emotion_counts = {}
        for item in self.data_list:
            emotion = item.get('emotion', 'Unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\nEmotion label distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples")

        # Verify prosody annotations
        prosody_count = sum(
            1 for item in self.data_list if 'prosody_annotations' in item)
        print(
            f"\nSamples with prosody annotations: {prosody_count}/{len(self.data_list)}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with preprocessed audio.
        """
        item = self.data_list[idx]

        # Get preprocessed audio data
        speech = item['audio_data']
        sr = item.get('audio_sr', 16000)  # Default to 16kHz if not specified

        # Ensure speech is numpy array
        if not isinstance(speech, np.ndarray):
            speech = np.array(speech, dtype=np.float32)

        # Process through feature extractor
        if hasattr(self.feature_extractor, 'feature_extractor'):
            # For models with separate feature extractor
            features = self.feature_extractor.feature_extractor(
                speech,
                sampling_rate=sr,
                return_tensors="pt"
            )
        else:
            # For models with unified processor
            features = self.feature_extractor(
                speech,
                sampling_rate=sr,
                return_tensors="pt"
            )

        # Extract the appropriate feature type
        if "input_values" in features:
            input_features = features.input_values.squeeze(0)
        elif "input_features" in features:
            input_features = features.input_features.squeeze(0)
        else:
            raise ValueError("Unknown feature format")

        # Get labels
        text = " ".join(item['words']) if 'words' in item else ""
        # Use emotion value directly since it's already encoded
        emotion = item['emotion']

        # Handle prosody annotations
        prosody = item.get('prosody_annotations', [])
        if not isinstance(prosody, list):
            prosody = [prosody]

        return {
            'input_features': input_features,
            'asr_target': item.get('words', []),
            'emotion_targets': emotion,
            'prosody_targets': torch.tensor(prosody, dtype=torch.float32),
            'words': item.get('words', []),
            'audio_duration': item.get('audio_duration', len(speech) / sr)
        }


class DataCollatorMTLWithPadding:
    """
    Data collator for multi-task learning that works with preprocessed features.
    Compatible with the updated MTLDataset that returns already extracted features.
    """

    def __init__(
        self,
        processor: Union[AutoProcessor, AutoFeatureExtractor],
        tokenizer: Optional[Any] = None,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: bool = True
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples with preprocessed features.

        Args:
            features: List of samples from MTLDataset (with input_features already extracted)

        Returns:
            Dictionary with batched tensors
        """
        # Prepare batch dictionary
        batch = {}

        # Extract input features (already preprocessed by MTLDataset)
        input_features_list = [feature["input_features"]
                               for feature in features]

        # Determine the type of features and pad accordingly
        # 2D features (e.g., mel-spectrogram for Whisper)
        if len(input_features_list[0].shape) == 2:
            # Find max dimensions
            max_freq = max(feat.shape[0] for feat in input_features_list)
            max_time = max(feat.shape[1] for feat in input_features_list)

            # Create padded tensor
            batch_size = len(features)
            padded_features = torch.zeros(batch_size, max_freq, max_time)
            attention_mask = torch.zeros(
                batch_size, max_time) if self.return_attention_mask else None

            # Fill with actual features
            for i, feat in enumerate(input_features_list):
                freq_dim, time_dim = feat.shape
                padded_features[i, :freq_dim, :time_dim] = feat
                if attention_mask is not None:
                    attention_mask[i, :time_dim] = 1

            batch["input_features"] = padded_features

        # 1D features (e.g., raw waveform for wav2vec2)
        elif len(input_features_list[0].shape) == 1:
            # Pad sequences
            max_length = max(feat.shape[0] for feat in input_features_list)

            # Apply pad_to_multiple_of if specified
            if self.pad_to_multiple_of is not None:
                max_length = ((max_length + self.pad_to_multiple_of - 1) //
                              self.pad_to_multiple_of) * self.pad_to_multiple_of

            batch_size = len(features)
            padded_features = torch.zeros(batch_size, max_length)
            attention_mask = torch.ones(
                batch_size, max_length) if self.return_attention_mask else None

            # Fill with actual features
            for i, feat in enumerate(input_features_list):
                length = feat.shape[0]
                padded_features[i, :length] = feat
                if attention_mask is not None:
                    attention_mask[i, length:] = 0

            batch["input_values"] = padded_features

        else:
            raise ValueError(
                f"Unexpected feature shape: {input_features_list[0].shape}")

        if self.return_attention_mask and attention_mask is not None:
            batch["attention_mask"] = attention_mask

        # Process labels for each task
        # 1. ASR labels (CTC)
        if self.tokenizer is not None:
            # Get text from words
            texts = []
            for feature in features:
                words = feature.get("asr_target", feature.get("words", []))
                text = " ".join(words) if isinstance(
                    words, list) else str(words)
                texts.append(text)

            # Tokenize texts
            tokenized = []
            for text in texts:
                token_ids = self.tokenizer.encode(
                    text, add_special_tokens=True)
                tokenized.append(torch.tensor(token_ids, dtype=torch.long))

            # Pad sequences
            if tokenized:
                asr_labels = torch.nn.utils.rnn.pad_sequence(
                    tokenized, batch_first=True, padding_value=-100
                )
                asr_lengths = torch.tensor(
                    [len(t) for t in tokenized], dtype=torch.long)
            else:
                asr_labels = None
                asr_lengths = None
        else:
            asr_labels = None
            asr_lengths = None

        # 2. SER labels (emotion classification)
        emotion_labels = torch.tensor(
            [feature.get("emotion_targets", 0) for feature in features],
            dtype=torch.long
        )

        # 3. Prosody labels (sequence labeling)
        prosody_sequences = []
        for i, feature in enumerate(features):
            # Get prosody targets with default empty list
            prosody = feature.get("prosody_targets", [])

            # Ensure it's a tensor
            if not isinstance(prosody, torch.Tensor):
                prosody = torch.tensor(prosody, dtype=torch.float32)

            # Determine target length based on input features
            if "input_values" in batch:
                # For wav2vec2-like models, approximate frame count
                target_len = batch["input_values"].shape[1] // 320
            else:
                # For whisper-like models, use time dimension
                target_len = batch["input_features"].shape[-1]

            # Adjust prosody length to match target
            if len(prosody) == 0:
                # No prosody labels, create zeros
                prosody_tensor = torch.zeros(target_len, dtype=torch.float32)
            else:
                prosody_tensor = prosody
                if len(prosody) != target_len:
                    # Simple nearest neighbor interpolation
                    if len(prosody) > 0:
                        indices = torch.linspace(
                            0, len(prosody) - 1, target_len).long()
                        indices = torch.clamp(indices, 0, len(prosody) - 1)
                        prosody_tensor = prosody[indices]
                    else:
                        prosody_tensor = torch.zeros(
                            target_len, dtype=torch.float32)

            prosody_sequences.append(prosody_tensor)

        # Pad prosody sequences
        if prosody_sequences:
            max_prosody_len = max(len(seq) for seq in prosody_sequences)
            prosody_labels = torch.zeros(
                len(features), max_prosody_len, dtype=torch.float32)
            for i, seq in enumerate(prosody_sequences):
                prosody_labels[i, :len(seq)] = seq
        else:
            prosody_labels = torch.zeros(len(features), 1, dtype=torch.float32)

        # Combine labels into tuple (following paper's approach)
        if asr_labels is not None:
            batch["labels"] = ((asr_labels, asr_lengths),
                               emotion_labels, prosody_labels)
        else:
            batch["labels"] = (None, emotion_labels, prosody_labels)

        return batch
