import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoProcessor, AutoFeatureExtractor


class MTLDataset(Dataset):
    """
    Improved dataset following the paper's approach.
    Handles audio loading and basic preprocessing.
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        processor: Union[AutoProcessor, AutoFeatureExtractor],
        target_sr: int = 16000,
        max_duration: Optional[float] = None,
        emotion_label_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_list: List of data samples with audio paths and labels
            processor: HuggingFace processor/feature extractor
            target_sr: Target sampling rate
            max_duration: Maximum audio duration in seconds
            emotion_label_map: Mapping from emotion strings to integers
        """
        self.data_list = data_list
        self.processor = processor
        self.target_sr = target_sr
        self.max_duration = max_duration
        
        # Default emotion mapping if not provided
        if emotion_label_map is None:
            self.emotion_label_map = {
                'Neutral': 0, 'Confusion': 1, 'Anger': 2, 'Disgust': 3,
                'Frustration': 4, 'Sadness': 5, 'Surprise': 6, 'Joy': 7, 'Fear': 8
            }
        else:
            self.emotion_label_map = emotion_label_map
        
        # Filter by duration if needed
        if max_duration is not None:
            self.data_list = self._filter_by_duration()
    
    def _filter_by_duration(self) -> List[Dict]:
        """Filter samples by maximum duration"""
        filtered = []
        for item in self.data_list:
            try:
                info = sf.info(item['audio_filepath'])
                duration = info.duration
                if duration <= self.max_duration:
                    filtered.append(item)
            except:
                # If we can't read the file info, skip it
                continue
        
        print(f"Filtered {len(self.data_list) - len(filtered)} samples exceeding {self.max_duration}s")
        return filtered
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - speech: raw audio array
                - sampling_rate: audio sampling rate
                - text: transcription text
                - emotion: emotion label (integer)
                - prosody: prosody labels (list of 0/1)
                - words: list of words
        """
        item = self.data_list[idx]
        
        # Load audio
        speech, sr = librosa.load(item['audio_filepath'], sr=self.target_sr)
        
        # Get labels
        text = " ".join(item['words']) if 'words' in item else ""
        emotion = self.emotion_label_map.get(item['emotion'], item['emotion'])
        
        # Handle prosody annotations
        prosody = item.get('prosody_annotations', [])
        if not isinstance(prosody, list):
            prosody = [prosody]
        
        return {
            'speech': speech,
            'sampling_rate': sr,
            'text': text,
            'words': item.get('words', []),
            'emotion': emotion,
            'prosody': prosody,
            'audio_path': item['audio_filepath']
        }


class DataCollatorMTLWithPadding:
    """
    Data collator for multi-task learning following the paper's approach.
    Handles padding and processing for all three tasks.
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
        Collate batch of samples.
        
        Args:
            features: List of samples from dataset
            
        Returns:
            Dictionary with batched tensors
        """
        # Extract audio arrays and sampling rates
        speech_list = [feature["speech"] for feature in features]
        sampling_rates = [feature["sampling_rate"] for feature in features]
        
        # Check all have same sampling rate
        assert len(set(sampling_rates)) == 1, "All samples must have same sampling rate"
        
        # Process audio through feature extractor
        if hasattr(self.processor, 'feature_extractor'):
            # For models with separate feature extractor (e.g., Wav2Vec2)
            audio_features = self.processor.feature_extractor(
                speech_list,
                sampling_rate=sampling_rates[0],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
                return_attention_mask=self.return_attention_mask
            )
        else:
            # For models with unified processor (e.g., Whisper)
            audio_features = self.processor(
                speech_list,
                sampling_rate=sampling_rates[0],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
                return_attention_mask=self.return_attention_mask
            )
        
        # Prepare batch dictionary
        batch = {}
        
        # Add audio features (input_values or input_features depending on model)
        if "input_values" in audio_features:
            batch["input_values"] = audio_features.input_values
        elif "input_features" in audio_features:
            batch["input_features"] = audio_features.input_features
        
        if self.return_attention_mask and "attention_mask" in audio_features:
            batch["attention_mask"] = audio_features.attention_mask
        
        # Process labels for each task
        # 1. ASR labels (CTC)
        if self.tokenizer is not None:
            texts = [feature["text"] for feature in features]
            
            # Tokenize texts
            with self.processor.as_target_processor() if hasattr(self.processor, 'as_target_processor') else nullcontext():
                text_features = self.tokenizer(
                    texts,
                    padding=self.padding,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                )
            
            # Replace padding with -100 for CTC loss
            asr_labels = text_features.input_ids.masked_fill(
                text_features.attention_mask.ne(1), -100
            )
            asr_lengths = text_features.attention_mask.sum(-1)
        else:
            asr_labels = None
            asr_lengths = None
        
        # 2. SER labels (emotion classification)
        emotion_labels = torch.tensor([feature["emotion"] for feature in features], dtype=torch.long)
        
        # 3. Prosody labels (sequence labeling)
        # Need to pad prosody sequences to match audio length
        prosody_sequences = []
        for i, feature in enumerate(features):
            prosody = feature["prosody"]
            
            # Determine target length based on audio features
            if "input_values" in batch:
                target_len = batch["input_values"].shape[1] // 320  # Approximate frame count
            else:
                target_len = batch["input_features"].shape[-1]  # For spectrogram features
            
            # Create prosody tensor with appropriate length
            if len(prosody) == 0:
                # No prosody labels, create zeros
                prosody_tensor = torch.zeros(target_len, dtype=torch.float)
            else:
                # Interpolate prosody to match target length
                prosody_tensor = torch.tensor(prosody, dtype=torch.float)
                if len(prosody) != target_len:
                    # Simple nearest neighbor interpolation
                    indices = torch.linspace(0, len(prosody) - 1, target_len).long()
                    prosody_tensor = prosody_tensor[indices]
            
            prosody_sequences.append(prosody_tensor)
        
        # Pad prosody sequences
        max_prosody_len = max(len(seq) for seq in prosody_sequences)
        prosody_labels = torch.zeros(len(features), max_prosody_len)
        for i, seq in enumerate(prosody_sequences):
            prosody_labels[i, :len(seq)] = seq
        
        # Combine labels into tuple (following paper's approach)
        if asr_labels is not None:
            batch["labels"] = ((asr_labels, asr_lengths), emotion_labels, prosody_labels)
        else:
            batch["labels"] = (None, emotion_labels, prosody_labels)
        
        return batch


# Utility context manager for cases without target processor
from contextlib import nullcontext