"""
Training Utilities
Helper functions for data processing, collation, and serialization
"""

import torch
import numpy as np
import json
from typing import List, Dict


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def collate_fn_mtl(batch: List[Dict], pad_token_id: int = 0, tokenizer=None, backbone_name: str = "whisper") -> Dict:
    """
    Custom collate function for MTL.
    Properly handles different backbone input shapes by finding
    the max size of all dimensions in the batch.
    """
    batch_size = len(batch)

    # Handle input features based on backbone type
    if backbone_name in ["whisper", "wav2vec2-bert"]:
        # Find the max dimensions for both frequency and time across the entire batch.
        max_n_mels = max(item['input_features'].shape[0] for item in batch)
        max_time = max(item['input_features'].shape[1] for item in batch)

        # Create a padded tensor using the maximum dimensions found.
        input_features = torch.zeros(batch_size, max_n_mels, max_time)

        # Copy each item into the correctly-sized padded tensor.
        for i, item in enumerate(batch):
            n_mels, time_len = item['input_features'].shape
            input_features[i, :n_mels, :time_len] = item['input_features']

    elif backbone_name in ["xlsr", "mms"]:
        # Handle 1D audio features
        max_len = max(item['input_features'].shape[0] for item in batch)
        input_features = torch.zeros(batch_size, max_len)
        for i, item in enumerate(batch):
            feat = item['input_features'].flatten()
            length = feat.shape[0]
            input_features[i, :length] = feat
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Pad prosody targets
    max_prosody_len = max(item['prosody_targets'].shape[0] for item in batch)
    prosody_targets = torch.zeros(batch_size, max_prosody_len)
    for i, item in enumerate(batch):
        prosody_len = item['prosody_targets'].shape[0]
        prosody_targets[i, :prosody_len] = item['prosody_targets']

    # Stack emotion targets
    emotion_targets = torch.tensor(
        [item['emotion_targets'] for item in batch], dtype=torch.long
    )

    # Collect words
    words_batch = [item['words'] for item in batch]

    # Tokenize and pad ASR targets
    if tokenizer:
        tokenized_ids = [
            torch.tensor(tokenizer.encode(
                " ".join(item['asr_target']), add_special_tokens=True), dtype=torch.long)
            for item in batch
        ]
        asr_targets = torch.nn.utils.rnn.pad_sequence(
            tokenized_ids, batch_first=True, padding_value=pad_token_id
        )
        asr_lengths = torch.tensor([len(ids)
                                   for ids in tokenized_ids], dtype=torch.long)
    else:
        asr_targets, asr_lengths = None, None

    return {
        'input_features': input_features,
        'words': words_batch,
        'asr_targets': asr_targets,
        'asr_lengths': asr_lengths,
        'prosody_targets': prosody_targets,
        'emotion_targets': emotion_targets
    }


def configure_cuda_memory():
    """Configure CUDA for optimal memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Allow TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True