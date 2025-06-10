#!/usr/bin/env python3
"""
Quick test to verify the fixes are working.
Save this as test_fixes.py and run it before full training.
"""

import torch
import numpy as np
from datasets import load_dataset, Audio
import os

# Import your modules
from mtl_config import MTLConfig
from mtl_dataset import MTLDataset
from backbone_models import BACKBONE_CONFIGS, BackboneModel
from train_mtl import collate_fn_mtl


def test_single_sample(audio_base_path, train_jsonl, backbone_name="xlsr"):
    """Test processing a single sample through the entire pipeline"""

    print(f"Testing {backbone_name} backbone...")

    # 1. Load one sample
    data_files = {"train": train_jsonl}
    dataset = load_dataset("json", data_files=data_files)
    dataset["train"] = dataset["train"].map(
        lambda batch: {"audio_filepath": [os.path.join(
            audio_base_path, path) for path in batch["audio_filepath"]]},
        batched=True,
        batch_size=1,
    )
    dataset["train"] = dataset["train"].cast_column(
        "audio_filepath", Audio(sampling_rate=16000))

    print(f"✓ Loaded dataset with {len(dataset['train'])} samples")

    # 2. Create MTL dataset
    config = MTLConfig(backbone_name=backbone_name)
    temp_backbone = BackboneModel(config.backbone_config)
    feature_extractor = temp_backbone.feature_extractor

    mtl_dataset = MTLDataset(
        dataset["train"],
        config=config,
        feature_extractor=feature_extractor
    )

    # 3. Get one item
    item = mtl_dataset[0]
    print(
        f"✓ Got item with input_features shape: {item['input_features'].shape}")

    # 4. Test collate function
    batch = [item]  # Single item batch

    # Mock tokenizer
    class MockTokenizer:
        pad_id = 0

        def encode(self, text, add_special_tokens=True):
            return list(range(10))  # Mock tokens

    collated = collate_fn_mtl(
        batch,
        pad_token_id=0,
        tokenizer=MockTokenizer(),
        backbone_name=backbone_name
    )

    print(f"✓ Collated input shape: {collated['input_features'].shape}")

    # 5. Test forward pass through backbone
    with torch.no_grad():
        input_features = collated['input_features']
        output = temp_backbone(input_features)
        print(f"✓ Backbone output shape: {output.shape}")

    # Expected shapes:
    if backbone_name in ["whisper", "wav2vec2-bert"]:
        assert input_features.ndim == 3  # (batch, n_mels, time)
    else:
        assert input_features.ndim == 2  # (batch, time)

    print(f"✅ All tests passed for {backbone_name}!")

    # Cleanup
    del temp_backbone
    torch.cuda.empty_cache()

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python test_fixes.py <audio_base_path> <train_jsonl> [backbone]")
        sys.exit(1)

    audio_base_path = sys.argv[1]
    train_jsonl = sys.argv[2]
    backbone = sys.argv[3] if len(sys.argv) > 3 else "xlsr"

    try:
        test_single_sample(audio_base_path, train_jsonl, backbone)
        print("\n✅ Everything looks good! You can now run the full training.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
