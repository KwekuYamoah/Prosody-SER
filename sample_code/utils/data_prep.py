"""
Data Preparation Utilities
Functions for dataset preparation and tokenizer setup
"""

import os
from datasets import load_dataset, Audio
from sample_code.scripts.tokenizer import SentencePieceTokenizer


def prepare_text_for_training(dataset_dict, output_path):
    """Extracts all text from the dataset and saves to a file for SentencePiece training"""
    all_text = []

    for split in ['train', 'val', 'test']:
        if split in dataset_dict:
            for sample in dataset_dict[split]:
                words = sample['words']
                text = " ".join(words)
                all_text.append(text)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in all_text:
            f.write(text + '\n')
    print(f"Saved {len(all_text)} text samples to {output_path}")


def setup_tokenizer_and_dataset(dataset_dict, vocab_size=4000, model_prefix='akan_mtl_tokenizer'):
    """Complete setup function for SentencePiece tokenizer"""
    # Step 1: Prepare text data
    text_file = "training_text.txt"
    prepare_text_for_training(dataset_dict, text_file)

    # Step 2: Train tokenizer
    tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
    tokenizer.train_tokenizer(text_file, model_prefix=model_prefix)

    return tokenizer


def load_and_prepare_datasets(train_jsonl, val_jsonl, test_jsonl, audio_base_path):
    """Load datasets and prepare audio paths"""
    data_files = {
        "train": train_jsonl,
        "val": val_jsonl,
        "test": test_jsonl
    }

    dataset_dict = load_dataset("json", data_files=data_files)

    # Process audio paths
    for split in ["train", "val", "test"]:
        dataset_dict[split] = dataset_dict[split].map(
            lambda batch: {"audio_filepath": [os.path.join(
                audio_base_path, path) for path in batch["audio_filepath"]]},
            batched=True,
            batch_size=500,
        )
        # Cast to Audio - this doesn't load the audio, just sets up the column type
        dataset_dict[split] = dataset_dict[split].cast_column(
            "audio_filepath", Audio(sampling_rate=16000))

    return dataset_dict