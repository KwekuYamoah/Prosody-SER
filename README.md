# Prosody-SER: Multi-Task Learning for Speech Emotion Recognition

This repository contains a Multi-Task Learning (MTL) system for Speech Emotion Recognition (SER) that combines Automatic Speech Recognition (ASR), Prosody Analysis, and Emotion Recognition tasks.

## Features

- Multi-task learning architecture with configurable backbone models
- Support for multiple backbone models:
  - Whisper
  - XLSR
  - MMS
  - Wav2Vec2-BERT
- SentencePiece tokenizer with CTC support
- Comprehensive training pipeline with:
  - Early stopping
  - Model checkpointing
  - Metrics tracking
  - Wandb integration
  - Training history visualization

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Prosody-SER.git
cd Prosody-SER
```

2. Install required dependencies:

```bash
pip install torch transformers sentencepiece matplotlib tqdm numpy scikit-learn wandb jiwer datasets
```

## Dataset Structure

The system expects your dataset to be organized in the following format:

1. Audio files in a base directory
2. JSONL files for train/val/test splits with the following structure:

```json
{
    "audio_filepath": "relative/path/to/audio/file.wav",
    "words": ["word1", "word2", "word3"],
    "prosody_annotations": [0, 1, 0],
    "emotion": 3
}
```

## Usage

### Training

The training script supports various command-line arguments for customization:

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 16 \
    --vocab_size 16000 \
    --num_epochs 100 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --use_wandb
```

#### Command-line Arguments

- `--audio_base_path`: Base directory containing audio files (required)
- `--train_jsonl`: Path to training JSONL file (required)
- `--val_jsonl`: Path to validation JSONL file (required)
- `--test_jsonl`: Path to test JSONL file (required)
- `--backbone`: Backbone model to use (default: "whisper")
  - Options: "whisper", "xlsr", "mms", "wav2vec2-bert"
- `--batch_size`: Batch size for training (default: 8)
- `--vocab_size`: Vocabulary size for tokenizer (default: 4000)
- `--num_epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--save_dir`: Directory to save checkpoints (default: "checkpoints")
- `--tokenizer_path`: Path to trained tokenizer (required)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 1)
- `--use_amp`: Enable automatic mixed precision training (default: True)
- `--use_enhanced_training`: Enable enhanced training with freeze/unfreeze strategy (default: True)
- `--freeze_encoder_initially`: Whether to freeze encoder initially (default: True)
- `--unfreeze_epoch_ratio`: Fraction of epochs to train with frozen encoder (default: 0.5)
- `--lr_reduction_factor`: Factor to reduce LR when unfreezing (default: 0.1)
- `--asr_lr_multiplier`: Learning rate multiplier for ASR head (default: 0.1)
- `--ctc_entropy_weight`: Entropy regularization weight for CTC (default: 0.01)
- `--ctc_blank_weight`: Maximum blank probability for CTC (default: 0.95)
- `--use_wandb`: Enable Wandb logging (optional flag)
- `--use_scheduler`: Use cosine annealing learning rate scheduler (optional flag)
- `--scale_lr_with_accumulation`: Scale learning rate with gradient accumulation (optional flag)

### Evaluation

To evaluate a trained model:

```bash
python -m sample_code.scripts.evaluate \
    --checkpoint_path "checkpoints/best_model.pt" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --audio_base_path "./AUDIO/" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --decode_method beam \
    --output_dir "eval_results"
```

#### Evaluation Arguments

- `--checkpoint_path`: Path to model checkpoint (required)
- `--tokenizer_path`: Path to trained tokenizer (required)
- `--audio_base_path`: Base path to audio files directory (required)
- `--test_jsonl`: Path to test JSONL file (required)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--decode_method`: Decoding method for ASR (default: "beam", choices: ["greedy", "beam"])
- `--output_dir`: Directory to save evaluation results (default: "eval_results")
- `--use_amp`: Use automatic mixed precision (default: True)
- `--device`: Device to use for evaluation (default: "cuda", choices: ["cuda", "cpu"])

The evaluation script will generate:
1. A JSON file with detailed metrics (`test_metrics.json`)
2. The evaluation configuration (`eval_config.json`)
3. A summary report (`summary.txt`)

### Example Usage

#### Google Colab

```bash
python -m sample_code.scripts.train \
    --audio_base_path "/content/drive/MyDrive/AKAN-SPEECH-EMOTION-DATA/AUDIO/" \
    --train_jsonl "/content/drive/MyDrive/AKAN-SPEECH-EMOTION-DATA/AUDIO/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "/content/drive/MyDrive/AKAN-SPEECH-EMOTION-DATA/AUDIO/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "/content/drive/MyDrive/AKAN-SPEECH-EMOTION-DATA/AUDIO/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 16 \
    --vocab_size 16000 \
    --num_epochs 100 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --use_wandb
```

#### HPC or Local Machine

```bash
python -m sample_code.scripts.train \
    --audio_base_path "/path/to/your/audio/files/" \
    --train_jsonl "/path/to/train.jsonl" \
    --val_jsonl "/path/to/val.jsonl" \
    --test_jsonl "/path/to/test.jsonl" \
    --backbone whisper \
    --batch_size 16 \
    --vocab_size 16000 \
    --num_epochs 100 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --use_wandb
```

## Output

The training process will:

1. Save model checkpoints in the specified `save_dir`
2. Generate training history plots
3. Save test results in JSON format
4. Log metrics to Wandb if enabled

## Model Architecture

The system uses a multi-task learning approach with:

- A shared backbone for feature extraction
- Task-specific heads for:
  - ASR (Automatic Speech Recognition)
  - Prosody Analysis
  - Emotion Recognition

## Tokenizer Management

The system uses SentencePiece for subword tokenization, which provides several benefits:

- Better handling of out-of-vocabulary words
- Consistent vocabulary size
- Improved handling of morphological variations

### Initial Tokenizer Training

Before training the MTL model, you need to train the SentencePiece tokenizer:

```python
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# Train new tokenizer
tokenizer = SentencePieceTokenizer(
    vocab_size=16000,  # Adjust based on your needs
    model_prefix='akan_mtl_tokenizer'
)
tokenizer.train_tokenizer(text_data=your_text_data)
```

This will:

1. Extract all text from your dataset
2. Train a new SentencePiece model
3. Save the model and vocabulary files

### Using Existing Tokenizer

For subsequent training runs, you can use an existing tokenizer:

```python
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# Load existing tokenizer
tokenizer = SentencePieceTokenizer(
    model_path='path/to/tokenizer.model',
    vocab_size=16000  # Should match the trained tokenizer
)
tokenizer.load_tokenizer()
```

### Tokenizer Configuration

The SentencePiece tokenizer includes special tokens:

- `pad_id`: 0 (Padding token)
- `unk_id`: 1 (Unknown token)
- `blank_id`: 2 (CTC blank token)
- `bos_id`: 3 (Beginning of sequence)
- `eos_id`: 4 (End of sequence)

### Tokenizer Best Practices

1. **Vocabulary Size**:

   - Start with a smaller vocabulary (e.g., 1000-2000) for initial testing
   - Increase to 8000-16000 for production use
   - Consider your language's characteristics when choosing size
2. **Training Data**:

   - Use a representative sample of your dataset
   - Include all possible word forms
   - Consider adding domain-specific terms
3. **Model Type**:

   - Uses BPE (Byte Pair Encoding) by default
   - Character coverage set to 0.995
   - No normalization applied (identity rule)
4. **Special Tokens**:

   - The system automatically handles special tokens
   - CTC blank token is essential for ASR
   - BOS/EOS tokens help with sequence modeling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
