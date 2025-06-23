# Prosody-SER: Multi-Task Learning for Speech Emotion Recognition

This repository contains a Multi-Task Learning (MTL) system for Speech Emotion Recognition (SER) that combines Automatic Speech Recognition (ASR), Prosody Analysis, and Emotion Recognition tasks. The system follows the methodology from "Speech Emotion Recognition with Multi-task Learning" paper.

## Features

- **Multi-task learning architecture** with configurable backbone models
- **Flexible task head control** - enable/disable individual task heads
- **Alpha-weighted loss computation** following paper methodology
- **Enhanced CTC regularization** to prevent blank prediction collapse
- **Support for multiple backbone models**:
  - Whisper (default)
  - XLSR
  - MMS
  - Wav2Vec2-BERT
- **SentencePiece tokenizer** with CTC support
- **Comprehensive training pipeline** with:
  - Early stopping
  - Model checkpointing
  - Metrics tracking
  - Wandb integration
  - Training history visualization
  - Ablation study support

## Key Changes in Current Version

### 1. Alpha Configuration System

The system now uses a flexible alpha weighting system where:

- **SER is the main task** (weight = 1.0)
- **ASR and Prosody are auxiliary tasks** (weighted by alpha values)
- **Loss formula**: `L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody`

### 2. Task Head Control

You can now control which task heads are active using boolean flags:

- `--use_alpha_ser`: Enable SER task head (main task)
- `--use_alpha_asr`: Enable ASR task head (auxiliary task)
- `--use_alpha_prosody`: Enable Prosody task head (auxiliary task)

### 3. Enhanced CTC Regularization

- **Entropy regularization** to prevent overconfidence
- **Strong blank penalty** (default: 50.0) to prevent blank prediction collapse
- **Configurable blank threshold** (default: 0.3)
- **Label smoothing** and confidence penalty options

### 4. Paper-Style Configuration

- Follows the experimental setup from the research paper
- Optimal alpha values: ASR=0.1, Prosody=0.1 (paper findings)
- Separate learning rates for backbone and task heads
- No freeze/unfreeze strategy (paper approach)

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

#### Basic Training Command

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 64 \
    --vocab_size 16000 \
    --num_epochs 5 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --alpha_ser 1.0 \
    --alpha_asr 0.1 \
    --alpha_prosody 0.1 \
    --use_alpha_asr \
    --use_alpha_ser \
    --use_alpha_prosody
```

#### Task Head Control Examples

**Full MTL (all tasks enabled):**

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 64 \
    --vocab_size 16000 \
    --num_epochs 5 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --alpha_ser 1.0 \
    --alpha_asr 0.1 \
    --alpha_prosody 0.1 \
    --use_alpha_asr \
    --use_alpha_ser \
    --use_alpha_prosody
```

**SER + ASR only (no prosody):**

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 64 \
    --vocab_size 16000 \
    --num_epochs 5 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --alpha_ser 1.0 \
    --alpha_asr 0.1 \
    --use_alpha_asr \
    --use_alpha_ser
```

**SER only (baseline):**

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 64 \
    --vocab_size 16000 \
    --num_epochs 5 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --alpha_ser 1.0 \
    --use_alpha_ser
```

#### Command-line Arguments

**Required Arguments:**

- `--audio_base_path`: Base directory containing audio files
- `--train_jsonl`: Path to training JSONL file
- `--val_jsonl`: Path to validation JSONL file
- `--test_jsonl`: Path to test JSONL file
- `--tokenizer_path`: Path to trained tokenizer

**Model Configuration:**

- `--backbone`: Backbone model to use (default: "whisper")
  - Options: "whisper", "xlsr", "mms", "wav2vec2-bert"
- `--vocab_size`: Vocabulary size for tokenizer (default: 4000)

**Alpha Configuration (Task Weights):**

- `--alpha_ser`: Alpha weight for SER main task (default: 1.0)
- `--alpha_asr`: Alpha weight for ASR auxiliary task (default: 0.1, paper optimal)
- `--alpha_prosody`: Alpha weight for Prosody auxiliary task (default: 0.1, paper optimal)

**Task Head Control (Boolean Flags):**

- `--use_alpha_ser`: Enable SER task head (main task)
- `--use_alpha_asr`: Enable ASR task head (auxiliary task)
- `--use_alpha_prosody`: Enable Prosody task head (auxiliary task)

**Enhanced CTC Regularization:**

- `--ctc_entropy_weight`: Entropy regularization weight for CTC loss (default: 0.01)
- `--ctc_blank_penalty`: Blank penalty weight for CTC loss (default: 50.0, very strong penalty)
- `--ctc_blank_threshold`: Threshold for blank penalty (default: 0.3, much better than 0.8)
- `--ctc_label_smoothing`: Label smoothing for CTC loss (default: 0.0)
- `--ctc_confidence_penalty`: Confidence penalty for CTC loss (default: 0.0)

**Training Configuration:**

- `--batch_size`: Batch size per GPU (default: 8)
- `--num_epochs`: Number of training epochs (default: 100)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency

**Learning Rates:**

- `--backbone_lr`: Learning rate for backbone (fine-tuning) (default: 1e-5)
- `--head_lr`: Learning rate for task heads (default: 5e-5)
- `--warmup_ratio`: Warmup ratio for learning rate schedule (default: 0.1)

**Training Options:**

- `--save_dir`: Directory to save checkpoints (default: "checkpoints")
- `--early_stopping_patience`: Patience for early stopping (default: 10)
- `--checkpoint_interval`: Save checkpoint every N epochs (default: 5)
- `--eval_steps`: Evaluate every N training steps (default: 500)
- `--log_steps`: Log every N training steps (default: 100)

**Experiment Tracking:**

- `--use_wandb`: Use Weights & Biases for experiment tracking
- `--wandb_project`: W&B project name (default: "mtl-akan-speech")
- `--experiment_name`: Experiment name for tracking

**Performance Options:**

- `--use_amp`: Use automatic mixed precision (default: True)
- `--num_workers`: Number of data loading workers (default: 4)
- `--pin_memory`: Pin memory for faster GPU transfer (default: True)
- `--prefetch_factor`: Number of batches to prefetch per worker (default: 2)

**Tokenizer Options:**

- `--retrain_tokenizer`: Whether to retrain the tokenizer

**Ablation Study:**

- `--run_ablation_study`: Run paper's ablation study with different alpha values
- `--ablation_alphas`: Alpha values to test in ablation study (default: [0.0, 0.001, 0.01, 0.1, 1.0])
- `--ablation_epochs`: Epochs per alpha in ablation study (default: 5)

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
    --alpha_ser 1.0 \
    --alpha_asr 0.1 \
    --alpha_prosody 0.1 \
    --use_alpha_asr \
    --use_alpha_ser \
    --use_alpha_prosody \
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
    --batch_size 64 \
    --vocab_size 16000 \
    --num_epochs 50 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --alpha_ser 1.0 \
    --alpha_asr 0.1 \
    --alpha_prosody 0.1 \
    --use_alpha_asr \
    --use_alpha_ser \
    --use_alpha_prosody \
    --use_wandb
```

## Output

The training process will:

1. Save model checkpoints in the specified `save_dir`
2. Generate training history plots
3. Save test results in JSON format
4. Log metrics to Wandb if enabled
5. Create comprehensive results summary with alpha values and CTC regularization settings

## Model Architecture

The system uses a multi-task learning approach with:

- **Shared backbone** for feature extraction (Whisper, XLSR, MMS, or Wav2Vec2-BERT)
- **Task-specific heads** for:
  - **SER** (Speech Emotion Recognition) - Main task with weight 1.0
  - **ASR** (Automatic Speech Recognition) - Auxiliary task weighted by α_ASR
  - **Prosody Analysis** - Auxiliary task weighted by α_Prosody

### Loss Computation

The model follows the paper's loss formula:

```
L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody
```

Where:

- **SER** is the main task (typically α_SER = 1.0)
- **ASR** and **Prosody** are auxiliary tasks (typically α_ASR = α_Prosody = 0.1)

### Enhanced CTC Regularization

The ASR task uses enhanced CTC loss with:

- **Entropy regularization** to prevent overconfidence
- **Strong blank penalty** (default: 50.0) to prevent blank prediction collapse
- **Configurable blank threshold** (default: 0.3)
- **Label smoothing** and confidence penalty options

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

- `unk_id`: 0 (Unknown token)
- `bos_id`: 1 (Beginning of sequence)
- `eos_id`: 2 (End of sequence)
- `pad_id`: 3 (Padding token)
- `blank_id`: 4 (CTC blank token)

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

## Ablation Study Support

The system includes built-in support for running ablation studies following the paper's methodology:

```bash
python -m sample_code.scripts.train \
    --audio_base_path "./AUDIO/" \
    --train_jsonl "./json_data/ser_audio_features_wav_train.jsonl" \
    --val_jsonl "./json_data/ser_audio_features_wav_val.jsonl" \
    --test_jsonl "./json_data/ser_audio_features_wav_test.jsonl" \
    --backbone whisper \
    --batch_size 32 \
    --vocab_size 16000 \
    --num_epochs 10 \
    --save_dir "checkpoints" \
    --tokenizer_path "akan_mtl_tokenizer.model" \
    --use_amp \
    --run_ablation_study \
    --ablation_alphas 0.0 0.001 0.01 0.1 1.0 \
    --ablation_epochs 3
```

This will:

1. Test different alpha values (0.0, 0.001, 0.01, 0.1, 1.0)
2. Train for specified epochs per alpha value
3. Find the optimal alpha configuration
4. Save results for analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
