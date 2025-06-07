# Multi-Task Learning Framework for Speech Processing

A flexible multi-task learning framework for speech processing tasks including ASR, prosodic prominence detection, and speech emotion recognition.

## Features

- **Multiple Backbone Support**: Easily switch between different backbone models (Whisper, XLSR, MMS, Wav2Vec2-BERT)
- **Task-Specific Heads**: Configurable heads for ASR, prosody, and emotion recognition
- **Dynamic Tokenizer**: SentencePiece tokenizer for improved ASR performance
- **Comprehensive Monitoring**: Training metrics, validation metrics, and visualization tools
- **Flexible Configuration**: Easy-to-use configuration system for model parameters

## Supported Backbone Models
- Whisper (large-v3)
- XLSR (wav2vec2-large-xlsr-53)
- MMS (mms-1b-all)
- Wav2Vec2-BERT (w2v-bert-2.0)

## Requirements

### Core Dependencies
- torch>=2.0.0
- transformers>=4.30.0
- sentencepiece>=0.1.99
- numpy>=1.24.0
- scikit-learn>=1.0.0
- jiwer>=3.0.0

### Optional Dependencies
- matplotlib>=3.7.0 (for visualization)
- wandb>=0.15.0 (for experiment tracking)
- tqdm>=4.65.0 (for progress bars)

## Project Structure

```
.
├── sample_code/
│   ├── backbone_models.py    # Backbone model implementations
│   ├── mtl_config.py         # Configuration management
│   ├── mtl_model.py          # MTL model implementation
│   ├── mtl_dataset.py        # Dataset handling
│   ├── tokenizer.py          # SentencePiece tokenizer
│   └── train_mtl.py          # Training script
├── checkpoints/              # Model checkpoints
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Quick Start

### Google Colab Setup

1. Clone the repository:
```bash
!git clone https://github.com/yourusername/Prosody-SER.git
%cd Prosody-SER
```

2. Install dependencies:
```bash
!pip install -r requirements.txt
```

3. Set up GPU:
```python
import torch
assert torch.cuda.is_available(), "GPU not available"
```

4. Load your dataset and create the MTL system:
```python
from sample_code.train_mtl import create_mtl_system

# Create MTL system with new tokenizer
model, train_loader, val_loader, test_loader, tokenizer = create_mtl_system(
    dataset_dict=your_dataset,
    backbone_name="whisper",
    vocab_size=4000,
    batch_size=8,
    retrain_tokenizer=True  # Set to False to use existing tokenizer
)

# Or load existing tokenizer
model, train_loader, val_loader, test_loader, tokenizer = create_mtl_system(
    dataset_dict=your_dataset,
    backbone_name="whisper",
    vocab_size=4000,
    batch_size=8,
    tokenizer_path="path/to/tokenizer.model"
)
```

### HPC Setup

1. Clone and install:
```bash
git clone https://github.com/yourusername/Prosody-SER.git
cd Prosody-SER
pip install -r requirements.txt
```

2. Run training:
```bash
python sample_code/train_mtl.py \
    --backbone whisper \
    --batch_size 8 \
    --vocab_size 4000 \
    --num_epochs 10 \
    --retrain_tokenizer  # Remove this flag to use existing tokenizer
```

## Training Options

### Command Line Arguments
- `--backbone`: Backbone model to use (whisper, xlsr, mms, wav2vec2-bert)
- `--batch_size`: Batch size for training
- `--vocab_size`: Vocabulary size for SentencePiece tokenizer
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--save_dir`: Directory to save checkpoints
- `--use_wandb`: Enable Weights & Biases tracking
- `--retrain_tokenizer`: Train new SentencePiece tokenizer
- `--tokenizer_path`: Path to existing tokenizer model

### Configuration Options
- `vocab_size`: Size of vocabulary (default: 4000)
- `emotion_classes`: Number of emotion classes (default: 9)
- `prosody_classes`: Number of prosody classes (default: 2)
- `freeze_encoder`: Whether to freeze backbone encoder (default: True)
- `loss_weights`: Weights for different task losses

## Tokenizer Management

The framework uses SentencePiece for subword tokenization, which provides several benefits:
- Better handling of out-of-vocabulary words
- Consistent vocabulary size
- Improved handling of morphological variations

### Training New Tokenizer
```python
from sample_code.train_mtl import setup_tokenizer_and_dataset

tokenizer = setup_tokenizer_and_dataset(
    dataset_dict=your_dataset,
    vocab_size=4000,
    model_prefix='whisper_mtl_tokenizer'
)
```

### Loading Existing Tokenizer
```python
from sample_code.tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer(model_path='path/to/tokenizer.model')
tokenizer.load_tokenizer()
```

## Monitoring and Visualization

### Weights & Biases
Enable W&B tracking with the `--use_wandb` flag to monitor:
- Training and validation losses
- Task-specific metrics
- Model parameters and gradients

### Training History
The framework automatically saves:
- Training history plots
- Model checkpoints
- Tokenizer model and vocabulary

## Model Checkpoints

Checkpoints are saved in the following format:
```
checkpoints/
├── best_model.pt           # Best model based on validation loss
├── checkpoint_epoch_X.pt   # Checkpoint for epoch X
├── final_model.pt          # Final model after training
└── training_history.json   # Training metrics history
```

## Best Practices

### Initial Testing
1. Start with a small dataset subset
2. Use a smaller vocabulary size (e.g., 1000)
3. Enable all monitoring tools
4. Use a smaller batch size

### Full Training
1. Use the complete dataset
2. Set appropriate vocabulary size
3. Enable early stopping
4. Use gradient accumulation for larger batches

### Memory Management
- Monitor GPU memory usage
- Use gradient accumulation for larger batches
- Clear cache between training runs

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Tokenizer Errors**: Ensure text data is properly preprocessed
3. **Training Instability**: Adjust learning rate or loss weights

### Performance Optimization
1. Use mixed precision training
2. Enable gradient accumulation
3. Optimize data loading with num_workers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 