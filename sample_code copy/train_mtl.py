import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb  # for experiment tracking
from jiwer import wer, cer  # for ASR metrics
from datasets import load_dataset, Audio  # Add datasets import

from mtl_config import MTLConfig
from mtl_model import MTLModel
from mtl_dataset import MTLDataset
from backbone_models import BACKBONE_CONFIGS
from tokenizer import SentencePieceTokenizer

class MTLEvaluator:
    """Class for evaluating MTL model performance"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def compute_asr_metrics(self, predictions, targets, words_batch):
        """Compute ASR-specific metrics (WER and CER)"""
        # Convert predictions to text
        pred_texts = []
        target_texts = []
        
        for pred, target, words in zip(predictions, targets, words_batch):
            # Convert token IDs to text
            pred_text = " ".join([words[i] for i in pred if i < len(words)])
            target_text = " ".join([words[i] for i in target if i < len(words)])
            
            pred_texts.append(pred_text)
            target_texts.append(target_text)
        
        # Compute WER and CER
        wer_score = wer(target_texts, pred_texts)
        cer_score = cer(target_texts, pred_texts)
        
        return {
            'wer': wer_score,
            'cer': cer_score
        }

    def flatten_and_filter_sequences(self, predictions, targets, max_length=None):
        """
        Flatten sequence predictions and targets, filtering out padding tokens
        """
        flat_preds = []
        flat_targets = []
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Convert to numpy if tensor
            if torch.is_tensor(pred_seq):
                pred_seq = pred_seq.cpu().numpy()
            if torch.is_tensor(target_seq):
                target_seq = target_seq.cpu().numpy()
            
            # Determine sequence length
            seq_len = len(target_seq) if max_length is None else min(len(target_seq), max_length)
            
            # Add valid predictions and targets
            for i in range(seq_len):
                # Skip padding tokens (assuming 0 is padding)
                if target_seq[i] != 0:  # Adjust this condition based on your padding token
                    flat_preds.append(pred_seq[i])
                    flat_targets.append(target_seq[i])
        
        return np.array(flat_preds), np.array(flat_targets)

    def evaluate(self, data_loader):
        """Evaluate model on a data loader"""
        all_predictions = {
            'asr': [], 'prosody': [], 'emotion': []
        }
        all_targets = {
            'asr': [], 'prosody': [], 'emotion': []
        }
        words_batch = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                asr_targets = batch['asr_targets'].to(self.device) if torch.is_tensor(batch['asr_targets']) else None
                prosody_targets = batch['prosody_targets'].to(self.device)
                emotion_targets = batch['emotion_targets'].to(self.device)

                # Get model predictions
                outputs = self.model(
                    input_features=input_features,
                    asr_targets=asr_targets,
                    prosody_targets=prosody_targets,
                    emotion_targets=emotion_targets
                )

                # Collect predictions and targets
                if 'asr_logits' in outputs and asr_targets is not None:
                    asr_preds = outputs['asr_logits'].argmax(dim=-1)
                    all_predictions['asr'].extend(asr_preds.cpu().numpy())
                    all_targets['asr'].extend(asr_targets.cpu().numpy())
                    words_batch.extend(batch['words'])

                if 'prosody_logits' in outputs:
                    prosody_preds = (outputs['prosody_logits'] > 0).float()
                    all_predictions['prosody'].extend(prosody_preds.cpu().numpy())
                    all_targets['prosody'].extend(prosody_targets.cpu().numpy())

                if 'emotion_logits' in outputs:
                    emotion_preds = outputs['emotion_logits'].argmax(dim=-1)
                    all_predictions['emotion'].extend(emotion_preds.cpu().numpy())
                    all_targets['emotion'].extend(emotion_targets.cpu().numpy())

        # Calculate metrics
        metrics = {}
        for task in ['asr', 'prosody', 'emotion']:
            if len(all_predictions[task]) > 0:
                if task == 'asr':
                    # Compute ASR-specific metrics
                    asr_metrics = self.compute_asr_metrics(
                        all_predictions[task],
                        all_targets[task],
                        words_batch
                    )
                    metrics[task] = asr_metrics
                elif task == 'prosody':
                    # For prosody (sequence labeling)
                    flat_preds, flat_targets = self.flatten_and_filter_sequences(
                        all_predictions[task], all_targets[task]
                    )
                    if len(flat_preds) > 0:
                        metrics[task] = {
                            'accuracy': accuracy_score(flat_targets, flat_preds),
                            'f1': f1_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'precision': precision_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'recall': recall_score(flat_targets, flat_preds, average='weighted', zero_division=0)
                        }
                    else:
                        metrics[task] = {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
                else:  # emotion
                    # For emotion classification (single label per sample)
                    preds = np.array(all_predictions[task])
                    targets = np.array(all_targets[task])
                    
                    # Ensure 1D arrays
                    if preds.ndim > 1:
                        preds = preds.flatten()
                    if targets.ndim > 1:
                        targets = targets.flatten()
                    
                    metrics[task] = {
                        'accuracy': accuracy_score(targets, preds),
                        'f1': f1_score(targets, preds, average='weighted', zero_division=0),
                        'precision': precision_score(targets, preds, average='weighted', zero_division=0),
                        'recall': recall_score(targets, preds, average='weighted', zero_division=0)
                    }

        return metrics

class MTLTrainer:
    """Enhanced trainer class with monitoring and visualization"""
    def __init__(self, model, device='cuda', use_wandb=False):
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_wandb = use_wandb
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_features = batch['input_features'].to(self.device)
        asr_targets = batch['asr_targets'].to(self.device) if torch.is_tensor(batch['asr_targets']) else None
        asr_lengths = batch['asr_lengths'].to(self.device)
        prosody_targets = batch['prosody_targets'].to(self.device)
        emotion_targets = batch['emotion_targets'].to(self.device)

        outputs = self.model(
            input_features=input_features,
            asr_targets=asr_targets,
            asr_lengths=asr_lengths,
            prosody_targets=prosody_targets,
            emotion_targets=emotion_targets,
            return_loss=True
        )

        return outputs

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        epoch_losses = {'total': 0, 'asr': 0, 'prosody': 0, 'emotion': 0}
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            outputs = self.train_step(batch)

            total_loss = outputs['total_loss']
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            if 'asr_loss' in outputs:
                epoch_losses['asr'] += outputs['asr_loss'].item()
            if 'prosody_loss' in outputs:
                epoch_losses['prosody'] += outputs['prosody_loss'].item()
            if 'emotion_loss' in outputs:
                epoch_losses['emotion'] += outputs['emotion_loss'].item()

            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        save_dir='checkpoints',
        early_stopping_patience=5
    ):
        """Complete training loop with monitoring and saving"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch(train_loader, optimizer)
            
            # Validation
            evaluator = MTLEvaluator(self.model, self.device)
            val_metrics = evaluator.evaluate(val_loader)
            
            # Calculate validation loss
            val_losses = self.evaluate_loss(val_loader)
            
            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_total_loss': train_losses['total'],
                'val_total_loss': val_losses['total']
            }
            
            # Add task-specific metrics
            for task in ['asr', 'prosody', 'emotion']:
                if task in train_losses:
                    metrics[f'train_{task}_loss'] = train_losses[task]
                if task in val_losses:
                    metrics[f'val_{task}_loss'] = val_losses[task]
                if task in val_metrics:
                    for metric_name, value in val_metrics[task].items():
                        metrics[f'val_{task}_{metric_name}'] = value

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            for task in ['asr', 'prosody', 'emotion']:
                if task in val_metrics:
                    print(f"  {task.capitalize()} Metrics:")
                    for metric_name, value in val_metrics[task].items():
                        print(f"    {metric_name}: {value:.4f}")

            # Save history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['train_metrics'].append(metrics)
            self.history['val_metrics'].append(val_metrics)

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                self.save_model(os.path.join(save_dir, 'best_model.pt'))
                print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Save final model and history
        self.save_model(os.path.join(save_dir, 'final_model.pt'))
        self.save_history(os.path.join(save_dir, 'training_history.json'))

    def evaluate_loss(self, data_loader):
        """Evaluate loss on a data loader"""
        self.model.eval()
        losses = {'total': 0, 'asr': 0, 'prosody': 0, 'emotion': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                asr_targets = batch['asr_targets'].to(self.device) if torch.is_tensor(batch['asr_targets']) else None
                asr_lengths = batch['asr_lengths'].to(self.device)
                prosody_targets = batch['prosody_targets'].to(self.device)
                emotion_targets = batch['emotion_targets'].to(self.device)

                outputs = self.model(
                    input_features=input_features,
                    asr_targets=asr_targets,
                    asr_lengths=asr_lengths,
                    prosody_targets=prosody_targets,
                    emotion_targets=emotion_targets,
                    return_loss=True
                )
                
                losses['total'] += outputs['total_loss'].item()
                if 'asr_loss' in outputs:
                    losses['asr'] += outputs['asr_loss'].item()
                if 'prosody_loss' in outputs:
                    losses['prosody'] += outputs['prosody_loss'].item()
                if 'emotion_loss' in outputs:
                    losses['emotion'] += outputs['emotion_loss'].item()
                
                num_batches += 1

        # Average losses
        for key in losses:
            losses[key] /= num_batches

        return losses

    def save_model(self, path):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config
        }, path)

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_history(self, path):
        """Save training history"""
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = []
                for item in value:
                    if isinstance(item, dict):
                        serializable_item = {}
                        for k, v in item.items():
                            if isinstance(v, (np.integer, np.floating)):
                                serializable_item[k] = float(v)
                            else:
                                serializable_item[k] = v
                        history_serializable[key].append(serializable_item)
                    else:
                        history_serializable[key].append(item)
            else:
                history_serializable[key] = value
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=4)

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        epochs = range(len(self.history['train_loss']))
        plt.plot(epochs, [x['total'] for x in self.history['train_loss']], label='Train Loss')
        plt.plot(epochs, [x['total'] for x in self.history['val_loss']], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot metrics
        plt.subplot(2, 1, 2)
        if len(self.history['val_metrics']) > 0:
            for task in ['asr', 'prosody', 'emotion']:
                if task in self.history['val_metrics'][0]:
                    metric_key = 'accuracy' if task != 'asr' else 'wer'
                    if metric_key in self.history['val_metrics'][0][task]:
                        values = [x[task][metric_key] for x in self.history['val_metrics']]
                        plt.plot(epochs, values, label=f'{task.capitalize()} {metric_key.upper()}')
        
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

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

def create_mtl_system(
    dataset_dict: Dict,
    backbone_name: str = "whisper",
    use_asr: bool = True,
    use_prosody: bool = True,
    use_ser: bool = True,
    vocab_size: int = 4000,
    batch_size: int = 8,
    retrain_tokenizer: bool = False,
    tokenizer_path: Optional[str] = None,
    **kwargs
):
    """
    Create complete MTL system with configurable backbone

    Args:
        dataset_dict: Your dataset dictionary
        backbone_name: Name of the backbone model to use
        use_asr, use_prosody, use_ser: Enable/disable specific heads
        vocab_size: Size of vocabulary
        batch_size: Batch size for training
        retrain_tokenizer: Whether to retrain the tokenizer
        tokenizer_path: Path to existing tokenizer model
        **kwargs: Additional arguments for MTLConfig
    """
    # Setup SentencePiece tokenizer
    if retrain_tokenizer or tokenizer_path is None:
        print("Training new SentencePiece tokenizer...")
        tokenizer = setup_tokenizer_and_dataset(dataset_dict, vocab_size=vocab_size)
        if tokenizer_path:
            # Save the newly trained tokenizer
            tokenizer.model_path = tokenizer_path
            tokenizer.sp.save(tokenizer_path)
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        tokenizer.load_tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens - Blank: {tokenizer.blank_id}, Pad: {tokenizer.pad_id}, UNK: {tokenizer.unk_id}")

    # Create config with correct vocabulary size
    config = MTLConfig(
        backbone_name=backbone_name,
        vocab_size=tokenizer.get_vocab_size(),
        emotion_classes=9,  # 9 classes dataset
        prosody_classes=2,  # Binary prominence
        **kwargs
    )

    # Create model
    model = MTLModel(
        config=config,
        use_asr=use_asr,
        use_prosody=use_prosody,
        use_ser=use_ser,
        tokenizer=tokenizer
    )

    print(f"Created MTL model with backbone: {backbone_name}")
    print(f"Active heads: {model.get_active_heads()}")

    # Create datasets
    train_dataset = MTLDataset(
        dataset_dict,
        split='train',
        config=config
    )

    val_dataset = MTLDataset(
        dataset_dict,
        split='val',
        config=config
    )

    test_dataset = MTLDataset(
        dataset_dict,
        split='test',
        config=config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_mtl(batch, pad_token_id=tokenizer.pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(batch, pad_token_id=tokenizer.pad_id)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(batch, pad_token_id=tokenizer.pad_id)
    )

    return model, train_loader, val_loader, test_loader, tokenizer

def collate_fn_mtl(batch, pad_token_id=0):
    """Custom collate function for MTL"""
    batch_size = len(batch)

    # Get maximum dimensions
    max_feature_len = max(item['input_features'].shape[-1] for item in batch)
    max_asr_len = max(item['asr_length'].item() if torch.is_tensor(item['asr_target']) else len(item['asr_target']) for item in batch)
    max_prosody_len = max(item['prosody_annotations'].shape[0] for item in batch)

    # Feature dimension for first item
    n_mels = batch[0]['input_features'].shape[0]

    # Initialize tensors
    input_features = torch.zeros(batch_size, n_mels, max_feature_len)
    asr_targets = torch.full((batch_size, max_asr_len), pad_token_id, dtype=torch.long)
    asr_lengths = torch.zeros(batch_size, dtype=torch.long)
    prosody_annotations = torch.zeros(batch_size, max_prosody_len)
    emotions = torch.zeros(batch_size, dtype=torch.long)

    words_batch = []

    for i, item in enumerate(batch):
        # Input features
        feat_len = item['input_features'].shape[-1]
        input_features[i, :, :feat_len] = item['input_features']

        # ASR targets
        if torch.is_tensor(item['asr_target']):
            asr_len = item['asr_target'].shape[0]
            asr_targets[i, :asr_len] = item['asr_target']
            asr_lengths[i] = asr_len
        else:
            asr_lengths[i] = len(item['asr_target'])

        # Prosody annotations
        prosody_len = item['prosody_annotations'].shape[0]
        prosody_annotations[i, :prosody_len] = item['prosody_annotations']

        # Emotions
        emotions[i] = item['emotion']

        # Words batch
        words_batch.append(item['words'])

    return {
        'input_features': input_features,
        'words': words_batch,
        'asr_targets': asr_targets,
        'asr_lengths': asr_lengths,
        'prosody_targets': prosody_annotations,
        'emotion_targets': emotions
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="whisper", choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                      help="Whether to retrain the tokenizer")
    parser.add_argument("--tokenizer_path", type=str,
                      help="Path to trained tokenizer/where to store it")
    # Add new arguments for data paths
    parser.add_argument("--audio_base_path", type=str, required=True,
                      help="Base path to audio files directory")
    parser.add_argument("--train_jsonl", type=str, required=True,
                      help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", type=str, required=True,
                      help="Path to validation JSONL file")
    parser.add_argument("--test_jsonl", type=str, required=True,
                      help="Path to test JSONL file")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="mtl-speech", config=vars(args))

    # Load dataset with specified paths
    data_files = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
        "test": args.test_jsonl
    }
    
    dataset_dict = load_dataset("json", data_files=data_files)
    
    # Process audio paths and convert to Audio objects
    for split in ["train", "val", "test"]:
        # Add audio base path to the audio_filepath
        dataset_dict[split] = dataset_dict[split].map(
            lambda x: {"audio_filepath": os.path.join(args.audio_base_path, x["audio_filepath"])}
        )
        # Cast to Audio with 16000Hz sampling rate
        dataset_dict[split] = dataset_dict[split].cast_column("audio_filepath", Audio(sampling_rate=16000))

    # Create MTL system
    model, train_loader, val_loader, test_loader, tokenizer = create_mtl_system(
        dataset_dict=dataset_dict,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        retrain_tokenizer=args.retrain_tokenizer,
        tokenizer_path=args.tokenizer_path
    )

    # Move model to device
    model = model.to(device)

    # Initialize trainer and optimizer
    trainer = MTLTrainer(model, device=device, use_wandb=args.use_wandb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

    # Plot training history
    trainer.plot_training_history(os.path.join(args.save_dir, 'training_history.png'))

    # Load best model and evaluate on test set
    trainer.load_model(os.path.join(args.save_dir, 'best_model.pt'))
    evaluator = MTLEvaluator(model, device)
    test_metrics = evaluator.evaluate(test_loader)
    
    print("\nTest Set Results:")
    for task, metrics in test_metrics.items():
        print(f"\n{task.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)