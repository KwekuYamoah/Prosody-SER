import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import gc
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb  # for experiment tracking
from jiwer import wer, cer  # for ASR metrics
from datasets import load_dataset, Audio
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import math


from mtl_config import MTLConfig
from mtl_model import MTLModel
from mtl_dataset import MTLDataset
from backbone_models import BACKBONE_CONFIGS
from backbone_models import BackboneModel
from tokenizer import SentencePieceTokenizer
from memory_utils import print_memory_usage, cleanup_memory, optimize_model_for_memory

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Add MTLConfig to safe globals for model loading
torch.serialization.add_safe_globals([MTLConfig])


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


class MTLEvaluator:
    """Class for evaluating MTL model performance"""

    def __init__(self, model, tokenizer, device='cuda', use_amp=True):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.use_amp = use_amp
        self.model.to(device)
        self.model.eval()

    def compute_asr_metrics(self, predictions: List, targets: List) -> Dict:
        """
        FIX: Compute ASR metrics (WER and CER) using the tokenizer.
        The previous implementation was incorrect. This version properly decodes
        token IDs into text before comparing them.
        """
        # Decode predicted token IDs into text
        pred_texts = []
        for pred_ids in predictions:
            # The tokenizer's decode method handles the conversion from IDs to string
            pred_texts.append(self.tokenizer.decode(pred_ids))

        # Decode target token IDs into text
        target_texts = []
        for target_ids in targets:
            # Filter out padding tokens before decoding targets
            filtered_ids = [id for id in target_ids if id !=
                            self.tokenizer.pad_id]
            target_texts.append(self.tokenizer.decode(filtered_ids))

        detailed_results = [
            {"predicted": p, "target": t} for p, t in zip(pred_texts, target_texts)
        ]

        # Calculate WER and CER on the decoded text
        wer_score = wer(target_texts, pred_texts)
        cer_score = cer(target_texts, pred_texts)

        return {
            "wer": wer_score,
            "cer": cer_score,
            "detailed_results": detailed_results,
        }

    def flatten_and_filter_sequences(self, predictions, targets):
        """Flatten sequence predictions and targets, filtering out padding tokens"""
        flat_preds, flat_targets = [], []

        for pred_seq, target_seq in zip(predictions, targets):
            if torch.is_tensor(pred_seq):
                pred_seq = pred_seq.cpu().numpy()
            if torch.is_tensor(target_seq):
                target_seq = target_seq.cpu().numpy()

            valid_mask = target_seq != 0  # Assuming padding value is 0
            flat_preds.extend(pred_seq[valid_mask])
            flat_targets.extend(target_seq[valid_mask])
        return np.array(flat_preds), np.array(flat_targets)

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a data loader with memory optimization"""
        all_predictions = {"asr": [], "prosody": [], "emotion": []}
        all_targets = {"asr": [], "prosody": [], "emotion": []}
        detailed_results = {"prosody": [], "emotion": []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                input_features = batch["input_features"].to(
                    self.device, non_blocking=True)
                asr_targets = batch["asr_targets"].to(
                    self.device, non_blocking=True)
                prosody_targets = batch["prosody_targets"].to(
                    self.device, non_blocking=True)
                emotion_targets = batch["emotion_targets"].to(
                    self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_features=input_features,
                        asr_targets=asr_targets,
                        prosody_targets=prosody_targets,
                        emotion_targets=emotion_targets,
                    )

                self._process_batch_predictions(
                    outputs, batch, all_predictions, all_targets, detailed_results
                )

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                del batch, outputs, input_features, asr_targets, prosody_targets, emotion_targets

        return self._calculate_metrics(all_predictions, all_targets, detailed_results)

    def _process_batch_predictions(self, outputs, batch, all_predictions, all_targets, detailed_results):
        """Process batch predictions efficiently"""
        if 'asr_logits' in outputs and batch['asr_targets'] is not None:
            asr_preds = outputs['asr_logits'].argmax(dim=-1).cpu().tolist()
            asr_targets = batch['asr_targets'].cpu().tolist()
            all_predictions['asr'].extend(asr_preds)
            all_targets['asr'].extend(asr_targets)

        if 'prosody_logits' in outputs:
            prosody_preds = (outputs['prosody_logits'] > 0).float().cpu()
            prosody_targets = batch['prosody_targets'].cpu()
            all_predictions['prosody'].extend(prosody_preds.numpy())
            all_targets['prosody'].extend(prosody_targets.numpy())
            # Detailed prosody results can be added here if needed

        if 'emotion_logits' in outputs:
            emotion_preds = outputs['emotion_logits'].argmax(
                dim=-1).cpu().numpy()
            emotion_targets = batch['emotion_targets'].cpu().numpy()
            all_predictions['emotion'].extend(emotion_preds)
            all_targets['emotion'].extend(emotion_targets)

    def _calculate_metrics(self, all_predictions, all_targets, detailed_results):
        """Calculate metrics for all tasks"""
        metrics = {}
        for task in ['asr', 'prosody', 'emotion']:
            if not all_predictions[task]:
                continue

            if task == 'asr':
                metrics[task] = self.compute_asr_metrics(
                    all_predictions['asr'], all_targets['asr'])
            elif task == 'prosody':
                flat_preds, flat_targets = self.flatten_and_filter_sequences(
                    all_predictions['prosody'], all_targets['prosody'])
                if len(flat_preds) > 0:
                    metrics[task] = {
                        'accuracy': accuracy_score(flat_targets, flat_preds),
                        'f1': f1_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                    }
            elif task == 'emotion':
                metrics[task] = {
                    'accuracy': accuracy_score(all_targets['emotion'], all_predictions['emotion']),
                    'f1': f1_score(all_targets['emotion'], all_predictions['emotion'], average='weighted', zero_division=0),
                }
        return metrics

    def print_detailed_results(self, metrics):
        """Print detailed results for each task"""
        print("\nDetailed Evaluation Results:")

        for task in ['asr', 'prosody', 'emotion']:
            if task in metrics:
                print(f"\n{task.upper()} Results:")
                for metric_name, value in metrics[task].items():
                    if metric_name != 'detailed_results':
                        print(f"{metric_name}: {value:.4f}")

                print("\nSample Predictions:")
                for i, result in enumerate(metrics[task]['detailed_results'][:5]):
                    print(f"\nExample {i+1}:")
                    if task == 'asr':
                        print(f"Target: {result['target']}")
                        print(f"Predicted: {result['predicted']}")
                    elif task == 'prosody':
                        print("Words:", " ".join(result['words']))
                        print("Target Prominence:", " ".join(
                            map(str, result['target'])))
                        print("Predicted Prominence:", " ".join(
                            map(str, result['predicted'])))
                    else:  # emotion
                        print(f"Target Emotion: {result['target']}")
                        print(f"Predicted Emotion: {result['predicted']}")


class MTLTrainer:
    """Enhanced trainer class with gradient accumulation support"""

    def __init__(self, model, device='cuda', use_wandb=False, use_amp=True, gradient_accumulation_steps=1):
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_wandb = use_wandb
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = GradScaler(enabled=use_amp)
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move batch to device
        input_features = batch['input_features'].to(self.device)
        asr_targets = batch['asr_targets'].to(
            self.device) if torch.is_tensor(batch['asr_targets']) else None
        asr_lengths = batch['asr_lengths'].to(self.device)
        prosody_targets = batch['prosody_targets'].to(self.device)
        emotion_targets = batch['emotion_targets'].to(self.device)

        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_features=input_features,
                    asr_targets=asr_targets,
                    asr_lengths=asr_lengths,
                    prosody_targets=prosody_targets,
                    emotion_targets=emotion_targets,
                    return_loss=True
                )
        else:
            outputs = self.model(
                input_features=input_features,
                asr_targets=asr_targets,
                asr_lengths=asr_lengths,
                prosody_targets=prosody_targets,
                emotion_targets=emotion_targets,
                return_loss=True
            )

        return outputs

    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch with gradient accumulation"""
        epoch_losses = {'total': 0, 'asr': 0, 'prosody': 0, 'emotion': 0}
        num_batches = 0

        # Zero gradients at the start
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Forward pass
            outputs = self.train_step(batch)

            # Scale loss by accumulation steps
            total_loss = outputs['total_loss'] / \
                self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.use_amp:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(optimizer)

                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    # Step optimizer
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Step scheduler if provided
                if scheduler is not None:
                    scheduler.step()

                # Zero gradients for next accumulation
                optimizer.zero_grad()

            # Accumulate losses (multiply back to get actual loss value)
            epoch_losses['total'] += total_loss.item() * \
                self.gradient_accumulation_steps
            if 'asr_loss' in outputs:
                epoch_losses['asr'] += outputs['asr_loss'].item()
            if 'prosody_loss' in outputs:
                epoch_losses['prosody'] += outputs['prosody_loss'].item()
            if 'emotion_loss' in outputs:
                epoch_losses['emotion'] += outputs['emotion_loss'].item()

            num_batches += 1

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Delete references to free memory
            del batch, outputs, total_loss

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate_loss(self, data_loader):
        """Evaluate loss on a data loader"""
        self.model.eval()
        losses = {'total': 0, 'asr': 0, 'prosody': 0, 'emotion': 0}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                asr_targets = batch['asr_targets'].to(
                    self.device) if torch.is_tensor(batch['asr_targets']) else None
                asr_lengths = batch['asr_lengths'].to(self.device)
                prosody_targets = batch['prosody_targets'].to(self.device)
                emotion_targets = batch['emotion_targets'].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_features=input_features,
                            asr_targets=asr_targets,
                            asr_lengths=asr_lengths,
                            prosody_targets=prosody_targets,
                            emotion_targets=emotion_targets,
                            return_loss=True
                        )
                else:
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

                # Clear cache periodically
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

                # Delete references
                del batch, outputs

        # Average losses
        for key in losses:
            losses[key] /= num_batches

        return losses

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        tokenizer,
        scheduler=None,
        save_dir='checkpoints',
        early_stopping_patience=5,
        checkpoint_interval=10
    ):
        """Complete training loop with improved checkpoint management"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch + 1}/{num_epochs}")

            # Training
            train_losses = self.train_epoch(train_loader, optimizer, scheduler)

            # Validation
            from train_mtl import MTLEvaluator  # Import here to avoid circular import
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp)
            val_metrics = evaluator.evaluate(val_loader)

            # Calculate validation loss
            val_losses = self.evaluate_loss(val_loader)

            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_total_loss': train_losses['total'],
                'val_total_loss': val_losses['total']
            }

            for task in ['asr', 'prosody', 'emotion']:
                if task in train_losses:
                    metrics[f'train_{task}_loss'] = train_losses[task]
                if task in val_losses:
                    metrics[f'val_{task}_loss'] = val_losses[task]
                if task in val_metrics:
                    for metric_name, value in val_metrics[task].items():
                        if metric_name != 'detailed_results':
                            metrics[f'val_{task}_{metric_name}'] = value

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
                        if metric_name != 'detailed_results' and not isinstance(value, list):
                            print(f"    {metric_name}: {value:.4f}")

            # Save history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['train_metrics'].append(metrics)
            self.history['val_metrics'].append(val_metrics)

            # Check for best model and save
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(os.path.join(
                    save_dir, 'best_model.pt'), epoch, optimizer, scheduler, is_best=True)
                print(
                    f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Save checkpoint at specified intervals
            if (epoch + 1) % checkpoint_interval == 0 and epoch != best_epoch:
                checkpoint_path = os.path.join(
                    save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(
                    checkpoint_path, epoch, optimizer, scheduler)
                print(f"  Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Final model is the best model
        best_path = os.path.join(save_dir, 'best_model.pt')
        final_path = os.path.join(save_dir, 'final_model.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
            print(
                f"Best model (epoch {best_epoch + 1}) copied to final_model.pt")

        self.save_history(os.path.join(save_dir, 'training_history.json'))

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None, is_best=False):
        """Save complete checkpoint with optimizer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model.config,
            'is_best': is_best,
            'history': self.history,
            'gradient_accumulation_steps': self.gradient_accumulation_steps
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """Load complete checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            if "weights_only" in str(e):
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False)
            else:
                raise e

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'gradient_accumulation_steps' in checkpoint:
            self.gradient_accumulation_steps = checkpoint['gradient_accumulation_steps']

        return checkpoint.get('epoch', 0)

    def save_history(self, path):
        """Save training history"""
        from train_mtl import NumpyEncoder  # Import the custom encoder

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4,
                      ensure_ascii=False, cls=NumpyEncoder)

    def plot_training_history(self, save_path=None):
        """Plot training history"""

        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return

        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 1, 1)
        epochs = range(len(self.history['train_loss']))
        plt.plot(epochs, [x['total']
                 for x in self.history['train_loss']], label='Train Loss')
        plt.plot(epochs, [x['total']
                 for x in self.history['val_loss']], label='Val Loss')
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
                        values = [x[task][metric_key]
                                  for x in self.history['val_metrics']]
                        plt.plot(
                            epochs, values, label=f'{task.capitalize()} {metric_key.upper()}')

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


def collate_fn_mtl(batch: List[Dict], pad_token_id: int = 0, tokenizer=None, backbone_name: str = "whisper") -> Dict:
    """
    Custom collate function for MTL.
    CRITICAL FIX: Properly handle different backbone input shapes.
    """
    batch_size = len(batch)

    # Handle input features based on backbone type
    if backbone_name == "whisper":
        # Whisper: input shape is (n_mels, time) for each sample
        # Find max time dimension
        max_time = max(item['input_features'].shape[1] for item in batch)
        n_mels = batch[0]['input_features'].shape[0]

        # Create padded tensor: (batch, n_mels, time)
        input_features = torch.zeros(batch_size, n_mels, max_time)

        for i, item in enumerate(batch):
            time_len = item['input_features'].shape[1]
            input_features[i, :, :time_len] = item['input_features']

    elif backbone_name in ["xlsr", "mms", "wav2vec2-bert"]:
        # Wav2Vec2: input shape is (time,) for each sample - 1D audio
        # Find max length
        max_len = max(item['input_features'].shape[0] for item in batch)

        # Create padded tensor: (batch, time)
        input_features = torch.zeros(batch_size, max_len)

        for i, item in enumerate(batch):
            feat = item['input_features']
            # Ensure it's 1D
            if feat.ndim > 1:
                feat = feat.flatten()
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
        tokenized_ids = []
        for item in batch:
            text_to_encode = " ".join(item['asr_target'])
            ids = tokenizer.encode(text_to_encode, add_special_tokens=True)
            tokenized_ids.append(torch.tensor(ids, dtype=torch.long))

        # Pad sequences
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="whisper",
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision training")
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--scale_lr_with_accumulation", action="store_true",
                        help="Scale learning rate with gradient accumulation")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use cosine annealing learning rate scheduler")
    args = parser.parse_args()

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"\nTraining Configuration:")
    print(f"  Per-GPU batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")

    base_lr = args.lr
    if args.scale_lr_with_accumulation and args.gradient_accumulation_steps > 1:
        # Square root scaling is more conservative and often works better
        lr_scale = math.sqrt(args.gradient_accumulation_steps)
        adjusted_lr = base_lr * lr_scale
        print(f"  Base learning rate: {base_lr}")
        print(f"  Adjusted learning rate: {adjusted_lr}")
    else:
        adjusted_lr = base_lr
        print(f"  Learning rate: {adjusted_lr}")

    # Set device and optimize CUDA settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print initial memory usage
    print_memory_usage("Initial memory usage:")

    if torch.cuda.is_available():
        # Optimize CUDA memory management
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # Reduce memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="mtl-speech", config=vars(args))

    # ============================================================================
    # PHASE 1: LOAD TRAINING DATA AND TRAIN MODEL
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 1: LOADING TRAINING DATA")
    print("="*50)

    # Load only train and val data first
    data_files = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
        "test": args.test_jsonl
    }

    dataset_dict = load_dataset("json", data_files=data_files)

    # Process audio paths
    for split in ["train", "val", "test"]:
        dataset_dict[split] = dataset_dict[split].map(
            lambda batch: {"audio_filepath": [os.path.join(
                args.audio_base_path, path) for path in batch["audio_filepath"]]},
            batched=True,
            batch_size=500,
        )
        # Cast to Audio - this doesn't load the audio, just sets up the column type
        dataset_dict[split] = dataset_dict[split].cast_column(
            "audio_filepath", Audio(sampling_rate=16000))

    print(f"Loaded train: {len(dataset_dict['train'])} samples")
    print(f"Loaded val: {len(dataset_dict['val'])} samples")

    print_memory_usage("After dataset loading:")

    # Setup tokenizer
    if args.retrain_tokenizer or args.tokenizer_path is None:
        print("Training new SentencePiece tokenizer...")
        tokenizer = setup_tokenizer_and_dataset(
            dataset_dict, vocab_size=args.vocab_size)
        if args.tokenizer_path:
            tokenizer.model_path = args.tokenizer_path
            tokenizer.sp.save(args.tokenizer_path)
    else:
        print(f"Loading existing tokenizer from {args.tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
        tokenizer.load_tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    # Create config
    config = MTLConfig(
        backbone_name=args.backbone,
        vocab_size=tokenizer.get_vocab_size(),
        emotion_classes=9,
        prosody_classes=2,
        loss_weights={
            'asr': 0.0,
            'prosody': 1.0,
            'ser': 1.0
        },

    )

    # Create model
    model = MTLModel(
        config=config,
        use_asr=False,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    ).to(device)

    print(f"Created MTL model with backbone: {args.backbone}")
    print(f"Active heads: {model.get_active_heads()}")

    # Apply memory optimizations
    model = optimize_model_for_memory(model)
    model = model.to(device)

    print_memory_usage("After model creation:")

    # Create feature extractor to share across datasets

    temp_backbone = BackboneModel(config.backbone_config)
    feature_extractor = temp_backbone.feature_extractor
    del temp_backbone
    cleanup_memory()

    # Create datasets with lazy loading
    train_dataset = MTLDataset(
        dataset_dict['train'],
        config=config,
        feature_extractor=feature_extractor
    )

    val_dataset = MTLDataset(
        dataset_dict['val'],
        config=config,
        feature_extractor=feature_extractor
    )

    # Create data loaders with memory-aware settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_mtl(
            batch,
            pad_token_id=tokenizer.pad_id,
            tokenizer=tokenizer,
            backbone_name=config.backbone_name  # ADD THIS LINE
        ),
        num_workers=0,  # CHANGE THIS TO 0
        pin_memory=False  # ADD THIS LINE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch,
            pad_token_id=tokenizer.pad_id,
            tokenizer=tokenizer,
            backbone_name=config.backbone_name  # ADD THIS LINE
        ),
        num_workers=0,  # CHANGE THIS TO 0
        pin_memory=False  # ADD THIS LINE
    )

    print_memory_usage("After data loader creation:")

    # ============================================================================
    # PHASE 2: TRAIN MODEL
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 2: TRAINING MODEL")
    print("="*50)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adjusted_lr,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # Initialize scheduler if requested
    scheduler = None
    if args.use_scheduler:
        # Calculate total training steps accounting for accumulation
        steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
        total_steps = steps_per_epoch * args.num_epochs

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        print(
            f"  Using cosine annealing scheduler with {total_steps} total steps")

    # Train with memory monitoring
    trainer = MTLTrainer(
        model,
        device=device,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        early_stopping_patience=5,
        checkpoint_interval=5
    )

    print_memory_usage("After training:")

    # Clean up training data before test evaluation
    del train_dataset, val_dataset, train_loader, val_loader
    cleanup_memory()

    # Plot training history
    trainer.plot_training_history(
        os.path.join(args.save_dir, 'training_history.png')
    )

    print("\nTraining completed!")

    # ============================================================================
    # PHASE 3: LOAD TEST DATA AND EVALUATE
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 3: LOADING TEST DATA AND EVALUATING")
    print("="*50)

    test_dataset = MTLDataset(
        dataset_dict['test'],
        config=config,
        feature_extractor=feature_extractor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch,
            pad_token_id=tokenizer.pad_id,
            tokenizer=tokenizer,
            backbone_name=config.backbone_name  # ADD THIS LINE
        ),
        num_workers=0,  # CHANGE THIS TO 0
        pin_memory=False  # ADD THIS LINE
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = MTLEvaluator(model, tokenizer, device, use_amp=args.use_amp)
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_detailed_results(test_metrics)

    # Save results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4, cls=NumpyEncoder)

    print_memory_usage("Final memory usage:")

    print("\n" + "="*50)
    print("EVALUATION COMPLETED!")
    print("="*50)

    if args.use_wandb:

        wandb.config.update({
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_gpu_batch_size": args.batch_size,
            "learning_rate": adjusted_lr,
            "base_learning_rate": base_lr,
            "use_scheduler": args.use_scheduler
        })
