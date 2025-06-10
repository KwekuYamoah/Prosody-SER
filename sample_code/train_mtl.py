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
from typing import Dict, Optional
import matplotlib.pyplot as plt
import math


from mtl_config import MTLConfig
from mtl_model import MTLModel
from mtl_dataset import MTLDataset
from backbone_models import BACKBONE_CONFIGS
from backbone_models import BackboneModel
from tokenizer import SentencePieceTokenizer
from memory_utils import print_memory_usage, cleanup_memory, optimize_model_for_memory

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

    def __init__(self, model, device='cuda', use_amp=True):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.model.to(device)
        self.model.eval()

    def compute_asr_metrics(self, predictions, targets, words_batch):
        """Compute ASR-specific metrics (WER and CER)"""
        pred_texts = []
        target_texts = []
        detailed_results = []

        for pred, target, words in zip(predictions, targets, words_batch):
            # Filter out padding tokens (ID 0) from predictions and targets
            target_no_pad = target[target != 0]  # Remove padding tokens
            pred_no_pad = pred[:len(target_no_pad)]

            # Convert token IDs to text
            pred_text = " ".join([words[i]
                                 for i in pred_no_pad if i < len(words)])
            target_text = " ".join([words[i]
                                   for i in target_no_pad if i < len(words)])

            pred_texts.append(pred_text)
            target_texts.append(target_text)

            detailed_results.append({
                'predicted': pred_text,
                'target': target_text,
                'words': words,
                'pred_length': len(pred_no_pad),
                'target_length': len(target_no_pad)
            })

        # Compute WER and CER only on non-empty targets
        valid_pairs = [(p, t)
                       for p, t in zip(pred_texts, target_texts) if t.strip()]
        if valid_pairs:
            valid_preds, valid_targets = zip(*valid_pairs)
            wer_score = wer(list(valid_targets), list(valid_preds))
            cer_score = cer(list(valid_targets), list(valid_preds))
        else:
            wer_score = 0.0
            cer_score = 0.0

        return {
            'wer': wer_score,
            'cer': cer_score,
            'detailed_results': detailed_results,
            'valid_samples': len(valid_pairs),
            'total_samples': len(predictions)
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

            # Create mask for non-padding positions
            # Assuming padding value is 0 for prosody targets
            valid_mask = target_seq != 0

            # Apply mask to get only valid positions
            valid_preds = pred_seq[valid_mask]
            valid_targets = target_seq[valid_mask]

            # Add to flattened arrays
            flat_preds.extend(valid_preds)
            flat_targets.extend(valid_targets)

        return np.array(flat_preds), np.array(flat_targets)

    def evaluate(self, data_loader):
        """Evaluate model on a data loader with memory optimization"""
        all_predictions = {'asr': [], 'prosody': [], 'emotion': []}
        all_targets = {'asr': [], 'prosody': [], 'emotion': []}
        words_batch = []
        detailed_results = {'asr': [], 'prosody': [], 'emotion': []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                # Move batch to device
                input_features = batch['input_features'].to(
                    self.device, non_blocking=True)
                asr_targets = batch['asr_targets'].to(
                    self.device, non_blocking=True) if torch.is_tensor(batch['asr_targets']) else None
                prosody_targets = batch['prosody_targets'].to(
                    self.device, non_blocking=True)
                emotion_targets = batch['emotion_targets'].to(
                    self.device, non_blocking=True)

                # Get model predictions with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_features=input_features,
                            asr_targets=asr_targets,
                            prosody_targets=prosody_targets,
                            emotion_targets=emotion_targets
                        )
                else:
                    outputs = self.model(
                        input_features=input_features,
                        asr_targets=asr_targets,
                        prosody_targets=prosody_targets,
                        emotion_targets=emotion_targets
                    )

                # Process predictions in smaller chunks to save memory
                self._process_batch_predictions(
                    outputs, batch, all_predictions, all_targets, words_batch, detailed_results)

                # Clear cache every few batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

                # Delete references
                del batch, outputs, input_features
                if asr_targets is not None:
                    del asr_targets
                del prosody_targets, emotion_targets

        return self._calculate_metrics(all_predictions, all_targets, words_batch, detailed_results)

    def _process_batch_predictions(self, outputs, batch, all_predictions, all_targets, words_batch, detailed_results):
        """Process batch predictions efficiently"""
        if 'asr_logits' in outputs and batch['asr_targets'] is not None:
            asr_preds = outputs['asr_logits'].argmax(dim=-1).cpu()
            asr_targets = batch['asr_targets'].cpu() if torch.is_tensor(
                batch['asr_targets']) else batch['asr_targets']
            all_predictions['asr'].extend(asr_preds.numpy())
            all_targets['asr'].extend(
                asr_targets.numpy() if torch.is_tensor(asr_targets) else asr_targets)
            words_batch.extend(batch['words'])
            del asr_preds

        if 'prosody_logits' in outputs:
            prosody_preds = (outputs['prosody_logits'] > 0).float().cpu()
            prosody_targets = batch['prosody_targets'].cpu()
            all_predictions['prosody'].extend(prosody_preds.numpy())
            all_targets['prosody'].extend(prosody_targets.numpy())

            for i, (pred, target, words) in enumerate(zip(prosody_preds, prosody_targets, batch['words'])):
                pred_np = pred.numpy()
                target_np = target.numpy()
                valid_mask = target_np != 0
                detailed_results['prosody'].append({
                    'words': words,
                    'predicted': pred_np,
                    'target': target_np,
                    'valid_mask': valid_mask,
                    'num_valid_tokens': valid_mask.sum(),
                    'accuracy': (pred_np[valid_mask] == target_np[valid_mask]).mean() if valid_mask.sum() > 0 else 0.0
                })
            del prosody_preds, prosody_targets

        if 'emotion_logits' in outputs:
            emotion_preds = outputs['emotion_logits'].argmax(dim=-1).cpu()
            emotion_targets = batch['emotion_targets'].cpu()
            all_predictions['emotion'].extend(emotion_preds.numpy())
            all_targets['emotion'].extend(emotion_targets.numpy())

            for i, (pred, target) in enumerate(zip(emotion_preds, emotion_targets)):
                detailed_results['emotion'].append({
                    'predicted': pred.item(),
                    'target': target.item()
                })
            del emotion_preds, emotion_targets

    def _calculate_metrics(self, all_predictions, all_targets, words_batch, detailed_results):
        """Calculate metrics efficiently"""
        metrics = {}
        for task in ['asr', 'prosody', 'emotion']:
            if len(all_predictions[task]) > 0:
                if task == 'asr':
                    metrics[task] = self.compute_asr_metrics(
                        all_predictions[task],
                        all_targets[task],
                        words_batch
                    )
                elif task == 'prosody':
                    flat_preds, flat_targets = self.flatten_and_filter_sequences(
                        all_predictions[task], all_targets[task]
                    )
                    if len(flat_preds) > 0:
                        metrics[task] = {
                            'accuracy': accuracy_score(flat_targets, flat_preds),
                            'f1': f1_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'precision': precision_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'recall': recall_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'detailed_results': detailed_results[task]
                        }
                    else:
                        metrics[task] = {
                            'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                            'detailed_results': detailed_results[task]
                        }
                else:  # emotion
                    preds = np.array(all_predictions[task])
                    targets = np.array(all_targets[task])
                    if preds.ndim > 1:
                        preds = preds.flatten()
                    if targets.ndim > 1:
                        targets = targets.flatten()

                    metrics[task] = {
                        'accuracy': accuracy_score(targets, preds),
                        'f1': f1_score(targets, preds, average='weighted', zero_division=0),
                        'precision': precision_score(targets, preds, average='weighted', zero_division=0),
                        'recall': recall_score(targets, preds, average='weighted', zero_division=0),
                        'detailed_results': detailed_results[task]
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
        self.scaler = GradScaler() if use_amp else None
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
            evaluator = MTLEvaluator(self.model, self.device, self.use_amp)
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


def collate_fn_mtl(batch, pad_token_id=0, tokenizer=None):
    """Custom collate function for MTL with tokenization"""
    batch_size = len(batch)

    # Get maximum dimensions
    max_feature_len = max(item['input_features'].shape[-1] for item in batch)
    max_prosody_len = max(item['prosody_annotations'].shape[0]
                          for item in batch)

    # Feature dimension for first item
    n_mels = batch[0]['input_features'].shape[0]

    # Initialize tensors
    input_features = torch.zeros(batch_size, n_mels, max_feature_len)
    prosody_annotations = torch.zeros(batch_size, max_prosody_len)
    emotions = torch.zeros(batch_size, dtype=torch.long)

    words_batch = []

    # Handle ASR tokenization if tokenizer is provided
    if tokenizer is not None:
        # Tokenize all words first to get max length
        tokenized_words = []
        for item in batch:
            words = item['asr_target']  # This is still the words list
            token_ids = tokenizer.encode_words(words, add_special_tokens=True)
            tokenized_words.append(token_ids)

        max_asr_len = max(len(tokens) for tokens in tokenized_words)
        asr_targets = torch.full(
            (batch_size, max_asr_len), pad_token_id, dtype=torch.long)
        asr_lengths = torch.zeros(batch_size, dtype=torch.long)
    else:
        asr_targets = None
        asr_lengths = None
        tokenized_words = [None] * batch_size

    for i, (item, tokens) in enumerate(zip(batch, tokenized_words)):
        # Input features
        feat_len = item['input_features'].shape[-1]
        input_features[i, :, :feat_len] = item['input_features']

        # ASR targets (if tokenizer provided)
        if tokenizer is not None and tokens is not None:
            asr_len = len(tokens)
            asr_targets[i, :asr_len] = torch.tensor(tokens, dtype=torch.long)
            asr_lengths[i] = asr_len

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
        dataset_dict,
        split='train',
        config=config,
        feature_extractor=feature_extractor
    )

    val_dataset = MTLDataset(
        dataset_dict,
        split='val',
        config=config,
        feature_extractor=feature_extractor
    )

    # Create data loaders with memory-aware settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer)
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
        dataset_dict,
        split='test',
        config=config,
        feature_extractor=feature_extractor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer)
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = MTLEvaluator(model, device, use_amp=args.use_amp)
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
