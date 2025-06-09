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

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def compute_asr_metrics(self, predictions, targets, words_batch):
        """Compute ASR-specific metrics (WER and CER)"""
        pred_texts = []
        target_texts = []
        detailed_results = []

        for pred, target, words in zip(predictions, targets, words_batch):
            # Filter out padding tokens (ID 0) from predictions and targets
            # Find the actual sequence length by removing padding
            target_no_pad = target[target != 0]  # Remove padding tokens
            # Truncate prediction to target length
            pred_no_pad = pred[:len(target_no_pad)]

            # Convert token IDs to text, ensuring we don't go out of bounds
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
        """Evaluate model on a data loader"""
        all_predictions = {
            'asr': [], 'prosody': [], 'emotion': []
        }
        all_targets = {
            'asr': [], 'prosody': [], 'emotion': []
        }
        words_batch = []
        detailed_results = {
            'asr': [],
            'prosody': [],
            'emotion': []
        }

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                asr_targets = batch['asr_targets'].to(
                    self.device) if torch.is_tensor(batch['asr_targets']) else None
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
                    all_predictions['prosody'].extend(
                        prosody_preds.cpu().numpy())
                    all_targets['prosody'].extend(
                        prosody_targets.cpu().numpy())

                    # Store detailed prosody results
                    for i, (pred, target, words) in enumerate(zip(prosody_preds, prosody_targets, batch['words'])):
                        pred_np = pred.cpu().numpy()
                        target_np = target.cpu().numpy()

                        # Create mask for valid (non-padded) positions
                        valid_mask = target_np != 0

                        detailed_results['prosody'].append({
                            'words': words,
                            'predicted': pred_np,
                            'target': target_np,
                            'valid_mask': valid_mask,
                            'num_valid_tokens': valid_mask.sum(),
                            'accuracy': (pred_np[valid_mask] == target_np[valid_mask]).mean() if valid_mask.sum() > 0 else 0.0
                        })

                if 'emotion_logits' in outputs:
                    emotion_preds = outputs['emotion_logits'].argmax(dim=-1)
                    all_predictions['emotion'].extend(
                        emotion_preds.cpu().numpy())
                    all_targets['emotion'].extend(
                        emotion_targets.cpu().numpy())

                    # Store detailed emotion results
                    for i, (pred, target) in enumerate(zip(emotion_preds, emotion_targets)):
                        detailed_results['emotion'].append({
                            'predicted': pred.item(),
                            'target': target.item()
                        })

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
                            'recall': recall_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                            'detailed_results': detailed_results[task]
                        }
                    else:
                        metrics[task] = {
                            'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
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
                        'recall': recall_score(targets, preds, average='weighted', zero_division=0),
                        'detailed_results': detailed_results[task]
                    }

        return metrics

    def evaluate_with_tracking(self, data_loader, track_predictions=False, save_dir='analysis'):
        """Evaluate model with optional prediction tracking"""
        tracker = PredictionTracker(save_dir) if track_predictions else None

        all_predictions = {'asr': [], 'prosody': [], 'emotion': []}
        all_targets = {'asr': [], 'prosody': [], 'emotion': []}
        words_batch = []
        detailed_results = {'asr': [], 'prosody': [],
                            'emotion': []}  # Add this line

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                asr_targets = batch['asr_targets'].to(
                    self.device) if torch.is_tensor(batch['asr_targets']) else None
                prosody_targets = batch['prosody_targets'].to(self.device)
                emotion_targets = batch['emotion_targets'].to(self.device)

                # Get model predictions
                outputs = self.model(
                    input_features=input_features,
                    asr_targets=asr_targets,
                    prosody_targets=prosody_targets,
                    emotion_targets=emotion_targets
                )

                # Track predictions if enabled
                if tracker:
                    tracker.add_batch_predictions(batch, outputs)

                # Collect predictions and targets
                if 'asr_logits' in outputs and asr_targets is not None:
                    asr_preds = outputs['asr_logits'].argmax(dim=-1)
                    all_predictions['asr'].extend(asr_preds.cpu().numpy())
                    all_targets['asr'].extend(asr_targets.cpu().numpy())
                    words_batch.extend(batch['words'])

                if 'prosody_logits' in outputs:
                    prosody_preds = (outputs['prosody_logits'] > 0).float()
                    all_predictions['prosody'].extend(
                        prosody_preds.cpu().numpy())
                    all_targets['prosody'].extend(
                        prosody_targets.cpu().numpy())

                    # Store detailed prosody results for print_detailed_results
                    for i, (pred, target, words) in enumerate(zip(prosody_preds, prosody_targets, batch['words'])):
                        pred_np = pred.cpu().numpy()
                        target_np = target.cpu().numpy()

                        # Create mask for valid (non-padded) positions
                        valid_mask = target_np != 0

                        detailed_results['prosody'].append({
                            'words': words,
                            'predicted': pred_np,
                            'target': target_np,
                            'valid_mask': valid_mask,
                            'num_valid_tokens': valid_mask.sum(),
                            'accuracy': (pred_np[valid_mask] == target_np[valid_mask]).mean() if valid_mask.sum() > 0 else 0.0
                        })

                if 'emotion_logits' in outputs:
                    emotion_preds = outputs['emotion_logits'].argmax(dim=-1)
                    all_predictions['emotion'].extend(
                        emotion_preds.cpu().numpy())
                    all_targets['emotion'].extend(
                        emotion_targets.cpu().numpy())

                    # Store detailed emotion results for print_detailed_results
                    for i, (pred, target) in enumerate(zip(emotion_preds, emotion_targets)):
                        detailed_results['emotion'].append({
                            'predicted': pred.item(),
                            'target': target.item()
                        })

        # Calculate metrics
        metrics = {}
        for task in ['asr', 'prosody', 'emotion']:
            if len(all_predictions[task]) > 0:
                if task == 'asr':
                    asr_metrics = self.compute_asr_metrics(
                        all_predictions[task], all_targets[task], words_batch
                    )
                    metrics[task] = asr_metrics
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
                            # Add detailed results
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
                        # Add detailed results
                        'detailed_results': detailed_results[task]
                    }

        # Save tracking results if enabled
        if tracker:
            tracker.save_predictions()
            analysis = tracker.analyze_predictions()
            print("\n=== Prediction Analysis Summary ===")
            for task, task_analysis in analysis.items():
                print(f"\n{task.upper()} Task:")
                for key, value in task_analysis.items():
                    if not isinstance(value, (list, dict)):
                        print(f"  {key}: {value}")

        return metrics

    def print_detailed_results(self, metrics):
        """Print detailed results for each task"""
        print("\nDetailed Evaluation Results:")

        # ASR Results
        if 'asr' in metrics:
            print("\nASR Results:")
            print(f"Overall WER: {metrics['asr']['wer']:.4f}")
            print(f"Overall CER: {metrics['asr']['cer']:.4f}")
            print("\nSample Predictions:")
            # Show first 5 examples
            for i, result in enumerate(metrics['asr']['detailed_results'][:5]):
                print(f"\nExample {i+1}:")
                print(f"Target: {result['target']}")
                print(f"Predicted: {result['predicted']}")

        # Prosody Results
        if 'prosody' in metrics:
            print("\nProsody Results:")
            print(f"Overall Accuracy: {metrics['prosody']['accuracy']:.4f}")
            print(f"Overall F1: {metrics['prosody']['f1']:.4f}")
            print("\nSample Predictions:")
            # Show first 5 examples
            for i, result in enumerate(metrics['prosody']['detailed_results'][:5]):
                print(f"\nExample {i+1}:")
                print("Words:", " ".join(result['words']))
                print("Target Prominence:", " ".join(
                    map(str, result['target'])))
                print("Predicted Prominence:", " ".join(
                    map(str, result['predicted'])))

        # Emotion Results
        if 'emotion' in metrics:
            print("\nEmotion Results:")
            print(f"Overall Accuracy: {metrics['emotion']['accuracy']:.4f}")
            print(f"Overall F1: {metrics['emotion']['f1']:.4f}")
            print("\nSample Predictions:")
            # Show first 5 examples
            for i, result in enumerate(metrics['emotion']['detailed_results'][:5]):
                print(f"\nExample {i+1}:")
                print(f"Target Emotion: {result['target']}")
                print(f"Predicted Emotion: {result['predicted']}")


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
        asr_targets = batch['asr_targets'].to(
            self.device) if torch.is_tensor(batch['asr_targets']) else None
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
            train_losses = self.train_epoch(train_loader, optimizer)

            # Validation
            evaluator = MTLEvaluator(self.model, self.device)
            val_metrics = evaluator.evaluate(val_loader)

            # Calculate validation loss
            val_losses = self.evaluate_loss(val_loader)

            # Log metrics (existing code...)
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

            # Print epoch summary (existing code...)
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            for task in ['asr', 'prosody', 'emotion']:
                if task in val_metrics:
                    print(f"  {task.capitalize()} Metrics:")
                    for metric_name, value in val_metrics[task].items():
                        if metric_name != 'detailed_results':
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
                    save_dir, 'best_model.pt'), epoch, optimizer, is_best=True)
                print(
                    f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Save checkpoint at specified intervals (but not if it's already the best)
            if (epoch + 1) % checkpoint_interval == 0 and epoch != best_epoch:
                checkpoint_path = os.path.join(
                    save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path, epoch, optimizer)
                print(f"  Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Final model is the best model (copy best to final for clarity)
        best_path = os.path.join(save_dir, 'best_model.pt')
        final_path = os.path.join(save_dir, 'final_model.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
            print(
                f"Best model (epoch {best_epoch + 1}) copied to final_model.pt")

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
                asr_targets = batch['asr_targets'].to(
                    self.device) if torch.is_tensor(batch['asr_targets']) else None
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

    def save_checkpoint(self, path, epoch, optimizer, is_best=False):
        """Save complete checkpoint with optimizer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model.config,
            'is_best': is_best,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
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

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        return checkpoint.get('epoch', 0)

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

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history_serializable, f, indent=4,
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


class PredictionTracker:
    """Class to track and analyze model predictions across tasks"""

    def __init__(self, save_dir='analysis'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.predictions = {
            'asr': [],
            'prosody': [],
            'emotion': []
        }

    def add_batch_predictions(self, batch_data, outputs):
        """Add predictions from a single batch"""

        # ASR predictions
        if 'asr_logits' in outputs and 'asr_targets' in batch_data:
            asr_preds = outputs['asr_logits'].argmax(dim=-1)
            asr_targets = batch_data['asr_targets']
            words_batch = batch_data['words']

            for i, (pred, target, words) in enumerate(zip(asr_preds, asr_targets, words_batch)):
                # Remove padding tokens
                target_no_pad = target[target != 0].cpu().numpy()
                pred_no_pad = pred[:len(target_no_pad)].cpu().numpy()

                pred_text = " ".join([words[j]
                                     for j in pred_no_pad if j < len(words)])
                target_text = " ".join([words[j]
                                        for j in target_no_pad if j < len(words)])

                # Calculate individual WER/CER only for non-empty targets
                if target_text.strip():
                    individual_wer = wer([target_text], [pred_text])
                    individual_cer = cer([target_text], [pred_text])
                else:
                    individual_wer = 0.0
                    individual_cer = 0.0

                self.predictions['asr'].append({
                    'sample_id': len(self.predictions['asr']),
                    'words': words,
                    'target_text': target_text,
                    'predicted_text': pred_text,
                    'wer': individual_wer,
                    'cer': individual_cer,
                    'target_ids': target_no_pad.tolist(),
                    'predicted_ids': pred_no_pad.tolist(),
                    'original_target_length': len(target),
                    'valid_target_length': len(target_no_pad)
                })

        # Prosody predictions
        if 'prosody_logits' in outputs:
            prosody_preds = (outputs['prosody_logits'] > 0).float()
            prosody_targets = batch_data['prosody_targets']
            words_batch = batch_data['words']

            for i, (pred, target, words) in enumerate(zip(prosody_preds, prosody_targets, words_batch)):
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()

                # Create mask for valid positions (non-padding)
                valid_positions = target_np != 0
                if valid_positions.sum() > 0:
                    word_accuracy = (pred_np[valid_positions]
                                     == target_np[valid_positions]).mean()
                else:
                    word_accuracy = 0.0

                self.predictions['prosody'].append({
                    'sample_id': len(self.predictions['prosody']),
                    'words': words,
                    'target_prominence': target_np.tolist(),
                    'predicted_prominence': pred_np.tolist(),
                    'valid_mask': valid_positions.tolist(),
                    'word_accuracy': word_accuracy,
                    'num_words': len(words),
                    'num_valid_positions': int(valid_positions.sum()),
                    'num_prominent_true': int(target_np[valid_positions].sum()) if valid_positions.sum() > 0 else 0,
                    'num_prominent_pred': int(pred_np[valid_positions].sum()) if valid_positions.sum() > 0 else 0
                })

        # Emotion predictions
        if 'emotion_logits' in outputs:
            emotion_preds = outputs['emotion_logits'].argmax(dim=-1)
            emotion_targets = batch_data['emotion_targets']
            words_batch = batch_data['words']

            for i, (pred, target, words) in enumerate(zip(emotion_preds, emotion_targets, words_batch)):
                self.predictions['emotion'].append({
                    'sample_id': len(self.predictions['emotion']),
                    'words': words,
                    'target_emotion': target.item(),
                    'predicted_emotion': pred.item(),
                    'correct': (pred.item() == target.item()),
                    'utterance_text': " ".join(words)
                })

    def save_predictions(self):
        """Save all predictions to files"""
        for task, preds in self.predictions.items():
            if preds:
                save_path = os.path.join(
                    self.save_dir, f'{task}_predictions.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(preds, f, indent=2,
                              ensure_ascii=False, cls=NumpyEncoder)
                print(f"Saved {len(preds)} {task} predictions to {save_path}")

    def analyze_predictions(self):
        """Generate analysis reports"""
        analysis = {}

        # ASR Analysis
        if self.predictions['asr']:
            asr_preds = self.predictions['asr']
            analysis['asr'] = {
                'total_samples': len(asr_preds),
                'average_wer': float(np.mean([p['wer'] for p in asr_preds])),
                'average_cer': float(np.mean([p['cer'] for p in asr_preds])),
                'worst_wer_samples': sorted(asr_preds, key=lambda x: x['wer'], reverse=True)[:5],
                'best_wer_samples': sorted(asr_preds, key=lambda x: x['wer'])[:5]
            }

        # Prosody Analysis
        if self.predictions['prosody']:
            prosody_preds = self.predictions['prosody']
            analysis['prosody'] = {
                'total_samples': len(prosody_preds),
                'average_word_accuracy': float(np.mean([p['word_accuracy'] for p in prosody_preds])),
                'prominence_precision': float(self._calculate_prominence_precision(prosody_preds)),
                'prominence_recall': float(self._calculate_prominence_recall(prosody_preds)),
                'worst_accuracy_samples': sorted(prosody_preds, key=lambda x: x['word_accuracy'])[:5]
            }

        # Emotion Analysis
        if self.predictions['emotion']:
            emotion_preds = self.predictions['emotion']
            analysis['emotion'] = {
                'total_samples': len(emotion_preds),
                'accuracy': float(np.mean([p['correct'] for p in emotion_preds])),
                'confusion_matrix': self._calculate_emotion_confusion(emotion_preds),
                'misclassified_samples': [p for p in emotion_preds if not p['correct']][:10]
            }

        # Save analysis
        analysis_path = os.path.join(self.save_dir, 'prediction_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2,
                      ensure_ascii=False, cls=NumpyEncoder)

        return analysis

    def _calculate_prominence_precision(self, prosody_preds):
        """Calculate precision for prominence detection"""
        true_pos = sum(
            np.sum((np.array(p['predicted_prominence']) == 1) &
                   (np.array(p['target_prominence']) == 1))
            for p in prosody_preds
        )
        pred_pos = sum(
            np.sum(np.array(p['predicted_prominence']) == 1) for p in prosody_preds)
        return true_pos / pred_pos if pred_pos > 0 else 0.0

    def _calculate_prominence_recall(self, prosody_preds):
        """Calculate recall for prominence detection"""
        true_pos = sum(
            np.sum((np.array(p['predicted_prominence']) == 1) &
                   (np.array(p['target_prominence']) == 1))
            for p in prosody_preds
        )
        actual_pos = sum(
            np.sum(np.array(p['target_prominence']) == 1) for p in prosody_preds)
        return true_pos / actual_pos if actual_pos > 0 else 0.0

    def _calculate_emotion_confusion(self, emotion_preds):
        """Calculate confusion matrix for emotions"""
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))

        for pred in emotion_preds:
            true_label = pred['target_emotion']
            pred_label = pred['predicted_emotion']
            confusion[true_label][pred_label] += 1

        return dict(confusion)


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
        tokenizer = setup_tokenizer_and_dataset(
            dataset_dict, vocab_size=vocab_size)
        if tokenizer_path:
            # Save the newly trained tokenizer
            tokenizer.model_path = tokenizer_path
            tokenizer.sp.save(tokenizer_path)
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        tokenizer.load_tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(
        f"Special tokens - Blank: {tokenizer.blank_id}, Pad: {tokenizer.pad_id}, UNK: {tokenizer.unk_id}")

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
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id)
    )

    return model, train_loader, val_loader, test_loader, tokenizer


def collate_fn_mtl(batch, pad_token_id=0):
    """Custom collate function for MTL"""
    batch_size = len(batch)

    # Get maximum dimensions
    max_feature_len = max(item['input_features'].shape[-1] for item in batch)
    max_asr_len = max(item['asr_length'].item() if torch.is_tensor(
        item['asr_target']) else len(item['asr_target']) for item in batch)
    max_prosody_len = max(item['prosody_annotations'].shape[0]
                          for item in batch)

    # Feature dimension for first item
    n_mels = batch[0]['input_features'].shape[0]

    # Initialize tensors
    input_features = torch.zeros(batch_size, n_mels, max_feature_len)
    asr_targets = torch.full((batch_size, max_asr_len),
                             pad_token_id, dtype=torch.long)
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
    parser.add_argument("--backbone", type=str, default="whisper",
                        choices=list(BACKBONE_CONFIGS.keys()))
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
            lambda x: {"audio_filepath": os.path.join(
                args.audio_base_path, x["audio_filepath"])}
        )
        # Cast to Audio with 16000Hz sampling rate
        dataset_dict[split] = dataset_dict[split].cast_column(
            "audio_filepath", Audio(sampling_rate=16000))

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
    trainer.plot_training_history(os.path.join(
        args.save_dir, 'training_history.png'))

    # Load best model and evaluate on test set
    trainer.load_checkpoint(os.path.join(args.save_dir, 'best_model.pt'))
    evaluator = MTLEvaluator(model, device)

    # Evaluate with detailed prediction tracking
    test_metrics = evaluator.evaluate_with_tracking(
        test_loader,
        track_predictions=True,
        save_dir=os.path.join(args.save_dir, 'predictions')
    )

    # Print detailed results
    evaluator.print_detailed_results(test_metrics)

    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4,
                  ensure_ascii=False, cls=NumpyEncoder)
