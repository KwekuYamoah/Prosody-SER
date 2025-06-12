import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import json
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


from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.mtl_model import MTLModel
from sample_code.scripts.mtl_dataset import MTLDataset
from sample_code.scripts.backbone_models import BACKBONE_CONFIGS
from sample_code.scripts.backbone_models import BackboneModel
from sample_code.scripts.tokenizer import SentencePieceTokenizer
from sample_code.scripts.ctc_decoder import CTCDecoder
from sample_code.scripts.memory_utils import print_memory_usage, cleanup_memory, optimize_model_for_memory

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

    def __init__(self, model, tokenizer, device='cuda', use_amp=True, decode_method='greedy'):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.use_amp = use_amp
        self.decode_method = decode_method
        self.model.to(device)
        self.model.eval()

        # Initialize CTC decoder
        self.ctc_decoder = CTCDecoder(blank_id=0, beam_width=10)

    def compute_asr_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor],
                            lengths: List[torch.Tensor] = None) -> Dict:
        """
        Compute ASR metrics (WER and CER) using proper CTC decoding.

        Args:
            predictions: List of logit tensors (not argmax'd)
            targets: List of target token ID tensors
            lengths: List of sequence lengths
        """
        # Decode predictions using CTC decoder
        pred_texts = []
        all_logits = torch.stack(predictions) if isinstance(
            predictions[0], torch.Tensor) else predictions

        # Use CTC decoder to get token sequences
        decoded_sequences = self.ctc_decoder.decode_batch(
            all_logits,
            lengths=lengths,
            method=self.decode_method
        )

        # Convert token sequences to text
        for decoded_ids in decoded_sequences:
            pred_texts.append(self.tokenizer.decode(
                decoded_ids, skip_special_tokens=True))

        # Decode target token IDs into text
        target_texts = []
        for i, target_ids in enumerate(targets):
            if torch.is_tensor(target_ids):
                target_ids = target_ids.cpu().numpy()

            # Get actual length and filter padding
            if lengths is not None and i < len(lengths):
                actual_length = lengths[i].item() if torch.is_tensor(
                    lengths[i]) else lengths[i]
                target_ids = target_ids[:actual_length]

            # Filter out padding tokens
            filtered_ids = [int(id)
                            for id in target_ids if id != self.tokenizer.pad_id]
            target_texts.append(self.tokenizer.decode(
                filtered_ids, skip_special_tokens=True))

        # Debug print for first few examples
        for i in range(min(3, len(pred_texts))):
            print(
                f"Debug - Target: '{target_texts[i]}', Predicted: '{pred_texts[i]}'")

        detailed_results = [
            {"predicted": p, "target": t} for p, t in zip(pred_texts, target_texts)
        ]

        # Calculate WER and CER on the decoded text
        wer_score = wer(target_texts, pred_texts) if pred_texts and any(
            pred_texts) else 1.0
        cer_score = cer(target_texts, pred_texts) if pred_texts and any(
            pred_texts) else 1.0

        return {
            "wer": wer_score,
            "cer": cer_score,
            # Keep first 5 for inspection
            "detailed_results": detailed_results[:5],
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
        """Evaluate model on a data loader with proper CTC handling"""
        all_predictions = {"asr": [], "prosody": [], "emotion": []}
        all_targets = {"asr": [], "prosody": [], "emotion": []}
        all_lengths = {"asr": []}
        detailed_results = {"prosody": [], "emotion": []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                input_features = batch["input_features"].to(
                    self.device, non_blocking=True)
                asr_targets = batch["asr_targets"].to(
                    self.device, non_blocking=True)
                asr_lengths = batch["asr_lengths"].to(
                    self.device, non_blocking=True)
                prosody_targets = batch["prosody_targets"].to(
                    self.device, non_blocking=True)
                emotion_targets = batch["emotion_targets"].to(
                    self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(
                        input_features=input_features,
                        asr_targets=asr_targets,
                        asr_lengths=asr_lengths,
                        prosody_targets=prosody_targets,
                        emotion_targets=emotion_targets,
                    )

                self._process_batch_predictions(
                    outputs, batch, all_predictions, all_targets, all_lengths, detailed_results
                )

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

                del batch, outputs, input_features, asr_targets, prosody_targets, emotion_targets

        return self._calculate_metrics(all_predictions, all_targets, all_lengths, detailed_results)

    def _process_batch_predictions(self, outputs, batch, all_predictions, all_targets,
                                   all_lengths, detailed_results):
        """Process batch predictions with proper CTC handling"""
        if 'asr_logits' in outputs and batch['asr_targets'] is not None:
            # Store logits for CTC decoding (not argmax)
            asr_logits = outputs['asr_logits'].detach().cpu()
            asr_targets = batch['asr_targets'].cpu()
            asr_lengths = batch['asr_lengths'].cpu()

            all_predictions['asr'].extend(asr_logits)
            all_targets['asr'].extend(asr_targets)
            all_lengths['asr'].extend(asr_lengths)

        if 'prosody_logits' in outputs:
            prosody_preds = (outputs['prosody_logits'] > 0).float().cpu()
            prosody_targets = batch['prosody_targets'].cpu()
            all_predictions['prosody'].extend(prosody_preds.numpy())
            all_targets['prosody'].extend(prosody_targets.numpy())

            # Add detailed results for prosody
            for i in range(len(prosody_preds)):
                if 'words' in batch and i < len(batch['words']):
                    detailed_results['prosody'].append({
                        'words': batch['words'][i],
                        'predicted': prosody_preds[i].numpy().tolist(),
                        'target': prosody_targets[i].numpy().tolist()
                    })

        if 'emotion_logits' in outputs:
            emotion_preds = outputs['emotion_logits'].argmax(
                dim=-1).cpu().numpy()
            emotion_targets = batch['emotion_targets'].cpu().numpy()
            all_predictions['emotion'].extend(emotion_preds)
            all_targets['emotion'].extend(emotion_targets)

            # Add detailed results for emotion
            emotion_labels = ['anger', 'contempt', 'disgust', 'fear',
                              'guilt', 'happy', 'sadness', 'shame', 'surprise']
            for i in range(len(emotion_preds)):
                detailed_results['emotion'].append({
                    'predicted': emotion_labels[emotion_preds[i]] if emotion_preds[i] < len(emotion_labels) else f"class_{emotion_preds[i]}",
                    'target': emotion_labels[emotion_targets[i]] if emotion_targets[i] < len(emotion_labels) else f"class_{emotion_targets[i]}"
                })

    def _calculate_metrics(self, all_predictions, all_targets, all_lengths, detailed_results):
        """Calculate metrics for all tasks with proper CTC decoding"""
        metrics = {}

        for task in ['asr', 'prosody', 'emotion']:
            if not all_predictions[task]:
                continue

            if task == 'asr':
                # Use proper CTC decoding for ASR metrics
                metrics[task] = self.compute_asr_metrics(
                    all_predictions['asr'],
                    all_targets['asr'],
                    all_lengths['asr']
                )
            elif task == 'prosody':
                flat_preds, flat_targets = self.flatten_and_filter_sequences(
                    all_predictions['prosody'], all_targets['prosody'])
                if len(flat_preds) > 0:
                    metrics[task] = {
                        'accuracy': accuracy_score(flat_targets, flat_preds),
                        'f1': f1_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                        'detailed_results': detailed_results['prosody'][:5]
                    }
            elif task == 'emotion':
                metrics[task] = {
                    'accuracy': accuracy_score(all_targets['emotion'], all_predictions['emotion']),
                    'f1': f1_score(all_targets['emotion'], all_predictions['emotion'], average='weighted', zero_division=0),
                    'detailed_results': detailed_results['emotion'][:5]
                }

        return metrics

    def print_detailed_results(self, metrics):
        """Print detailed results for each task"""
        print("\nDetailed Evaluation Results:")

        for task in ['asr', 'prosody', 'emotion']:
            if task not in metrics:
                continue

            print(f"\n{task.upper()} Results:")

            # Print metrics
            for metric_name, value in metrics[task].items():
                if metric_name != 'detailed_results' and not isinstance(value, list):
                    print(f"{metric_name}: {value:.4f}")

            # Print sample predictions if available
            if 'detailed_results' in metrics[task] and metrics[task]['detailed_results']:
                print("\nSample Predictions:")
                for i, result in enumerate(metrics[task]['detailed_results'][:5]):
                    print(f"\nExample {i+1}:")
                    if task == 'asr':
                        print(f"Target: {result['target']}")
                        print(f"Predicted: {result['predicted']}")
                    elif task == 'prosody':
                        if 'words' in result:
                            print("Words:", " ".join(result['words']))
                        print("Target Prominence:", " ".join(
                            map(str, result['target'])))
                        print("Predicted Prominence:", " ".join(
                            map(str, result['predicted'])))
                    else:  # emotion
                        print(f"Target Emotion: {result['target']}")
                        print(f"Predicted Emotion: {result['predicted']}")
            else:
                print("\nNo detailed results available for this task.")


class MTLTrainer:
    """Enhanced trainer class with gradient accumulation support"""

    def __init__(self, model, device='cuda', use_wandb=False, use_amp=True, gradient_accumulation_steps=1):
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_wandb = use_wandb
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
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
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
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
                        self.model.parameters(), max_norm=0.5)

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

            if 'loss_details' in outputs:
                if 'loss_details' not in epoch_losses:
                    epoch_losses['loss_details'] = {}
                for task, details in outputs['loss_details'].items():
                    if task not in epoch_losses['loss_details']:
                        epoch_losses['loss_details'][task] = {}
                    for key, value in details.items():
                        if key not in epoch_losses['loss_details'][task]:
                            epoch_losses['loss_details'][task][key] = 0
                        epoch_losses['loss_details'][task][key] += value.item(
                        ) if torch.is_tensor(value) else value

            num_batches += 1

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Delete references to free memory
            del batch, outputs, total_loss

        # Average losses
        # for key in epoch_losses:
        #     epoch_losses[key] /= num_batches

        # if 'loss_details' in epoch_losses:
        #     for task in epoch_losses['loss_details']:
        #         for key in epoch_losses['loss_details'][task]:
        #             epoch_losses['loss_details'][task][key] /= num_batches

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
                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
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

    def create_optimizer_with_differential_lr(self, model, base_lr: float,
                                              asr_lr_multiplier: float = 0.1) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for ASR head"""
        asr_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:  # Only include trainable parameters
                if 'asr_head' in name or 'asr_layer_norm' in name:
                    asr_params.append(param)
                else:
                    other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append(
                {'params': other_params, 'lr': base_lr, 'name': 'other'})
        if asr_params:
            param_groups.append(
                {'params': asr_params, 'lr': base_lr * asr_lr_multiplier, 'name': 'asr'})

        return torch.optim.AdamW(param_groups, weight_decay=0.01, eps=1e-8)

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

            # Validation - FIXED: No circular import needed
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp, decode_method='greedy')
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

    def train_with_freeze_unfreeze(self, train_loader, val_loader, num_epochs, tokenizer, base_lr=1e-4,
                                   save_dir='checkpoints', early_stopping_patience=7, checkpoint_interval=5, unfreeze_epoch_ratio=0.5,
                                   lr_reduction_factor=0.1, dynamic_loss_weights=True
                                   ):
        """
        Complete training loop with freeze/unfreeze strategy.

        Args:
            unfreeze_epoch_ratio: Fraction of epochs to train with frozen encoder
            lr_reduction_factor: Factor to reduce LR when unfreezing
            dynamic_loss_weights: Whether to adjust loss weights during training
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        unfreeze_epoch = int(num_epochs * unfreeze_epoch_ratio)

        # Initial setup
        print(f"\nTraining Configuration:")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Unfreeze at epoch: {unfreeze_epoch}")
        print(
            f"  Initial encoder state: {'Frozen' if self.model.config.freeze_encoder else 'Unfrozen'}")

        # Freeze encoder initially
        if self.model.config.freeze_encoder:
            self.model.backbone.freeze_encoder()
            print(
                f"  Trainable parameters: {self.model.backbone.get_num_trainable_params():,}")
            print(
                f"  Total parameters: {self.model.backbone.get_num_total_params():,}")

        # Create optimizer with differential learning rates
        optimizer = self.create_optimizer_with_differential_lr(
            self.model,
            base_lr=base_lr,
            asr_lr_multiplier=self.model.config.asr_lr_multiplier
        )

        # Create scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Add to history tracking
        if 'freeze_status' not in self.history:
            self.history['freeze_status'] = []
        if 'loss_weights' not in self.history:
            self.history['loss_weights'] = []

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # Check if we should unfreeze
            if epoch == unfreeze_epoch and self.model.backbone.is_frozen():
                print(f"\n Unfreezing encoder at epoch {epoch + 1}")
                self.model.backbone.unfreeze_encoder()

                # Reduce learning rate for all parameters
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= lr_reduction_factor
                    print(
                        f"  {param_group['name']} LR: {old_lr:.2e} → {param_group['lr']:.2e}")

                print(
                    f"  Trainable parameters: {self.model.backbone.get_num_trainable_params():,}")

            # Dynamic loss weight adjustment
            if dynamic_loss_weights:
                if epoch < unfreeze_epoch:
                    # Stage 1: Higher ASR weight
                    loss_weights = {'asr': 1.0, 'prosody': 0.3, 'ser': 0.3}
                elif epoch < unfreeze_epoch + 5:
                    # Transition period: Gradually balance weights
                    progress = (epoch - unfreeze_epoch) / 5
                    asr_weight = 1.0 - 0.3 * progress  # 1.0 → 0.7
                    other_weight = 0.3 + 0.4 * progress  # 0.3 → 0.7
                    loss_weights = {'asr': asr_weight,
                                    'prosody': other_weight, 'ser': other_weight}
                else:
                    # Stage 2: Balanced weights
                    loss_weights = {'asr': 0.7, 'prosody': 0.7, 'ser': 0.7}

                self.model.config.update_loss_weights(loss_weights)
                print(f"  Loss weights: ASR={loss_weights['asr']:.2f}, "
                      f"Prosody={loss_weights['prosody']:.2f}, SER={loss_weights['ser']:.2f}")

            # Training
            train_losses = self.train_epoch(train_loader, optimizer)

            # Print CTC loss details if available
            if 'loss_details' in train_losses and 'asr' in train_losses['loss_details']:
                print(f"\nCTC Loss Details:")
                print(
                    f"  CTC Loss: {train_losses['loss_details']['asr']['ctc_loss']:.4f}")
                print(
                    f"  Entropy Loss: {train_losses['loss_details']['asr']['entropy_loss']:.4f}")
                print(
                    f"  Blank Penalty: {train_losses['loss_details']['asr']['blank_penalty']:.4f}")

            # Validation
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp, decode_method='greedy')
            val_metrics = evaluator.evaluate(val_loader)
            val_losses = self.evaluate_loss(val_loader)

            # Update scheduler
            scheduler.step(val_losses['total'])

            # Print results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            evaluator.print_detailed_results(val_metrics)

            # Update history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['val_metrics'].append(val_metrics)
            self.history['freeze_status'].append(
                self.model.backbone.is_frozen())
            self.history['loss_weights'].append(
                self.model.config.loss_weights.copy())

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pt'),
                    epoch, optimizer, scheduler, is_best=True
                )
                print(
                    f"\n New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Periodic checkpoints
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(
                    checkpoint_path, epoch, optimizer, scheduler)
                print(f" Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n Early stopping triggered after {epoch + 1} epochs")
                break

        # Copy best model to final
        best_path = os.path.join(save_dir, 'best_model.pt')
        final_path = os.path.join(save_dir, 'final_model.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
            print(
                f"\n Best model (epoch {best_epoch + 1}) copied to final_model.pt")

        # Save training history
        self.save_history(os.path.join(save_dir, 'training_history.json'))

        # Plot enhanced training history
        self.plot_enhanced_training_history(
            os.path.join(save_dir, 'training_history.png'))

    def plot_enhanced_training_history(self, save_path=None):
        """Plot enhanced training history with freeze/unfreeze markers"""
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return

        plt.figure(figsize=(20, 12))
        epochs = range(len(self.history['train_loss']))

        # Plot 1: Total Loss with freeze/unfreeze regions
        plt.subplot(3, 2, 1)
        train_total = [x['total'] for x in self.history['train_loss']]
        val_total = [x['total'] for x in self.history['val_loss']]

        plt.plot(epochs, train_total, label='Train Loss', linewidth=2)
        plt.plot(epochs, val_total, label='Val Loss', linewidth=2)

        # Add freeze/unfreeze shading
        if 'freeze_status' in self.history:
            frozen_epochs = [i for i, frozen in enumerate(
                self.history['freeze_status']) if frozen]
            if frozen_epochs:
                plt.axvspan(0, max(frozen_epochs) + 1, alpha=0.2,
                            color='blue', label='Encoder Frozen')

        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Individual Task Losses
        plt.subplot(3, 2, 2)
        for task in ['asr', 'prosody', 'emotion']:
            if task in self.history['train_loss'][0]:
                values = [x.get(task, 0) for x in self.history['train_loss']]
                plt.plot(epochs, values,
                         label=f'{task.upper()} Loss', linewidth=2)

        plt.title('Individual Task Losses (Training)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: ASR Metrics
        plt.subplot(3, 2, 3)
        if len(self.history['val_metrics']) > 0 and 'asr' in self.history['val_metrics'][0]:
            wer_values = [x['asr'].get('wer', 1.0)
                          for x in self.history['val_metrics']]
            cer_values = [x['asr'].get('cer', 1.0)
                          for x in self.history['val_metrics']]
            plt.plot(epochs, wer_values, label='WER', linewidth=2)
            plt.plot(epochs, cer_values, label='CER', linewidth=2)

        plt.title('ASR Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Prosody & Emotion Metrics
        plt.subplot(3, 2, 4)
        for task in ['prosody', 'emotion']:
            if len(self.history['val_metrics']) > 0 and task in self.history['val_metrics'][0]:
                acc_values = [x[task].get('accuracy', 0)
                              for x in self.history['val_metrics']]
                plt.plot(epochs, acc_values,
                         label=f'{task.capitalize()} Accuracy', linewidth=2)

        plt.title('Classification Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 5: Loss Weights Evolution
        plt.subplot(3, 2, 5)
        if 'loss_weights' in self.history and self.history['loss_weights']:
            asr_weights = [x['asr'] for x in self.history['loss_weights']]
            prosody_weights = [x['prosody']
                               for x in self.history['loss_weights']]
            ser_weights = [x['ser'] for x in self.history['loss_weights']]

            plt.plot(epochs, asr_weights, label='ASR Weight', linewidth=2)
            plt.plot(epochs, prosody_weights,
                     label='Prosody Weight', linewidth=2)
            plt.plot(epochs, ser_weights, label='SER Weight', linewidth=2)

        plt.title('Loss Weight Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 6: Learning Rate Evolution
        plt.subplot(3, 2, 6)
        plt.text(0.5, 0.5, 'Learning Rate Plot\n(Add LR tracking to history)',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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

    def save_checkpoint_enhanced(self, path, epoch, optimizer, scheduler=None, is_best=False):
        """Save complete checkpoint with freeze status"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model.config,
            'is_best': is_best,
            'history': self.history,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'is_encoder_frozen': self.model.backbone.is_frozen()
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

    def load_checkpoint_enhanced(self, path, optimizer=None, scheduler=None):
        """Load complete checkpoint with freeze status"""
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'is_encoder_frozen' in checkpoint and checkpoint['is_encoder_frozen']:
            self.model.backbone.freeze_encoder()

        return checkpoint.get('epoch', 0)

    def save_history(self, path):
        """Save training history"""
        from sample_code.train_mtl import NumpyEncoder  # Import the custom encoder

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
    CRITICAL FIX: Properly handle different backbone input shapes by finding
    the max size of all dimensions in the batch.
    """
    batch_size = len(batch)

    # Handle input features based on backbone type
    if backbone_name in ["whisper", "wav2vec2-bert"]:
        # --- FIX STARTS HERE ---
        # Find the max dimensions for both frequency and time across the entire batch.
        max_n_mels = max(item['input_features'].shape[0] for item in batch)
        max_time = max(item['input_features'].shape[1] for item in batch)

        # Create a padded tensor using the maximum dimensions found.
        input_features = torch.zeros(batch_size, max_n_mels, max_time)

        # Copy each item into the correctly-sized padded tensor.
        for i, item in enumerate(batch):
            n_mels, time_len = item['input_features'].shape
            input_features[i, :n_mels, :time_len] = item['input_features']
        # --- FIX ENDS HERE ---

    elif backbone_name in ["xlsr", "mms"]:
        # This part for 1D audio is likely correct but kept for completeness.
        max_len = max(item['input_features'].shape[0] for item in batch)
        input_features = torch.zeros(batch_size, max_len)
        for i, item in enumerate(batch):
            feat = item['input_features'].flatten()
            length = feat.shape[0]
            input_features[i, :length] = feat
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # --- The rest of the function remains the same ---

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

    # New arguments for freeze/unfreeze strategy
    parser.add_argument("--freeze_encoder_initially", action="store_true", default=True,
                        help="Whether to freeze encoder initially")
    parser.add_argument("--unfreeze_epoch_ratio", type=float, default=0.5,
                        help="Fraction of epochs to train with frozen encoder")
    parser.add_argument("--lr_reduction_factor", type=float, default=0.1,
                        help="Factor to reduce LR when unfreezing")
    parser.add_argument("--ctc_entropy_weight", type=float, default=0.01,
                        help="Entropy regularization weight for CTC")
    parser.add_argument("--ctc_blank_weight", type=float, default=0.95,
                        help="Maximum blank probability for CTC")
    parser.add_argument("--asr_lr_multiplier", type=float, default=0.1,
                        help="Learning rate multiplier for ASR head")
    parser.add_argument("--use_enhanced_training", action="store_true", default=True,
                        help="Use enhanced training with freeze/unfreeze strategy")

    args = parser.parse_args()

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"\nTraining Configuration:")
    print(f"  Per-GPU batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Enhanced training: {args.use_enhanced_training}")

    if args.use_enhanced_training:
        print(f"  Freeze/Unfreeze Strategy:")
        print(
            f"    Initial state: {'Frozen' if args.freeze_encoder_initially else 'Unfrozen'}")
        print(
            f"    Unfreeze at: {int(args.num_epochs * args.unfreeze_epoch_ratio)} epochs")
        print(f"    LR reduction: {args.lr_reduction_factor}x")
        print(f"  CTC Configuration:")
        print(f"    Entropy weight: {args.ctc_entropy_weight}")
        print(f"    Blank weight: {args.ctc_blank_weight}")
        print(f"    ASR LR multiplier: {args.asr_lr_multiplier}")

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
        config_dict = vars(args)
        config_dict['effective_batch_size'] = effective_batch_size
        wandb.init(project="mtl-speech", config=config_dict)

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
    else:
        print(f"\nLoading existing tokenizer from {args.tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
        tokenizer.load_tokenizer()

    # Test tokenizer
    text = "hello world test"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"\nTokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer blank ID: {tokenizer.blank_id}")
    print(f"Sample encoding test: {ids}")
    print(f"Sample decode test: {decoded}\n")

    if args.use_enhanced_training:
        config = MTLConfig(
            backbone_name=args.backbone,
            vocab_size=tokenizer.get_vocab_size(),
            emotion_classes=9,
            prosody_classes=2,
            freeze_encoder=args.freeze_encoder_initially,
            loss_weights={
                'asr': 1.0,      # Higher weight initially
                'prosody': 0.3,  # Lower weight initially
                'ser': 0.3       # Lower weight initially
            },
            # CTC parameters
            ctc_entropy_weight=args.ctc_entropy_weight,
            ctc_blank_weight=args.ctc_blank_weight,
            asr_lr_multiplier=args.asr_lr_multiplier,
            warmup_steps=1000
        )
    else:
        # Original config for standard training
        config = MTLConfig(
            backbone_name=args.backbone,
            vocab_size=tokenizer.get_vocab_size(),
            emotion_classes=9,
            prosody_classes=2,
            loss_weights={
                'asr': 0.3,
                'prosody': 0.3,
                'ser': 0.4
            },
        )

    # Create model
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    ).to(device)

    print(f"\nCreated MTL model with backbone: {args.backbone}")
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

    # Train with memory monitoring
    trainer = MTLTrainer(
        model,
        device=device,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Add the enhanced methods to the trainer instance
    # trainer.create_optimizer_with_differential_lr = lambda model, base_lr, asr_lr_multiplier: create_optimizer_with_differential_lr(
    #     trainer, model, base_lr, asr_lr_multiplier)
    # trainer.train_with_freeze_unfreeze = lambda *args, **kwargs: train_with_freeze_unfreeze(
    #     trainer, *args, **kwargs)
    # trainer.plot_enhanced_training_history = lambda *args, **kwargs: plot_enhanced_training_history(
    #     trainer, *args, **kwargs)
    # trainer.save_checkpoint = lambda *args, **kwargs: save_checkpoint_enhanced(
    #     trainer, *args, **kwargs)
    # trainer.load_checkpoint = lambda *args, **kwargs: load_checkpoint_enhanced(
    #     trainer, *args, **kwargs)

    print("\nStarting training...")

    if args.use_enhanced_training:
        # Use enhanced training with freeze/unfreeze strategy
        trainer.train_with_freeze_unfreeze(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            tokenizer=tokenizer,
            base_lr=adjusted_lr,
            save_dir=args.save_dir,
            early_stopping_patience=7,  # More patience for two-stage training
            checkpoint_interval=5,
            unfreeze_epoch_ratio=args.unfreeze_epoch_ratio,
            lr_reduction_factor=args.lr_reduction_factor,
            dynamic_loss_weights=True
        )
    else:
        # Use original training method
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
            steps_per_epoch = len(
                train_loader) // args.gradient_accumulation_steps
            total_steps = steps_per_epoch * args.num_epochs

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
            print(
                f"  Using cosine annealing scheduler with {total_steps} total steps")

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
    if args.use_enhanced_training:
        trainer.plot_enhanced_training_history(
            os.path.join(args.save_dir, 'training_history.png')
        )
    else:
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
    # Load best model and evaluate on test set
    trainer.load_checkpoint(os.path.join(args.save_dir, 'best_model.pt'))

    evaluator = MTLEvaluator(model, tokenizer, device,
                             use_amp=args.use_amp, decode_method='beam')
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_detailed_results(test_metrics)

    # Save results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
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
            "use_scheduler": args.use_scheduler,
            "enhanced_training": args.use_enhanced_training
        })

        if args.use_enhanced_training:
            wandb.config.update({
                "freeze_encoder_initially": args.freeze_encoder_initially,
                "unfreeze_epoch_ratio": args.unfreeze_epoch_ratio,
                "lr_reduction_factor": args.lr_reduction_factor,
                "ctc_entropy_weight": args.ctc_entropy_weight,
                "ctc_blank_weight": args.ctc_blank_weight,
                "asr_lr_multiplier": args.asr_lr_multiplier
            })

        wandb.finish()
