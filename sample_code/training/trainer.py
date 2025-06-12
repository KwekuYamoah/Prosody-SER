"""
MTL Trainer Module
Handles model training, optimization, and checkpointing
"""

import torch
import os
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional
import wandb

from .evaluator import MTLEvaluator
from .utils import NumpyEncoder


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
        for key in epoch_losses:
            if key != 'loss_details':
                epoch_losses[key] /= num_batches

        if 'loss_details' in epoch_losses:
            for task in epoch_losses['loss_details']:
                for key in epoch_losses['loss_details'][task]:
                    epoch_losses['loss_details'][task][key] /= num_batches

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

            # Validation
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
                                   save_dir='checkpoints', early_stopping_patience=7, checkpoint_interval=5,
                                   unfreeze_epoch_ratio=0.5, lr_reduction_factor=0.1, dynamic_loss_weights=True):
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
                print(f"\n⚡ Unfreezing encoder at epoch {epoch + 1}")
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
                    f"\n🎯 New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Periodic checkpoints
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(
                    checkpoint_path, epoch, optimizer, scheduler)
                print(f"💾 Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(
                    f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

        # Copy best model to final
        best_path = os.path.join(save_dir, 'best_model.pt')
        final_path = os.path.join(save_dir, 'final_model.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
            print(
                f"\n✅ Best model (epoch {best_epoch + 1}) copied to final_model.pt")

        # Save training history
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

        # Add freeze status if using enhanced training
        if hasattr(self.model.backbone, 'is_frozen'):
            checkpoint['is_encoder_frozen'] = self.model.backbone.is_frozen()

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

        # Restore freeze status if available
        if 'is_encoder_frozen' in checkpoint and checkpoint['is_encoder_frozen']:
            if hasattr(self.model.backbone, 'freeze_encoder'):
                self.model.backbone.freeze_encoder()

        return checkpoint.get('epoch', 0)

    def save_history(self, path):
        """Save training history"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4,
                      ensure_ascii=False, cls=NumpyEncoder)
