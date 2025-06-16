"""
MTL Trainer Module with Paper-Style Alpha Control
Handles model training with SER as main task and ASR/Prosody as auxiliary tasks
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
    """Enhanced trainer class with paper-style alpha control for auxiliary tasks"""

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
            'train_metrics': [], 'val_metrics': [],
            'alpha_history': []  # Track alpha values over time
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

    def train_epoch_with_alpha_logging(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch with alpha value logging"""
        epoch_losses = {'total': 0, 'ser': 0, 'asr': 0, 'prosody': 0}
        alpha_values = {'alpha_asr': 0, 'alpha_prosody': 0}
        num_batches = 0

        # Zero gradients at the start
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Forward pass
            outputs = self.train_step(batch)

            # Scale loss by accumulation steps
            total_loss = outputs['total_loss'] / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()

            # Accumulate losses
            epoch_losses['total'] += total_loss.item() * self.gradient_accumulation_steps
            if 'emotion_loss' in outputs:
                epoch_losses['ser'] += outputs['emotion_loss'].item()
            if 'asr_loss' in outputs:
                epoch_losses['asr'] += outputs['asr_loss'].item()
            if 'prosody_loss' in outputs:
                epoch_losses['prosody'] += outputs['prosody_loss'].item()

            # Track alpha values
            if 'alpha_values' in outputs:
                alpha_values['alpha_asr'] = outputs['alpha_values']['alpha_asr']
                alpha_values['alpha_prosody'] = outputs['alpha_values']['alpha_prosody']

            num_batches += 1

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            del batch, outputs, total_loss

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses, alpha_values

    def train_with_alpha_control(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        tokenizer,
        save_dir='checkpoints',
        early_stopping_patience=5,
        checkpoint_interval=10,
        alpha_asr=0.1,
        alpha_prosody=0.1
    ):
        """Training loop with fixed alpha values following the paper"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        print(f"\nStarting training with paper-style alpha control:")
        print(f"  SER (main task): weight = 1.0")
        print(f"  ASR (auxiliary): alpha = {alpha_asr}")
        print(f"  Prosody (auxiliary): alpha = {alpha_prosody}")

        # Update model's alpha values
        self.model.config.update_alpha_weights(alpha_asr, alpha_prosody)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            train_losses, alpha_vals = self.train_epoch_with_alpha_logging(train_loader, optimizer)

            # Validation
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp, decode_method='greedy')
            val_metrics = evaluator.evaluate(val_loader)
            val_losses = self.evaluate_loss(val_loader)

            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_total_loss': train_losses['total'],
                'val_total_loss': val_losses['total'],
                'alpha_asr': alpha_vals['alpha_asr'],
                'alpha_prosody': alpha_vals['alpha_prosody']
            }

            for task in ['ser', 'asr', 'prosody']:
                task_map = {'ser': 'emotion', 'asr': 'asr', 'prosody': 'prosody'}
                actual_task = task_map[task]
                
                if task in train_losses:
                    metrics[f'train_{task}_loss'] = train_losses[task]
                if task in val_losses:
                    metrics[f'val_{task}_loss'] = val_losses[task]
                if actual_task in val_metrics:
                    for metric_name, value in val_metrics[actual_task].items():
                        if metric_name != 'detailed_results':
                            metrics[f'val_{task}_{metric_name}'] = value

            if self.use_wandb:
                wandb.log(metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Alpha ASR: {alpha_vals['alpha_asr']}, Alpha Prosody: {alpha_vals['alpha_prosody']}")
            
            evaluator.print_detailed_results(val_metrics)

            # Save history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['train_metrics'].append(metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['alpha_history'].append(alpha_vals)

            # Check for best model and save
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch, optimizer, None, is_best=True)
                print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Save checkpoint at specified intervals
            if (epoch + 1) % checkpoint_interval == 0 and epoch != best_epoch:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path, epoch, optimizer, None)
                print(f"  Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        self.save_history(os.path.join(save_dir, 'training_history.json'))

    def train_with_freeze_unfreeze_alpha(
        self,
        train_loader,
        val_loader,
        num_epochs,
        tokenizer,
        base_lr=1e-4,
        save_dir='checkpoints',
        early_stopping_patience=7,
        checkpoint_interval=5,
        unfreeze_epoch_ratio=0.5,
        lr_reduction_factor=0.1,
        alpha_asr=0.1,
        alpha_prosody=0.1
    ):
        """Enhanced training with freeze/unfreeze strategy and paper-style alpha control"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        unfreeze_epoch = int(num_epochs * unfreeze_epoch_ratio)

        print(f"\nPaper-Style Enhanced Training Configuration:")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Unfreeze at epoch: {unfreeze_epoch}")
        print(f"  SER (main task): weight = 1.0")
        print(f"  ASR (auxiliary): alpha = {alpha_asr}")
        print(f"  Prosody (auxiliary): alpha = {alpha_prosody}")

        # Update model's alpha values
        self.model.config.update_alpha_weights(alpha_asr, alpha_prosody)

        # Freeze encoder initially
        if self.model.config.freeze_encoder:
            self.model.backbone.freeze_encoder()
            print(f"  Trainable parameters: {self.model.backbone.get_num_trainable_params():,}")

        # Create optimizer with differential learning rates
        optimizer = self.create_optimizer_with_differential_lr(
            self.model,
            base_lr=base_lr,
            asr_lr_multiplier=self.model.config.asr_lr_multiplier
        )

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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
                    print(f"  {param_group['name']} LR: {old_lr:.2e} → {param_group['lr']:.2e}")

                print(f"  Trainable parameters: {self.model.backbone.get_num_trainable_params():,}")

            # Training
            train_losses, alpha_vals = self.train_epoch_with_alpha_logging(train_loader, optimizer)

            # Validation
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp, decode_method='greedy')
            val_metrics = evaluator.evaluate(val_loader)
            val_losses = self.evaluate_loss(val_loader)

            scheduler.step(val_losses['total'])

            # Print results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Alpha Values - ASR: {alpha_vals['alpha_asr']}, Prosody: {alpha_vals['alpha_prosody']}")
            evaluator.print_detailed_results(val_metrics)

            # Update history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['val_metrics'].append(val_metrics)
            self.history['alpha_history'].append(alpha_vals)

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pt'),
                    epoch, optimizer, scheduler, is_best=True
                )
                print(f"\n🎯 New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Periodic checkpoints
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)
                print(f"💾 Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

        # Save training history
        self.save_history(os.path.join(save_dir, 'training_history.json'))

    def evaluate_loss(self, data_loader):
        """Evaluate loss on a data loader"""
        self.model.eval()
        losses = {'total': 0, 'ser': 0, 'asr': 0, 'prosody': 0}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
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
                if 'emotion_loss' in outputs:
                    losses['ser'] += outputs['emotion_loss'].item()
                if 'asr_loss' in outputs:
                    losses['asr'] += outputs['asr_loss'].item()
                if 'prosody_loss' in outputs:
                    losses['prosody'] += outputs['prosody_loss'].item()

                num_batches += 1

                # Clear cache periodically
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

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
            if param.requires_grad:
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

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None, is_best=False):
        """Save complete checkpoint with optimizer state and alpha values"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model.config,
            'is_best': is_best,
            'history': self.history,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'alpha_values': {
                'alpha_asr': self.model.config.alpha_asr,
                'alpha_prosody': self.model.config.alpha_prosody
            }
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

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

        # Restore alpha values
        if 'alpha_values' in checkpoint:
            alpha_vals = checkpoint['alpha_values']
            self.model.config.update_alpha_weights(
                alpha_vals['alpha_asr'], 
                alpha_vals['alpha_prosody']
            )
            print(f"Restored alpha values: ASR={alpha_vals['alpha_asr']}, Prosody={alpha_vals['alpha_prosody']}")

        # Restore freeze status if available
        if 'is_encoder_frozen' in checkpoint and checkpoint['is_encoder_frozen']:
            if hasattr(self.model.backbone, 'freeze_encoder'):
                self.model.backbone.freeze_encoder()

        return checkpoint.get('epoch', 0)

    def save_history(self, path):
        """Save training history with alpha tracking"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4,
                      ensure_ascii=False, cls=NumpyEncoder)

    def get_alpha_summary(self) -> dict:
        """Get summary of alpha values used during training"""
        if not self.history['alpha_history']:
            return {}
        
        alpha_asr_values = [h['alpha_asr'] for h in self.history['alpha_history']]
        alpha_prosody_values = [h['alpha_prosody'] for h in self.history['alpha_history']]
        
        return {
            'alpha_asr': {
                'final': alpha_asr_values[-1] if alpha_asr_values else 0,
                'mean': sum(alpha_asr_values) / len(alpha_asr_values) if alpha_asr_values else 0,
                'constant': len(set(alpha_asr_values)) == 1
            },
            'alpha_prosody': {
                'final': alpha_prosody_values[-1] if alpha_prosody_values else 0,
                'mean': sum(alpha_prosody_values) / len(alpha_prosody_values) if alpha_prosody_values else 0,
                'constant': len(set(alpha_prosody_values)) == 1
            }
        }