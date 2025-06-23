"""
MTL Trainer Module
Paper-style trainer with proper alpha control and backbone handling
Following "Speech Emotion Recognition with Multi-task Learning" methodology
"""

import torch
import os
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from typing import Dict, Optional
import wandb

from sample_code.training.evaluator import MTLEvaluator
from sample_code.training.utils import NumpyEncoder
from sample_code.scripts.mtl_model import MTLModel

from sample_code.utils.visualization import plot_training_history, plot_task_metrics_comparison


class MTLTrainer:
    """
    Trainer for MTL with proper alpha control and backbone handling.

    Key features:
    - Optimizer with differential learning rates
    - Enhanced training step with AMP support
    - Alpha experimentation support
    - Proper logging following methodology
    """

    def __init__(self, model: MTLModel, device='cuda', use_wandb=False, use_amp=True, gradient_accumulation_steps=1):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = GradScaler(enabled=use_amp)
        self.model.to(device)

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': [],
            'alpha_history': [], 'task_loss_history': []
        }

        print(f"MTL Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP enabled: {use_amp}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  W&B logging: {use_wandb}")

    def create_paper_style_optimizer(self, backbone_lr: float = 1e-5, head_lr: float = 5e-5) -> torch.optim.Optimizer:
        """
        Create optimizer following paper's approach:
        - Lower learning rate for backbone (fine-tuning)
        - Higher learning rate for task heads (from scratch)
        """
        param_groups = []

        # Backbone parameters (lower LR for fine-tuning)
        backbone_params = list(self.model.backbone.parameters())
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })

        # Task head parameters (higher LR)
        head_params = []
        for name, module in self.model.named_children():
            if name != 'backbone':  # All non-backbone modules
                head_params.extend(list(module.parameters()))

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'task_heads'
            })

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=0.01, eps=1e-8)

        print(f"Optimizer created:")
        print(f"  Backbone LR: {backbone_lr}")
        print(f"  Task heads LR: {head_lr}")
        print(f"  Parameter groups: {len(param_groups)}")

        return optimizer

    def train_step(self, batch):
        """Single training step following paper's methodology"""
        self.model.train()

        # Move batch to device
        input_features = batch['input_features'].to(self.device)
        asr_targets = batch['asr_targets'].to(
            self.device) if torch.is_tensor(batch['asr_targets']) else None
        asr_lengths = batch['asr_lengths'].to(self.device)
        prosody_targets = batch['prosody_targets'].to(self.device)
        emotion_targets = batch['emotion_targets'].to(self.device)

        # Forward pass with AMP if enabled
        if self.use_amp:
            with torch.amp.autocast(device_type="cuda"):
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
        epoch_losses = {'total': 0, 'ser': 0, 'asr': 0, 'prosody': 0}
        ctc_stats_epoch = []
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
            if 'emotion_loss' in outputs:
                epoch_losses['ser'] += outputs['emotion_loss'].item()
            if 'asr_loss' in outputs:
                epoch_losses['asr'] += outputs['asr_loss'].item()

                # Track CTC statistics
                if 'loss_details' in outputs and 'asr' in outputs['loss_details']:
                    ctc_stats_epoch.append(outputs['loss_details']['asr'])

            if 'prosody_loss' in outputs:
                epoch_losses['prosody'] += outputs['prosody_loss'].item()

            num_batches += 1

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Delete references to free memory
            del batch, outputs, total_loss

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Average CTC statistics
        avg_ctc_stats = {}
        if ctc_stats_epoch:
            for key in ctc_stats_epoch[0].keys():
                avg_ctc_stats[key] = sum(
                    stats[key] for stats in ctc_stats_epoch) / len(ctc_stats_epoch)
            epoch_losses['ctc_stats'] = avg_ctc_stats

        return epoch_losses

    def evaluate_loss(self, data_loader):
        """Evaluate loss on a data loader"""
        self.model.eval()
        losses = {'total': 0, 'ser': 0, 'asr': 0, 'prosody': 0}
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

                # Delete references
                del batch, outputs

        # Average losses
        for key in losses:
            losses[key] /= num_batches

        return losses

    def train_paper_style(self, train_loader, val_loader, num_epochs, tokenizer,
                          backbone_lr=1e-5, head_lr=5e-5, save_dir='checkpoints',
                          early_stopping_patience=10, checkpoint_interval=5):
        """
        Complete training loop following paper's methodology.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            tokenizer: Tokenizer for ASR evaluation
            backbone_lr: Learning rate for backbone (fine-tuning)
            head_lr: Learning rate for task heads (training from scratch)
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            checkpoint_interval: Save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        best_ser_accuracy = 0.0
        patience_counter = 0
        best_epoch = 0

        print(f"\n{'='*60}")
        print(f"MTL TRAINING")
        print(f"{'='*60}")
        print(f"Main task: SER (α = {self.model.config.alpha_ser})")
        print(
            f"Auxiliary tasks: ASR (α = {self.model.config.alpha_asr}), Prosody (α = {self.model.config.alpha_prosody})")
        print(f"Loss formula: L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody")
        print(f"Total epochs: {num_epochs}")
        print(f"Backbone LR: {backbone_lr}, Head LR: {head_lr}")
        print(f"\n{'='*60}")
        # Create paper-style optimizer
        optimizer = self.create_paper_style_optimizer(backbone_lr, head_lr)

        # Create scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
        )

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # Training
            train_losses = self.train_epoch(train_loader, optimizer, None)

            # Print CTC loss details if available
            if 'ctc_stats' in train_losses:
                print(f"\nCTC Statistics:")
                ctc_stats = train_losses['ctc_stats']
                print(f"  CTC Loss: {ctc_stats.get('ctc_loss', 0):.4f}")
                print(
                    f"  Entropy Loss: {ctc_stats.get('entropy_loss', 0):.4f}")
                print(
                    f"  Blank Penalty: {ctc_stats.get('blank_penalty', 0):.4f}")
                print(f"  More Info:")
                print(
                    f"    Avg Blank Prob: {ctc_stats.get('avg_blank_prob', 0):.4f}")
                print(
                    f"    Blank Threshold: {ctc_stats.get('blank_threshold', 0):.4f}")
                print(
                    f"    Blank Penalty Weight: {ctc_stats.get('blank_penalty_weight', 0):.4f}")
                print(
                    f"    Entropy Weight: {ctc_stats.get('entropy_weight', 0):.4f}")

            # Validation
            evaluator = MTLEvaluator(
                self.model, tokenizer, self.device, self.use_amp, decode_method='greedy'
            )
            val_metrics = evaluator.evaluate(val_loader)
            val_losses = self.evaluate_loss(val_loader)

            # Update scheduler
            scheduler.step(val_losses['total'])

            # Print results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    SER: {train_losses['ser']:.4f} (main task)")
            print(
                f"    ASR: {train_losses['asr']:.4f} (α={self.model.config.alpha_asr})")
            print(
                f"    Prosody: {train_losses['prosody']:.4f} (α={self.model.config.alpha_prosody})")
            print(f"  Val Loss: {val_losses['total']:.4f}")

            # Print validation metrics
            ser_accuracy = 0.0
            evaluator.print_detailed_results(val_metrics)

            

            # Update history
            self.history['train_loss'].append(train_losses)
            self.history['val_loss'].append(val_losses)
            self.history['val_metrics'].append(val_metrics)
            self.history['alpha_history'].append(
                self.model.config.get_alpha_values())

            # Log to wandb if enabled
            if self.use_wandb:
                self.log_paper_style_metrics(
                    {
                        'total_loss': torch.tensor(train_losses['total']),
                        'emotion_loss': torch.tensor(train_losses['ser']),
                        'asr_loss': torch.tensor(train_losses['asr']),
                        'prosody_loss': torch.tensor(train_losses['prosody']),
                        'alpha_values': self.model.config.get_alpha_values()
                    },
                    epoch, 'train'
                )

                wandb.log({
                    'epoch': epoch,
                    'train_total_loss': train_losses['total'],
                    'val_total_loss': val_losses['total'],
                    'val_ser_accuracy': ser_accuracy,
                    'alpha_asr': self.model.config.alpha_asr,
                    'alpha_prosody': self.model.config.alpha_prosody
                })

            # Check for best model (prioritize validation loss for the whole task)
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(
                    os.path.join(save_dir, self.model.config.backbone_name+'_best_model.pt'),
                    epoch, optimizer, scheduler, is_best=True
                )
                print(
                    f"\n New best MTL Model: {best_val_loss:.4f} (saved)")
            else:
                patience_counter += 1

            # Periodic checkpoints
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    save_dir, f'{self.model.config.backbone_name}_checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(
                    checkpoint_path, epoch, optimizer, scheduler)
                print(f"Checkpoint saved at epoch {epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(
                    f"\n Early stopping triggered after {epoch + 1} epochs")
                break

        # Copy best model to final
        best_path = os.path.join(save_dir, self.model.config.backbone_name+'_best_model.pt')
        final_path = os.path.join(save_dir, self.model.config.backbone_name+'_final_model.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
            print(
                f"\n✅ Best model (epoch {best_epoch + 1}) copied to final_model.pt")

        # Save training history
        self.save_history(os.path.join(save_dir, self.model.config.backbone_name+'_training_history.json'))

        print(f"\n🎉 Training completed!")
        print(
            f"Best MTL Model with Val Loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        
        # Plot training history
        plot_training_history(self.history, os.path.join(save_dir, self.model.config.backbone_name+'_training_history.png'))
        # Plot task metrics comparison
        plot_task_metrics_comparison(self.history, os.path.join(save_dir, self.model.config.backbone_name+'_task_metrics_comparison.png'))

    def run_paper_ablation_study(self, train_loader, val_loader, tokenizer,
                                 alpha_values=[0.0, 0.001, 0.01, 0.1, 1.0],
                                 epochs_per_alpha=5, save_dir='ablation_results'):
        """
        Run ablation study following paper's methodology.
        Tests different alpha values to find optimal auxiliary task contribution.
        """
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY")
        print(f"{'='*60}")
        print(f"Testing alpha values: {alpha_values}")
        print(f"Epochs per alpha: {epochs_per_alpha}")

        os.makedirs(save_dir, exist_ok=True)
        ablation_results = {}

        for alpha in alpha_values:
            print(f"\n--- Testing Alpha = {alpha} ---")

            # Update model's alpha values
            # Same alpha for both auxiliary tasks
            self.model.update_alpha_values(alpha, alpha, alpha)

            # Create optimizer for this alpha configuration
            optimizer = self.create_paper_style_optimizer(
                backbone_lr=1e-5, head_lr=5e-5)

            # Reset model state for fair comparison
            # Note: In practice, you might want to start from a pre-trained checkpoint

            best_ser_acc = 0.0
            alpha_losses = []

            for epoch in range(epochs_per_alpha):
                # Training
                train_losses = self.train_epoch(train_loader, optimizer)

                # Quick evaluation
                self.model.eval()
                with torch.no_grad():
                    evaluator = MTLEvaluator(
                        self.model, tokenizer, self.device, self.use_amp, decode_method='greedy'
                    )
                    val_metrics = evaluator.evaluate(val_loader)
                    ser_acc = val_metrics.get(
                        'emotion', {}).get('accuracy', 0.0)

                    if ser_acc > best_ser_acc:
                        best_ser_acc = ser_acc

                    alpha_losses.append(train_losses['total'])

                print(
                    f"  Epoch {epoch+1}: SER Acc = {ser_acc:.4f}, Loss = {train_losses['total']:.4f}")

            ablation_results[alpha] = {
                'best_ser_accuracy': best_ser_acc,
                'final_ser_accuracy': ser_acc,
                'avg_loss': sum(alpha_losses) / len(alpha_losses),
                'loss_history': alpha_losses
            }

            print(f"Alpha {alpha} -> Best SER Accuracy: {best_ser_acc:.4f}")

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    f'ablation_alpha_{alpha}_best_accuracy': best_ser_acc,
                    f'ablation_alpha_{alpha}_avg_loss': ablation_results[alpha]['avg_loss']
                })

        # Find optimal alpha
        best_alpha = max(ablation_results.keys(),
                         key=lambda a: ablation_results[a]['best_ser_accuracy'])

        print(f"\n🎯 ABLATION STUDY RESULTS:")
        print(f"{'Alpha':<8} {'Best Acc':<10} {'Avg Loss':<10}")
        print("-" * 30)
        for alpha in alpha_values:
            results = ablation_results[alpha]
            marker = " <- BEST" if alpha == best_alpha else ""
            print(
                f"{alpha:<8} {results['best_ser_accuracy']:<10.4f} {results['avg_loss']:<10.4f}{marker}")

        print(f"\n Optimal Alpha: {best_alpha}")
        print(
            f"   Best SER Accuracy: {ablation_results[best_alpha]['best_ser_accuracy']:.4f}")
        print(f"   Paper's optimal: 0.1 (for comparison)")

        # Save ablation results
        ablation_path = os.path.join(save_dir, 'ablation_results.json')
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=4)

        # Set model to optimal alpha
        self.model.update_alpha_values(best_alpha, best_alpha)

        return ablation_results, best_alpha

    def log_paper_style_metrics(self, outputs, epoch, mode='train'):
        """Log metrics following paper's reporting style"""
        if self.use_wandb:
            log_dict = {
                f'{mode}_total_loss': outputs['total_loss'].item(),
                'epoch': epoch
            }

            # Individual task losses
            if 'emotion_loss' in outputs:
                log_dict[f'{mode}_ser_loss'] = outputs['emotion_loss'].item()
            if 'asr_loss' in outputs:
                log_dict[f'{mode}_asr_loss'] = outputs['asr_loss'].item()
            if 'prosody_loss' in outputs:
                log_dict[f'{mode}_prosody_loss'] = outputs['prosody_loss'].item()

            # Alpha values
            if 'alpha_values' in outputs:
                for key, value in outputs['alpha_values'].items():
                    log_dict[f'alpha_{key}'] = value

            # CTC regularization details
            if 'loss_details' in outputs and 'asr' in outputs['loss_details']:
                asr_details = outputs['loss_details']['asr']
                for key, value in asr_details.items():
                    log_dict[f'asr_{key}'] = value

            wandb.log(log_dict)

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None, is_best=False):
        """Save complete checkpoint with optimizer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model.config,
            'is_best': is_best,
            'history': self.history,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'alpha_values': self.model.config.get_alpha_values(),
            'paper_info': self.model.get_paper_training_info()
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

        # Restore alpha values if available
        if 'alpha_values' in checkpoint:
            alpha_vals = checkpoint['alpha_values']
            self.model.update_alpha_values(
                alpha_vals.get('alpha_ser', 1.0),  # Default to 1.0 for main task
                alpha_vals.get('alpha_asr', 0.1),
                alpha_vals.get('alpha_prosody', 0.1)
            )

        return checkpoint.get('epoch', 0)

    def save_history(self, path):
        """Save training history"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4,
                      ensure_ascii=False, cls=NumpyEncoder)

    def get_training_summary(self):
        """Get summary of training configuration and results"""
        return {
            'model_info': self.model.get_paper_training_info(),
            'training_config': {
                'device': str(self.device),
                'use_amp': self.use_amp,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'use_wandb': self.use_wandb
            },
            'current_alpha_values': self.model.config.get_alpha_values(),
            'loss_formula': 'L = α_SER * L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody',
            'training_history_length': len(self.history['train_loss']),
            'paper_reference': 'Speech Emotion Recognition with Multi-task Learning'
        }
