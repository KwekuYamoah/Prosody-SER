"""
Main Training Script for MTL Model
Clean and modular implementation
"""

import torch
import os
import argparse
import math
import json
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import core modules
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.mtl_model import MTLModel
from sample_code.scripts.mtl_dataset import MTLDataset
from sample_code.scripts.backbone_models import BACKBONE_CONFIGS, BackboneModel
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# Import utilities
from sample_code.scripts.memory_utils import print_memory_usage, cleanup_memory, optimize_model_for_memory

# Import training modules
from sample_code.training import MTLTrainer, MTLEvaluator, collate_fn_mtl, configure_cuda_memory, NumpyEncoder

from sample_code.utils import (
    setup_tokenizer_and_dataset,
    load_and_prepare_datasets,
    plot_training_history,
    plot_enhanced_training_history
)

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.serialization.add_safe_globals([MTLConfig])


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MTL Speech Model")

    # Model configuration
    parser.add_argument("--backbone", type=str, default="whisper",
                        choices=list(BACKBONE_CONFIGS.keys()),
                        help="Backbone model to use")
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="Vocabulary size for tokenizer")

    # Data configuration
    parser.add_argument("--audio_base_path", type=str, required=True,
                        help="Base path to audio files directory")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", type=str, required=True,
                        help="Path to validation JSONL file")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Advanced training options
    parser.add_argument("--use_enhanced_training", action="store_true", default=True,
                        help="Use enhanced training with freeze/unfreeze strategy")
    parser.add_argument("--freeze_encoder_initially", action="store_true", default=True,
                        help="Whether to freeze encoder initially")
    parser.add_argument("--unfreeze_epoch_ratio", type=float, default=0.5,
                        help="Fraction of epochs to train with frozen encoder")
    parser.add_argument("--lr_reduction_factor", type=float, default=0.1,
                        help="Factor to reduce LR when unfreezing")
    parser.add_argument("--asr_lr_multiplier", type=float, default=0.1,
                        help="Learning rate multiplier for ASR head")

    # CTC configuration
    parser.add_argument("--ctc_entropy_weight", type=float, default=0.01,
                        help="Entropy regularization weight for CTC")
    parser.add_argument("--ctc_blank_weight", type=float, default=0.95,
                        help="Maximum blank probability for CTC")

    # Tokenizer options
    parser.add_argument("--tokenizer_path", type=str,
                        help="Path to trained tokenizer")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                        help="Whether to retrain the tokenizer")

    # Other options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision training")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use cosine annealing learning rate scheduler")
    parser.add_argument("--scale_lr_with_accumulation", action="store_true",
                        help="Scale learning rate with gradient accumulation")

    return parser.parse_args()


def setup_config(args, tokenizer):
    """Setup model configuration based on arguments"""
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
    return config


def create_data_loaders(dataset_dict, config, tokenizer, feature_extractor, batch_size):
    """Create train, validation, and test data loaders"""
    # Create datasets
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_mtl(
            batch,
            pad_token_id=tokenizer.pad_id,
            tokenizer=tokenizer,
            backbone_name=config.backbone_name
        ),
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch,
            pad_token_id=tokenizer.pad_id,
            tokenizer=tokenizer,
            backbone_name=config.backbone_name
        ),
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


def main():
    """Main training function"""
    args = parse_arguments()

    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Per-GPU batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Enhanced training: {args.use_enhanced_training}")

    if args.use_enhanced_training:
        print(f"\nFreeze/Unfreeze Strategy:")
        print(
            f"  Initial state: {'Frozen' if args.freeze_encoder_initially else 'Unfrozen'}")
        print(
            f"  Unfreeze at: {int(args.num_epochs * args.unfreeze_epoch_ratio)} epochs")
        print(f"  LR reduction: {args.lr_reduction_factor}x")
        print(f"\nCTC Configuration:")
        print(f"  Entropy weight: {args.ctc_entropy_weight}")
        print(f"  Blank weight: {args.ctc_blank_weight}")
        print(f"  ASR LR multiplier: {args.asr_lr_multiplier}")

    # Calculate adjusted learning rate
    base_lr = args.lr
    if args.scale_lr_with_accumulation and args.gradient_accumulation_steps > 1:
        lr_scale = math.sqrt(args.gradient_accumulation_steps)
        adjusted_lr = base_lr * lr_scale
        print(f"\nBase learning rate: {base_lr}")
        print(f"Adjusted learning rate: {adjusted_lr}")
    else:
        adjusted_lr = base_lr
        print(f"\nLearning rate: {adjusted_lr}")

    # Setup device and CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        configure_cuda_memory()
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print_memory_usage("\nInitial memory usage:")

    # Initialize wandb if enabled
    if args.use_wandb:
        config_dict = vars(args)
        config_dict['effective_batch_size'] = effective_batch_size
        wandb.init(project="mtl-speech", config=config_dict)

    # ============================================================================
    # PHASE 1: DATA LOADING AND PREPARATION
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 1: DATA LOADING AND PREPARATION")
    print("="*50)

    # Load datasets
    dataset_dict = load_and_prepare_datasets(
        args.train_jsonl, args.val_jsonl, args.test_jsonl, args.audio_base_path
    )

    print(f"\nLoaded datasets:")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Val: {len(dataset_dict['val'])} samples")
    print(f"  Test: {len(dataset_dict['test'])} samples")

    # Setup tokenizer
    if args.retrain_tokenizer or args.tokenizer_path is None:
        print("\nTraining new SentencePiece tokenizer...")
        tokenizer = setup_tokenizer_and_dataset(
            dataset_dict, vocab_size=args.vocab_size
        )
    else:
        print(f"\nLoading existing tokenizer from {args.tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
        tokenizer.load_tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer blank ID: {tokenizer.blank_id}")

    # ============================================================================
    # PHASE 2: MODEL SETUP
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 2: MODEL SETUP")
    print("="*50)

    # Setup configuration
    config = setup_config(args, tokenizer)

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
    print_memory_usage("\nAfter model creation:")

    # Create feature extractor
    temp_backbone = BackboneModel(config.backbone_config)
    feature_extractor = temp_backbone.feature_extractor
    del temp_backbone
    cleanup_memory()

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_dict, config, tokenizer, feature_extractor, args.batch_size
    )

    print_memory_usage("\nAfter data loader creation:")

    # ============================================================================
    # PHASE 3: TRAINING
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 3: TRAINING")
    print("="*50)

    # Create trainer
    trainer = MTLTrainer(
        model,
        device=device,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

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
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval,
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
            steps_per_epoch = len(
                train_loader) // args.gradient_accumulation_steps
            total_steps = steps_per_epoch * args.num_epochs
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
            print(
                f"Using cosine annealing scheduler with {total_steps} total steps")

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            tokenizer=tokenizer,
            save_dir=args.save_dir,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval
        )

    print_memory_usage("\nAfter training:")

    # Clean up training data before test evaluation
    del train_loader, val_loader
    cleanup_memory()

    # Plot training history
    if args.use_enhanced_training:
        plot_enhanced_training_history(
            trainer.history,
            os.path.join(args.save_dir, 'training_history.png')
        )
    else:
        plot_training_history(
            trainer.history,
            os.path.join(args.save_dir, 'training_history.png')
        )

    print("\nTraining completed!")

    # ============================================================================
    # PHASE 4: TEST EVALUATION
    # ============================================================================
    print("\n" + "="*50)
    print("PHASE 4: TEST EVALUATION")
    print("="*50)

    # Create test dataset and loader
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
            backbone_name=config.backbone_name
        ),
        num_workers=0,
        pin_memory=False
    )

    # Load best model
    print("\nLoading best model for evaluation...")
    trainer.load_checkpoint(os.path.join(args.save_dir, 'best_model.pt'))

    # Evaluate on test set
    evaluator = MTLEvaluator(model, tokenizer, device,
                             use_amp=args.use_amp, decode_method='beam')
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_detailed_results(test_metrics)

    # Save results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4, cls=NumpyEncoder)

    print_memory_usage("\nFinal memory usage:")

    print("\n" + "="*50)
    print("TRAINING AND EVALUATION COMPLETED!")
    print("="*50)

    # Finish wandb run
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


if __name__ == "__main__":
    main()
