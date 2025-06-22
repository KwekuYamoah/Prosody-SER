"""
Training Script for MTL Model
Following "Speech Emotion Recognition with Multi-task Learning" methodology
"""

import torch
import os
import argparse
import json
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import core modules
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.mtl_model import MTLModel
from sample_code.scripts.mtl_dataset import MTLDataset
from sample_code.scripts.backbone_models import BACKBONE_CONFIGS, BackboneModel
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# Import training components
from sample_code.training.trainer import MTLTrainer
from sample_code.training.evaluator import MTLEvaluator
from sample_code.training.utils import collate_fn_mtl, configure_cuda_memory, NumpyEncoder

# Import utilities
from sample_code.scripts.memory_utils import print_memory_usage, cleanup_memory
from sample_code.utils import load_and_prepare_datasets, setup_tokenizer_and_dataset

# Set environment variables for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Avoid deadlocks with DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.serialization.add_safe_globals([MTLConfig])


def parse_arguments():
    """Parse command line arguments following experimental setup"""
    parser = argparse.ArgumentParser(
        description="MTL Training for Speech Emotion Recognition")

    # Model configuration
    parser.add_argument("--backbone", type=str, default="whisper",
                        choices=list(BACKBONE_CONFIGS.keys()),
                        help="Backbone model (whisper, mms, xlsr, wav2vec2-bert)")
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="Vocabulary size for tokenizer")

    # alpha configuration
    parser.add_argument("--alpha_asr", type=float, default=0.1,
                        help="Alpha weight for ASR auxiliary task (paper optimal: 0.1)")
    parser.add_argument("--alpha_prosody", type=float, default=0.1,
                        help="Alpha weight for Prosody auxiliary task (paper optimal: 0.1)")

    # Enhanced CTC regularization parameters (NEW)
    parser.add_argument("--ctc_entropy_weight", type=float, default=0.01,
                        help="Entropy regularization weight for CTC loss")
    parser.add_argument("--ctc_blank_penalty", type=float, default=50.0,
                        help="Blank penalty weight for CTC loss (very strong penalty: 50.0)")
    parser.add_argument("--ctc_blank_threshold", type=float, default=0.3,
                        help="Threshold for blank penalty (default: 0.3, much better than 0.8)")
    parser.add_argument("--ctc_label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CTC loss")
    parser.add_argument("--ctc_confidence_penalty", type=float, default=0.0,
                        help="Confidence penalty for CTC loss")

    # Alpha experimentation
    parser.add_argument("--run_ablation_study", action="store_true",
                        help="Run paper's ablation study with different alpha values")
    parser.add_argument("--ablation_alphas", nargs='+', type=float,
                        default=[0.0, 0.001, 0.01, 0.1, 1.0],
                        help="Alpha values to test in ablation study")
    parser.add_argument("--ablation_epochs", type=int, default=5,
                        help="Epochs per alpha in ablation study")

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
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory efficiency")

    # learning rates
    parser.add_argument("--backbone_lr", type=float, default=1e-5,
                        help="Learning rate for backbone (fine-tuning)")
    parser.add_argument("--head_lr", type=float, default=5e-5,
                        help="Learning rate for task heads")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate schedule")

    # Training options
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N training steps")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log every N training steps")

    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="mtl-akan-speech",
                        help="W&B project name")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for tracking")

    # Performance options
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Pin memory for faster GPU transfer")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch per worker")

    # Tokenizer options
    parser.add_argument("--tokenizer_path", type=str,
                        help="Path to trained tokenizer")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                        help="Whether to retrain the tokenizer")

    return parser.parse_args()


def setup_paper_style_config(args, tokenizer):
    """Setup configuration following methodology with enhanced CTC"""
    config = MTLConfig.create_paper_config(
        backbone_name=args.backbone,
        alpha_asr=args.alpha_asr,
        alpha_prosody=args.alpha_prosody
    )

    # Update with tokenizer and task settings
    config.vocab_size = tokenizer.get_vocab_size()
    config.emotion_classes = 9
    config.prosody_classes = 2

    # Enhanced CTC regularization settings
    config.ctc_entropy_weight = args.ctc_entropy_weight
    config.ctc_blank_penalty = args.ctc_blank_penalty
    config.ctc_blank_threshold = args.ctc_blank_threshold
    config.ctc_label_smoothing = args.ctc_label_smoothing
    config.ctc_confidence_penalty = args.ctc_confidence_penalty

    # Learning rate settings
    config.backbone_learning_rate = args.backbone_lr
    config.task_head_learning_rate = args.head_lr

    return config


def create_efficient_data_loaders(dataset_dict, config, tokenizer, feature_extractor, args):
    """Create memory-efficient data loaders with proper settings"""
    train_dataset = MTLDataset(
        dataset_dict['train'], config=config, feature_extractor=feature_extractor)
    val_dataset = MTLDataset(
        dataset_dict['val'], config=config, feature_extractor=feature_extractor)

    # DataLoader with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer,
            backbone_name=config.backbone_name
        ),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        # Larger batch for validation (no gradients)
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_mtl(
            batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer,
            backbone_name=config.backbone_name
        ),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0
    )

    return train_loader, val_loader


def log_ctc_statistics(trainer, epoch):
    """Log CTC blank statistics for monitoring"""
    if hasattr(trainer.model, 'asr_loss') and hasattr(trainer.model.asr_loss, 'get_blank_statistics'):
        # This would need to be called during forward pass
        # For now, we'll log during training
        pass


def main():
    """Main training function with enhancements"""
    args = parse_arguments()

    # Print configuration
    print("\n" + "="*60)
    print("MTL TRAINING WITH ENHANCED CTC")
    print("="*60)
    print("Following: 'Speech Emotion Recognition with Multi-task Learning'")
    print(f"Main task: SER (weight = 1.0)")
    print(
        f"Auxiliary tasks: ASR (α = {args.alpha_asr}), Prosody (α = {args.alpha_prosody})")
    print(f"Loss formula: L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody")
    print(f"Backbone: {args.backbone}")
    print(f"Learning rates: Backbone={args.backbone_lr}, Heads={args.head_lr}")
    print(f"\nEnhanced CTC Regularization:")
    print(f"  Entropy weight: {args.ctc_entropy_weight}")
    print(
        f"  Blank penalty: {args.ctc_blank_penalty} (VERY strong penalty to prevent blank collapse)")
    print(
        f"  Blank threshold: {args.ctc_blank_threshold} (much better than 0.8!)")
    print(f"  Label smoothing: {args.ctc_label_smoothing}")
    print(f"  Confidence penalty: {args.ctc_confidence_penalty}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        configure_cuda_memory()
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for A100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_memory_usage("Initial memory usage:")

    # Initialize wandb if enabled
    if args.use_wandb:
        experiment_name = args.experiment_name or f"mtl_akan_{args.alpha_asr}_{args.alpha_prosody}_ctc_enhanced"
        wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            config=vars(args),
            tags=["mtl-akan", "enhanced-ctc", f"alpha-{args.alpha_asr}"]
        )

    # Load datasets
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)

    dataset_dict = load_and_prepare_datasets(
        args.train_jsonl, args.val_jsonl, args.test_jsonl, args.audio_base_path
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Val: {len(dataset_dict['val'])} samples")
    print(f"  Test: {len(dataset_dict['test'])} samples")

    # Setup tokenizer
    if args.retrain_tokenizer or args.tokenizer_path is None:
        print("\nTraining new SentencePiece tokenizer...")
        tokenizer = setup_tokenizer_and_dataset(
            dataset_dict, vocab_size=args.vocab_size)
        tokenizer_save_path = os.path.join(args.save_dir, "akan_mtl_tokenizer.model")
        os.makedirs(args.save_dir, exist_ok=True)
        # Save tokenizer for future use
        import shutil
        shutil.copy(tokenizer.model_path, tokenizer_save_path)
        print(f"Tokenizer saved to: {tokenizer_save_path}")
    else:
        print(f"\nLoading existing tokenizer from {args.tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
        tokenizer.load_tokenizer()

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer blank ID: {tokenizer.blank_id}")

    # Setup configuration and model
    print("\n" + "="*60)
    print("MODEL SETUP")
    print("="*60)

    config = setup_paper_style_config(args, tokenizer)
    print("\nConfiguration summary:")
    summary = config.get_paper_summary()
    print(json.dumps(summary, indent=2))

    # Create paper-style model with enhanced CTC
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    ).to(device)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing and hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled for memory efficiency")

    print(f"\nModel info:")
    print(f"  Active heads: {model.get_active_heads()}")
    print(
        f"  Training approach: {model.get_paper_training_info()['model_type']}")
    print(
        f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create feature extractor
    temp_backbone = BackboneModel(config.backbone_config)
    feature_extractor = temp_backbone.feature_extractor
    del temp_backbone
    cleanup_memory()

    # Create efficient data loaders
    train_loader, val_loader = create_efficient_data_loaders(
        dataset_dict, config, tokenizer, feature_extractor, args
    )

    print_memory_usage("After data loader creation:")

    # Create trainer with enhancements
    trainer = MTLTrainer(
        model=model,
        device=device,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    print("\nTrainer created with enhanced CTC loss")

    # Run ablation study if requested
    if args.run_ablation_study:
        print("\n" + "="*60)
        print("RUNNING ABLATION STUDY")
        print("="*60)

        ablation_results, optimal_alpha = trainer.run_paper_ablation_study(
            train_loader, val_loader, tokenizer,
            alpha_values=args.ablation_alphas,
            epochs_per_alpha=args.ablation_epochs,
            save_dir=os.path.join(args.save_dir, 'ablation')
        )

        print(f"\nAblation study completed! Optimal alpha: {optimal_alpha}")

        # Update config with optimal alpha
        config.update_alpha_weights(optimal_alpha, optimal_alpha)

        # Log ablation results to wandb
        if args.use_wandb:
            wandb.log({"ablation_results": ablation_results})
            for alpha, results in ablation_results.items():
                wandb.log({
                    f'ablation/alpha_{alpha}/best_accuracy': results['best_ser_accuracy'],
                    f'ablation/alpha_{alpha}/avg_loss': results['avg_loss']
                })

    # Main training with enhanced monitoring
    print("\n" + "="*60)
    print("MAIN TRAINING")
    print("="*60)
    print(
        f"Current alpha values: ASR={config.alpha_asr}, Prosody={config.alpha_prosody}")
    print(
        f"CTC regularization: entropy={config.ctc_entropy_weight}, blank_penalty={config.ctc_blank_penalty}, threshold={config.ctc_blank_threshold}")

    # Train with new approach
    trainer.train_paper_style(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        tokenizer=tokenizer,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_interval=args.checkpoint_interval
    )

    print_memory_usage("After training:")

    # Clean up before test evaluation
    del train_loader, val_loader
    cleanup_memory()

    # # Final evaluation on test set
    # print("\n" + "="*60)
    # print("FINAL TEST EVALUATION")
    # print("="*60)

    # # Load best model
    # best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    # if os.path.exists(best_model_path):
    #     checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # # Create test dataset and loader
    # test_dataset = MTLDataset(dataset_dict['test'], config=config, feature_extractor=feature_extractor)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size * 2,  # Larger batch for test (no gradients)
    #     shuffle=False,
    #     collate_fn=lambda batch: collate_fn_mtl(
    #         batch, pad_token_id=tokenizer.pad_id, tokenizer=tokenizer,
    #         backbone_name=config.backbone_name
    #     ),
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory and torch.cuda.is_available(),
    #     persistent_workers=args.num_workers > 0
    # )

    # # Final evaluation with beam search for better ASR
    # evaluator = MTLEvaluator(model, tokenizer, device, use_amp=args.use_amp, decode_method='beam')
    # test_metrics = evaluator.evaluate(test_loader)
    # evaluator.print_detailed_results(test_metrics)

    # # Prepare comprehensive results
    # final_results = {
    #     'paper_methodology': 'Speech Emotion Recognition with Multi-task Learning',
    #     'loss_formula': 'L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody',
    #     'alpha_values': config.get_alpha_values(),
    #     'ctc_regularization': {
    #         'entropy_weight': config.ctc_entropy_weight,
    #         'blank_penalty': config.ctc_blank_penalty,
    #         'label_smoothing': config.ctc_label_smoothing,
    #         'confidence_penalty': config.ctc_confidence_penalty
    #     },
    #     'test_metrics': test_metrics,
    #     'training_summary': trainer.get_training_summary(),
    #     'config_summary': config.get_paper_summary(),
    #     'model_stats': {
    #         'total_params': sum(p.numel() for p in model.parameters()),
    #         'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     }
    # }

    # # Save results
    # results_path = os.path.join(args.save_dir, 'final_results.json')
    # with open(results_path, 'w') as f:
    #     json.dump(final_results, f, indent=4, cls=NumpyEncoder)

    # print(f"\n✅ Training completed successfully!")
    # print(f"Results saved to: {args.save_dir}")

    # # Extract and display key metrics
    # ser_accuracy = test_metrics.get('emotion', {}).get('accuracy', 0.0)
    # asr_wer = test_metrics.get('asr', {}).get('wer', 1.0)
    # prosody_accuracy = test_metrics.get('prosody', {}).get('accuracy', 0.0)

    # print(f"\n📊 FINAL RESULTS:")
    # print(f"  SER Accuracy (main task): {ser_accuracy:.4f}")
    # print(f"  ASR WER (auxiliary): {asr_wer:.4f}")
    # print(f"  Prosody Accuracy (auxiliary): {prosody_accuracy:.4f}")
    # print(f"  Alpha values: ASR={config.alpha_asr}, Prosody={config.alpha_prosody}")

    # # Log final results to wandb
    # if args.use_wandb:
    #     wandb.log({
    #         'test/final_ser_accuracy': ser_accuracy,
    #         'test/final_asr_wer': asr_wer,
    #         'test/final_prosody_accuracy': prosody_accuracy
    #     })

    #     # Log a summary table
    #     wandb.run.summary['best_ser_accuracy'] = ser_accuracy
    #     wandb.run.summary['best_asr_wer'] = asr_wer
    #     wandb.run.summary['optimal_alpha'] = config.alpha_asr

    #     wandb.finish()

    print_memory_usage("Final memory usage:")

    # return final_results


if __name__ == "__main__":
    try:
        # final_results = main()
        main()
        print("\n🎉 Training script completed successfully!")
        exit(0)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if wandb.run is not None:
            wandb.finish()
        exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        exit(1)
