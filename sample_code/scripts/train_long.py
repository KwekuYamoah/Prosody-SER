"""
Main Training Script for MTL Model following the paper's approach
Clean and modular implementation with alpha-controlled auxiliary tasks
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
    """Parse command line arguments with paper-style alpha control"""
    parser = argparse.ArgumentParser(description="Train MTL Speech Model following the paper")

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

    # Paper-style alpha control
    parser.add_argument("--alpha_asr", type=float, default=0.1,
                        help="Alpha weight for ASR auxiliary task (paper style)")
    parser.add_argument("--alpha_prosody", type=float, default=0.1,
                        help="Alpha weight for Prosody auxiliary task (paper style)")
    parser.add_argument("--alpha_search", action="store_true",
                        help="Perform grid search over alpha values like in the paper")
    parser.add_argument("--alpha_values", nargs='+', type=float, 
                        default=[0.0, 0.001, 0.01, 0.1, 1.0],
                        help="Alpha values to search over (if alpha_search enabled)")

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


def setup_config_with_alpha(args, tokenizer, alpha_asr=None, alpha_prosody=None):
    """Setup model configuration with paper-style alpha control"""
    # Use provided alphas or defaults from args
    alpha_asr = alpha_asr if alpha_asr is not None else args.alpha_asr
    alpha_prosody = alpha_prosody if alpha_prosody is not None else args.alpha_prosody
    
    config = MTLConfig(
        backbone_name=args.backbone,
        vocab_size=tokenizer.get_vocab_size(),
        emotion_classes=9,
        prosody_classes=2,
        freeze_encoder=args.freeze_encoder_initially,
        
        # Paper-style alpha control
        alpha_asr=alpha_asr,
        alpha_prosody=alpha_prosody,
        
        # CTC parameters
        ctc_entropy_weight=args.ctc_entropy_weight,
        ctc_blank_weight=args.ctc_blank_weight,
        asr_lr_multiplier=args.asr_lr_multiplier,
        warmup_steps=1000
    )
    
    print(f"\nConfiguration created with:")
    print(f"  SER (main task): weight = 1.0")
    print(f"  ASR (auxiliary): alpha = {alpha_asr}")
    print(f"  Prosody (auxiliary): alpha = {alpha_prosody}")
    print(f"  Total auxiliary weight: {alpha_asr + alpha_prosody}")
    
    return config


def run_single_experiment(args, dataset_dict, tokenizer, feature_extractor, 
                         alpha_asr, alpha_prosody, experiment_name=""):
    """Run a single training experiment with given alpha values"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Alpha ASR: {alpha_asr}, Alpha Prosody: {alpha_prosody}")
    print(f"{'='*60}")
    
    # Setup configuration
    config = setup_config_with_alpha(args, tokenizer, alpha_asr, alpha_prosody)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    ).to(device)

    print(f"Created MTL model with backbone: {args.backbone}")
    print(f"Active heads: {model.get_active_heads()}")
    print(f"Task roles: {model.get_task_roles()}")

    # Apply memory optimizations
    model = optimize_model_for_memory(model)

    # Create data loaders
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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

    # Create trainer
    trainer = MTLTrainer(
        model,
        device=device,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Experiment-specific save directory
    exp_save_dir = os.path.join(args.save_dir, experiment_name.replace(" ", "_"))
    os.makedirs(exp_save_dir, exist_ok=True)

    # Training
    if args.use_enhanced_training:
        trainer.train_with_freeze_unfreeze_alpha(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            tokenizer=tokenizer,
            base_lr=args.lr,
            save_dir=exp_save_dir,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval,
            unfreeze_epoch_ratio=args.unfreeze_epoch_ratio,
            lr_reduction_factor=args.lr_reduction_factor,
            alpha_asr=alpha_asr,
            alpha_prosody=alpha_prosody
        )
    else:
        # Standard training with alpha control
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        trainer.train_with_alpha_control(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            tokenizer=tokenizer,
            save_dir=exp_save_dir,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval,
            alpha_asr=alpha_asr,
            alpha_prosody=alpha_prosody
        )

    # Evaluation
    print(f"\nEvaluating experiment: {experiment_name}")
    trainer.load_checkpoint(os.path.join(exp_save_dir, 'best_model.pt'))
    
    # Create test dataset
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

    evaluator = MTLEvaluator(model, tokenizer, device, use_amp=args.use_amp, decode_method='beam')
    test_metrics = evaluator.evaluate(test_loader)
    
    # Save results
    with open(os.path.join(exp_save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4, cls=NumpyEncoder)

    # Extract key metrics for comparison
    ser_acc = test_metrics.get('emotion', {}).get('accuracy', 0.0)
    asr_wer = test_metrics.get('asr', {}).get('wer', 1.0)
    prosody_acc = test_metrics.get('prosody', {}).get('accuracy', 0.0)
    
    result_summary = {
        'experiment_name': experiment_name,
        'alpha_asr': alpha_asr,
        'alpha_prosody': alpha_prosody,
        'ser_accuracy': ser_acc,
        'asr_wer': asr_wer,
        'prosody_accuracy': prosody_acc,
        'combined_score': ser_acc - 0.1 * asr_wer + 0.1 * prosody_acc  # Paper-style combined metric
    }
    
    print(f"\nResults for {experiment_name}:")
    print(f"  SER Accuracy: {ser_acc:.4f}")
    print(f"  ASR WER: {asr_wer:.4f}")
    print(f"  Prosody Accuracy: {prosody_acc:.4f}")
    print(f"  Combined Score: {result_summary['combined_score']:.4f}")
    
    # Cleanup
    del model, trainer, train_loader, val_loader, test_loader
    cleanup_memory()
    
    return result_summary


def run_alpha_search(args, dataset_dict, tokenizer, feature_extractor):
    """Run grid search over alpha values following the paper's methodology"""
    
    print(f"\n{'='*60}")
    print("ALPHA GRID SEARCH (Following Paper Methodology)")
    print(f"{'='*60}")
    print(f"Testing alpha values: {args.alpha_values}")
    print("Note: SER is the main task (weight=1.0), ASR and Prosody are auxiliary tasks")
    
    results = []
    
    # Test different combinations of alpha values
    for alpha_asr in args.alpha_values:
        for alpha_prosody in args.alpha_values:
            experiment_name = f"alpha_asr_{alpha_asr}_alpha_prosody_{alpha_prosody}"
            
            try:
                result = run_single_experiment(
                    args, dataset_dict, tokenizer, feature_extractor,
                    alpha_asr, alpha_prosody, experiment_name
                )
                results.append(result)
                
                # Log to wandb if enabled
                if args.use_wandb:
                    wandb.log({
                        f"alpha_search/ser_acc_asr_{alpha_asr}_prosody_{alpha_prosody}": result['ser_accuracy'],
                        f"alpha_search/asr_wer_asr_{alpha_asr}_prosody_{alpha_prosody}": result['asr_wer'],
                        f"alpha_search/prosody_acc_asr_{alpha_asr}_prosody_{alpha_prosody}": result['prosody_accuracy'],
                        f"alpha_search/combined_score_asr_{alpha_asr}_prosody_{alpha_prosody}": result['combined_score']
                    })
                    
            except Exception as e:
                print(f"Error in experiment {experiment_name}: {str(e)}")
                continue
    
    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x['combined_score'])
        
        print(f"\n{'='*60}")
        print("ALPHA SEARCH RESULTS")
        print(f"{'='*60}")
        
        print("\nAll Results:")
        for result in sorted(results, key=lambda x: x['combined_score'], reverse=True):
            print(f"  α_ASR={result['alpha_asr']:.3f}, α_Prosody={result['alpha_prosody']:.3f} -> "
                  f"SER: {result['ser_accuracy']:.4f}, Combined: {result['combined_score']:.4f}")
        
        print(f"\nBest Configuration:")
        print(f"  Alpha ASR: {best_result['alpha_asr']}")
        print(f"  Alpha Prosody: {best_result['alpha_prosody']}")
        print(f"  SER Accuracy: {best_result['ser_accuracy']:.4f}")
        print(f"  ASR WER: {best_result['asr_wer']:.4f}")
        print(f"  Prosody Accuracy: {best_result['prosody_accuracy']:.4f}")
        print(f"  Combined Score: {best_result['combined_score']:.4f}")
        
        # Save comprehensive results
        with open(os.path.join(args.save_dir, 'alpha_search_results.json'), 'w') as f:
            json.dump({
                'all_results': results,
                'best_result': best_result,
                'search_config': {
                    'alpha_values_tested': args.alpha_values,
                    'backbone': args.backbone,
                    'methodology': 'paper_style_main_auxiliary'
                }
            }, f, indent=4, cls=NumpyEncoder)
            
        return best_result
    else:
        print("No successful experiments in alpha search!")
        return None


def main():
    """Main training function with paper-style alpha control"""
    args = parse_arguments()

    print("\n" + "="*60)
    print("MTL TRAINING - PAPER STYLE ALPHA CONTROL")
    print("="*60)
    print(f"SER: Main Task (weight = 1.0)")
    print(f"ASR: Auxiliary Task (alpha = {args.alpha_asr})")
    print(f"Prosody: Auxiliary Task (alpha = {args.alpha_prosody})")
    
    if args.alpha_search:
        print(f"Alpha search enabled with values: {args.alpha_values}")

    # Setup device and CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        configure_cuda_memory()
        torch.backends.cudnn.benchmark = True

    print_memory_usage("\nInitial memory usage:")

    # Initialize wandb if enabled
    if args.use_wandb:
        config_dict = vars(args)
        config_dict['methodology'] = 'paper_style_alpha_control'
        config_dict['main_task'] = 'SER'
        config_dict['auxiliary_tasks'] = ['ASR', 'Prosody']
        wandb.init(project="mtl-speech-alpha-control", config=config_dict)

    # Load datasets
    print("\n" + "="*60)
    print("DATA LOADING AND PREPARATION")
    print("="*60)

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

    # Create feature extractor
    temp_backbone = BackboneModel(BACKBONE_CONFIGS[args.backbone])
    feature_extractor = temp_backbone.feature_extractor
    del temp_backbone
    cleanup_memory()

    # Run experiments
    if args.alpha_search:
        # Run alpha grid search
        best_result = run_alpha_search(args, dataset_dict, tokenizer, feature_extractor)
        
        if best_result and args.use_wandb:
            wandb.log({
                'best_alpha_asr': best_result['alpha_asr'],
                'best_alpha_prosody': best_result['alpha_prosody'],
                'best_ser_accuracy': best_result['ser_accuracy'],
                'best_combined_score': best_result['combined_score']
            })
    else:
        # Run single experiment with specified alphas
        result = run_single_experiment(
            args, dataset_dict, tokenizer, feature_extractor,
            args.alpha_asr, args.alpha_prosody, 
            f"single_exp_asr_{args.alpha_asr}_prosody_{args.alpha_prosody}"
        )
        
        if args.use_wandb:
            wandb.log({
                'final_ser_accuracy': result['ser_accuracy'],
                'final_asr_wer': result['asr_wer'],
                'final_prosody_accuracy': result['prosody_accuracy'],
                'final_combined_score': result['combined_score']
            })

    print_memory_usage("\nFinal memory usage:")

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()