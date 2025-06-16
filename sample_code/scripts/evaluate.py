"""
Standalone Evaluation Script for MTL Model with Paper-Style Alpha Analysis
"""

import torch
import os
import argparse
import json
from torch.utils.data import DataLoader

# Import core modules
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.mtl_model import MTLModel
from sample_code.scripts.mtl_dataset import MTLDataset
from sample_code.scripts.backbone_models import BackboneModel
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# Import utilities
from sample_code.scripts.memory_utils import print_memory_usage

# Import training modules
from sample_code.training import MTLEvaluator, collate_fn_mtl, NumpyEncoder

from sample_code.utils import load_and_prepare_datasets

# Set environment variables
torch.serialization.add_safe_globals([MTLConfig])


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate MTL Speech Model with Alpha Analysis")

    # Model configuration
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to trained tokenizer")

    # Data configuration
    parser.add_argument("--audio_base_path", type=str, required=True,
                        help="Base path to audio files directory")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")

    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--decode_method", type=str, default="beam",
                        choices=["greedy", "beam"],
                        help="Decoding method for ASR")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use for evaluation")

    # Alpha analysis options
    parser.add_argument("--analyze_alpha_impact", action="store_true",
                        help="Analyze impact of different alpha values on performance")
    parser.add_argument("--alpha_test_values", nargs='+', type=float,
                        default=[0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
                        help="Alpha values to test for impact analysis")

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print alpha information from checkpoint
    if 'alpha_values' in checkpoint:
        alpha_vals = checkpoint['alpha_values']
        print(f"Model trained with alpha values:")
        print(f"  Alpha ASR: {alpha_vals['alpha_asr']}")
        print(f"  Alpha Prosody: {alpha_vals['alpha_prosody']}")
        print(f"  SER (main task): weight = 1.0")
    else:
        print("No alpha information found in checkpoint")

    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Active heads: {model.get_active_heads()}")
    print(f"Task roles: {model.get_task_roles()}")

    return model, config, checkpoint


def evaluate_with_alpha_values(model, config, test_loader, tokenizer, device, 
                              alpha_asr, alpha_prosody, decode_method, use_amp):
    """Evaluate model with specific alpha values"""
    print(f"\nEvaluating with Alpha ASR={alpha_asr}, Alpha Prosody={alpha_prosody}")
    
    # Update model's alpha values
    original_alpha_asr = model.config.alpha_asr
    original_alpha_prosody = model.config.alpha_prosody
    
    model.config.update_alpha_weights(alpha_asr, alpha_prosody)
    
    evaluator = MTLEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_amp=use_amp,
        decode_method=decode_method
    )

    # Evaluate
    metrics = evaluator.evaluate(test_loader)
    
    # Add alpha values to metrics
    metrics['alpha_values'] = {
        'alpha_asr': alpha_asr,
        'alpha_prosody': alpha_prosody,
        'main_task_weight': 1.0
    }
    
    # Restore original alpha values
    model.config.update_alpha_weights(original_alpha_asr, original_alpha_prosody)
    
    return metrics


def analyze_alpha_impact(model, config, test_loader, tokenizer, device, 
                        alpha_test_values, decode_method, use_amp):
    """Analyze the impact of different alpha values on model performance"""
    print("\n" + "="*60)
    print("ALPHA IMPACT ANALYSIS")
    print("="*60)
    print("Testing different alpha combinations following the paper's methodology")
    
    results = []
    
    for alpha_asr in alpha_test_values:
        for alpha_prosody in alpha_test_values:
            print(f"\nTesting Alpha ASR={alpha_asr}, Alpha Prosody={alpha_prosody}")
            
            try:
                metrics = evaluate_with_alpha_values(
                    model, config, test_loader, tokenizer, device,
                    alpha_asr, alpha_prosody, decode_method, use_amp
                )
                
                # Extract key metrics
                ser_acc = metrics.get('emotion', {}).get('accuracy', 0.0)
                asr_wer = metrics.get('asr', {}).get('wer', 1.0) if 'asr' in metrics else 1.0
                prosody_acc = metrics.get('prosody', {}).get('accuracy', 0.0) if 'prosody' in metrics else 0.0
                
                result = {
                    'alpha_asr': alpha_asr,
                    'alpha_prosody': alpha_prosody,
                    'ser_accuracy': ser_acc,
                    'asr_wer': asr_wer,
                    'prosody_accuracy': prosody_acc,
                    'combined_score': ser_acc - 0.1 * asr_wer + 0.1 * prosody_acc,
                    'full_metrics': metrics
                }
                
                results.append(result)
                
                print(f"  Results: SER={ser_acc:.4f}, ASR_WER={asr_wer:.4f}, Prosody={prosody_acc:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
    
    # Analyze results
    if results:
        print(f"\n{'='*60}")
        print("ALPHA IMPACT ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print("\nTop 10 Alpha Combinations:")
        for i, result in enumerate(results[:10]):
            print(f"{i+1:2d}. α_ASR={result['alpha_asr']:.3f}, α_Prosody={result['alpha_prosody']:.3f} "
                  f"-> SER={result['ser_accuracy']:.4f}, Combined={result['combined_score']:.4f}")
        
        # Find best for each task
        best_ser = max(results, key=lambda x: x['ser_accuracy'])
        best_asr = min(results, key=lambda x: x['asr_wer']) if any('asr' in r['full_metrics'] for r in results) else None
        best_prosody = max(results, key=lambda x: x['prosody_accuracy']) if any('prosody' in r['full_metrics'] for r in results) else None
        best_combined = max(results, key=lambda x: x['combined_score'])
        
        print(f"\nBest Alpha Values by Task:")
        print(f"  SER: α_ASR={best_ser['alpha_asr']}, α_Prosody={best_ser['alpha_prosody']} -> Acc={best_ser['ser_accuracy']:.4f}")
        if best_asr:
            print(f"  ASR: α_ASR={best_asr['alpha_asr']}, α_Prosody={best_asr['alpha_prosody']} -> WER={best_asr['asr_wer']:.4f}")
        if best_prosody:
            print(f"  Prosody: α_ASR={best_prosody['alpha_asr']}, α_Prosody={best_prosody['alpha_prosody']} -> Acc={best_prosody['prosody_accuracy']:.4f}")
        print(f"  Combined: α_ASR={best_combined['alpha_asr']}, α_Prosody={best_combined['alpha_prosody']} -> Score={best_combined['combined_score']:.4f}")
        
        # Analyze patterns
        print(f"\nAlpha Impact Analysis:")
        
        # Effect of ASR alpha on SER
        ser_by_asr_alpha = {}
        for result in results:
            alpha_asr = result['alpha_asr']
            if alpha_asr not in ser_by_asr_alpha:
                ser_by_asr_alpha[alpha_asr] = []
            ser_by_asr_alpha[alpha_asr].append(result['ser_accuracy'])
        
        print(f"  SER Performance by ASR Alpha:")
        for alpha in sorted(ser_by_asr_alpha.keys()):
            avg_ser = sum(ser_by_asr_alpha[alpha]) / len(ser_by_asr_alpha[alpha])
            print(f"    α_ASR={alpha:.3f}: Avg SER Accuracy = {avg_ser:.4f}")
        
        # Effect of Prosody alpha on SER
        ser_by_prosody_alpha = {}
        for result in results:
            alpha_prosody = result['alpha_prosody']
            if alpha_prosody not in ser_by_prosody_alpha:
                ser_by_prosody_alpha[alpha_prosody] = []
            ser_by_prosody_alpha[alpha_prosody].append(result['ser_accuracy'])
        
        print(f"  SER Performance by Prosody Alpha:")
        for alpha in sorted(ser_by_prosody_alpha.keys()):
            avg_ser = sum(ser_by_prosody_alpha[alpha]) / len(ser_by_prosody_alpha[alpha])
            print(f"    α_Prosody={alpha:.3f}: Avg SER Accuracy = {avg_ser:.4f}")
    
    return results


def save_results(metrics, output_dir, args, alpha_analysis_results=None):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save main metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    print(f"\nMetrics saved to {metrics_path}")

    # Save alpha analysis if performed
    if alpha_analysis_results:
        alpha_analysis_path = os.path.join(output_dir, 'alpha_analysis_results.json')
        with open(alpha_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(alpha_analysis_results, f, indent=4, cls=NumpyEncoder)
        print(f"Alpha analysis results saved to {alpha_analysis_path}")

    # Save evaluation configuration
    config_path = os.path.join(output_dir, 'eval_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Evaluation configuration saved to {config_path}")

    # Create summary report
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PAPER-STYLE MTL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Test dataset: {args.test_jsonl}\n")
        f.write(f"Decode method: {args.decode_method}\n\n")

        if 'alpha_values' in metrics:
            f.write("ALPHA VALUES:\n")
            f.write("-"*30 + "\n")
            f.write(f"Alpha ASR (auxiliary): {metrics['alpha_values']['alpha_asr']}\n")
            f.write(f"Alpha Prosody (auxiliary): {metrics['alpha_values']['alpha_prosody']}\n")
            f.write(f"SER weight (main task): {metrics['alpha_values']['main_task_weight']}\n\n")

        f.write("TASK PERFORMANCE:\n")
        f.write("-"*30 + "\n")

        for task, task_metrics in metrics.items():
            if task in ['emotion', 'asr', 'prosody']:
                task_name = {'emotion': 'SER (Main Task)', 'asr': 'ASR (Auxiliary)', 'prosody': 'Prosody (Auxiliary)'}[task]
                f.write(f"\n{task_name}:\n")
                for metric_name, value in task_metrics.items():
                    if metric_name != 'detailed_results' and not isinstance(value, list):
                        f.write(f"  {metric_name}: {value:.4f}\n")

        if alpha_analysis_results:
            f.write(f"\n\nALPHA ANALYSIS SUMMARY:\n")
            f.write("-"*30 + "\n")
            best_result = max(alpha_analysis_results, key=lambda x: x['combined_score'])
            f.write(f"Best Alpha Combination:\n")
            f.write(f"  Alpha ASR: {best_result['alpha_asr']}\n")
            f.write(f"  Alpha Prosody: {best_result['alpha_prosody']}\n")
            f.write(f"  SER Accuracy: {best_result['ser_accuracy']:.4f}\n")
            f.write(f"  Combined Score: {best_result['combined_score']:.4f}\n")

    print(f"Summary saved to {summary_path}")


def main():
    """Main evaluation function"""
    args = parse_arguments()

    print("\n" + "="*60)
    print("MTL MODEL EVALUATION - PAPER STYLE ALPHA ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test data: {args.test_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Decode method: {args.decode_method}")
    print(f"Alpha analysis: {args.analyze_alpha_impact}")

    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = torch.device('cpu')

    print_memory_usage("\nInitial memory usage:")

    # Load model
    model, config, checkpoint = load_model_from_checkpoint(args.checkpoint_path, device)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}")
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
    tokenizer.load_tokenizer()
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    # Load test dataset
    print("\nLoading test dataset...")
    dataset_dict = load_and_prepare_datasets(
        train_jsonl=None,
        val_jsonl=None,
        test_jsonl=args.test_jsonl,
        audio_base_path=args.audio_base_path
    )

    dataset_dict = {k: v for k, v in dataset_dict.items() if v is not None}
    print(f"Test dataset size: {len(dataset_dict['test'])} samples")

    # Create feature extractor
    temp_backbone = BackboneModel(config.backbone_config)
    feature_extractor = temp_backbone.feature_extractor
    del temp_backbone

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

    print_memory_usage("\nAfter data loading:")

    # Standard evaluation with original alpha values
    print("\n" + "="*60)
    print("STANDARD EVALUATION")
    print("="*60)
    
    evaluator = MTLEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_amp=args.use_amp,
        decode_method=args.decode_method
    )

    metrics = evaluator.evaluate(test_loader)
    
    # Add alpha information to metrics
    metrics['alpha_values'] = {
        'alpha_asr': model.config.alpha_asr,
        'alpha_prosody': model.config.alpha_prosody,
        'main_task_weight': 1.0
    }
    
    evaluator.print_detailed_results(metrics)

    # Alpha impact analysis
    alpha_analysis_results = None
    if args.analyze_alpha_impact:
        alpha_analysis_results = analyze_alpha_impact(
            model, config, test_loader, tokenizer, device,
            args.alpha_test_values, args.decode_method, args.use_amp
        )

    # Save results
    save_results(metrics, args.output_dir, args, alpha_analysis_results)

    print_memory_usage("\nFinal memory usage:")

    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()