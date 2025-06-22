"""
Standalone Evaluation Script for MTL Model
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
    parser = argparse.ArgumentParser(description="Evaluate MTL Speech Model")

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

    print(
        f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Active heads: {model.get_active_heads()}")

    return model, config


def evaluate_on_dataset(model, config, test_loader, tokenizer, device, decode_method, use_amp):
    """Evaluate model on test dataset"""
    print("\nStarting evaluation...")

    evaluator = MTLEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_amp=use_amp,
        decode_method=decode_method
    )

    # Evaluate
    metrics = evaluator.evaluate(test_loader)

    # Print results
    evaluator.print_detailed_results(metrics)

    return metrics


def save_results(metrics, output_dir, args):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    print(f"\nMetrics saved to {metrics_path}")

    # Save evaluation configuration
    config_path = os.path.join(output_dir, 'eval_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Evaluation configuration saved to {config_path}")

    # Create summary report
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Test dataset: {args.test_jsonl}\n")
        f.write(f"Decode method: {args.decode_method}\n\n")

        f.write("METRICS:\n")
        f.write("-"*30 + "\n")

        for task, task_metrics in metrics.items():
            f.write(f"\n{task.upper()}:\n")
            for metric_name, value in task_metrics.items():
                if metric_name != 'detailed_results' and not isinstance(value, list):
                    f.write(f"  {metric_name}: {value:.4f}\n")

    print(f"Summary saved to {summary_path}")


def main():
    """Main evaluation function"""
    args = parse_arguments()

    print("\n" + "="*50)
    print("EVALUATION CONFIGURATION")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test data: {args.test_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Decode method: {args.decode_method}")
    print(f"Device: {args.device}")

    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = torch.device('cpu')

    print_memory_usage("\nInitial memory usage:")

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint_path, device)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}")
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_path)
    tokenizer.load_tokenizer()
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    # Load test dataset
    print("\nLoading test dataset...")
    dataset_dict = load_and_prepare_datasets(
        train_jsonl=None,  # We only need test data
        val_jsonl=None,
        test_jsonl=args.test_jsonl,
        audio_base_path=args.audio_base_path
    )

    # Filter out None values from dataset_dict
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

    # Evaluate
    metrics = evaluate_on_dataset(
        model=model,
        config=config,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        decode_method=args.decode_method,
        use_amp=args.use_amp
    )

    # Save results
    save_results(metrics, args.output_dir, args)

    print_memory_usage("\nFinal memory usage:")

    print("\n" + "="*50)
    print("EVALUATION COMPLETED!")
    print("="*50)


if __name__ == "__main__":
    main()