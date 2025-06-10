import torch
from torch.utils.data import DataLoader
import gc
import psutil
import os


def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
    else:
        gpu_memory = 0
        gpu_memory_cached = 0

    process = psutil.Process(os.getpid())
    ram_memory = process.memory_info().rss / 1e9

    return {
        'ram_gb': ram_memory,
        'gpu_allocated_gb': gpu_memory,
        'gpu_cached_gb': gpu_memory_cached
    }


def print_memory_usage(message=""):
    """Print current memory usage"""
    info = get_memory_info()
    print(f"\n{message}")
    print(f"RAM Usage: {info['ram_gb']:.2f} GB")
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {info['gpu_allocated_gb']:.2f} GB allocated, {info['gpu_cached_gb']:.2f} GB cached")


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimize_model_for_memory(model):
    """Apply memory optimizations to model"""
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Set model to use less memory during inference
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

    return model


def create_data_loader_with_memory_limit(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=2,
    max_memory_gb=None,
    **kwargs
):
    """Create a data loader with memory usage monitoring"""

    if max_memory_gb is not None:
        # Start with the requested batch size
        current_batch_size = batch_size

        while current_batch_size > 1:
            try:
                # Try creating the loader
                loader = DataLoader(
                    dataset,
                    batch_size=current_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    **kwargs
                )

                # Test loading one batch
                test_batch = next(iter(loader))

                # Check memory usage
                memory_info = get_memory_info()
                if memory_info['ram_gb'] < max_memory_gb:
                    print(
                        f"Successfully created loader with batch size {current_batch_size}")
                    return loader
                else:
                    print(
                        f"Batch size {current_batch_size} uses too much memory, reducing...")
                    current_batch_size //= 2
                    del loader, test_batch
                    cleanup_memory()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"OOM with batch size {current_batch_size}, reducing...")
                    current_batch_size //= 2
                    cleanup_memory()
                else:
                    raise e

    # If no memory limit or couldn't determine optimal size, use original
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
