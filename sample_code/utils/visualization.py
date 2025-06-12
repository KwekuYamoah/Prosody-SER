"""
Visualization Utilities
Functions for plotting training history and results
"""

import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot basic training history"""
    if len(history['train_loss']) == 0:
        print("No training history to plot")
        return

    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    epochs = range(len(history['train_loss']))
    plt.plot(epochs, [x['total']
             for x in history['train_loss']], label='Train Loss')
    plt.plot(epochs, [x['total']
             for x in history['val_loss']], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot metrics
    plt.subplot(2, 1, 2)
    if len(history['val_metrics']) > 0:
        for task in ['asr', 'prosody', 'emotion']:
            if task in history['val_metrics'][0]:
                metric_key = 'accuracy' if task != 'asr' else 'wer'
                if metric_key in history['val_metrics'][0][task]:
                    values = [x[task][metric_key]
                              for x in history['val_metrics']]
                    plt.plot(epochs, values,
                             label=f'{task.capitalize()} {metric_key.upper()}')

    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_enhanced_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot enhanced training history with freeze/unfreeze markers"""
    if len(history['train_loss']) == 0:
        print("No training history to plot")
        return

    plt.figure(figsize=(20, 12))
    epochs = range(len(history['train_loss']))

    # Plot 1: Total Loss with freeze/unfreeze regions
    plt.subplot(3, 2, 1)
    train_total = [x['total'] for x in history['train_loss']]
    val_total = [x['total'] for x in history['val_loss']]

    plt.plot(epochs, train_total, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_total, label='Val Loss', linewidth=2)

    # Add freeze/unfreeze shading
    if 'freeze_status' in history:
        frozen_epochs = [i for i, frozen in enumerate(
            history['freeze_status']) if frozen]
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
        if task in history['train_loss'][0]:
            values = [x.get(task, 0) for x in history['train_loss']]
            plt.plot(epochs, values, label=f'{task.upper()} Loss', linewidth=2)

    plt.title('Individual Task Losses (Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: ASR Metrics
    plt.subplot(3, 2, 3)
    if len(history['val_metrics']) > 0 and 'asr' in history['val_metrics'][0]:
        wer_values = [x['asr'].get('wer', 1.0) for x in history['val_metrics']]
        cer_values = [x['asr'].get('cer', 1.0) for x in history['val_metrics']]
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
        if len(history['val_metrics']) > 0 and task in history['val_metrics'][0]:
            acc_values = [x[task].get('accuracy', 0)
                          for x in history['val_metrics']]
            plt.plot(epochs, acc_values,
                     label=f'{task.capitalize()} Accuracy', linewidth=2)

    plt.title('Classification Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Loss Weights Evolution
    plt.subplot(3, 2, 5)
    if 'loss_weights' in history and history['loss_weights']:
        asr_weights = [x['asr'] for x in history['loss_weights']]
        prosody_weights = [x['prosody'] for x in history['loss_weights']]
        ser_weights = [x['ser'] for x in history['loss_weights']]

        plt.plot(epochs, asr_weights, label='ASR Weight', linewidth=2)
        plt.plot(epochs, prosody_weights, label='Prosody Weight', linewidth=2)
        plt.plot(epochs, ser_weights, label='SER Weight', linewidth=2)

    plt.title('Loss Weight Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Learning Rate Evolution (placeholder for now)
    plt.subplot(3, 2, 6)
    if 'lr_history' in history:
        # TODO: Add actual LR tracking
        plt.plot(epochs, history['lr_history'],
                 label='Learning Rate', linewidth=2)
    else:
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


def plot_task_metrics_comparison(history: Dict, save_path: Optional[str] = None):
    """Plot detailed comparison of task metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(len(history['train_loss']))

    # ASR Loss vs Metrics
    ax = axes[0, 0]
    if 'asr' in history['train_loss'][0]:
        ax.plot(epochs, [x['asr']
                for x in history['train_loss']], 'b-', label='Train Loss')
        ax.set_ylabel('Loss', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        ax2 = ax.twinx()
        if len(history['val_metrics']) > 0 and 'asr' in history['val_metrics'][0]:
            wer_values = [x['asr'].get('wer', 1.0)
                          for x in history['val_metrics']]
            ax2.plot(epochs, wer_values, 'r-', label='WER')
            ax2.set_ylabel('WER', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('ASR Performance')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    # Prosody Performance
    ax = axes[0, 1]
    if 'prosody' in history['train_loss'][0]:
        ax.plot(epochs, [x['prosody']
                for x in history['train_loss']], label='Train Loss')
        ax.plot(epochs, [x['prosody']
                for x in history['val_loss']], label='Val Loss')
        if len(history['val_metrics']) > 0 and 'prosody' in history['val_metrics'][0]:
            acc_values = [x['prosody'].get('accuracy', 0)
                          for x in history['val_metrics']]
            ax.plot(epochs, acc_values, label='Accuracy')
    ax.set_title('Prosody Performance')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Emotion Performance
    ax = axes[1, 0]
    if 'emotion' in history['train_loss'][0]:
        ax.plot(epochs, [x['emotion']
                for x in history['train_loss']], label='Train Loss')
        ax.plot(epochs, [x['emotion']
                for x in history['val_loss']], label='Val Loss')
        if len(history['val_metrics']) > 0 and 'emotion' in history['val_metrics'][0]:
            acc_values = [x['emotion'].get('accuracy', 0)
                          for x in history['val_metrics']]
            ax.plot(epochs, acc_values, label='Accuracy')
    ax.set_title('Emotion Performance')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined F1 Scores
    ax = axes[1, 1]
    for task in ['prosody', 'emotion']:
        if len(history['val_metrics']) > 0 and task in history['val_metrics'][0]:
            f1_values = [x[task].get('f1', 0) for x in history['val_metrics']]
            ax.plot(epochs, f1_values, label=f'{task.capitalize()} F1')
    ax.set_title('F1 Scores Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
