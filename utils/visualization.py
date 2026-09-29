"""
Visualization utilities for training results.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_training_history(training_history: List[Dict[str, List[float]]], 
                         save_path: Path,
                         title: str = 'Training History') -> None:
    """
    Plot training history for each fold.
    
    Args:
        training_history: List of history dicts per fold, each containing
                         'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Directory to save the plot
        title: Plot title
    """
    if not training_history:
        print("Warning: Training history is empty. Skipping plot.")
        return
    
    num_folds = len(training_history)
    fig, axes = plt.subplots(2, num_folds, figsize=(5 * num_folds, 8), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    for fold_idx, fold_history in enumerate(training_history):
        epochs = range(1, len(fold_history['train_loss']) + 1)
        
        # Plot Loss
        ax_loss = axes[0, fold_idx]
        ax_loss.plot(epochs, fold_history['train_loss'], 'b-', label='Train Loss')
        ax_loss.plot(epochs, fold_history['val_loss'], 'r-', label='Val Loss')
        ax_loss.set_title(f'Fold {fold_idx + 1}')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Plot Accuracy
        ax_acc = axes[1, fold_idx]
        ax_acc.plot(epochs, fold_history['train_acc'], 'b-', label='Train Acc')
        ax_acc.plot(epochs, fold_history['val_acc'], 'r-', label='Val Acc')
        ax_acc.set_title(f'Fold {fold_idx + 1}')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plot_file = save_path / 'training_history.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training history plot saved to: {plot_file}")


def plot_subject_comparison(subject_results: List[Dict[str, Any]], 
                           save_path: Path,
                           metric_name: str = 'Validation Accuracy') -> None:
    """
    Plot comparison of metrics across subjects.
    
    Args:
        subject_results: List of result dicts per subject, each containing
                        'subject_id', 'avg_val_acc', 'std_val_acc'
        save_path: Directory to save the plot
        metric_name: Name of the metric being plotted
    """
    if not subject_results:
        print("Warning: No subject results for comparison plot.")
        return
    
    subject_ids = [res['subject_id'] for res in subject_results]
    avg_values = [res['avg_val_acc'] for res in subject_results]
    std_values = [res.get('std_val_acc', 0) for res in subject_results]
    
    overall_mean = np.mean(avg_values)
    overall_std = np.std(avg_values)
    
    fig, ax = plt.subplots(figsize=(max(6, len(subject_ids) * 0.6), 6))
    
    bars = ax.bar(subject_ids, avg_values, yerr=std_values,
                  capsize=5, color='skyblue', edgecolor='black', alpha=0.7,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_xlabel('Subject ID')
    ax.set_ylabel(f'Average {metric_name}')
    ax.set_title(f'Model Performance by Subject\nOverall: {overall_mean:.3f} ± {overall_std:.3f}')
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val, std in zip(bars, avg_values, std_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = save_path / f'subject_comparison_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Subject comparison plot saved to: {plot_file}")


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[Path] = None,
                         title: str = 'Confusion Matrix',
                         normalize: bool = False) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close(fig)
