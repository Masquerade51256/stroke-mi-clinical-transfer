"""
Leave One Subject Out (LOSO) cross-validation trainer.

This trainer implements subject-independent evaluation by:
1. Using N-1 subjects for training
2. Using the left-out subject for testing
3. Repeating for all subjects

This is the gold standard for evaluating EEG-BCI model generalization
across different subjects.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer
from utils.visualization import plot_subject_comparison


class ConcatSubjectDataset(Dataset):
    """
    Dataset that concatenates multiple subjects' data.
    
    Used for LOSO training to combine N-1 subjects into a single training set.
    """
    
    def __init__(self, datasets: List[Dataset]):
        """
        Initialize with a list of subject datasets.
        
        Args:
            datasets: List of dataset instances, one per subject
        """
        self.datasets = datasets
        self.cumulative_sizes = self._calculate_cumulative_sizes()
        
        # Extract and combine data for compatibility with GroupKFold
        self._combine_data()
    
    def _calculate_cumulative_sizes(self) -> List[int]:
        """Calculate cumulative sizes for indexing."""
        sizes = [0]
        for ds in self.datasets:
            sizes.append(sizes[-1] + len(ds))
        return sizes
    
    def _combine_data(self):
        """Combine data, labels and group_ids from all datasets."""
        all_data = []
        all_labels = []
        all_group_ids = []
        
        for ds_idx, ds in enumerate(self.datasets):
            # Access underlying numpy arrays
            all_data.append(ds.data)
            all_labels.append(ds.labels)
            # Offset group_ids by dataset index to ensure uniqueness
            all_group_ids.append(ds.group_ids + ds_idx * 1000)
        
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.group_ids = np.concatenate(all_group_ids, axis=0)
    
    def __len__(self) -> int:
        """Return total number of samples across all subjects."""
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample by global index."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes[1:], 1):
            if idx < size:
                dataset_idx = i - 1
                local_idx = idx - self.cumulative_sizes[i - 1]
                break
        else:
            # Should not reach here if idx is valid
            raise IndexError(f"Index {idx} out of range")
        
        return self.datasets[dataset_idx][local_idx]
    
    def get_subject_distribution(self) -> Dict[int, int]:
        """Get sample count per subject."""
        return {
            i: len(ds) for i, ds in enumerate(self.datasets)
        }


class LOSOTrainer(BaseTrainer):
    """
    Leave One Subject Out cross-validation trainer.
    
    For each subject:
    - Training: All other subjects combined
    - Testing: The left-out subject
    
    This evaluates how well the model generalizes to unseen subjects,
    which is crucial for real-world BCI applications.
    
    Example:
        >>> trainer = LOSOTrainer(
        ...     model=model,
        ...     config=config,
        ...     device=device,
        ...     paths=paths,
        ...     logger=logger
        ... )
        >>> results = trainer.train(datasets)
    """
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        """
        Initialize the LOSO trainer.
        
        Args:
            model: The neural network model to train
            config: Training configuration
            device: Computation device
            paths: Path manager for saving outputs
            logger: Logger instance
            **kwargs: Additional LOSO specific args:
                - inner_k_folds: Number of folds for cross-validation within training set (optional)
                - use_group_kfold: Whether to use GroupKFold for training split (default: True)
                - balance_classes: Whether to balance classes across subjects (default: False)
        """
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # LOSO configuration - from kwargs (passed via trainer.args) or config
        self.inner_k_folds = kwargs.get('inner_k_folds', 
                                       self.config.get('trainer.args.inner_k_folds', 0))
        self.use_group_kfold = kwargs.get('use_group_kfold', 
                                          self.config.get('trainer.args.use_group_kfold', True))
        self.balance_classes = kwargs.get('balance_classes', 
                                          self.config.get('trainer.args.balance_classes', False))
        
        # Standard training configuration
        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.learning_rate = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)
        
        # Model configuration
        self.model_type = self.config.get('model.type')
        self.model_args = self.config.get('model.args', {})
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint configuration
        self.save_checkpoints = self.config.get('training.save_checkpoints', True)
        self.use_fp16_compression = self.config.get('training.use_fp16_compression', True)
        self._checkpoint_warned = False
        self.fallback_checkpoint_dir = self.config.get('paths.fallback_checkpoint_dir', None)
        self._using_fallback_path = False
        
    def _run_loso_on_subjects(self, subjects: List[int], dataset_info: Dict,
                              dataset_cls, group_name: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run LOSO on a given list of subjects.
        
        Args:
            subjects: List of subject IDs to run LOSO on
            dataset_info: Dataset information
            dataset_cls: Dataset class for loading
            group_name: Optional group name for logging
            
        Returns:
            Tuple of (all_results, all_histories)
        """
        all_results = []
        all_histories = []
        prefix = f"[{group_name}] " if group_name else ""
        
        for test_subject_idx, test_subject_id in enumerate(subjects):
            self._log(f"\n{prefix}[{test_subject_idx + 1}/{len(subjects)}] Test Subject: {test_subject_id}")
            self._log("-" * 40)
            
            train_subject_ids = [s for s in subjects if s != test_subject_id]
            
            self._log(f"Loading training data from subjects: {train_subject_ids}")
            train_datasets = []
            for train_subject_id in train_subject_ids:
                ds = dataset_cls(subject_id=train_subject_id, dataset_info=dataset_info)
                train_datasets.append(ds)
            
            train_dataset = ConcatSubjectDataset(train_datasets)
            self._log(f"Training set: {len(train_dataset)} samples from {len(train_datasets)} subjects")
            
            test_dataset = dataset_cls(subject_id=test_subject_id, dataset_info=dataset_info)
            self._log(f"Test set: {len(test_dataset)} samples from subject {test_subject_id}")
            
            subject_results, subject_history = self._train_loso_round(
                test_subject_id, train_dataset, test_dataset
            )
            
            if group_name:
                subject_results['group'] = group_name
            
            all_results.append(subject_results)
            all_histories.append({
                'test_subject_id': test_subject_id,
                'history': subject_history,
                'group': group_name
            })
            
            subject_viz_dir = self.paths.get_subject_viz_dir(test_subject_id)
            self._plot_loso_history(subject_history, subject_viz_dir, test_subject_id)
            
            del train_dataset, test_dataset, train_datasets
            import gc
            gc.collect()
        
        return all_results, all_histories
    
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run LOSO cross-validation training.
        
        Supports standard LOSO across all subjects, or stratified LOSO
        where subjects are grouped by a clinical field (e.g., ParalysisSide)
        and LOSO is performed within each group.
        
        Args:
            datasets: Dictionary containing:
                - subjects: List of subject IDs
                - info: Dataset information
                - dataset_cls: Dataset class for loading
                - subject_groups: Optional dict mapping group names to subject lists
                
        Returns:
            Dictionary containing LOSO results for all subjects
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')
        subject_groups = datasets.get('subject_groups')
        
        if not dataset_cls:
            raise ValueError("LOSO trainer requires 'dataset_cls' for lazy loading")
        
        if subject_groups:
            all_results = []
            all_histories = []
            group_stats = {}
            
            self._log("=" * 60)
            self._log("Starting Stratified LOSO Training")
            self._log(f"Stratification groups: {list(subject_groups.keys())}")
            self._log("=" * 60)
            
            for group_name, group_subjects in subject_groups.items():
                self._log(f"\n{'='*60}")
                self._log(f"Group: {group_name} ({len(group_subjects)} subjects)")
                self._log(f"{'='*60}")
                
                group_results, group_histories = self._run_loso_on_subjects(
                    group_subjects, dataset_info, dataset_cls, group_name
                )
                all_results.extend(group_results)
                all_histories.extend(group_histories)
                
                group_accs = [r['test_acc'] for r in group_results]
                group_stats[group_name] = {
                    'mean': float(np.mean(group_accs)),
                    'std': float(np.std(group_accs)),
                    'min': float(np.min(group_accs)),
                    'max': float(np.max(group_accs)),
                    'n': len(group_accs),
                    'subject_ids': group_subjects
                }
            
            self._plot_loso_comparison(all_results, group_stats=group_stats)
            
            test_accuracies = [r['test_acc'] for r in all_results]
            final_results = {
                'subjects': all_results,
                'group_stats': group_stats,
                'overall_mean': float(np.mean(test_accuracies)),
                'overall_std': float(np.std(test_accuracies)),
                'overall_min': float(np.min(test_accuracies)),
                'overall_max': float(np.max(test_accuracies)),
                'loso_config': {
                    'num_subjects': len(subjects),
                    'subject_ids': subjects,
                    'stratified': True,
                    'stratify_groups': list(subject_groups.keys()),
                }
            }
            
            self._log("\n" + "=" * 60)
            self._log("Stratified LOSO Training Complete")
            for group_name, stats in group_stats.items():
                self._log(f"  {group_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
            self._log(f"  Overall: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
            self._log("=" * 60)
            
            return final_results
        
        else:
            self._log("=" * 60)
            self._log("Starting LOSO (Leave One Subject Out) Training")
            self._log(f"Total subjects: {len(subjects)}")
            self._log(f"Each subject will be tested once, trained on {len(subjects)-1} others")
            self._log("=" * 60)
            
            all_results, all_histories = self._run_loso_on_subjects(
                subjects, dataset_info, dataset_cls
            )
            
            self._plot_loso_comparison(all_results)
            
            test_accuracies = [r['test_acc'] for r in all_results]
            final_results = {
                'subjects': all_results,
                'overall_mean': float(np.mean(test_accuracies)),
                'overall_std': float(np.std(test_accuracies)),
                'overall_min': float(np.min(test_accuracies)),
                'overall_max': float(np.max(test_accuracies)),
                'loso_config': {
                    'num_subjects': len(subjects),
                    'subject_ids': subjects,
                }
            }
            
            self._log("\n" + "=" * 60)
            self._log("LOSO Training Complete")
            self._log(f"Mean Accuracy: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
            self._log(f"Range: [{final_results['overall_min']:.4f}, {final_results['overall_max']:.4f}]")
            self._log("=" * 60)
            
            return final_results
    
    def _train_loso_round(self, test_subject_id: int, 
                          train_dataset: ConcatSubjectDataset,
                          test_dataset: Dataset) -> Tuple[Dict, Dict]:
        """
        Train and evaluate one LOSO round.
        
        Args:
            test_subject_id: ID of the test subject
            train_dataset: Combined training data from other subjects
            test_dataset: Test data from the left-out subject
            
        Returns:
            Tuple of (results_dict, training_history)
        """
        # Create model
        model = self._create_model(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device='cpu'),
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=torch.Generator(device='cpu')
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        best_test_acc = 0.0
        best_epoch = 0
        
        # Progress bar
        epoch_progress_bar = tqdm(
            range(self.epochs),
            desc=f'LOSO Sub{test_subject_id}',
            bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )
        
        for epoch in epoch_progress_bar:
            self.current_epoch = epoch
            
            # Cosine annealing learning rate
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
            
            # Evaluate on test subject
            test_loss, test_acc = self._validate(model, test_loader)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Update progress bar
            epoch_progress_bar.set_postfix_str(
                f"LR:{lr:.1e} Tr:{train_loss:.3f}/{train_acc:.3f} Te:{test_loss:.3f}/{test_acc:.3f}"
            )
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                if self.save_checkpoints:
                    self._save_checkpoint(model, optimizer, test_subject_id, epoch, test_acc)
            
            self.global_step += 1
        
        epoch_progress_bar.close()
        
        # Final evaluation
        final_test_loss, final_test_acc = self._validate(model, test_loader)
        
        results = {
            'test_subject_id': test_subject_id,
            'test_acc': final_test_acc,
            'test_loss': final_test_loss,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        }
        
        self._log(f"    Best: {best_test_acc:.4f} (epoch {best_epoch}), Final: {final_test_acc:.4f}")
        
        return results, history
    
    def _create_model(self, dataset) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            dataset: Dataset for inferring model parameters
            
        Returns:
            New model instance on device
        """
        from core.registry import MODELS, build_from_config
        
        # Build model config
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        # Auto-populate model arguments from dataset info
        if 'num_channels' not in model_config['args']:
            model_config['args']['num_channels'] = dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = dataset.data.shape[3]
        
        # Build model
        model = build_from_config(model_config, MODELS)
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"    Model: {total_params:,} params ({trainable_params:,} trainable)")
        
        return model
    
    def _compute_lr(self, epoch: int) -> float:
        """
        Compute learning rate with cosine annealing.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Learning rate for this epoch
        """
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.learning_rate / 2
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        test_subject_id: int, epoch: int, metric: float) -> None:
        """Save checkpoint."""
        if not self.save_checkpoints:
            return
        
        # Compress model weights
        state_dict = model.state_dict()
        
        if self.use_fp16_compression:
            compressed_state_dict = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in state_dict.items()
            }
        else:
            compressed_state_dict = state_dict
        
        checkpoint = {
            'epoch': epoch,
            'test_subject_id': test_subject_id,
            'model_state_dict': compressed_state_dict,
            'metric': metric,
            'compressed': self.use_fp16_compression
        }
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=test_subject_id,
            fold=0,  # LOSO doesn't use folds, use 0
            metric=metric
        )
        
        # Rename to indicate LOSO
        checkpoint_path = checkpoint_path.with_name(
            f"loso_subject{test_subject_id}_acc{metric:.4f}.pt"
        )
        
        try:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
        except (RuntimeError, IOError, OSError) as e:
            if not self._checkpoint_warned:
                self._log(f"Warning: Checkpoint save failed: {e}", level='warning')
                self._checkpoint_warned = True
    
    def _plot_loso_history(self, history: Dict, save_dir: Path, test_subject_id: int):
        """Plot training history for one LOSO round."""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax = axes[0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Subject {test_subject_id} LOSO Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[1]
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Subject {test_subject_id} LOSO Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / f'loso_training_history.png'
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    
    def _plot_loso_comparison(self, results: List[Dict], group_stats: Optional[Dict] = None):
        """Plot LOSO results comparison across subjects."""
        import matplotlib.pyplot as plt
        
        subject_ids = [r['test_subject_id'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        fig, ax = plt.subplots(figsize=(max(8, len(subject_ids) * 0.5), 6))
        
        if group_stats:
            # Color by group using tab10 palette
            group_names = list(group_stats.keys())
            group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
            group_color_map = {name: group_colors[i] for i, name in enumerate(group_names)}
            
            colors = []
            for r in results:
                g = r.get('group')
                if g and g in group_color_map:
                    colors.append(group_color_map[g])
                else:
                    colors.append('steelblue')
            
            bars = ax.bar(range(len(subject_ids)), test_accs,
                         color=colors, edgecolor='black', alpha=0.7)
            
            # Add legend for groups
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=group_color_map[g], edgecolor='black', label=g)
                              for g in group_names]
            ax.legend(handles=legend_handles, title='Group', loc='upper right')
        else:
            bars = ax.bar(range(len(subject_ids)), test_accs, 
                         color='steelblue', edgecolor='black', alpha=0.7)
            
            # Color bars by performance
            for i, (bar, acc) in enumerate(zip(bars, test_accs)):
                if acc >= 0.8:
                    bar.set_color('green')
                elif acc >= 0.6:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        ax.set_xlabel('Test Subject ID')
        ax.set_ylabel('Test Accuracy')
        title = 'Stratified LOSO Cross-Validation Results' if group_stats else 'LOSO Cross-Validation Results'
        ax.set_title(title)
        ax.set_xticks(range(len(subject_ids)))
        ax.set_xticklabels([f'Sub{s}' for s in subject_ids], rotation=45)
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, test_accs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add mean line
        mean_acc = np.mean(test_accs)
        ax.axhline(y=mean_acc, color='r', linestyle='--', 
                  label=f'Mean: {mean_acc:.3f}')
        if not group_stats:
            ax.legend()
        
        plt.tight_layout()
        save_path = self.paths.viz_dir / 'loso_comparison.png'
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        self._log(f"LOSO comparison plot saved to: {save_path}")
