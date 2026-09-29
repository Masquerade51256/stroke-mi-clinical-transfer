"""
Supervised learning trainer with k-fold cross-validation.
"""

import math
import copy
from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer
from utils.visualization import plot_training_history, plot_subject_comparison


class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised learning with k-fold cross-validation.
    
    Supports subject-wise training with group k-fold cross-validation
to prevent data leakage from overlapping windows.
    """
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        """
        Initialize the supervised trainer.
        
        Args:
            model: The neural network model to train
            config: Training configuration
            device: Computation device
            paths: Path manager for saving outputs
            logger: Logger instance
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # Training configuration
        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.learning_rate = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)
        self.k_folds = self.config.get('training.k_folds', 5)
        self.scheduler_type = self.config.get('training.scheduler.type', 'cosine')
        
        # Store model class and args for reinitialization
        self.model_type = self.config.get('model.type')
        self.model_args = self.config.get('model.args', {})
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint configuration
        self.save_checkpoints = self.config.get('training.save_checkpoints', True)
        self.save_optimizer_state = self.config.get('training.save_optimizer_state', False)
        self.use_fp16_compression = self.config.get('training.use_fp16_compression', True)  # 默认启用 FP16 压缩
        self._checkpoint_warned = False  # Track if we've already warned about save failures
        
        # 备用路径配置（当主路径保存失败时使用）
        self.fallback_checkpoint_dir = self.config.get('paths.fallback_checkpoint_dir', None)
        self._using_fallback_path = False  # 标记是否正在使用备用路径
        
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run training for all subjects with lazy data loading.
        
        Each subject's data is loaded on-demand to minimize memory usage.
        
        Args:
            datasets: Dictionary containing 'subjects', 'info', and 'dataset_cls'
                     (for lazy loading) or 'instances' (for pre-loaded data)
            
        Returns:
            Dictionary containing training results
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        
        # Support both lazy loading and pre-loaded instances
        use_lazy_loading = 'dataset_cls' in datasets
        dataset_cls = datasets.get('dataset_cls')
        subject_instances = datasets.get('instances', {})
        
        all_results = []
        all_histories = []
        
        self._log("=" * 60)
        self._log("Starting Supervised Training")
        self._log(f"Subjects: {subjects}")
        self._log(f"Epochs: {self.epochs}, K-folds: {self.k_folds}")
        self._log("=" * 60)
        
        for subject_id in subjects:
            self._log(f"\nProcessing Subject: {subject_id}")
            self._log("-" * 40)
            
            # Load data on-demand if using lazy loading
            if use_lazy_loading:
                dataset = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            else:
                dataset = subject_instances[subject_id]
            
            # Train this subject
            subject_results, subject_history = self._train_subject(subject_id, dataset)
            
            all_results.append(subject_results)
            all_histories.append({
                'subject_id': subject_id,
                'history': subject_history
            })
            
            # Plot individual subject history
            subject_viz_dir = self.paths.get_subject_viz_dir(subject_id)
            plot_training_history(subject_history, subject_viz_dir, 
                                 title=f'Subject {subject_id} Training History')
            
            # Free memory after training this subject
            if use_lazy_loading:
                del dataset
                import gc
                gc.collect()
        
        # Plot comparison across subjects
        plot_subject_comparison(all_results, self.paths.viz_dir)
        
        # Calculate overall statistics
        final_results = {
            'subjects': all_results,
            'overall_mean': np.mean([r['avg_val_acc'] for r in all_results]),
            'overall_std': np.std([r['avg_val_acc'] for r in all_results]),
        }
        
        self._log("\n" + "=" * 60)
        self._log("Training Complete")
        self._log(f"Overall: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)
        
        return final_results
    
    def _train_subject(self, subject_id: int, dataset) -> tuple:
        """
        Train a single subject with k-fold cross-validation.
        
        Args:
            subject_id: Subject identifier
            dataset: Dataset instance for this subject
            
        Returns:
            Tuple of (results_dict, training_history)
        """
        # Setup train/validation splitting
        fold_splits = []
        if self.k_folds == 1:
            # Simple train-test split (80/20) when k_folds=1
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(dataset))
            labels = getattr(dataset, 'labels', None)
            stratify = labels if labels is not None and len(np.unique(labels)) >= 2 else None
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=stratify
            )
            fold_splits.append((train_idx, val_idx))
            self._log("  Train/Test split (80/20)")
        else:
            kfold = GroupKFold(n_splits=self.k_folds)
            fold_splits = list(kfold.split(dataset, groups=dataset.group_ids))
        
        fold_histories = []
        fold_best_accuracies = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            if self.k_folds > 1:
                self._log(f"  Fold {fold_idx + 1}/{self.k_folds}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator()
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator()
            )
            
            # Create fresh model for this fold
            fold_model = self._create_model_for_fold(dataset)
            optimizer = torch.optim.Adam(
                fold_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Training loop
            fold_history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }
            best_val_acc = 0.0
            
            # Single progress bar for epochs with real-time metrics
            epoch_progress_bar = tqdm(
                range(self.epochs),
                desc=f'Sub{subject_id} Fold{fold_idx+1}',
                bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
            
            for epoch in epoch_progress_bar:
                self.current_epoch = epoch
                
                # Learning rate scheduling
                if self.scheduler_type == 'cosine':
                    lr = self._compute_lr(epoch)
                else:
                    lr = self.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Train and validate
                train_loss, train_acc = self._train_epoch(fold_model, train_loader, optimizer)
                val_loss, val_acc = self._validate(fold_model, val_loader)
                
                # Record history
                fold_history['train_loss'].append(train_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_loss'].append(val_loss)
                fold_history['val_acc'].append(val_acc)
                
                # Update progress bar with all metrics in compact format
                epoch_progress_bar.set_postfix_str(
                    f"LR:{lr:.1e} T:{train_loss:.3f}/{train_acc:.3f} V:{val_loss:.3f}/{val_acc:.3f}"
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_fold_checkpoint(fold_model, optimizer, subject_id, 
                                              fold_idx, epoch, val_acc)
                
                self.global_step += 1
            
            epoch_progress_bar.close()
            fold_histories.append(fold_history)
            fold_best_accuracies.append(best_val_acc)
            self._log(f"    Best Val Acc: {best_val_acc:.4f}")
        
        # Calculate subject statistics
        subject_avg = np.mean(fold_best_accuracies)
        subject_std = np.std(fold_best_accuracies)
        
        results = {
            'subject_id': subject_id,
            'avg_val_acc': subject_avg,
            'std_val_acc': subject_std,
            'fold_accuracies': fold_best_accuracies
        }
        
        return results, fold_histories
    
    def _create_model_for_fold(self, dataset) -> nn.Module:
        """
        Create a fresh model instance for a new fold.
        
        Args:
            dataset: Dataset instance (for inferring model parameters)
            
        Returns:
            New model instance
        """
        from core.registry import MODELS, build_from_config
        
        # Build model config
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        # Auto-populate model arguments from dataset info
        if 'num_channels' not in model_config['args']:
            # Get from data shape: (N, bands, channels, time)
            model_config['args']['num_channels'] = dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = dataset.data.shape[3]
        
        # Build model
        model = build_from_config(model_config, MODELS)
        return model.to(self.device)
    
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
                    optimizer: torch.optim.Optimizer) -> tuple:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
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
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> tuple:
        """
        Validate the model.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
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
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_fold_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                             subject_id: int, fold: int, epoch: int, 
                             metric: float) -> None:
        """
        Save checkpoint for a specific fold with aggressive compression.
        
        Uses FP16 quantization and high compression to minimize disk usage.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            subject_id: Subject identifier
            fold: Fold number
            epoch: Current epoch
            metric: Performance metric
        """
        if not self.save_checkpoints:
            return
        
        # 压缩模型权重：将 float32 转换为 float16（节省 50% 空间）
        state_dict = model.state_dict()
        
        if self.use_fp16_compression:
            compressed_state_dict = {}
            for key, value in state_dict.items():
                if value.dtype == torch.float32:
                    # 转换为 float16，可节省约 50% 空间
                    compressed_state_dict[key] = value.half()
                else:
                    compressed_state_dict[key] = value
        else:
            compressed_state_dict = state_dict
        
        # 构建最小化 checkpoint（不包含优化器状态）
        checkpoint = {
            'epoch': epoch,
            'subject_id': subject_id,
            'fold': fold,
            'model_state_dict': compressed_state_dict,
            'metric': metric,
            'compressed': self.use_fp16_compression  # 标记为压缩格式
        }
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=subject_id,
            fold=fold,
            metric=metric
        )
        
        # 尝试保存到主路径
        try:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
            return
        except (RuntimeError, IOError, OSError) as e:
            # 主路径保存失败，尝试备用路径
            if self.fallback_checkpoint_dir and not self._using_fallback_path:
                import os
                os.makedirs(self.fallback_checkpoint_dir, exist_ok=True)
                self._using_fallback_path = True
                self._log(f"    Info: Switching to fallback checkpoint dir: {self.fallback_checkpoint_dir}", level='info')
            elif not self.fallback_checkpoint_dir:
                # 没有备用路径，静默失败
                if not self._checkpoint_warned:
                    self._log(f"    Warning: Checkpoint save failed (insufficient disk space). Consider setting fallback_checkpoint_dir.", level='warning')
                    self._checkpoint_warned = True
                return
        
        # 尝试保存到备用路径
        if self._using_fallback_path:
            from pathlib import Path
            fallback_path = Path(self.fallback_checkpoint_dir) / checkpoint_path.name
            try:
                torch.save(checkpoint, fallback_path, _use_new_zipfile_serialization=True)
                self._log(f"    Info: Checkpoint saved to fallback: {fallback_path}", level='info')
            except (RuntimeError, IOError, OSError) as e:
                if not self._checkpoint_warned:
                    self._log(f"    Warning: Fallback checkpoint save also failed: {str(e)[:60]}...", level='warning')
                    self._checkpoint_warned = True
