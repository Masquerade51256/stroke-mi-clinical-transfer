"""
Transfer learning trainer with support for various fine-tuning strategies.

This trainer implements:
- Loading pretrained weights from checkpoint files
- Multiple layer freezing strategies (none, all, gradual, selective)
- Two-stage training: frozen feature extraction + fine-tuning
- Cross-subject and within-subject transfer learning
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer
from utils.visualization import plot_training_history, plot_subject_comparison


class TransferLearningTrainer(BaseTrainer):
    """
    Trainer for transfer learning with flexible freezing strategies.
    
    Supports loading pretrained models and applying various transfer learning
    strategies including feature extraction (frozen backbone) and fine-tuning.
    
    Example:
        >>> trainer = TransferLearningTrainer(
        ...     model=model,
        ...     config=config,
        ...     device=device,
        ...     paths=paths,
        ...     logger=logger
        ... )
        >>> results = trainer.train(datasets)
    """
    
    # 定义 freezing 策略常量
    FREEZE_NONE = "none"           # 不冻结，直接微调
    FREEZE_ALL = "all"             # 冻结所有特征层，只训练分类头
    FREEZE_GRADUAL = "gradual"     # 分阶段：先冻结训练，后解冻微调
    FREEZE_SELECTIVE = "selective" # 选择性冻结（保留部分层可训练）
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        """
        Initialize the transfer learning trainer.
        
        Args:
            model: The neural network model to train
            config: Training configuration
            device: Computation device
            paths: Path manager for saving outputs
            logger: Logger instance
            **kwargs: Transfer learning specific args:
                - pretrained_path: Path to pretrained checkpoint
                - freeze_strategy: Freezing strategy (none/all/gradual/selective)
                - freeze_epochs: Epochs to train with frozen layers (for gradual)
                - unfreeze_epochs: Epochs to train after unfreezing (for gradual)
                - learning_rate_frozen: Learning rate for frozen phase
                - learning_rate_finetune: Learning rate for fine-tuning phase
                - layers_to_unfreeze: List of layer names to unfreeze (for selective)
                - warmup_epochs: Warmup epochs before main training
        """
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # Transfer learning configuration
        self.pretrained_path = self.config.get('trainer.args.pretrained_path')
        self.freeze_strategy = self.config.get('trainer.args.freeze_strategy', 'gradual')
        
        # Training phases configuration
        self.freeze_epochs = self.config.get('trainer.args.freeze_epochs', 50)
        self.unfreeze_epochs = self.config.get('trainer.args.unfreeze_epochs', 100)
        
        # Learning rates for different phases
        self.learning_rate_frozen = self.config.get(
            'trainer.args.learning_rate_frozen', 
            self.config.get('training.optimizer.lr', 1e-3)
        )
        self.learning_rate_finetune = self.config.get(
            'trainer.args.learning_rate_finetune',
            self.config.get('training.optimizer.lr', 1e-3) * 0.1  # 默认使用较小的学习率微调
        )
        
        # Selective freezing configuration
        self.layers_to_unfreeze = self.config.get('trainer.args.layers_to_unfreeze', [])
        
        # Warmup configuration
        self.warmup_epochs = self.config.get('trainer.args.warmup_epochs', 0)
        
        # Standard training configuration
        self.epochs = self.config.get('training.epochs', 150)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.learning_rate = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)
        self.k_folds = self.config.get('training.k_folds', 5)
        
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
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate transfer learning configuration."""
        valid_strategies = [self.FREEZE_NONE, self.FREEZE_ALL, self.FREEZE_GRADUAL, self.FREEZE_SELECTIVE]
        if self.freeze_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown freeze_strategy '{self.freeze_strategy}'. "
                f"Available: {valid_strategies}"
            )
        
        if self.pretrained_path is None:
            self._log("WARNING: No pretrained_path specified. Training from scratch.", level='warning')
        elif not Path(self.pretrained_path).exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_path}")
        
        if self.freeze_strategy == self.FREEZE_GRADUAL:
            total_epochs = self.freeze_epochs + self.unfreeze_epochs
            if total_epochs != self.epochs:
                self._log(
                    f"Adjusting epochs to match gradual strategy: "
                    f"freeze={self.freeze_epochs}, unfreeze={self.unfreeze_epochs}, "
                    f"total={total_epochs} (was {self.epochs})",
                    level='info'
                )
                self.epochs = total_epochs
    
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run transfer learning training for all subjects.
        
        Args:
            datasets: Dictionary containing:
                - subjects: List of subject IDs
                - info: Dataset information
                - dataset_cls: Dataset class for lazy loading
                - instances: Pre-loaded dataset instances (optional)
                
        Returns:
            Dictionary containing training results for all subjects
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
        self._log("Starting Transfer Learning Training")
        self._log(f"Strategy: {self.freeze_strategy}")
        self._log(f"Pretrained: {self.pretrained_path or 'None (from scratch)'}")
        self._log(f"Subjects: {subjects}")
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
                                 title=f'Subject {subject_id} Transfer Learning History')
            
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
            'transfer_learning_config': {
                'strategy': self.freeze_strategy,
                'pretrained_path': self.pretrained_path,
                'freeze_epochs': self.freeze_epochs if self.freeze_strategy == self.FREEZE_GRADUAL else None,
                'unfreeze_epochs': self.unfreeze_epochs if self.freeze_strategy == self.FREEZE_GRADUAL else None,
            }
        }
        
        self._log("\n" + "=" * 60)
        self._log("Transfer Learning Training Complete")
        self._log(f"Overall: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)
        
        return final_results
    
    def _train_subject(self, subject_id: int, dataset) -> Tuple[Dict, List]:
        """
        Train a single subject with k-fold cross-validation.
        
        Args:
            subject_id: Subject identifier
            dataset: Dataset instance for this subject
            
        Returns:
            Tuple of (results_dict, training_history)
        """
        # Setup k-fold
        kfold = GroupKFold(n_splits=self.k_folds)
        
        fold_histories = []
        fold_best_accuracies = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(dataset, groups=dataset.group_ids)
        ):
            self._log(f"  Fold {fold_idx + 1}/{self.k_folds}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator(device='cpu')
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator(device='cpu')
            )
            
            # Create model and load pretrained weights
            fold_model = self._create_model_for_fold(dataset)
            
            # Apply transfer learning strategy
            if self.freeze_strategy == self.FREEZE_NONE:
                # 直接微调，不冻结
                fold_history, best_val_acc = self._train_finetune(
                    fold_model, train_loader, val_loader, subject_id, fold_idx
                )
                
            elif self.freeze_strategy == self.FREEZE_ALL:
                # 冻结所有层，只训练分类头
                self._freeze_all_layers(fold_model)
                fold_history, best_val_acc = self._train_finetune(
                    fold_model, train_loader, val_loader, subject_id, fold_idx
                )
                
            elif self.freeze_strategy == self.FREEZE_GRADUAL:
                # 分阶段训练：先冻结，后解冻
                fold_history, best_val_acc = self._train_gradual_unfreeze(
                    fold_model, train_loader, val_loader, subject_id, fold_idx
                )
                
            elif self.freeze_strategy == self.FREEZE_SELECTIVE:
                # 选择性冻结
                self._apply_selective_freezing(fold_model)
                fold_history, best_val_acc = self._train_finetune(
                    fold_model, train_loader, val_loader, subject_id, fold_idx
                )
            
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
        Create a model instance and load pretrained weights.
        
        Args:
            dataset: Dataset instance for inferring model parameters
            
        Returns:
            Model with pretrained weights loaded
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
        
        # Load pretrained weights if specified
        if self.pretrained_path:
            self._load_pretrained_weights(model)
        
        return model
    
    def _load_pretrained_weights(self, model: nn.Module):
        """
        Load pretrained weights from checkpoint file.
        
        Args:
            model: Model to load weights into
        """
        self._log(f"Loading pretrained weights from: {self.pretrained_path}")
        
        checkpoint = torch.load(self.pretrained_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle FP16 compressed checkpoints
        if checkpoint.get('compressed', False):
            state_dict = {k: v.float() if v.dtype == torch.float16 else v
                         for k, v in state_dict.items()}
        
        # Load weights with flexible matching
        model_dict = model.state_dict()
        
        # Filter out incompatible layers (e.g., final classifier with different num_classes)
        compatible_dict = {}
        incompatible_keys = []
        
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    incompatible_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                incompatible_keys.append(f"{k} (not in model)")
        
        # Load compatible weights
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
        
        self._log(f"Loaded {len(compatible_dict)}/{len(state_dict)} layers from pretrained model")
        if incompatible_keys:
            self._log(f"Skipped {len(incompatible_keys)} incompatible layers", level='warning')
    
    def _freeze_all_layers(self, model: nn.Module):
        """
        Freeze all layers except the final classifier.
        
        Args:
            model: Model to freeze
        """
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻分类头（通常是最后一层或最后几层）
        # 这里假设分类头名为 'classifier', 'fc', 'dense', 或 'output'
        unfrozen_layers = []
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['classifier', 'fc', 'dense', 'output', 'final']):
                for param in module.parameters():
                    param.requires_grad = True
                unfrozen_layers.append(name)
        
        self._log(f"Frozen all layers except: {unfrozen_layers}")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self._log(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                 f"({100 * trainable_params / total_params:.1f}%)")
    
    def _apply_selective_freezing(self, model: nn.Module):
        """
        Apply selective freezing based on configuration.
        
        Args:
            model: Model to apply freezing to
        """
        # 首先冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻指定的层
        unfrozen = []
        for name, param in model.named_parameters():
            for layer_pattern in self.layers_to_unfreeze:
                if layer_pattern in name:
                    param.requires_grad = True
                    unfrozen.append(name)
                    break
        
        self._log(f"Selective freezing: unfrozen {len(unfrozen)} parameter groups")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self._log(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                 f"({100 * trainable_params / total_params:.1f}%)")
    
    def _train_gradual_unfreeze(self, model: nn.Module, train_loader: DataLoader,
                                val_loader: DataLoader, subject_id: int, fold_idx: int
                                ) -> Tuple[Dict, float]:
        """
        Two-stage training: frozen feature extraction followed by fine-tuning.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            subject_id: Subject identifier
            fold_idx: Fold number
            
        Returns:
            Tuple of (training_history, best_validation_accuracy)
        """
        # ===== Stage 1: Frozen Training =====
        self._log(f"    Stage 1: Frozen training for {self.freeze_epochs} epochs")
        self._freeze_all_layers(model)
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate_frozen,
            weight_decay=self.weight_decay
        )
        
        stage1_history = self._train_epochs(
            model, train_loader, val_loader, optimizer,
            self.freeze_epochs, subject_id, fold_idx, stage="frozen"
        )
        
        # ===== Stage 2: Fine-tuning =====
        self._log(f"    Stage 2: Fine-tuning for {self.unfreeze_epochs} epochs")
        
        # 解冻所有层
        for param in model.parameters():
            param.requires_grad = True
        
        # 使用较小的学习率进行微调
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate_finetune,
            weight_decay=self.weight_decay
        )
        
        stage2_history = self._train_epochs(
            model, train_loader, val_loader, optimizer,
            self.unfreeze_epochs, subject_id, fold_idx, stage="finetune"
        )
        
        # 合并历史记录
        combined_history = {
            'train_loss': stage1_history['train_loss'] + stage2_history['train_loss'],
            'val_loss': stage1_history['val_loss'] + stage2_history['val_loss'],
            'train_acc': stage1_history['train_acc'] + stage2_history['train_acc'],
            'val_acc': stage1_history['val_acc'] + stage2_history['val_acc'],
            'stage_transition': self.freeze_epochs  # 标记阶段转换点
        }
        
        # 找到最佳验证准确率
        best_val_acc = max(combined_history['val_acc'])
        
        return combined_history, best_val_acc
    
    def _train_finetune(self, model: nn.Module, train_loader: DataLoader,
                        val_loader: DataLoader, subject_id: int, fold_idx: int
                        ) -> Tuple[Dict, float]:
        """
        Standard fine-tuning without freezing.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            subject_id: Subject identifier
            fold_idx: Fold number
            
        Returns:
            Tuple of (training_history, best_validation_accuracy)
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        history = self._train_epochs(
            model, train_loader, val_loader, optimizer,
            self.epochs, subject_id, fold_idx, stage="standard"
        )
        
        best_val_acc = max(history['val_acc'])
        return history, best_val_acc
    
    def _train_epochs(self, model: nn.Module, train_loader: DataLoader,
                      val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                      num_epochs: int, subject_id: int, fold_idx: int,
                      stage: str = "standard"
                      ) -> Dict:
        """
        Train for a specified number of epochs.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            num_epochs: Number of epochs to train
            subject_id: Subject identifier
            fold_idx: Fold number
            stage: Training stage name for logging
            
        Returns:
            Training history dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Progress bar
        desc = f"Sub{subject_id} Fold{fold_idx+1} {stage}"
        epoch_progress_bar = tqdm(
            range(num_epochs),
            desc=desc,
            bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )
        
        best_val_acc = 0.0
        
        for epoch in epoch_progress_bar:
            self.current_epoch = epoch
            
            # Compute learning rate with cosine annealing
            lr = self._compute_lr(epoch, num_epochs, optimizer.param_groups[0]['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train and validate
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
            val_loss, val_acc = self._validate(model, val_loader)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update progress bar
            epoch_progress_bar.set_postfix_str(
                f"LR:{lr:.1e} T:{train_loss:.3f}/{train_acc:.3f} V:{val_loss:.3f}/{val_acc:.3f}"
            )
            
            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.save_checkpoints:
                    self._save_fold_checkpoint(model, optimizer, subject_id, 
                                              fold_idx, epoch, val_acc, stage)
            
            self.global_step += 1
        
        epoch_progress_bar.close()
        return history
    
    def _compute_lr(self, epoch: int, total_epochs: int, initial_lr: float) -> float:
        """
        Compute learning rate with cosine annealing.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            initial_lr: Initial learning rate
            
        Returns:
            Learning rate for this epoch
        """
        return (1 + math.cos(epoch * math.pi / total_epochs)) * initial_lr / 2
    
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
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
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
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_fold_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                             subject_id: int, fold: int, epoch: int,
                             metric: float, stage: str = "") -> None:
        """
        Save checkpoint for a specific fold.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (not saved to save space)
            subject_id: Subject identifier
            fold: Fold number
            epoch: Current epoch
            metric: Performance metric
            stage: Training stage name
        """
        if not self.save_checkpoints:
            return
        
        # Compress model weights
        state_dict = model.state_dict()
        
        if self.use_fp16_compression:
            compressed_state_dict = {}
            for key, value in state_dict.items():
                if value.dtype == torch.float32:
                    compressed_state_dict[key] = value.half()
                else:
                    compressed_state_dict[key] = value
        else:
            compressed_state_dict = state_dict
        
        # Build checkpoint
        checkpoint = {
            'epoch': epoch,
            'subject_id': subject_id,
            'fold': fold,
            'model_state_dict': compressed_state_dict,
            'metric': metric,
            'stage': stage,
            'compressed': self.use_fp16_compression
        }
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=subject_id,
            fold=fold,
            metric=metric
        )
        
        # Add stage suffix to filename if specified
        if stage:
            checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_{stage}.pt"
            )
        
        # Try to save to primary path
        try:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
            return
        except (RuntimeError, IOError, OSError) as e:
            if self.fallback_checkpoint_dir and not self._using_fallback_path:
                import os
                os.makedirs(self.fallback_checkpoint_dir, exist_ok=True)
                self._using_fallback_path = True
                self._log(f"    Info: Switching to fallback checkpoint dir: {self.fallback_checkpoint_dir}",
                         level='info')
            elif not self.fallback_checkpoint_dir:
                if not self._checkpoint_warned:
                    self._log(f"    Warning: Checkpoint save failed (insufficient disk space).",
                             level='warning')
                    self._checkpoint_warned = True
                return
        
        # Try fallback path
        if self._using_fallback_path:
            from pathlib import Path
            fallback_path = Path(self.fallback_checkpoint_dir) / checkpoint_path.name
            try:
                torch.save(checkpoint, fallback_path, _use_new_zipfile_serialization=True)
                self._log(f"    Info: Checkpoint saved to fallback: {fallback_path}", level='info')
            except (RuntimeError, IOError, OSError) as e:
                if not self._checkpoint_warned:
                    self._log(f"    Warning: Fallback checkpoint save also failed.", level='warning')
                    self._checkpoint_warned = True
