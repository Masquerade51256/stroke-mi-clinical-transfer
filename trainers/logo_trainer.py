"""
LOGO (Leave One Group Out) cross-validation trainer.

For heterogeneous datasets (e.g., stroke patients with varying severity):
- Group similar subjects together based on metadata (severity, age, etc.)
- Leave one group out for testing
- Train on remaining groups

This provides a more realistic evaluation for clinical applications where
patients may form distinct subgroups.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer


class LOGOTrainer(BaseTrainer):
    """
    Leave One Group Out cross-validation trainer.
    
    Groups subjects based on metadata (e.g., stroke severity, age, etc.)
    and performs leave-one-group-out cross-validation.
    
    This is particularly useful for heterogeneous datasets like stroke patients
    where subjects can be grouped by clinical characteristics.
    
    Example:
        >>> # Group by severity (mild, moderate, severe)
        >>> trainer = LOGOTrainer(
        ...     model=model,
        ...     config=config,
        ...     device=device,
        ...     paths=paths,
        ...     logger=logger,
        ...     grouping_func=lambda sid, info: info['severity'][sid]
        ... )
    """
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # LOGO configuration
        self.grouping_key = kwargs.get('grouping_key',
                                       self.config.get('trainer.args.grouping_key', 'severity'))
        self.streaming = kwargs.get('streaming',
                                    self.config.get('trainer.args.streaming', True))
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps',
                                                       self.config.get('trainer.args.gradient_accumulation_steps', 1))
        
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
        
        # Will be populated during training
        self.groups = {}
        
    def _create_groups(self, subjects: List[int], dataset_info: Dict) -> Dict[str, List[int]]:
        """
        Create subject groups based on metadata.
        
        Args:
            subjects: List of subject IDs
            dataset_info: Dataset configuration containing grouping info
            
        Returns:
            Dictionary mapping group names to subject ID lists
        """
        groups = {}
        
        # Try to get grouping info from dataset_info
        subject_info = dataset_info.get('dataset', {}).get('subject_info', {})
        
        if not subject_info:
            # If no grouping info, create artificial groups (e.g., by subject ID ranges)
            self._log("No grouping metadata found, creating groups by subject ID")
            n_groups = 5  # Default: 5 groups
            group_size = len(subjects) // n_groups
            
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = start_idx + group_size if i < n_groups - 1 else len(subjects)
                group_name = f"group_{i+1}"
                groups[group_name] = subjects[start_idx:end_idx]
        else:
            # Group by specified key
            for subject_id in subjects:
                subject_meta = subject_info.get(str(subject_id), {})
                group_key = subject_meta.get(self.grouping_key, 'unknown')
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(subject_id)
        
        return groups
    
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run LOGO cross-validation training.
        
        Args:
            datasets: Dictionary containing:
                - subjects: List of subject IDs
                - info: Dataset information (should contain grouping metadata)
                - dataset_cls: Dataset class for loading
                
        Returns:
            Dictionary containing LOGO results
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')
        
        if not dataset_cls:
            raise ValueError("LOGOTrainer requires 'dataset_cls'")
        
        # Create groups
        self.groups = self._create_groups(subjects, dataset_info)
        
        self._log("=" * 60)
        self._log("Starting LOGO (Leave One Group Out) Training")
        self._log(f"Grouping key: {self.grouping_key}")
        self._log(f"Number of groups: {len(self.groups)}")
        for group_name, group_subjects in self.groups.items():
            self._log(f"  {group_name}: {len(group_subjects)} subjects - {group_subjects}")
        self._log("=" * 60)
        
        all_results = []
        all_histories = []
        
        # LOGO loop: each group is the test set once
        for test_group_idx, (test_group_name, test_group_subjects) in enumerate(self.groups.items()):
            self._log(f"\n[{test_group_idx + 1}/{len(self.groups)}] Test Group: {test_group_name}")
            self._log(f"Test subjects: {test_group_subjects}")
            self._log("-" * 40)
            
            # Get training groups (all except test group)
            train_subject_ids = []
            for group_name, group_subjects in self.groups.items():
                if group_name != test_group_name:
                    train_subject_ids.extend(group_subjects)
            
            self._log(f"Training subjects: {len(train_subject_ids)} from {len(self.groups) - 1} groups")
            
            # Train and evaluate
            if self.streaming:
                subject_results, subject_history = self._train_logo_streaming(
                    test_group_name, test_group_subjects, train_subject_ids,
                    dataset_cls, dataset_info
                )
            else:
                subject_results, subject_history = self._train_logo_standard(
                    test_group_name, test_group_subjects, train_subject_ids,
                    dataset_cls, dataset_info
                )
            
            all_results.append(subject_results)
            all_histories.append({
                'test_group': test_group_name,
                'history': subject_history
            })
            
            # Plot history
            group_viz_dir = self.paths.exp_dir / f'group_{test_group_name}'
            group_viz_dir.mkdir(exist_ok=True)
            self._plot_history(subject_history, group_viz_dir, test_group_name)
            
            import gc
            gc.collect()
        
        # Final comparison
        self._plot_comparison(all_results)
        
        # Statistics
        test_accuracies = [r['test_acc'] for r in all_results]
        final_results = {
            'groups': all_results,
            'overall_mean': np.mean(test_accuracies),
            'overall_std': np.std(test_accuracies),
            'group_info': {name: len(subjs) for name, subjs in self.groups.items()},
            'logo_config': {
                'grouping_key': self.grouping_key,
                'num_groups': len(self.groups),
                'streaming': self.streaming
            }
        }
        
        self._log("\n" + "=" * 60)
        self._log("LOGO Training Complete")
        self._log(f"Mean Accuracy: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)
        
        return final_results
    
    def _train_logo_streaming(self, test_group_name: str, test_subjects: List[int],
                              train_subject_ids: List[int], dataset_cls, dataset_info: Dict
                              ) -> Tuple[Dict, Dict]:
        """Train LOGO round with streaming (memory-efficient)."""
        # Similar to StreamingLOSOTrainer but with group-based evaluation
        
        # Create model from sample
        sample_ds = dataset_cls(subject_id=train_subject_ids[0], dataset_info=dataset_info)
        model = self._create_model(sample_ds)
        del sample_ds
        
        # Load test group data
        test_datasets = []
        for subject_id in test_subjects:
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            test_datasets.append(ds)
        test_dataset = ConcatDataset(test_datasets)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        best_test_acc = 0.0
        best_epoch = 0
        
        epoch_bar = tqdm(range(self.epochs), desc=f'Group{test_group_name}',
                        bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} {postfix}')
        
        for epoch in epoch_bar:
            self.current_epoch = epoch
            
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train with streaming
            train_loss, train_acc = self._train_epoch_streaming(
                model, optimizer, train_subject_ids, dataset_cls, dataset_info
            )
            
            # Evaluate on test group
            test_loss, test_acc = self._validate(model, test_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_bar.set_postfix_str(
                f"LR:{lr:.1e} Tr:{train_loss:.3f}/{train_acc:.3f} Te:{test_loss:.3f}/{test_acc:.3f}"
            )
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                if self.save_checkpoints:
                    self._save_checkpoint(model, test_group_name, epoch, test_acc)
            
            self.global_step += 1
        
        epoch_bar.close()
        
        final_loss, final_acc = self._validate(model, test_loader)
        
        return {
            'test_group': test_group_name,
            'test_acc': final_acc,
            'test_loss': final_loss,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch,
            'train_subjects': len(train_subject_ids),
            'test_subjects': len(test_subjects),
            'test_subject_ids': test_subjects
        }, history
    
    def _train_epoch_streaming(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                               train_subject_ids: List[int], dataset_cls, dataset_info: Dict
                               ) -> Tuple[float, float]:
        """Train one epoch with streaming."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        subject_order = train_subject_ids.copy()
        np.random.shuffle(subject_order)
        
        accumulation_counter = 0
        
        for subject_id in subject_order:
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            
            indices = list(range(len(ds)))
            np.random.shuffle(indices)
            
            for i in range(0, len(ds), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data, label = ds[idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                inputs = torch.stack(batch_data).to(self.device)
                labels = torch.stack(batch_labels).to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                accumulation_counter += 1
                
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * inputs.size(0) * self.gradient_accumulation_steps
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            del ds
            import gc
            gc.collect()
        
        if accumulation_counter % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _create_model(self, sample_dataset) -> nn.Module:
        """Create model."""
        from core.registry import MODELS, build_from_config
        
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        if 'num_channels' not in model_config['args']:
            model_config['args']['num_channels'] = sample_dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(sample_dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = sample_dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = sample_dataset.data.shape[3]
        
        model = build_from_config(model_config, MODELS)
        model = model.to(self.device)
        
        return model
    
    def _compute_lr(self, epoch: int) -> float:
        """Cosine annealing."""
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.learning_rate / 2
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate."""
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
    
    def _save_checkpoint(self, model: nn.Module, test_group_name: str, 
                        epoch: int, metric: float) -> None:
        """Save checkpoint."""
        if not self.save_checkpoints:
            return
        
        state_dict = model.state_dict()
        if self.use_fp16_compression:
            state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in state_dict.items()}
        
        checkpoint = {
            'epoch': epoch,
            'test_group': test_group_name,
            'model_state_dict': state_dict,
            'metric': metric,
            'compressed': self.use_fp16_compression
        }
        
        checkpoint_path = self.paths.exp_dir / f'logo_group{test_group_name}_acc{metric:.4f}.pt'
        
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception:
            pass
    
    def _plot_history(self, history: Dict, save_dir: Path, test_group_name: str):
        """Plot history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['test_loss'], 'r-', label='Test')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Group {test_group_name} Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, history['test_acc'], 'r-', label='Test')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Group {test_group_name} Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'logo_history.png', dpi=150)
        plt.close()
    
    def _plot_comparison(self, results: List[Dict]):
        """Plot comparison."""
        import matplotlib.pyplot as plt
        
        group_names = [r['test_group'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if acc >= 0.8 else 'yellow' if acc >= 0.6 else 'red' 
                  for acc in test_accs]
        
        bars = ax.bar(range(len(group_names)), test_accs, color=colors,
                     edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Test Group')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('LOGO Cross-Validation Results by Group')
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=45)
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis='y', alpha=0.3)
        
        mean_acc = np.mean(test_accs)
        ax.axhline(y=mean_acc, color='r', linestyle='--',
                  label=f'Mean: {mean_acc:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.paths.viz_dir / 'logo_comparison.png', dpi=150)
        plt.close()
