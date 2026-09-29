"""
Streaming LOSO (Leave One Subject Out) trainer with memory-efficient data loading.

For datasets with many subjects (e.g., 50 subjects in XW Stroke):
- Instead of loading all N-1 subjects into memory at once
- Stream data subject by subject during each epoch
- Use gradient accumulation to simulate large batch training

This reduces memory from O(N) to O(1) subjects while maintaining the same LOSO evaluation.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer
from utils.clinical_source_selection import (
    load_clinical_metadata,
    select_source_subjects
)


class StreamingSubjectDataset(Dataset):
    """
    Memory-efficient dataset that loads subjects on-demand.
    
    Instead of loading all data into memory, it maintains a list of subject IDs
    and loads data batch by batch during iteration.
    """
    
    def __init__(self, subject_ids: List[int], dataset_cls, dataset_info: Dict,
                 subject_buffer_size: int = 1, device: str = 'cpu'):
        """
        Initialize streaming dataset.
        
        Args:
            subject_ids: List of subject IDs to include
            dataset_cls: Dataset class for loading subjects
            dataset_info: Dataset configuration
            subject_buffer_size: Number of subjects to keep in memory (default: 1)
            device: Device for tensor storage
        """
        self.subject_ids = subject_ids
        self.dataset_cls = dataset_cls
        self.dataset_info = dataset_info
        self.subject_buffer_size = subject_buffer_size
        self.device = device
        
        # Cache for loaded subjects (LRU style)
        self._subject_cache = {}
        self._cache_order = []
        
        # Pre-compute total size without loading all data
        self._total_size = None
        self._subject_sizes = {}
    
    def _get_subject_data(self, subject_id: int):
        """Load subject data with caching."""
        if subject_id in self._subject_cache:
            # Move to end (most recently used)
            self._cache_order.remove(subject_id)
            self._cache_order.append(subject_id)
            return self._subject_cache[subject_id]
        
        # Load subject
        ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
        
        # Cache management
        if len(self._subject_cache) >= self.subject_buffer_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._subject_cache[oldest]
        
        self._subject_cache[subject_id] = ds
        self._cache_order.append(subject_id)
        
        return ds
    
    def __len__(self):
        """Get total samples (lazy computation)."""
        if self._total_size is None:
            # Estimate or compute total size
            # For efficiency, we can estimate or load metadata only
            self._total_size = 0
            for subject_id in self.subject_ids:
                # Load briefly to get size
                ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
                self._subject_sizes[subject_id] = len(ds)
                self._total_size += len(ds)
        return self._total_size
    
    def __getitem__(self, idx):
        """
        Get item by global index.
        
        Note: For streaming training, we recommend using StreamingDataLoader
        instead of standard DataLoader with this Dataset.
        """
        # Find which subject this index belongs to
        cumulative = 0
        for subject_id in self.subject_ids:
            size = self._subject_sizes.get(subject_id)
            if size is None:
                ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
                size = len(ds)
                self._subject_sizes[subject_id] = size
            
            if idx < cumulative + size:
                local_idx = idx - cumulative
                ds = self._get_subject_data(subject_id)
                return ds[local_idx]
            cumulative += size
        
        raise IndexError(f"Index {idx} out of range")


class StreamingDataLoader:
    """
    Memory-efficient data loader that streams subjects.
    
    Instead of loading all data into memory, it iterates through subjects
    one by one, yielding batches.
    """
    
    def __init__(self, subject_ids: List[int], dataset_cls, dataset_info: Dict,
                 batch_size: int = 32, shuffle: bool = True, device: str = 'cpu'):
        self.subject_ids = subject_ids
        self.dataset_cls = dataset_cls
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    
    def __iter__(self):
        """Iterate through all subjects, yielding batches."""
        subject_order = self.subject_ids.copy()
        if self.shuffle:
            np.random.shuffle(subject_order)
        
        for subject_id in subject_order:
            # Load one subject at a time
            ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
            
            # Create batches from this subject
            indices = list(range(len(ds)))
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(ds), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Load batch
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data, label = ds[idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                yield torch.stack(batch_data), torch.stack(batch_labels)
            
            # Delete subject data to free memory
            del ds
            import gc
            gc.collect()
    
    def __len__(self):
        """Estimate number of batches."""
        # This is approximate
        total_samples = 0
        for subject_id in self.subject_ids:
            ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
            total_samples += len(ds)
            del ds
        return (total_samples + self.batch_size - 1) // self.batch_size


class StreamingLOSOTrainer(BaseTrainer):
    """
    Memory-efficient LOSO trainer for large-scale datasets.
    
    Key features:
    - Streams training data subject-by-subject (constant memory)
    - Gradient accumulation for effective large batch training
    - Supports mixed precision training for speed
    - Handles class imbalance across subjects
    
    Example:
        >>> trainer = StreamingLOSOTrainer(
        ...     model=model,
        ...     config=config,
        ...     device=device,
        ...     paths=paths,
        ...     logger=logger,
        ...     gradient_accumulation_steps=4  # Simulate 4x larger batch
        ... )
    """
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # Streaming configuration
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 
                                                      self.config.get('trainer.args.gradient_accumulation_steps', 1))
        self.subject_buffer_size = kwargs.get('subject_buffer_size',
                                              self.config.get('trainer.args.subject_buffer_size', 1))
        self.mixed_precision = kwargs.get('mixed_precision',
                                          self.config.get('trainer.args.mixed_precision', False))
        
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
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None

        # Clinical source-subject selection configuration
        self.source_selection = self.config.get('trainer.args.source_selection', None)
        self._clinical_metadata = None
        
        # Incremental results saving / resume configuration
        self._incremental_results_path = self.paths.results_dir / "incremental_results.json"
        self._resume_enabled = self.config.get('trainer.args.resume', True)

    def _select_train_subjects(self, test_subject_id: int, all_subjects: List[int]) -> List[int]:
        """Select training subjects for a LOSO round."""
        if not self.source_selection or not self.source_selection.get('enabled', False):
            return [s for s in all_subjects if s != test_subject_id]

        k = self.source_selection.get('k', 10)
        require_same_location = self.source_selection.get('require_same_location', False)
        location_weight = self.source_selection.get('location_weight', 1.0)
        duration_weight = self.source_selection.get('duration_weight', 1.0)
        nihss_weight = self.source_selection.get('nihss_weight', 1.0)

        candidate_sources = [s for s in all_subjects if s != test_subject_id]
        selected = select_source_subjects(
            target_subject_id=test_subject_id,
            candidate_source_ids=candidate_sources,
            metadata_df=self._clinical_metadata,
            k=k,
            require_same_location=require_same_location,
            location_weight=location_weight,
            duration_weight=duration_weight,
            nihss_weight=nihss_weight
        )
        self._log(f"  Selected {len(selected)} source subjects for target {test_subject_id}: {selected}")
        return selected

    def _load_incremental_results(self) -> Optional[Dict[str, Any]]:
        """Load previously saved incremental results if available."""
        if not self._incremental_results_path.exists():
            return None
        try:
            with open(self._incremental_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._log(f"  Found incremental results: {len(data.get('subjects', []))} subjects completed")
            return data
        except Exception as e:
            self._log(f"  Warning: Failed to load incremental results: {e}", level="warning")
            return None
    
    def _save_incremental_results(self, all_results: List[Dict], all_histories: List[Dict],
                                   group_name: Optional[str] = None):
        """Save incremental results after each subject to enable resume."""
        try:
            data = {
                'subjects': all_results,
                'histories': all_histories,
                'group': group_name,
                'num_completed': len(all_results),
                'config': {
                    'model_type': self.model_type,
                    'num_subjects_expected': self.config.get('data.subjects', []),
                }
            }
            # Atomic write to avoid corruption
            tmp_path = self._incremental_results_path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self._convert_for_json(data), f, indent=2)
            tmp_path.replace(self._incremental_results_path)
        except Exception as e:
            self._log(f"  Warning: Failed to save incremental results: {e}", level="warning")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Recursively convert numpy/torch types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        return obj
    
    def _run_loso_on_subjects_streaming(self, subjects: List[int], dataset_info: Dict,
                                        dataset_cls, group_name: Optional[str] = None):
        """
        Run streaming LOSO on a given list of subjects.
        
        Args:
            subjects: List of subject IDs
            dataset_info: Dataset information
            dataset_cls: Dataset class for loading
            group_name: Optional group name for logging
            
        Returns:
            Tuple of (all_results, all_histories)
        """
        all_results = []
        all_histories = []
        prefix = f"[{group_name}] " if group_name else ""
        
        # Resume from incremental results if enabled
        remaining_subjects = list(subjects)
        if self._resume_enabled:
            incremental = self._load_incremental_results()
            if incremental is not None:
                completed_ids = {r['test_subject_id'] for r in incremental.get('subjects', [])}
                # Only resume if the group matches and subjects are consistent
                if incremental.get('group') == group_name:
                    all_results = incremental.get('subjects', [])
                    all_histories = incremental.get('histories', [])
                    remaining_subjects = [s for s in subjects if s not in completed_ids]
                    if remaining_subjects:
                        self._log(f"  Resuming LOSO: {len(completed_ids)} subjects already done, "
                                  f"{len(remaining_subjects)} remaining")
                    else:
                        self._log(f"  All {len(completed_ids)} subjects already completed; skipping training")
                    
        for test_subject_idx, test_subject_id in enumerate(remaining_subjects):
            self._log(f"\n{prefix}[{test_subject_idx + 1}/{len(subjects)}] Test Subject: {test_subject_id}")
            self._log("-" * 40)
            
            train_subject_ids = self._select_train_subjects(test_subject_id, subjects)

            subject_results, subject_history = self._train_loso_round_streaming(
                test_subject_id, train_subject_ids, dataset_cls, dataset_info
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
            self._plot_history(subject_history, subject_viz_dir, test_subject_id)
            
            # Save incremental results after each subject
            self._save_incremental_results(all_results, all_histories, group_name)
            
            import gc
            gc.collect()
        
        return all_results, all_histories
    
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run streaming LOSO cross-validation training.
        
        Supports standard LOSO across all subjects, or stratified LOSO
        where subjects are grouped by a clinical field.
        
        Args:
            datasets: Dictionary containing:
                - subjects: List of subject IDs
                - info: Dataset information
                - dataset_cls: Dataset class for loading
                - subject_groups: Optional dict mapping group names to subject lists
                
        Returns:
            Dictionary containing LOSO results
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')
        subject_groups = datasets.get('subject_groups')

        # Lazy-load clinical metadata if source selection is enabled
        if self.source_selection and self.source_selection.get('enabled', False) and self._clinical_metadata is None:
            metadata_path = self.source_selection.get('metadata_path')
            if not metadata_path:
                # Fallback to participants_tsv from dataset_info (user must provide a stratified CSV)
                metadata_path = dataset_info.get('dataset', {}).get('participants_tsv')
            if metadata_path:
                self._clinical_metadata = load_clinical_metadata(metadata_path)
                self._log(f"Clinical source selection enabled (k={self.source_selection.get('k', 10)})")
            else:
                self._log("Warning: source_selection enabled but no metadata_path found; disabling.", level="warning")
                self.source_selection = None
        
        if not dataset_cls:
            raise ValueError("StreamingLOSOTrainer requires 'dataset_cls'")
        
        if subject_groups:
            all_results = []
            all_histories = []
            group_stats = {}
            
            self._log("=" * 60)
            self._log("Starting Stratified Streaming LOSO Training")
            self._log(f"Stratification groups: {list(subject_groups.keys())}")
            self._log("=" * 60)
            
            for group_name, group_subjects in subject_groups.items():
                self._log(f"\n{'='*60}")
                self._log(f"Group: {group_name} ({len(group_subjects)} subjects)")
                self._log(f"{'='*60}")
                
                group_results, group_histories = self._run_loso_on_subjects_streaming(
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
            
            self._plot_comparison(all_results, group_stats=group_stats)
            
            test_accuracies = [r['test_acc'] for r in all_results]
            final_results = {
                'subjects': all_results,
                'group_stats': group_stats,
                'overall_mean': float(np.mean(test_accuracies)),
                'overall_std': float(np.std(test_accuracies)),
                'overall_min': float(np.min(test_accuracies)),
                'overall_max': float(np.max(test_accuracies)),
                'streaming_config': {
                    'num_subjects': len(subjects),
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                    'subject_buffer_size': self.subject_buffer_size,
                    'mixed_precision': self.mixed_precision,
                    'stratified': True,
                    'stratify_groups': list(subject_groups.keys()),
                }
            }
            
            self._log("\n" + "=" * 60)
            self._log("Stratified Streaming LOSO Training Complete")
            for group_name, stats in group_stats.items():
                self._log(f"  {group_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
            self._log(f"  Overall: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
            self._log("=" * 60)
            
            self._save_histories_if_needed(all_histories)
            return final_results
        
        else:
            all_results = []
            all_histories = []
            
            self._log("=" * 60)
            self._log("Starting Streaming LOSO Training")
            self._log(f"Total subjects: {len(subjects)}")
            self._log(f"Memory-efficient mode: {self.subject_buffer_size} subject(s) in memory")
            self._log(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
            if self.mixed_precision:
                self._log("Mixed precision training: Enabled")
            self._log("=" * 60)
            
            all_results, all_histories = self._run_loso_on_subjects_streaming(
                subjects, dataset_info, dataset_cls
            )
            
            self._plot_comparison(all_results)
            
            test_accuracies = [r['test_acc'] for r in all_results]
            final_results = {
                'subjects': all_results,
                'overall_mean': float(np.mean(test_accuracies)),
                'overall_std': float(np.std(test_accuracies)),
                'overall_min': float(np.min(test_accuracies)),
                'overall_max': float(np.max(test_accuracies)),
                'streaming_config': {
                    'num_subjects': len(subjects),
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                    'subject_buffer_size': self.subject_buffer_size,
                    'mixed_precision': self.mixed_precision
                }
            }
            
            self._log("\n" + "=" * 60)
            self._log("Streaming LOSO Training Complete")
            self._log(f"Mean Accuracy: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
            self._log("=" * 60)
            
            self._save_histories_if_needed(all_histories)
            return final_results
    
    def _save_histories_if_needed(self, all_histories):
        """Save training histories if configured."""
        save_hist = self.config.get("trainer.args.save_training_history", False)
        if save_hist and all_histories:
            try:
                import json
                hist_path = self.paths.logs_dir / "training_histories.json"
                def _convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, "item"):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: _convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [_convert(v) for v in obj]
                    return obj
                with open(hist_path, "w", encoding="utf-8") as f:
                    json.dump(_convert(all_histories), f, indent=2)
                self._log(f"Training histories saved to: {hist_path}")
            except Exception as e:
                self._log(f"Warning: Failed to save training histories: {e}", level="warning")
    
    def _build_cached_train_loader(self, train_subject_ids: List[int],
                                    dataset_cls, dataset_info: Dict):
        """
        Preload all training subjects into memory and build an efficient DataLoader.
        
        This eliminates the massive redundant preprocessing (loadmat -> resample -> 
        filter bank -> windowing) that occurs on every epoch when streaming.
        """
        import time
        start = time.time()
        
        all_data = []
        all_labels = []
        augmentor = None
        
        for i, subject_id in enumerate(train_subject_ids):
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            all_data.append(ds.data)
            all_labels.append(ds.labels)
            # Capture augmentor from first subject (same config for all)
            if i == 0 and hasattr(ds, 'augmentor'):
                augmentor = ds.augmentor
        
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Inner class to keep augmentation support while being in-memory
        class _InMemoryDataset(Dataset):
            def __init__(self, data, labels, augmentor=None):
                self.data = data
                self.labels = labels
                self.augmentor = augmentor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                data = self.data[idx]
                label = self.labels[idx]
                if self.augmentor is not None:
                    data = self.augmentor.process(data)
                return torch.from_numpy(data).float(), torch.tensor(label).long()
        
        train_dataset = _InMemoryDataset(all_data, all_labels, augmentor)
        
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': False,
        }
        
        # Enable GPU-optimized data loading when CUDA is available
        if self.device.type == 'cuda':
            loader_kwargs['pin_memory'] = self.config.get('training.pin_memory', True)
            num_workers = self.config.get('training.num_workers', 0)
            if num_workers > 0:
                loader_kwargs['num_workers'] = num_workers
                loader_kwargs['persistent_workers'] = True
                loader_kwargs['prefetch_factor'] = self.config.get('training.prefetch_factor', 2)
        
        train_loader = DataLoader(train_dataset, **loader_kwargs)
        
        elapsed = time.time() - start
        self._log(f"  Preloaded {len(train_subject_ids)} subjects ({len(train_dataset)} samples) in {elapsed:.1f}s")
        
        return train_loader
    
    def _train_epoch_cached(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                            train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch using a cached in-memory DataLoader.
        """
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        accumulation_counter = 0
        use_acc = self.gradient_accumulation_steps > 1
        
        for inputs, labels in train_loader:
            # non_blocking transfer for faster GPU loading with pin_memory
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    if use_acc:
                        loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                if use_acc:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if use_acc:
                accumulation_counter += 1
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            else:
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Statistics (undo accumulation division for logging)
            total_loss += loss.item() * inputs.size(0) * (self.gradient_accumulation_steps if use_acc else 1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Handle remaining gradients
        if use_acc and accumulation_counter % self.gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _train_loso_round_streaming(self, test_subject_id: int,
                                    train_subject_ids: List[int],
                                    dataset_cls, dataset_info: Dict,
                                    val_subject_id: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Train one LOSO round with streaming data loading.
        
        Implements validation-subject early stopping for rigorous cross-subject evaluation:
        - Test subject: completely held out, never seen during training or model selection
        - Validation subject: randomly selected from train subjects, used for early stopping
        - Train subjects: remaining N-2 subjects used for actual training
        
        Args:
            val_subject_id: If provided, use this specific subject as validation instead of
                           randomly selecting one. Useful for validation-based pipeline selection.
        """
        # Early stopping config
        es_patience = self.config.get('trainer.args.early_stopping_patience', 15)
        use_val_es = self.config.get('trainer.args.use_validation_early_stopping', False)
        
        # Split train subjects into train/val if enabled and feasible
        train_subject_ids_inner = train_subject_ids.copy()
        val_loader = None
        
        if val_subject_id is not None:
            # Explicit validation subject (e.g., for pipeline selection)
            if val_subject_id in train_subject_ids:
                train_subject_ids_inner = [s for s in train_subject_ids if s != val_subject_id]
                val_dataset = dataset_cls(subject_id=val_subject_id, dataset_info=dataset_info)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                self._log(f"  Val subject (specified): {val_subject_id} (ES patience={es_patience})")
            else:
                self._log(f"  Warning: specified val_subject {val_subject_id} not in train subjects; ignoring", level="warning")
                val_subject_id = None
        elif use_val_es and len(train_subject_ids) >= 2:
            # Deterministic random selection based on test_subject_id + seed
            import random
            seed = self.config.get('experiment.seed', 42)
            rng = random.Random(test_subject_id + seed)
            val_subject_id = rng.choice(train_subject_ids)
            train_subject_ids_inner = [s for s in train_subject_ids if s != val_subject_id]
            
            # Build validation loader
            val_dataset = dataset_cls(subject_id=val_subject_id, dataset_info=dataset_info)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            self._log(f"  Val subject: {val_subject_id} (ES patience={es_patience})")
        else:
            self._log(f"  No val split (train={len(train_subject_ids)} subjects)")
        
        # Create model
        sample_ds = dataset_cls(subject_id=train_subject_ids_inner[0], dataset_info=dataset_info)
        model = self._create_model(sample_ds)
        del sample_ds
        
        # Create test dataset
        test_dataset = dataset_cls(subject_id=test_subject_id, dataset_info=dataset_info)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': []
        }
        
        # Best tracking (val-based for model selection, test for reporting only)
        best_val_acc = -1.0
        best_val_epoch = 0
        best_val_state = None
        epochs_no_improve = 0
        stopped_early = False
        
        # Preload training data
        train_loader = self._build_cached_train_loader(
            train_subject_ids_inner, dataset_cls, dataset_info
        )
        
        # Progress bar
        desc = f'Sub{test_subject_id}'
        if val_subject_id:
            desc += f'(Val{val_subject_id})'
        epoch_bar = tqdm(range(self.epochs), desc=desc,
                        bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} {postfix}')
        
        for epoch in epoch_bar:
            self.current_epoch = epoch
            
            # Update learning rate
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train
            train_loss, train_acc = self._train_epoch_cached(model, optimizer, train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation (for early stopping & model selection)
            if val_loader is not None:
                val_loss, val_acc = self._validate(model, val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss, val_acc = float('nan'), float('nan')
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # Test (for monitoring only, NEVER used for model selection)
            test_loss, test_acc = self._validate(model, test_loader)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Postfix
            postfix = f"LR:{lr:.1e} Tr:{train_loss:.3f}/{train_acc:.3f}"
            if val_loader:
                postfix += f" Va:{val_loss:.3f}/{val_acc:.3f}"
            postfix += f" Te:{test_loss:.3f}/{test_acc:.3f}"
            epoch_bar.set_postfix_str(postfix)
            
            # Model selection based on VALIDATION accuracy
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    epochs_no_improve = 0
                    # Save best model state in memory
                    best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                
                # Early stopping check
                if epochs_no_improve >= es_patience:
                    self._log(f"  Early stopping @ epoch {epoch} (best val @ {best_val_epoch})")
                    stopped_early = True
                    break
            else:
                # Fallback: no val, no early stopping
                pass
            
            self.global_step += 1
        
        epoch_bar.close()
        
        # Final evaluation: load best val model and evaluate on test
        if val_loader is not None and best_val_state is not None:
            model.load_state_dict(best_val_state)
            self._log(f"  Loaded best val model (epoch {best_val_epoch}, val_acc={best_val_acc:.4f})")
        
        final_loss, final_acc = self._validate(model, test_loader)
        
        # Also compute final metrics on val for reporting
        final_val_loss, final_val_acc = float('nan'), float('nan')
        if val_loader is not None:
            final_val_loss, final_val_acc = self._validate(model, val_loader)
        
        return {
            'test_subject_id': test_subject_id,
            'test_acc': final_acc,           # ← best-val-model on TEST (rigorous)
            'test_loss': final_loss,
            'val_acc': final_val_acc,        # ← best-val-model on VAL
            'val_loss': final_val_loss,
            'best_val_acc': best_val_acc,    # ← peak val acc during training
            'best_val_epoch': best_val_epoch,
            'stopped_early': stopped_early,
            'actual_epochs': epoch + 1 if stopped_early else self.epochs,
            'train_subjects': len(train_subject_ids_inner),
            'val_subject': val_subject_id,
            'test_samples': len(test_dataset)
        }, history
    
    def _train_epoch_streaming(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                               train_subject_ids: List[int], dataset_cls, dataset_info: Dict
                               ) -> Tuple[float, float]:
        """
        Train one epoch by streaming through subjects.
        """
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle subjects each epoch
        subject_order = train_subject_ids.copy()
        np.random.shuffle(subject_order)
        
        # Gradient accumulation counter
        accumulation_counter = 0
        
        for subject_id in subject_order:
            # Load subject
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            
            # Create batches
            indices = list(range(len(ds)))
            np.random.shuffle(indices)
            
            for i in range(0, len(ds), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Load batch
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data, label = ds[idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                inputs = torch.stack(batch_data).to(self.device)
                labels = torch.stack(batch_labels).to(self.device)
                
                # Forward
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = self.criterion(outputs, labels)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                
                # Accumulation
                accumulation_counter += 1
                
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Statistics
                total_loss += loss.item() * inputs.size(0) * self.gradient_accumulation_steps
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            # Free memory
            del ds
            import gc
            gc.collect()
        
        # Handle remaining gradients
        if accumulation_counter % self.gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _create_model(self, sample_dataset) -> nn.Module:
        """Create model from sample dataset."""
        from core.registry import MODELS, build_from_config
        
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        # Auto-populate
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
        """Cosine annealing learning rate."""
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.learning_rate / 2
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, model: nn.Module, test_subject_id: int, 
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
            'test_subject_id': test_subject_id,
            'model_state_dict': state_dict,
            'metric': metric,
            'compressed': self.use_fp16_compression
        }
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=test_subject_id, fold=0, metric=metric
        )
        checkpoint_path = checkpoint_path.with_name(
            f"streaming_loso_subject{test_subject_id}_acc{metric:.4f}.pt"
        )
        
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            pass
    
    def _plot_history(self, history: Dict, save_dir: Path, test_subject_id: int):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['test_loss'], 'r-', label='Test')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Subject {test_subject_id} Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, history['test_acc'], 'r-', label='Test')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Subject {test_subject_id} Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'streaming_loso_history.png', dpi=150)
        plt.close()
    
    def _plot_comparison(self, results: List[Dict], group_stats: Optional[Dict] = None):
        """Plot comparison across subjects."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        subject_ids = [r['test_subject_id'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        fig, ax = plt.subplots(figsize=(max(8, len(subject_ids) * 0.4), 6))
        
        if group_stats:
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
            
            bars = ax.bar(range(len(subject_ids)), test_accs, color=colors,
                         edgecolor='black', alpha=0.7)
            
            legend_handles = [Patch(facecolor=group_color_map[g], edgecolor='black', label=g)
                              for g in group_names]
            ax.legend(handles=legend_handles, title='Group', loc='upper right')
        else:
            colors = ['green' if acc >= 0.8 else 'yellow' if acc >= 0.6 else 'red' 
                      for acc in test_accs]
            
            bars = ax.bar(range(len(subject_ids)), test_accs, color=colors, 
                         edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Test Subject ID')
        ax.set_ylabel('Test Accuracy')
        title = 'Stratified Streaming LOSO Results' if group_stats else 'Streaming LOSO Cross-Validation Results'
        ax.set_title(title)
        ax.set_xticks(range(len(subject_ids)))
        ax.set_xticklabels([f'Sub{s}' for s in subject_ids], rotation=45)
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis='y', alpha=0.3)
        
        mean_acc = np.mean(test_accs)
        ax.axhline(y=mean_acc, color='r', linestyle='--', 
                  label=f'Mean: {mean_acc:.3f}')
        if not group_stats:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.paths.viz_dir / 'streaming_loso_comparison.png', dpi=150)
        plt.close()
