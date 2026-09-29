"""
Abstract base class for datasets.

All dataset implementations should inherit from this class and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all EEG datasets.
    
    Provides common functionality for loading and preprocessing EEG data.
    Specific datasets should implement the _load_raw_data method.
    
    Example:
        >>> class MyDataset(BaseDataset):
        ...     def _load_raw_data(self):
        ...         # Load data from files
        ...         return data, labels
    """
    
    def __init__(self, subject_id: int, dataset_info: Dict[str, Any], 
                 transform: Optional[Any] = None, mode: str = "train"):
        """
        Initialize the dataset.
        
        Args:
            subject_id: Subject identifier
            dataset_info: Dataset configuration dictionary
            transform: Optional transform to apply to samples
            mode: 'train', 'val', or 'test'
        """
        self.subject_id = subject_id
        self.dataset_info = dataset_info
        self.transform = transform
        self.mode = mode
        
        # Extract configuration
        self.data_dir = dataset_info.get('dataset', {}).get('data_dir', '')
        self.original_sr = dataset_info.get('dataset', {}).get('original_sr', 250)
        self.target_sr = dataset_info.get('preprocessing', {}).get('resample', {}).get('target_sr', 250)
        
        # Channel selection
        self.channels_selected = dataset_info.get('preprocessing', {}).get('channel_selection', {}).get('channels_to_select', [])
        
        # Filter banks
        self.filter_banks = dataset_info.get('preprocessing', {}).get('filter_bank', {}).get('filter_banks', [])
        self.n_bands = len(self.filter_banks)
        
        # Load and process data
        self.data, self.labels, self.group_ids = self._load_and_process_data()
        
        # Data augmentation (sample-level, applied in __getitem__ for train mode)
        self.augmentor = None
        augmentation_config = dataset_info.get('preprocessing', {}).get('augmentation', [])
        if augmentation_config:
            from preprocessing.factory import ProcessorFactory
            # Filter out zero-probability methods to avoid unnecessary overhead
            active_methods = [m for m in augmentation_config if m.get('probability', 0) > 0]
            if active_methods:
                self.augmentor = ProcessorFactory.create_processor(
                    name="sample_augmentation",
                    processor_type="augmentation",
                    methods=active_methods
                )
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label) as tensors
        """
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.mode == "train" and self.augmentor is not None:
            data = self.augmentor.process(data)
        
        if self.transform:
            data = self.transform(data)
        
        return torch.from_numpy(data).float(), torch.tensor(label).long()
    
    def _load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the data.
        
        This method orchestrates the preprocessing pipeline:
        1. Load raw data
        2. Resample
        3. Channel selection
        4. Filter bank processing
        5. Sliding window segmentation
        
        Returns:
            Tuple of (data, labels, group_ids)
        """
        # Load raw data (implemented by subclasses)
        raw_data, raw_labels = self._load_raw_data()
        
        # Ensure numpy arrays
        raw_data = np.array(raw_data)
        raw_labels = np.array(raw_labels)
        
        # Remove singleton dimensions
        if raw_data.shape[0] == 1:
            raw_data = raw_data[0]
        if raw_labels.shape[0] == 1:
            raw_labels = raw_labels[0]
        
        # Resample
        from preprocessing.resample import ResampleProcessor
        resampler = ResampleProcessor(target_sfreq=self.target_sr, original_sfreq=self.original_sr)
        resampled_data = resampler.process(raw_data)
        
        # Artifact removal pipeline (optional)
        # Configured under dataset_info['preprocessing']['artifact_removal_pipeline']
        artifact_pipeline_config = self.dataset_info.get('preprocessing', {}).get('artifact_removal_pipeline')
        if artifact_pipeline_config:
            import copy
            from preprocessing.factory import ProcessorFactory
            
            # Deep copy to avoid mutating the original config dict
            artifact_pipeline_config = copy.deepcopy(artifact_pipeline_config)
            
            # Auto-inject EOG channel names from dataset.EOG_channels if available
            eog_channel_indices = self.dataset_info.get('dataset', {}).get('EOG_channels')
            if eog_channel_indices is not None:
                n_channels = resampled_data.shape[-2]
                if n_channels == 32:
                    from preprocessing.artifact_removal.base import BaseArtifactRemovalProcessor
                    ch_names = list(BaseArtifactRemovalProcessor.DEFAULT_XWSTROKE_CH_NAMES)
                else:
                    ch_names = [f'EEG{i}' for i in range(n_channels)]
                eog_ch_names = [ch_names[i] for i in eog_channel_indices if 0 <= i < len(ch_names)]
                
                for proc_config in artifact_pipeline_config:
                    if 'eog_channels' not in proc_config:
                        proc_config['eog_channels'] = eog_ch_names
            
            artifact_pipeline = ProcessorFactory.create_pipeline_from_config(artifact_pipeline_config)
            resampled_data = artifact_pipeline.process(resampled_data)
        
        # Channel selection
        selected_data = resampled_data[:, self.channels_selected, :]
        
        # Filter bank processing
        from preprocessing.filter_banks import FilterBankProcessor
        filter_bank = FilterBankProcessor(filter_banks=self.filter_banks, sample_rate=self.target_sr)
        filtered_data = filter_bank(selected_data)
        
        # Euclidean Alignment (optional)
        # Recommended placement: after temporal filtering (filter bank),
        # before sliding window segmentation / model input.
        ea_config = self.dataset_info.get('preprocessing', {}).get('euclidean_alignment')
        if ea_config is not None and ea_config.get('enabled', False):
            from preprocessing.euclidean_alignment import EuclideanAlignmentProcessor
            eps = ea_config.get('eps', 1e-10)
            ea_proc = EuclideanAlignmentProcessor(name="ea", eps=eps)
            filtered_data = ea_proc.process(filtered_data)
        
        # Adjust labels if needed (convert from 1-based to 0-based indexing)
        labels = raw_labels
        if len(labels) > 0 and np.min(labels) >= 1:
            labels = labels - 1
        
        # Optional dataset-specific label remapping
        labels = self._remap_labels(labels)
        
        # Sliding window segmentation
        from preprocessing.slide_windows import SlidingWindowSegmenter
        window_len = int(self.dataset_info.get('preprocessing', {}).get('windowing', {}).get('window_length_sec', 3) * self.target_sr)
        window_stride = int(self.dataset_info.get('preprocessing', {}).get('windowing', {}).get('window_stride_sec', 1) * self.target_sr)
        
        segmenter = SlidingWindowSegmenter(window_len, window_stride)
        final_data, final_labels, group_ids = segmenter(filtered_data, labels=labels)
        
        # Log final processed data shape for verification (debug level to avoid cluttering progress bars)
        import logging
        logging.debug(f"[Subject {self.subject_id}] Data loaded: shape={final_data.shape}, n_classes={len(np.unique(final_labels))}, n_samples={len(final_labels)}")
        
        return final_data, final_labels, group_ids
    
    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Optional label remapping hook.
        
        Override in subclasses to implement dataset-specific label mapping
        (e.g., left/right -> affected/unaffected for stroke datasets).
        
        Args:
            labels: Array of labels (0-based indexing)
            
        Returns:
            Remapped labels
        """
        return labels
    
    @abstractmethod
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from files.
        
        This method must be implemented by subclasses to load
        dataset-specific data from files.
        
        Returns:
            Tuple of (data, labels) where:
            - data: Array of shape (n_trials, n_channels, n_timepoints) or (n_channels, n_timepoints)
            - labels: Array of shape (n_trials,) or scalar
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'subject_id': self.subject_id,
            'n_samples': len(self),
            'data_shape': self.data.shape if len(self.data) > 0 else None,
            'n_classes': len(np.unique(self.labels)) if len(self.labels) > 0 else 0,
            'sample_rate': self.target_sr,
        }
