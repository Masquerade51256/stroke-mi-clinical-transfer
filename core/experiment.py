"""
Experiment management for standardized EEG-BCI experiments.

This module provides the ExperimentManager class that orchestrates
the entire experiment lifecycle: setup, training, evaluation, and logging.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import torch
import numpy as np

from .config import Config
from .registry import DATASETS, MODELS, TRAINERS
from utils.path_manager import PathManager
from utils.logging import setup_logger


class ExperimentManager:
    """
    Manages the complete lifecycle of an experiment.
    
    This class handles:
    - Experiment initialization and configuration
    - Directory structure setup
    - Logging setup
    - Model and dataset instantiation
    - Training orchestration
    - Result saving and artifact management
    
    Example:
        >>> config = Config.fromfile('configs/experiment.yaml')
        >>> exp = ExperimentManager(config, exp_name='my_experiment')
        >>> exp.setup()
        >>> exp.run()
    """
    
    def __init__(self, config: Config, exp_name: Optional[str] = None, 
                 resume_from: Optional[str] = None):
        """
        Initialize the experiment manager.
        
        Args:
            config: Experiment configuration
            exp_name: Experiment name (for directory naming). If None, auto-generated.
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.resume_from = resume_from
        
        # Generate experiment name if not provided
        if exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = config.get('data.dataset', 'unknown')
            model_name = config.get('model.type', 'unknown')
            exp_name = f"{dataset_name}_{model_name}_{timestamp}"
        self.exp_name = exp_name
        
        # Initialize path manager
        self.paths = PathManager(exp_name, config.get('paths.root_dir', './experiments'))
        
        # State variables
        self.logger: Optional[logging.Logger] = None
        self.model: Optional[torch.nn.Module] = None
        self.datasets: Dict[str, Any] = {}
        self.trainer: Optional[Any] = None
        self.results: Dict[str, Any] = {}
        
        # Device configuration
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup and return the computation device."""
        device_str = self.config.get('training.device', 'cuda')
        if device_str == 'cuda' and not torch.cuda.is_available():
            self._warn("CUDA not available, falling back to CPU")
            device_str = 'cpu'
        
        device = torch.device(device_str)
        return device
    
    def setup(self) -> None:
        """
        Setup the complete experiment environment.
        
        This includes:
        - Creating directory structure
        - Setting up logging
        - Saving configuration
        - Setting random seeds
        """
        # Create directories
        self.paths.create_directories()
        
        # Setup logging
        log_level = self.config.get('logging.level', 'INFO')
        self.logger = setup_logger(
            name=self.exp_name,
            log_file=self.paths.log_file,
            level=log_level,
            console=self.config.get('logging.console', True)
        )
        
        self._info(f"Experiment: {self.exp_name}")
        self._info(f"Device: {self.device}")
        self._info(f"Output directory: {self.paths.exp_dir}")
        
        # Save configuration
        self._save_config()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Log configuration summary
        self._log_config_summary()
        
    def _save_config(self) -> None:
        """Save the configuration to the experiment directory."""
        config_path = self.paths.exp_dir / 'config.yaml'
        self.config.dump(config_path)
        self._info(f"Configuration saved to: {config_path}")
        
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.get('experiment.seed', 42)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CuDNN deterministic (may impact performance)
            if self.config.get('training.deterministic', False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
        self._info(f"Random seed set to: {seed}")
        
    def _log_config_summary(self) -> None:
        """Log a summary of the configuration."""
        self._info("=" * 60)
        self._info("Configuration Summary:")
        self._info("=" * 60)
        
        # Dataset info
        dataset_type = self.config.get('data.dataset', 'unknown')
        self._info(f"Dataset: {dataset_type}")
        
        # Model info
        model_type = self.config.get('model.type', 'unknown')
        self._info(f"Model: {model_type}")
        
        # Training info
        epochs = self.config.get('training.epochs', 'unknown')
        batch_size = self.config.get('training.batch_size', 'unknown')
        lr = self.config.get('training.optimizer.lr', 'unknown')
        self._info(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
        self._info("=" * 60)
        
    def build_dataset(self) -> None:
        """
        Build dataset configuration without loading data.
        
        Data is loaded on-demand during training to minimize memory usage.
        """
        dataset_config = self.config.get('data', {})
        dataset_type = dataset_config.get('dataset')
        
        if dataset_type is None:
            raise ValueError("Dataset type not specified in config (data.dataset)")
        
        self._info(f"Building dataset: {dataset_type}")
        
        # Get dataset class from registry
        try:
            dataset_cls = DATASETS.get(dataset_type)
        except KeyError:
            available = DATASETS.list_keys()
            raise ValueError(f"Unknown dataset '{dataset_type}'. Available: {available}")
        
        # Load dataset info/config
        dataset_info_path = dataset_config.get('info_path')
        if dataset_info_path and os.path.exists(dataset_info_path):
            from .config import Config
            dataset_info = Config.fromfile(dataset_info_path).to_dict()
        else:
            dataset_info = dataset_config.get('info', {})
        
        # Get target subjects
        target_subjects = dataset_config.get('subjects', 'all')
        if target_subjects == 'all':
            target_subjects = dataset_info.get('dataset', {}).get('subjects', [1])
        
        # Store configuration for lazy loading
        self.datasets = {
            'subjects': target_subjects,
            'info': dataset_info,
            'dataset_cls': dataset_cls,
            # instances will be loaded on-demand during training
        }
        
        # Subject stratification: group subjects by a clinical field (e.g., ParalysisSide)
        stratify_field = dataset_config.get('stratify_field')
        if stratify_field:
            from data.datasets import load_subject_groups
            subject_groups = load_subject_groups(
                target_subjects, dataset_info, field=stratify_field
            )
            self.datasets['subject_groups'] = subject_groups
            group_summary = {k: len(v) for k, v in subject_groups.items()}
            self._info(f"Stratified by '{stratify_field}': {group_summary}")
        
        self._info(f"Configured for {len(target_subjects)} subjects (lazy loading)")
        
    def build_model(self) -> None:
        """
        Build the model based on configuration.
        """
        model_config = self.config.get('model', {})
        model_type = model_config.get('type')
        
        if model_type is None:
            raise ValueError("Model type not specified in config (model.type)")
        
        self._info(f"Building model: {model_type}")
        
        # Get model class from registry
        try:
            model_cls = MODELS.get(model_type)
        except KeyError:
            available = MODELS.list_keys()
            raise ValueError(f"Unknown model '{model_type}'. Available: {available}")
        
        # Prepare model arguments (handle None case)
        model_args = (model_config.get('args') or {}).copy()
        
        # Auto-populate from dataset info if available
        if 'datasets' in self.__dict__ and 'info' in self.datasets:
            dataset_info = self.datasets['info']
            if 'num_channels' not in model_args:
                model_args['num_channels'] = len(
                    dataset_info.get('preprocessing', {})
                    .get('channel_selection', {})
                    .get('channels_to_select', [])
                )
            if 'num_classes' not in model_args:
                model_args['num_classes'] = dataset_info.get('dataset', {}).get('num_classes', 4)
            if 'input_length' not in model_args:
                sr = dataset_info.get('preprocessing', {}).get('resample', {}).get('target_sr', 250)
                window_len = dataset_info.get('preprocessing', {}).get('windowing', {}).get('window_length_sec', 3)
                model_args['input_length'] = int(sr * window_len)
            if 'num_bands' not in model_args:
                model_args['num_bands'] = len(
                    dataset_info.get('preprocessing', {}).get('filter_bank', {}).get('filter_banks', [])
                )
        
        # Instantiate model
        self.model = model_cls(**model_args)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def build_trainer(self) -> None:
        """
        Build the trainer based on configuration.
        """
        trainer_config = self.config.get('trainer', {})
        trainer_type = trainer_config.get('type', 'SupervisedTrainer')
        
        self._info(f"Building trainer: {trainer_type}")
        
        # Get trainer class from registry
        try:
            trainer_cls = TRAINERS.get(trainer_type)
        except KeyError:
            available = TRAINERS.list_keys()
            raise ValueError(f"Unknown trainer '{trainer_type}'. Available: {available}")
        
        # Prepare trainer arguments
        trainer_args = trainer_config.get('args', {}).copy()
        trainer_args.update({
            'model': self.model,
            'config': self.config,
            'device': self.device,
            'paths': self.paths,
            'logger': self.logger
        })
        
        self.trainer = trainer_cls(**trainer_args)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        self._info("=" * 60)
        self._info("Starting Experiment")
        self._info("=" * 60)
        
        # Build components
        self.build_dataset()
        self.build_model()
        self.build_trainer()
        
        # Run training
        self.results = self.trainer.train(self.datasets)
        
        # Save results
        self._save_results()
        
        self._info("=" * 60)
        self._info("Experiment Completed")
        self._info("=" * 60)
        
        return self.results
    
    def _save_results(self) -> None:
        """Save experiment results to disk."""
        results_path = self.paths.results_dir / 'results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json.dump(self._convert_to_serializable(self.results), f, indent=2)
        self._info(f"Results saved to: {results_path}")
        
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
    
    def _info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _warn(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def _error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
