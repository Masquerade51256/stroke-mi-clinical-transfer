"""
Abstract base class for trainers.

All trainer implementations should inherit from this class and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

import torch
from torch import nn

from ..config import Config


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Trainers are responsible for the training loop, optimization,
    validation, and checkpoint management.
    
    Example:
        >>> class MyTrainer(BaseTrainer):
        ...     def train(self, datasets):
        ...         # Implementation
        ...         pass
    """
    
    def __init__(self, model: nn.Module, config: Config, 
                 device: torch.device, paths: Any, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            config: Training configuration
            device: Computation device
            paths: Path manager for saving outputs
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device
        self.paths = paths
        self.logger = logger
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        
    @abstractmethod
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            datasets: Dictionary containing datasets
            
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    def _log(self, message: str, level: str = 'info') -> None:
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error')
        """
        if self.logger is None:
            print(message)
            return
            
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
    
    def save_checkpoint(self, filepath: str, **kwargs) -> None:
        """
        Save a training checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            **kwargs: Additional data to save in the checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
        }
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
        self._log(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Supports both standard checkpoints and FP16 compressed checkpoints.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        state_dict = checkpoint['model_state_dict']
        
        # 处理 FP16 压缩的 checkpoint：转换回 float32
        if checkpoint.get('compressed', False):
            state_dict = {k: v.float() if v.dtype == torch.float16 else v 
                         for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        
        self._log(f"Checkpoint loaded: {filepath}")
        return checkpoint
