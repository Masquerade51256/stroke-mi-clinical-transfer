"""
Path management for experiments.

Provides centralized path handling for experiment outputs,
including checkpoints, logs, and visualizations.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime


class PathManager:
    """
    Manages all paths for an experiment.
    
    Creates and provides access to:
    - Experiment root directory
    - Checkpoints directory
    - Logs directory
    - Visualizations directory
    - Results directory
    
    Example:
        >>> paths = PathManager('my_experiment', './experiments')
        >>> paths.create_directories()
        >>> print(paths.checkpoints_dir)
    """
    
    def __init__(self, exp_name: str, root_dir: str = './experiments'):
        """
        Initialize the path manager.
        
        Args:
            exp_name: Name of the experiment
            root_dir: Root directory for all experiments
        """
        self.exp_name = exp_name
        self.root_dir = Path(root_dir)
        
        # Main experiment directory
        self.exp_dir = self.root_dir / exp_name
        
        # Subdirectories
        self.checkpoints_dir = self.exp_dir / 'checkpoints'
        self.logs_dir = self.exp_dir / 'logs'
        self.viz_dir = self.exp_dir / 'visualizations'
        self.results_dir = self.exp_dir / 'results'
        
        # Specific files
        self.log_file = self.logs_dir / 'training.log'
        self.config_file = self.exp_dir / 'config.yaml'
        self.results_file = self.results_dir / 'results.json'
        
    def create_directories(self) -> None:
        """Create all necessary directories."""
        for directory in [self.exp_dir, self.checkpoints_dir, self.logs_dir, 
                         self.viz_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, name: Optional[str] = None, 
                           subject_id: Optional[int] = None,
                           fold: Optional[int] = None,
                           metric: Optional[float] = None) -> Path:
        """
        Generate a checkpoint path.
        
        Args:
            name: Custom checkpoint name
            subject_id: Subject identifier
            fold: Fold number
            metric: Performance metric (e.g., accuracy)
            
        Returns:
            Path object for the checkpoint
        """
        if name:
            return self.checkpoints_dir / name
        
        parts = []
        if subject_id is not None:
            parts.append(f'subject_{subject_id}')
        if fold is not None:
            parts.append(f'fold_{fold}')
        if metric is not None:
            parts.append(f'acc_{metric:.4f}')
        
        filename = '_'.join(parts) + '.pt' if parts else 'checkpoint.pt'
        return self.checkpoints_dir / filename
    
    def get_viz_path(self, name: str) -> Path:
        """
        Get path for a visualization file.
        
        Args:
            name: Filename for the visualization
            
        Returns:
            Path object
        """
        return self.viz_dir / name
    
    def get_subject_viz_dir(self, subject_id: int) -> Path:
        """
        Get visualization directory for a specific subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Path object for the subject's visualization directory
        """
        subject_dir = self.viz_dir / f'subject_{subject_id}'
        subject_dir.mkdir(parents=True, exist_ok=True)
        return subject_dir
