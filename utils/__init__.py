"""
Utility functions for the EEG-BCI experiment framework.
"""

import yaml

from .path_manager import PathManager
from .logging import setup_logger
from .visualization import plot_training_history, plot_subject_comparison


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


__all__ = [
    'PathManager',
    'setup_logger',
    'plot_training_history',
    'plot_subject_comparison',
    'load_config',
]
