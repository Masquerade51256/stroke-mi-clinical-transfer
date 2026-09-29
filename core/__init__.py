"""
Core module for EEG-BCI Experiment Framework.

This module provides the foundational components for building and running
standardized EEG-based Brain-Computer Interface experiments.
"""

from .registry import Registry, DATASETS, MODELS, TRAINERS
from .config import Config
from .experiment import ExperimentManager

__all__ = [
    'Registry',
    'DATASETS', 
    'MODELS',
    'TRAINERS',
    'Config',
    'ExperimentManager',
]
