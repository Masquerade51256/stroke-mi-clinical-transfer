"""
Data module for EEG-BCI experiments.
"""

from core.registry import DATASETS
from core.base.base_dataset import BaseDataset

# Import and register datasets
try:
    from .datasets import BCICIV2aDataset, XWStrokeDataset, XWStrokeEDFDataset, LowerStrokeDataset, FiveFingerDataset, HSStrokeDataset, SHUDataset

    # Register datasets
    DATASETS.register('BCICIV2a')(BCICIV2aDataset)
    DATASETS.register('BCICIV2b')(BCICIV2aDataset)  # Same structure as 2a
    DATASETS.register('XWStroke')(XWStrokeDataset)
    DATASETS.register('XWStrokeEDF')(XWStrokeEDFDataset)
    DATASETS.register('LowerStroke')(LowerStrokeDataset)
    DATASETS.register('5F')(FiveFingerDataset)
    DATASETS.register('FiveFinger')(FiveFingerDataset)
    DATASETS.register('HSStroke')(HSStrokeDataset)
    DATASETS.register('SHU')(SHUDataset)
    
except ImportError as e:
    print(f"Warning: Could not register datasets: {e}")

__all__ = [
    'DATASETS',
    'BaseDataset',
    'BCICIV2aDataset',
    'XWStrokeDataset',
    'XWStrokeEDFDataset',
    'LowerStrokeDataset',
    'FiveFingerDataset',
    'HSStrokeDataset',
    'SHUDataset',
]
