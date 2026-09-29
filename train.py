#!/usr/bin/env python3
"""
Unified training script for EEG-BCI experiments.

This is the main entry point for running experiments.
For backward compatibility, it supports both old-style constant_value.py
configuration and new-style YAML configuration.

Usage:
    # Run with default configuration
    python train.py
    
    # Run with specific config file
    python train.py --config configs/experiment/my_experiment.yaml
    
    # Run with specific dataset and model
    python train.py --dataset XWStroke --model EEGNet
    
    # List available options
    python train.py --list-datasets
    python train.py --list-models

Note:
    The old constant_value.py configuration is deprecated.
    Please migrate to YAML configuration files.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import new framework components
from experiments.run_experiment import main, parse_args
from core import Config

# Import and register all components
from models import register_all_models
from data import DATASETS
from trainers import TRAINERS

if __name__ == '__main__':
    # Register models before running
    register_all_models()
    
    # Run the main entry point
    main()
