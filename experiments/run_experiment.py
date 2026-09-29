#!/usr/bin/env python3
"""
Main entry point for running EEG-BCI experiments.

This script provides a unified interface for running experiments
with different configurations, datasets, and models.

Usage:
    # Run with default config
    python experiments/run_experiment.py
    
    # Run with specific config file
    python experiments/run_experiment.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml
    
    # Run with custom experiment name
    python experiments/run_experiment.py --name my_experiment
    
    # Resume from checkpoint
    python experiments/run_experiment.py --resume experiments/my_experiment/checkpoints/subject_1_fold_0_acc_0.8500.pt
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and register components first
from models import register_all_models, MODELS
from data import DATASETS
from trainers import TRAINERS

from core import Config, ExperimentManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run EEG-BCI Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default configuration
  python experiments/run_experiment.py
  
  # Run with specific configuration
  python experiments/run_experiment.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml
  
  # Run with custom name
  python experiments/run_experiment.py --name my_test --config my_config.yaml
  
  # Resume training
  python experiments/run_experiment.py --resume checkpoints/model.pt
        '''
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml',
        help='Path to configuration file (default: configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml)'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated from config)'
    )
    
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--list-trainers',
        action='store_true',
        help='List available trainers and exit'
    )
    
    return parser.parse_args()


def create_default_config():
    """Create a default configuration."""
    return Config({
        'experiment': {
            'name': 'default_experiment',
            'seed': 42
        },
        'data': {
            'dataset': 'XWStroke',
            'subjects': 'all',
            'info_path': 'configs/dataset/XWStroke.yaml'
        },
        'model': {
            'type': 'EEGNet',
            'args': {}
        },
        'training': {
            'device': 'cuda',
            'epochs': 100,
            'batch_size': 32,
            'k_folds': 5,
            'optimizer': {
                'lr': 0.001,
                'weight_decay': 0.01
            }
        },
        'trainer': {
            'type': 'SupervisedTrainer'
        },
        'paths': {
            'root_dir': './experiments'
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    })


def main():
    """Main entry point."""
    args = parse_args()
    
    # Ensure models are registered
    register_all_models()
    
    # Handle list commands
    if args.list_datasets:
        print("Available datasets:")
        for name in DATASETS.list_keys():
            print(f"  - {name}")
        return
    
    if args.list_models:
        print("Available models:")
        for name in MODELS.list_keys():
            print(f"  - {name}")
        return
    
    if args.list_trainers:
        print("Available trainers:")
        for name in TRAINERS.list_keys():
            print(f"  - {name}")
        return
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = Config.fromfile(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Creating a default configuration...")
        config = create_default_config()
    
    # Override config with command line args
    if args.device:
        config.set('training.device', args.device)
    
    # Create and run experiment
    try:
        experiment = ExperimentManager(
            config=config,
            exp_name=args.name,
            resume_from=args.resume
        )
        
        # Setup
        experiment.setup()
        
        # Run
        results = experiment.run()
        
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print(f"Results saved to: {experiment.paths.results_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
