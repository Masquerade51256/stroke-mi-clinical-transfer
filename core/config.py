"""
Configuration management for the experiment framework.

Provides a unified interface for loading, validating, and accessing
experiment configurations from YAML files.
"""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
import copy


class Config:
    """
    Configuration container with dict-like and attribute-like access.
    
    Supports loading from YAML files and provides convenient access patterns.
    
    Example:
        >>> config = Config.fromfile('configs/experiment.yaml')
        >>> print(config.training.batch_size)
        >>> print(config['training']['batch_size'])
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, 
                 filename: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
            filename: Path to the configuration file (for reference)
        """
        self._config_dict = config_dict or {}
        self._filename = filename
        
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config['key']"""
        return self._config_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment: config['key'] = value"""
        self._config_dict[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-like access: config.key"""
        if name.startswith('_'):
            # Avoid recursion for private attributes
            return object.__getattribute__(self, name)
        try:
            value = self._config_dict[name]
            # Recursively convert dicts to Config objects
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-like assignment: config.key = value"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._config_dict[name] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'key' in config"""
        return key in self._config_dict
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config_dict})"
    
    def __str__(self) -> str:
        """Pretty string representation."""
        return yaml.dump(self._config_dict, default_flow_style=False, allow_unicode=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config_dict
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'training.batch_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config_dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to plain dictionary."""
        return copy.deepcopy(self._config_dict)
    
    def merge(self, other: Union[Dict[str, Any], 'Config']) -> 'Config':
        """
        Merge another config into this one.
        
        Args:
            other: Another Config object or dict to merge
            
        Returns:
            Self for chaining
        """
        if isinstance(other, Config):
            other = other.to_dict()
        self._deep_merge(self._config_dict, other)
        return self
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
    
    def validate(self, required_keys: list) -> bool:
        """
        Validate that required keys exist in the config.
        
        Args:
            required_keys: List of required key paths (supports dot notation)
            
        Returns:
            True if all keys exist
            
        Raises:
            ValueError: If any required key is missing
        """
        missing = []
        for key in required_keys:
            if self.get(key) is None:
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")
        return True
    
    @property
    def filename(self) -> Optional[str]:
        """Get the configuration file path."""
        return self._filename
    
    @classmethod
    def fromfile(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from a YAML file.
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            Config object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict, str(filepath))
    
    @classmethod
    def from_string(cls, config_str: str) -> 'Config':
        """
        Load configuration from a YAML string.
        
        Args:
            config_str: YAML configuration string
            
        Returns:
            Config object
        """
        config_dict = yaml.safe_load(config_str)
        return cls(config_dict)
    
    def dump(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            filepath: Output file path. If None, use the original filename.
        """
        if filepath is None:
            if self._filename is None:
                raise ValueError("No filename specified for dumping")
            filepath = self._filename
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self._config_dict, f, default_flow_style=False, allow_unicode=True)


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configurations, with later configs overriding earlier ones.
    
    Args:
        *configs: Variable number of Config objects
        
    Returns:
        Merged Config object
    """
    if not configs:
        return Config()
    
    result = Config(configs[0].to_dict())
    for config in configs[1:]:
        result.merge(config)
    return result
