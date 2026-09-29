"""
Registry system for managing datasets, models, and trainers.

This module implements a registry pattern that allows components to be
registered and retrieved by name, enabling a plugin-like architecture.
"""

from typing import Any, Callable, Dict, Optional, Type
import inspect


class Registry:
    """
    Universal registry for managing components (datasets, models, trainers, etc.).
    
    Example:
        >>> DATASETS = Registry('datasets')
        >>> @DATASETS.register('my_dataset')
        ... class MyDataset:
        ...     pass
        >>> dataset_class = DATASETS.get('my_dataset')
    """
    
    def __init__(self, name: str):
        """
        Initialize a registry.
        
        Args:
            name: Name of the registry (for identification purposes)
        """
        self._name = name
        self._module_dict: Dict[str, Any] = {}
        
    def __len__(self) -> int:
        """Return the number of registered modules."""
        return len(self._module_dict)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is registered."""
        return key in self._module_dict
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"
    
    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name
    
    @property
    def modules(self) -> Dict[str, Any]:
        """Get all registered modules."""
        return self._module_dict.copy()
    
    def get(self, key: str) -> Any:
        """
        Get a registered module by key.
        
        Args:
            key: The registration key
            
        Returns:
            The registered module
            
        Raises:
            KeyError: If the key is not registered
        """
        if key not in self._module_dict:
            raise KeyError(f"'{key}' is not registered in '{self._name}' registry. "
                          f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]
    
    def register(self, name: Optional[str] = None, force: bool = False) -> Callable:
        """
        Register a module.
        
        Args:
            name: Registration name. If None, use the class/function name.
            force: Whether to override an existing registration.
            
        Returns:
            Decorator function
        """
        def _register(module: Any) -> Any:
            module_name = name if name is not None else module.__name__
            
            if module_name in self._module_dict:
                if not force:
                    # Silently skip duplicate registrations
                    return module
                # If force=True, continue to override
            
            self._module_dict[module_name] = module
            return module
        
        return _register
    
    def unregister(self, name: str) -> None:
        """
        Unregister a module.
        
        Args:
            name: The registration key to remove
        """
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in '{self._name}'")
        del self._module_dict[name]
    
    def list_keys(self) -> list:
        """Return a list of all registered keys."""
        return list(self._module_dict.keys())


# Global registries for the framework
DATASETS = Registry('datasets')
MODELS = Registry('models')
TRAINERS = Registry('trainers')
TRANSFORMS = Registry('transforms')


def build_from_config(config: Dict[str, Any], registry: Registry, 
                     default_args: Optional[Dict[str, Any]] = None) -> Any:
    """
    Build an object from a configuration dictionary.
    
    Args:
        config: Configuration dict. Must contain a 'type' key indicating
                the registered name, and optionally an 'args' dict for constructor args.
        registry: The registry to search for the type
        default_args: Default arguments to use if not specified in config
        
    Returns:
        The constructed object
        
    Example:
        >>> config = {'type': 'EEGNet', 'args': {'num_channels': 22, 'num_classes': 4}}
        >>> model = build_from_config(config, MODELS)
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config)}")
    
    if 'type' not in config:
        raise KeyError("config must contain a 'type' key")
    
    obj_type = config['type']
    obj_cls = registry.get(obj_type)
    
    args = config.get('args', {}).copy()
    if default_args is not None:
        for key, value in default_args.items():
            args.setdefault(key, value)
    
    # Check if it's a class or a function
    if inspect.isclass(obj_cls):
        return obj_cls(**args)
    else:
        return obj_cls(**args)
