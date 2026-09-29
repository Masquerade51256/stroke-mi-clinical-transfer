# src/preprocessing/factory.py
from typing import Dict, Any, Type
from preprocessing.base_processor import BaseProcessor
from preprocessing.filter_banks import FilterBankProcessor
from preprocessing.resample import ResampleProcessor
from preprocessing.artifact_removal import ArtifactRemovalProcessor, EMARProcessor, SASICAProcessor, MARAProcessor
from preprocessing.data_augmentation import DataAugmentationProcessor
from preprocessing.euclidean_alignment import EuclideanAlignmentProcessor


class ProcessorFactory:
    """
    Factory class for creating processor instances from configuration.
    Uses registry pattern for extensibility.
    """
    
    # Registry mapping processor type names to class implementations
    _processor_registry: Dict[str, Type[BaseProcessor]] = {
        'filterbank': FilterBankProcessor,
        'resample': ResampleProcessor,
        'artifact_removal': ArtifactRemovalProcessor,
        'emar': EMARProcessor,
        'sasica': SASICAProcessor,
        'mara': MARAProcessor,
        'augmentation': DataAugmentationProcessor,
        'euclidean_alignment': EuclideanAlignmentProcessor,
    }
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[BaseProcessor]) -> None:
        """Register a new processor type."""
        if name in cls._processor_registry:
            raise ValueError(f"Processor '{name}' is already registered")
        cls._processor_registry[name] = processor_class
    
    @classmethod
    def create_processor(cls, name: str, processor_type: str, **config) -> BaseProcessor:
        """
        Create a processor instance from configuration.
        
        Args:
            name: Unique name for this processor instance.
            processor_type: Type identifier (must be in registry).
            **config: Processor-specific configuration.
            
        Returns:
            Configured processor instance.
        """
        if processor_type not in cls._processor_registry:
            available = list(cls._processor_registry.keys())
            raise ValueError(f"Unknown processor type: '{processor_type}'. "
                           f"Available types: {available}")
        
        processor_class = cls._processor_registry[processor_type]
        return processor_class(name=name, **config)
    
    @classmethod
    def create_pipeline_from_config(cls, pipeline_config: list) -> 'ProcessingPipeline':
        """
        Create a complete processing pipeline from configuration list.
        
        Args:
            pipeline_config: List of processor configurations, e.g.:
                [
                    {
                        'name': 'resample_250hz',
                        'type': 'resample',
                        'target_sfreq': 250,
                        'original_sfreq': 1000
                    },
                    {
                        'name': 'filterbank_mu_beta',
                        'type': 'filterbank',
                        'filter_banks': [[4, 8], [8, 12], [12, 30]],
                        'sample_rate': 250
                    }
                ]
                
        Returns:
            Configured ProcessingPipeline instance.
        """
        from preprocessing.base_processor import ProcessingPipeline
        
        pipeline = ProcessingPipeline()
        
        for proc_config in pipeline_config:
            # Extract processor type and name
            proc_name = proc_config['name']
            proc_type = proc_config['type']
            
            # Remove name and type from config (they're not processor parameters)
            proc_params = {k: v for k, v in proc_config.items() 
                          if k not in ['name', 'type']}
            
            # Create processor instance
            processor = cls.create_processor(
                name=proc_name,
                processor_type=proc_type,
                **proc_params
            )
            
            pipeline.add_processor(processor)
        
        return pipeline
