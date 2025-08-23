"""
Feature Factory Registry for creating and managing different feature implementations.

This factory provides a centralized way to register and create feature instances
with appropriate initialization parameters.
"""
from typing import Dict, Type, Any, Optional, Callable
from .base import BaseFeature


class FeatureFactory:
    """
    Factory class for creating feature instances with a registry pattern.
    
    Provides a clean interface for registering feature classes and creating
    instances with appropriate parameters.
    """
    
    _registry: Dict[str, Type[BaseFeature]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, feature_class: Type[BaseFeature], default_config: Optional[Dict[str, Any]] = None):
        """
        Register a feature class with the factory.
        
        Args:
            name: Name to register the feature under (e.g., 'caption', 'detection')
            feature_class: The feature class to register
            default_config: Optional default configuration parameters
        """
        cls._registry[name.lower()] = feature_class
        cls._default_configs[name.lower()] = default_config or {}
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseFeature:
        """
        Create a feature instance by name.
        
        Args:
            name: Name of the feature to create
            config: Configuration dictionary to pass to the feature constructor
            
        Returns:
            BaseFeature: Instance of the requested feature
            
        Raises:
            ValueError: If the feature name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Feature '{name}' not found. Available features: {list(cls._registry.keys())}")
        
        feature_class = cls._registry[name]
        default_config = cls._default_configs[name].copy()
        
        # Merge default config with provided config
        final_config = {**default_config}
        if config:
            final_config.update(config)
        
        return feature_class(final_config)
    
    @classmethod
    def get_available_features(cls) -> list[str]:
        """
        Get list of available feature names.
        
        Returns:
            List of registered feature names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_feature_class(cls, name: str) -> Type[BaseFeature]:
        """
        Get the feature class by name without instantiating.
        
        Args:
            name: Name of the feature
            
        Returns:
            Feature class
            
        Raises:
            ValueError: If the feature name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Feature '{name}' not found. Available features: {list(cls._registry.keys())}")
        
        return cls._registry[name]
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered features. Primarily for testing."""
        cls._registry.clear()
        cls._default_configs.clear()

    # Decorator-based registration for convenience
    @classmethod
    def decorator(cls, name: str, default_config: Optional[Dict[str, Any]] = None) -> Callable[[Type[BaseFeature]], Type[BaseFeature]]:
        """Return a decorator that registers a feature class under `name`."""
        def _decorator(feature_class: Type[BaseFeature]) -> Type[BaseFeature]:
            cls.register(name, feature_class, default_config)
            return feature_class
        return _decorator


def register_all_features():
    """
    Register all available feature implementations with the factory.
    This function should be called to populate the factory registry.
    """
    # Import features locally to avoid circular imports
    from .caption.captioner import CaptionFeature
    from .detection.detector import DetectionFeature
    from .depth.estimator import DepthFeature
    from .feature_extraction.extractor import FeatureExtractionFeature
    from .listing.lister import ListingFeature
    
    # Import segmentation feature if available
    try:
        from .segmentation.segmenter import SegmentationFeature
        has_segmentation = True
    except ImportError:
        has_segmentation = False
    
    # Import description feature if available
    try:
        from .description.describer import DescriptionFeature
        has_description = True
    except ImportError:
        has_description = False
    
    # Register each feature with appropriate default configs
    FeatureFactory.register('caption', CaptionFeature, {
        'model': 'vikhyatk/moondream2',
        'revision': '2025-06-21',
        'caption_length': 'long'
    })
    
    FeatureFactory.register('detection', DetectionFeature, {
        'model': 'IDEA-Research/grounding-dino-base',
        'box_threshold': 0.3,
        'text_threshold': 0.3,
        'default_prompts': ['object']
    })
    
    FeatureFactory.register('depth', DepthFeature, {})
    
    FeatureFactory.register('feature_extraction', FeatureExtractionFeature, {
        'crop_padding': 10
    })
    
    FeatureFactory.register('listing', ListingFeature, {
        'model': 'vikhyatk/moondream2'
    })
    
    # Register optional features if available
    if has_segmentation:
        FeatureFactory.register('segmentation', SegmentationFeature, {})
    
    if has_description:
        FeatureFactory.register('description', DescriptionFeature, {
            'model': 'vikhyatk/moondream2'
        })


# Convenience functions for easy feature creation
def create_caption_feature(
    model: str = "vikhyatk/moondream2",
    revision: str = "2025-06-21",
    caption_length: str = "long",
    **kwargs
) -> BaseFeature:
    """Create a caption feature with specified parameters."""
    config = {
        'model': model,
        'revision': revision,
        'caption_length': caption_length,
        **kwargs
    }
    return FeatureFactory.create('caption', config)


def create_detection_feature(
    model: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    default_prompts: list = None,
    **kwargs
) -> BaseFeature:
    """Create a detection feature with specified parameters."""
    config = {
        'model': model,
        'box_threshold': box_threshold,
        'text_threshold': text_threshold,
        'default_prompts': default_prompts or ['object'],
        **kwargs
    }
    return FeatureFactory.create('detection', config)


def create_depth_feature(**kwargs) -> BaseFeature:
    """Create a depth estimation feature with specified parameters."""
    return FeatureFactory.create('depth', kwargs)


def create_feature_extraction_feature(
    crop_padding: int = 10,
    **kwargs
) -> BaseFeature:
    """Create a feature extraction feature with specified parameters."""
    config = {
        'crop_padding': crop_padding,
        **kwargs
    }
    return FeatureFactory.create('feature_extraction', config)


def create_listing_feature(
    model: str = "vikhyatk/moondream2",
    **kwargs
) -> BaseFeature:
    """Create a listing feature with specified parameters."""
    config = {
        'model': model,
        **kwargs
    }
    return FeatureFactory.create('listing', config)


def create_segmentation_feature(**kwargs) -> BaseFeature:
    """Create a segmentation feature with specified parameters."""
    return FeatureFactory.create('segmentation', kwargs)


def create_description_feature(
    model: str = "vikhyatk/moondream2",
    **kwargs
) -> BaseFeature:
    """Create a description feature with specified parameters."""
    config = {
        'model': model,
        **kwargs
    }
    return FeatureFactory.create('description', config)
