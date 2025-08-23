"""Main Cortexia SDK class"""

import os
from typing import Any, Dict, List, Optional, Type, Union

from ..core.registry import 
from ..core.config.manager import ConfigManager
from .exceptions import CortexiaError, FeatureNotFoundError


class Cortexia:
    """
    Main entry point for the Cortexia Video SDK.
    
    Provides access to independent annotation features through a registry system.
    """
    
    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize Cortexia SDK.
        
        Args:
            config: Configuration file path or dictionary
        """
        # Initialize configuration
        if isinstance(config, str):
            self.config_manager = ConfigManager(config_file_path=config)
        elif isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config_manager.config_data = config
        else:
            self.config_manager = ConfigManager()
        
        # Load configuration if file path provided
        if isinstance(config, str):
            self.config_manager.load_config()
        
        # Initialize feature registry
        self.registry = FeatureRegistry(self.config_manager)
        self._register_default_features()
    
    @classmethod
    def from_config(cls, config_path: str) -> "Cortexia":
        """
        Create Cortexia instance from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Cortexia instance
        """
        return cls(config=config_path)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Cortexia":
        """
        Create Cortexia instance from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Cortexia instance
        """
        return cls(config=config_dict)
    
    def _register_default_features(self):
        """Register default features"""
        # Import here to avoid circular imports
        from ..features.detection.detector import DetectionFeature
        from ..features.segmentation.segmenter import SegmentationFeature
        from ..features.description.describer import DescriptionFeature
        from ..features.listing.lister import ListingFeature
        from ..features.feature_extraction.extractor import FeatureExtractionFeature
        from ..features.caption.captioner import CaptionFeature
        from ..features.depth.estimator import DepthFeature
        
        # Register features
        self.registry.register("detection", DetectionFeature)
        self.registry.register("segmentation", SegmentationFeature)
        self.registry.register("description", DescriptionFeature)
        self.registry.register("listing", ListingFeature)
        self.registry.register("feature_extraction", FeatureExtractionFeature)
        self.registry.register("caption", CaptionFeature)
        self.registry.register("depth", DepthFeature)
    
    def detection(self):
        """Get detection feature instance"""
        return self.registry.get_feature("detection")
    
    def segmentation(self):
        """Get segmentation feature instance"""
        return self.registry.get_feature("segmentation")
    
    def description(self):
        """Get description feature instance"""
        return self.registry.get_feature("description")
    
    def listing(self):
        """Get listing feature instance"""
        return self.registry.get_feature("listing")
    
    def feature_extraction(self):
        """Get feature extraction instance"""
        return self.registry.get_feature("feature_extraction")
    
    def caption(self):
        """Get caption feature instance"""
        return self.registry.get_feature("caption")
    
    def depth(self):
        """Get depth estimation feature instance"""
        return self.registry.get_feature("depth")
    
    def get_feature(self, name: str):
        """
        Get feature instance by name.
        
        Args:
            name: Feature name
            
        Returns:
            Feature instance
            
        Raises:
            FeatureNotFoundError: If feature not found
        """
        return self.registry.get_feature(name)
    
    def list_features(self) -> List[str]:
        """
        List available features.
        
        Returns:
            List of feature names
        """
        return self.registry.list_features()
    
    def process_video_with_features(self, video_path: str, features: List[str]) -> List[Any]:
        """
        Process video with specified features.
        
        Args:
            video_path: Path to video file
            features: List of feature names to apply
            
        Returns:
            List of processed results
        """
        # This is a convenience method for processing video with multiple features
        # Each feature is applied independently to each frame
        results = []
        
        for feature_name in features:
            feature = self.get_feature(feature_name)
            # Process video with this feature
            # Implementation will depend on video loading utilities
            pass
        
        return results