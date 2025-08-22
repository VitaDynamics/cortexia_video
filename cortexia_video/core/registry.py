"""Feature registry for managing independent annotation features"""

from typing import Any, Dict, List, Type

from ..api.exceptions import FeatureNotFoundError
from ..features.base import BaseFeature


class FeatureRegistry:
    """
    Registry for managing feature instances and classes.
    
    Provides lazy initialization of features and ensures each feature
    is only instantiated once per registry instance.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize feature registry.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._feature_classes: Dict[str, Type[BaseFeature]] = {}
        self._feature_instances: Dict[str, BaseFeature] = {}
    
    def register(self, name: str, feature_class: Type[BaseFeature]):
        """
        Register a feature class.
        
        Args:
            name: Feature name for registration
            feature_class: Feature class to register
        """
        self._feature_classes[name] = feature_class
    
    def get_feature(self, name: str) -> BaseFeature:
        """
        Get feature instance by name.
        
        Creates instance if not already created (lazy initialization).
        
        Args:
            name: Feature name
            
        Returns:
            Feature instance
            
        Raises:
            FeatureNotFoundError: If feature not registered
        """
        if name not in self._feature_classes:
            raise FeatureNotFoundError(f"Feature '{name}' not found in registry")
        
        # Return existing instance if available
        if name in self._feature_instances:
            return self._feature_instances[name]
        
        # Create new instance
        feature_class = self._feature_classes[name]
        
        # Get feature-specific config if available
        feature_config = self._get_feature_config(name) if self.config_manager else {}
        
        # Create instance
        instance = feature_class(config=feature_config)
        self._feature_instances[name] = instance
        
        return instance
    
    def _get_feature_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific feature.
        
        Args:
            name: Feature name
            
        Returns:
            Feature-specific configuration dictionary
        """
        if not self.config_manager:
            return {}
        
        # Try to get feature-specific config section
        feature_config_key = f"features.{name}"
        feature_config = self.config_manager.get_param(feature_config_key, {})
        
        return feature_config
    
    def list_features(self) -> List[str]:
        """
        List all registered feature names.
        
        Returns:
            List of feature names
        """
        return list(self._feature_classes.keys())
    
    def has_feature(self, name: str) -> bool:
        """
        Check if a feature is registered.
        
        Args:
            name: Feature name
            
        Returns:
            True if feature is registered
        """
        return name in self._feature_classes
    
    def clear_instances(self):
        """Clear all feature instances (useful for testing or reinitialization)"""
        self._feature_instances.clear()
    
    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered feature.
        
        Args:
            name: Feature name
            
        Returns:
            Dictionary with feature information
            
        Raises:
            FeatureNotFoundError: If feature not registered
        """
        if name not in self._feature_classes:
            raise FeatureNotFoundError(f"Feature '{name}' not found in registry")
        
        feature_class = self._feature_classes[name]
        instance = self._feature_instances.get(name)
        
        return {
            "name": name,
            "class": feature_class.__name__,
            "module": feature_class.__module__,
            "initialized": instance is not None,
            "config": self._get_feature_config(name) if self.config_manager else {}
        }