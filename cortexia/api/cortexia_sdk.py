"""Main Cortexia Video SDK class

Provides a unified interface for accessing all features and gates.
"""

from typing import Any, Dict, List, Optional, Union

from .exceptions import FeatureNotFoundError, ConfigurationError
from ..features import feature_registry
from ..gates import gate_registry


class Cortexia:
    """Main Cortexia SDK class for video processing.
    
    Provides unified access to all registered features and gates.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Cortexia with optional configuration.
        
        Args:
            config: Global configuration dictionary
        """
        self.config = config or {}
        self._feature_instances = {}
        self._gate_instances = {}
    
    @property
    def features(self) -> List[str]:
        """Get list of available feature names."""
        return list(feature_registry.keys())
    
    @property
    def gates(self) -> List[str]:
        """Get list of available gate names."""
        return list(gate_registry.keys())
    
    def get_feature(self, name: str, **kwargs) -> Any:
        """Get or create a feature instance.
        
        Args:
            name: Feature name
            **kwargs: Arguments for feature constructor
            
        Returns:
            Feature instance
            
        Raises:
            FeatureNotFoundError: If feature is not registered
        """
        if name not in feature_registry:
            raise FeatureNotFoundError(f"Feature '{name}' not found")
        
        # Cache instances by name and kwargs
        cache_key = (name, frozenset(kwargs.items()))
        if cache_key not in self._feature_instances:
            feature_class = feature_registry.require(name)
            self._feature_instances[cache_key] = feature_class(**kwargs)
        
        return self._feature_instances[cache_key]
    
    def get_gate(self, name: str, **kwargs) -> Any:
        """Get or create a gate instance.
        
        Args:
            name: Gate name
            **kwargs: Arguments for gate constructor
            
        Returns:
            Gate instance
            
        Raises:
            FeatureNotFoundError: If gate is not registered
        """
        if name not in gate_registry:
            raise FeatureNotFoundError(f"Gate '{name}' not found")
        
        # Cache instances by name and kwargs
        cache_key = (name, frozenset(kwargs.items()))
        if cache_key not in self._gate_instances:
            gate_class = gate_registry.require(name)
            self._gate_instances[cache_key] = gate_class(**kwargs)
        
        return self._gate_instances[cache_key]
    
    def create_feature(self, name: str, **kwargs) -> Any:
        """Create a new feature instance (no caching).
        
        Args:
            name: Feature name
            **kwargs: Arguments for feature constructor
            
        Returns:
            New feature instance
        """
        if name not in feature_registry:
            raise FeatureNotFoundError(f"Feature '{name}' not found")
        
        feature_class = feature_registry.require(name)
        return feature_class(**kwargs)
    
    def create_gate(self, name: str, **kwargs) -> Any:
        """Create a new gate instance (no caching).
        
        Args:
            name: Gate name
            **kwargs: Arguments for gate constructor
            
        Returns:
            New gate instance
        """
        if name not in gate_registry:
            raise FeatureNotFoundError(f"Gate '{name}' not found")
        
        gate_class = gate_registry.require(name)
        return gate_class(**kwargs)
    
    def list_features(self) -> List[str]:
        """List all available feature names."""
        return self.features
    
    def list_gates(self) -> List[str]:
        """List all available gate names."""
        return self.gates
    
    def feature_info(self, name: str) -> Dict[str, Any]:
        """Get information about a feature.
        
        Args:
            name: Feature name
            
        Returns:
            Dictionary with feature information
        """
        if name not in feature_registry:
            raise FeatureNotFoundError(f"Feature '{name}' not found")
        
        feature_class = feature_registry.require(name)
        return {
            "name": name,
            "class": feature_class.__name__,
            "module": feature_class.__module__,
            "doc": feature_class.__doc__,
        }
    
    def gate_info(self, name: str) -> Dict[str, Any]:
        """Get information about a gate.
        
        Args:
            name: Gate name
            
        Returns:
            Dictionary with gate information
        """
        if name not in gate_registry:
            raise FeatureNotFoundError(f"Gate '{name}' not found")
        
        gate_class = gate_registry.require(name)
        return {
            "name": name,
            "class": gate_class.__name__,
            "module": gate_class.__module__,
            "doc": gate_class.__doc__,
        }