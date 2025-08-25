"""Main Cortexia Video SDK API

This module provides a unified interface to access all features and gates
from the cortexia package through registries.
"""

from .cortexia_sdk import Cortexia
from ..features import feature_registry
from ..gates import gate_registry

__all__ = [
    "Cortexia",
    "feature_registry",
    "gate_registry",
    "get_feature",
    "get_gate",
    "list_features",
    "list_gates",
    "create_feature",
    "create_gate",
]


def get_feature(name: str):
    """Get a feature class by name from the feature registry.
    
    Args:
        name: The feature name (e.g., "caption", "detection", "depth")
        
    Returns:
        The feature class
        
    Raises:
        KeyError: If the feature is not found
    """
    return feature_registry.require(name)


def get_gate(name: str):
    """Get a gate class by name from the gate registry.
    
    Args:
        name: The gate name (e.g., "blur", "entropy", "clip")
        
    Returns:
        The gate class
        
    Raises:
        KeyError: If the gate is not found
    """
    return gate_registry.require(name)


def list_features():
    """List all available feature names.
    
    Returns:
        List of feature names
    """
    return list(feature_registry.keys())


def list_gates():
    """List all available gate names.
    
    Returns:
        List of gate names
    """
    return list(gate_registry.keys())


def create_feature(name: str, **kwargs):
    """Create a feature instance by name.
    
    Args:
        name: The feature name
        **kwargs: Arguments to pass to the feature constructor
        
    Returns:
        Feature instance
    """
    feature_class = get_feature(name)
    return feature_class(**kwargs)


def create_gate(name: str, **kwargs):
    """Create a gate instance by name.
    
    Args:
        name: The gate name
        **kwargs: Arguments to pass to the gate constructor
        
    Returns:
        Gate instance
    """
    gate_class = get_gate(name)
    return gate_class(**kwargs)