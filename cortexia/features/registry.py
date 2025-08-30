"""feature registry using decorator pattern.

This module integrates feature schemas with the core registry system
using the decorator pattern for flexible, self-registering classes.
"""

from ..core.registry import Registry

# Global registry instance for feature schemas
feature_registry = Registry("features")

# Import the registry instance for easy access
__all__ = ["feature_registry"]