"""gate registry using decorator pattern.

This module integrates gate schemas with the core registry system
using the decorator pattern for flexible, self-registering classes.
"""

from ..core.registry import Registry

# Global registry instance for gate schemas
gate_registry = Registry("gates")

# Import the registry instance for easy access
__all__ = ["gate_registry"]