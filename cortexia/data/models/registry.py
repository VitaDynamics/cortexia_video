"""Result schema registry using decorator pattern.

This module integrates result schemas with the core registry system
using the decorator pattern for flexible, self-registering classes.
"""

from ...core.registry import Registry

# Global registry instance for scehma schemas
schema_registry = Registry("schema")


def get_schema(schema_name: str):
    """
    Get a schema class by name from the schema registry.
    
    Args:
        schema_name: Name of the schema to retrieve
        
    Returns:
        The schema class
        
    Raises:
        ValueError: If schema is not found
    """
    return schema_registry.get(schema_name)

# Import the registry instances for easy access
__all__ = ["schema_registry", "get_schema"]