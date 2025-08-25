"""Centralized schema registry system for unified data flow management.

This module provides a centralized registry for all result schemas used in the
Cortexia Video SDK, enabling consistent schema resolution across features and gates.
"""

from typing import Dict, Type, List, Optional
from pathlib import Path
import importlib

from .result.base_result import BaseResult


class SchemaRegistry:
    """
    Centralized registry for managing result schemas across the SDK.
    
    This registry maintains mappings between schema names and their corresponding
    classes, enabling consistent schema resolution for features, gates, and
    dataset operations.
    """
    
    _instance: Optional['SchemaRegistry'] = None
    _schemas: Dict[str, Type[BaseResult]] = {}
    _schema_mapping: Dict[str, str] = {
        # Core result schemas
        "CaptionResult": "caption_result.CaptionResult",
        "TaggingResult": "tagging_result.TaggingResult", 
        "DetectionResult": "detection.DetectionResult",
        "SegmentationResult": "segmentation.SegmentationResult",
        "DepthResult": "depth_result.DepthResult",
        "FeatureExtractionResult": "feature_extraction_result.FeatureExtractionResult",
        "DescriptionResult": "description_result.DescriptionResult",
        "GateResult": "gate_result.GateResult",
    }
    
    def __new__(cls) -> 'SchemaRegistry':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'SchemaRegistry':
        """Get the singleton instance of the schema registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_schema(self, schema_name: str, schema_class: Type[BaseResult]) -> None:
        """
        Register a schema class with the registry.
        
        Args:
            schema_name: Name identifier for the schema
            schema_class: The schema class to register
            
        Raises:
            ValueError: If schema_name already exists or schema_class is invalid
        """
        if schema_name in self._schemas:
            existing_class = self._schemas[schema_name]
            if existing_class != schema_class:
                raise ValueError(
                    f"Schema '{schema_name}' already registered with different class: "
                    f"{existing_class} != {schema_class}"
                )
        
        if not issubclass(schema_class, BaseResult):
            raise ValueError(f"Schema class must inherit from BaseResult, got {schema_class}")
        
        self._schemas[schema_name] = schema_class
    
    def get_schema(self, schema_name: str) -> Type[BaseResult]:
        """
        Get a schema class by name.
        
        Args:
            schema_name: Name of the schema to retrieve
            
        Returns:
            The schema class
            
        Raises:
            ValueError: If schema is not registered
        """
        # First check if already loaded
        if schema_name in self._schemas:
            return self._schemas[schema_name]
        
        # Try to load from mapping
        if schema_name in self._schema_mapping:
            schema_class = self._load_schema_class(schema_name)
            self._schemas[schema_name] = schema_class
            return schema_class
        
        raise ValueError(f"Unknown schema: {schema_name}")
    
    def _load_schema_class(self, schema_name: str) -> Type[BaseResult]:
        """
        Dynamically load a schema class from the mapping.
        
        Args:
            schema_name: Name of the schema to load
            
        Returns:
            The loaded schema class
            
        Raises:
            ImportError: If the schema class cannot be imported
        """
        module_path = self._schema_mapping[schema_name]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            # Import the module
            module = importlib.import_module(f"..{module_name}", __name__)
            schema_class = getattr(module, class_name)
            
            if not issubclass(schema_class, BaseResult):
                raise ImportError(f"Class {class_name} is not a BaseResult subclass")
            
            return schema_class
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load schema {schema_name}: {e}")
    
    def list_schemas(self) -> List[str]:
        """
        Get a list of all registered schema names.
        
        Returns:
            List of schema names
        """
        # Combine loaded and available schemas
        all_schemas = set(self._schemas.keys())
        all_schemas.update(self._schema_mapping.keys())
        return sorted(list(all_schemas))
    
    def is_registered(self, schema_name: str) -> bool:
        """
        Check if a schema is registered.
        
        Args:
            schema_name: Name of the schema to check
            
        Returns:
            True if schema is registered, False otherwise
        """
        return schema_name in self._schemas or schema_name in self._schema_mapping
    
    def unregister_schema(self, schema_name: str) -> bool:
        """
        Unregister a schema from the registry.
        
        Args:
            schema_name: Name of the schema to unregister
            
        Returns:
            True if schema was unregistered, False if not found
        """
        removed = False
        if schema_name in self._schemas:
            del self._schemas[schema_name]
            removed = True
        if schema_name in self._schema_mapping:
            del self._schema_mapping[schema_name]
            removed = True
        return removed
    
    def clear_registry(self) -> None:
        """Clear all registered schemas. Use with caution."""
        self._schemas.clear()
        self._schema_mapping.clear()
    
    def get_schema_info(self, schema_name: str) -> Dict[str, str]:
        """
        Get information about a schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Dictionary with schema information
        """
        if not self.is_registered(schema_name):
            raise ValueError(f"Schema '{schema_name}' not registered")
        
        info = {"name": schema_name}
        
        # Check if loaded
        if schema_name in self._schemas:
            schema_class = self._schemas[schema_name]
            info["class"] = f"{schema_class.__module__}.{schema_class.__name__}"
            info["status"] = "loaded"
        elif schema_name in self._schema_mapping:
            info["class"] = self._schema_mapping[schema_name]
            info["status"] = "available"
        
        return info
    
    def validate_schema(self, schema_name: str, data: dict) -> bool:
        """
        Validate data against a schema.
        
        Args:
            schema_name: Name of the schema to validate against
            data: Data to validate
            
        Returns:
            True if validation succeeds
            
        Raises:
            ValueError: If schema not found or validation fails
        """
        schema_class = self.get_schema(schema_name)
        try:
            # Attempt to create instance with data
            schema_class(**data)
            return True
        except Exception as e:
            raise ValueError(f"Schema validation failed for {schema_name}: {e}")


# Global registry instance
registry = SchemaRegistry()


# Convenience functions
def get_schema(schema_name: str) -> Type[BaseResult]:
    """
    Get a schema class by name.
    
    Args:
        schema_name: Name of the schema
        
    Returns:
        The schema class
    """
    return registry.get_schema(schema_name)


def register_schema(schema_name: str, schema_class: Type[BaseResult]) -> None:
    """
    Register a new schema with the global registry.
    
    Args:
        schema_name: Name identifier for the schema
        schema_class: The schema class to register
    """
    registry.register_schema(schema_name, schema_class)


def list_schemas() -> List[str]:
    """Get a list of all available schema names."""
    return registry.list_schemas()


def is_schema_registered(schema_name: str) -> bool:
    """Check if a schema is registered."""
    return registry.is_registered(schema_name)