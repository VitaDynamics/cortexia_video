"""Base result schema for all feature and gate outputs.

This module provides the foundation class that all result schemas inherit from,
ensuring consistent serialization and deserialization for Lance dataset storage.
"""

from abc import ABC
from typing import Any, Dict, Type, TypeVar
from io import BytesIO

import numpy as np
from pydantic import BaseModel

# Type variable for generic deserialization
T = TypeVar('T', bound='BaseResult')


class BaseResult(BaseModel, ABC):
    """
    Base class for all feature and gate result schemas.
    
    Provides standardized dict serialization/deserialization that works
    seamlessly with Lance dataset storage through LanceAdapter.
    """
    
    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays and custom types
        
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert result to dictionary for Lance storage.
        
        This method handles serialization of complex types like numpy arrays
        by converting them to bytes or appropriate serializable formats.
        
        Args:
            **kwargs: Additional arguments for Pydantic dict method
            
        Returns:
            Dictionary representation suitable for Lance storage
        """
        data = super().dict(**kwargs)
        return self._serialize_special_types(data)
    
    def _serialize_special_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle serialization of special types like numpy arrays.
        
        This method can be overridden by subclasses to handle feature-specific
        serialization needs while maintaining the base pattern.
        
        Args:
            data: Dictionary with potentially unserializable values
            
        Returns:
            Dictionary with all values serialized appropriately
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Serialize numpy arrays to bytes
                buffer = BytesIO()
                np.save(buffer, value)
                serialized[f"{key}_numpy_bytes"] = buffer.getvalue()
                serialized[f"{key}_numpy_shape"] = list(value.shape)
                serialized[f"{key}_numpy_dtype"] = str(value.dtype)
                # Don't include the original numpy array
            elif value is None:
                serialized[key] = None
            else:
                serialized[key] = value
        return serialized
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Reconstruct result from dictionary loaded from Lance.
        
        This method handles deserialization of complex types and should be
        overridden by subclasses to provide proper reconstruction logic.
        
        Args:
            data: Dictionary loaded from Lance dataset
            
        Returns:
            Reconstructed result instance
        """
        # Deserialize numpy arrays
        deserialized_data = cls._deserialize_special_types(data)
        
        # Use Pydantic's model validation for reconstruction
        return cls(**deserialized_data)
    
    @classmethod
    def _deserialize_special_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle deserialization of special types like numpy arrays.
        
        Args:
            data: Dictionary from Lance with serialized values
            
        Returns:
            Dictionary with values deserialized to proper types
        """
        deserialized = {}
        numpy_fields = set()
        
        # Collect numpy array metadata
        for key in data:
            if key.endswith('_numpy_bytes'):
                field_name = key[:-12]  # Remove '_numpy_bytes' suffix
                numpy_fields.add(field_name)
        
        # Reconstruct numpy arrays
        for field_name in numpy_fields:
            bytes_key = f"{field_name}_numpy_bytes"
            shape_key = f"{field_name}_numpy_shape" 
            dtype_key = f"{field_name}_numpy_dtype"
            
            if bytes_key in data and data[bytes_key] is not None:
                buffer = BytesIO(data[bytes_key])
                array = np.load(buffer)
                
                # Reshape if shape info available
                if shape_key in data:
                    array = array.reshape(data[shape_key])
                
                # Cast to correct dtype if available
                if dtype_key in data:
                    array = array.astype(data[dtype_key])
                
                deserialized[field_name] = array
        
        # Copy non-numpy fields
        for key, value in data.items():
            if not any(key.endswith(suffix) for suffix in ['_numpy_bytes', '_numpy_shape', '_numpy_dtype']):
                # Only include if it's not a numpy field that we already reconstructed
                field_name = key
                if field_name not in numpy_fields:
                    deserialized[key] = value
        
        return deserialized
    
    @staticmethod
    def _serialize_image_data(frame_data: np.ndarray) -> tuple[bytes, list[int], str]:
        """
        Serialize image frame data for storage.
        
        Args:
            frame_data: Numpy array containing image data
            
        Returns:
            Tuple of (serialized_data, shape, dtype_string)
        """
        buffer = BytesIO()
        np.save(buffer, frame_data)
        serialized = buffer.getvalue()
        shape = list(frame_data.shape)
        dtype_str = str(frame_data.dtype)
        return serialized, shape, dtype_str
    
    @staticmethod  
    def _deserialize_image_data(data: bytes, shape: list[int], dtype_str: str) -> np.ndarray:
        """
        Deserialize image frame data from storage.
        
        Args:
            data: Serialized image data
            shape: Original array shape  
            dtype_str: Original array dtype as string
            
        Returns:
            Reconstructed numpy array
        """
        buffer = BytesIO(data)
        arr = np.load(buffer)
        return arr.reshape(shape).astype(dtype_str)
    
    def get_schema_name(self) -> str:
        """
        Get the schema name for this result type.
        
        Returns:
            String identifier for this schema type
        """
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        """String representation showing schema type and key fields."""
        return f"{self.get_schema_name()}({self._get_repr_fields()})"
    
    def _get_repr_fields(self) -> str:
        """
        Get key fields for repr display. Can be overridden by subclasses.
        
        Returns:
            String representation of key fields
        """
        # Show first few fields by default
        fields = []
        data = self.dict()
        for key, value in list(data.items())[:3]:  # First 3 fields
            if isinstance(value, (int, float, str, bool)) and value is not None:
                fields.append(f"{key}={value}")
        return ", ".join(fields)