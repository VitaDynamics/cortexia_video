"""Base result schema for all feature and gate outputs.

This module provides the foundation class that all result schemas inherit from,
ensuring consistent serialization and deserialization with PyArrow struct support
for Lance integration.
"""

from abc import ABC
from typing import Any, Dict, Type, TypeVar, List, Optional, Union
from io import BytesIO
import json

import numpy as np
import pyarrow as pa
import pyarrow.types as pat



T = TypeVar('T', bound='BaseResult')


class BaseResult(ABC):
    """
    Base class for all feature and gate result schemas.
    
    Provides standardized dict serialization/deserialization that works
    with numpy arrays and other complex types, plus PyArrow struct support
    for Lance integration.
    """
        
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert result to dictionary representation.
        
        This method handles serialization of complex types like numpy arrays
        by converting them to bytes or appropriate serializable formats.
        
        Args:
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Dictionary representation
        """
        # Get all instance attributes
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                data[key] = value
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
        Reconstruct result from dictionary
        
        This method handles deserialization of complex types and should be
        overridden by subclasses to provide proper reconstruction logic.
        
        Args:
            data: Dictionary 
            
        Returns:
            Reconstructed result instance
        """
        # Deserialize numpy arrays
        deserialized_data = cls._deserialize_special_types(data)
        
        # Create instance directly without pydantic validation
        instance = cls.__new__(cls)
        for key, value in deserialized_data.items():
            setattr(instance, key, value)
        return instance
    
    @classmethod
    def _deserialize_special_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle deserialization of special types like numpy arrays.
        
        Args:
            data: Dictionary
            
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
    
    def to_pyarrow_struct(self) -> pa.StructArray:
        """
        Convert result to PyArrow struct array for Lance integration.
        
        Returns:
            PyArrow StructArray containing all result data
        """
        fields = []
        values = []
        
        # Get all instance attributes
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                field_name, field_value = self._convert_field_to_arrow(key, value)
                if field_name is not None:
                    fields.append(field_name)
                    values.append(field_value)
        
        # Create struct array from fields and values
        struct_array = pa.StructArray.from_arrays(
            arrays=values,
            names=fields
        )
        
        return struct_array
    
    def _convert_field_to_arrow(self, key: str, value: Any) -> tuple[Optional[str], Optional[pa.Array]]:
        """
        Convert a single field to PyArrow format.
        
        Args:
            key: Field name
            value: Field value
            
        Returns:
            Tuple of (field_name, arrow_array) or (None, None) if field should be skipped
        """
        if value is None:
            return key, pa.array([None], type=pa.null())
        
        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            # Convert numpy array to binary format for struct field
            buffer = BytesIO()
            np.save(buffer, value)
            serialized_data = buffer.getvalue()
            return f"{key}_numpy", pa.array([serialized_data], type=pa.binary())
        
        # Handle basic types
        elif isinstance(value, bool):
            return key, pa.array([value], type=pa.bool_())
        elif isinstance(value, int):
            return key, pa.array([value], type=pa.int64())
        elif isinstance(value, float):
            return key, pa.array([value], type=pa.float64())
        elif isinstance(value, str):
            return key, pa.array([value], type=pa.string())
        
        # Handle lists
        elif isinstance(value, list):
            if not value:
                return key, pa.array([], type=pa.list_(pa.string()))
            
            # Infer list type from first element
            first_elem = value[0]
            if isinstance(first_elem, str):
                return key, pa.array([value], type=pa.list_(pa.string()))
            elif isinstance(first_elem, int):
                return key, pa.array([value], type=pa.list_(pa.int64()))
            elif isinstance(first_elem, float):
                return key, pa.array([value], type=pa.list_(pa.float64()))
            elif isinstance(first_elem, bool):
                return key, pa.array([value], type=pa.list_(pa.bool_()))
            else:
                # For complex objects, convert to string representation
                return key, pa.array([[str(x) for x in value]], type=pa.list_(pa.string()))
        
        # Handle dictionaries
        elif isinstance(value, dict):
            # Convert dict to JSON string for simplicity
            json_str = json.dumps(value)
            return key, pa.array([json_str], type=pa.string())
        
        # Handle custom objects - convert to dict representation
        elif hasattr(value, '__dict__'):
            # Convert object to dict for serialization, then to JSON string
            obj_dict = value.__dict__
            json_str = json.dumps(obj_dict)
            return key, pa.array([json_str], type=pa.string())
        
        # Handle other objects - convert to string
        else:
            return key, pa.array([str(value)], type=pa.string())
    
    @classmethod
    def from_pyarrow_struct(cls: Type[T], struct_array: pa.StructArray) -> T:
        """
        Reconstruct result from PyArrow struct array.
        
        Args:
            struct_array: PyArrow StructArray containing result data
            
        Returns:
            Reconstructed result instance
        """
        if len(struct_array) == 0:
            return cls()
        
        # Get the first (and only) element from the struct array
        struct_data = struct_array[0]
        
        # Convert back to dictionary format
        data = {}
        for field_name in struct_array.type.names:
            field_value = struct_data[field_name]
            
            # Handle numpy arrays
            if field_name.endswith('_numpy'):
                original_field_name = field_name[:-6]  # Remove '_numpy' suffix
                if field_value is not None:
                    # Deserialize numpy array from binary data
                    binary_data = field_value.as_py()
                    buffer = BytesIO(binary_data)
                    array = np.load(buffer)
                    data[original_field_name] = array
                else:
                    data[original_field_name] = None
            else:
                # Handle basic types, lists, and structs
                if field_value is not None:
                    py_value = field_value.as_py()
                    
                    # Handle struct fields (for custom objects)
                    if isinstance(py_value, dict):
                        # This is a struct that represents a custom object
                        # We'll store it as a dict and let the specific from_dict method handle reconstruction
                        data[field_name] = py_value
                    elif isinstance(py_value, str):
                        # This might be a string representation of a dict
                        try:
                            # Try to parse as dict
                            import json
                            parsed_dict = json.loads(py_value)
                            data[field_name] = parsed_dict
                        except (json.JSONDecodeError, TypeError):
                            # Not a JSON string, treat as regular string
                            data[field_name] = py_value
                    else:
                        data[field_name] = py_value
                else:
                    data[field_name] = None
        
        return cls.from_dict(data)
    
    def get_pyarrow_schema(self) -> pa.StructType:
        """
        Get the PyArrow schema for this result type.
        
        Returns:
            PyArrow StructType representing the schema
        """
        fields = []
        
        # Get all instance attributes
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                field_type = self._get_arrow_field_type(key, value)
                if field_type is not None:
                    fields.append(pa.field(key, field_type))
        
        return pa.struct(fields)
    
    def _get_arrow_field_type(self, key: str, value: Any) -> Optional[pa.DataType]:
        """
        Get the PyArrow field type for a given field.
        
        Args:
            key: Field name
            value: Field value
            
        Returns:
            PyArrow DataType or None if field should be skipped
        """
        if value is None:
            return pa.null()
        
        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            return pa.binary()
        
        # Handle basic types
        elif isinstance(value, bool):
            return pa.bool_()
        elif isinstance(value, int):
            return pa.int64()
        elif isinstance(value, float):
            return pa.float64()
        elif isinstance(value, str):
            return pa.string()
        
        # Handle lists
        elif isinstance(value, list):
            if not value:
                return pa.list_(pa.string())
            
            # Infer list type from first element
            first_elem = value[0]
            if isinstance(first_elem, str):
                return pa.list_(pa.string())
            elif isinstance(first_elem, int):
                return pa.list_(pa.int64())
            elif isinstance(first_elem, float):
                return pa.list_(pa.float64())
            elif isinstance(first_elem, bool):
                return pa.list_(pa.bool_())
            else:
                return pa.list_(pa.string())
        
        # Handle dictionaries
        elif isinstance(value, dict):
            dict_fields = []
            for dict_key, dict_value in value.items():
                dict_field_type = self._get_arrow_field_type(dict_key, dict_value)
                if dict_field_type is not None:
                    dict_fields.append(pa.field(dict_key, dict_field_type))
            
            if dict_fields:
                return pa.struct(dict_fields)
            else:
                return None
        
        # Handle other objects - convert to string
        else:
            return pa.string()

    def __init__(self, **kwargs):
        """
        Initialize the result with provided fields.
        
        Args:
            **kwargs: Field values to set on the instance
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    # It shoud useful for type infer in python. 
    def __str__(self) -> str:
        """
        User-friendly string representation of the schema.
        
        Returns:
            Formatted string showing schema structure and values
        """
        return self._format_schema()
    
    def __call__(self) -> str:
        """
        Call method that returns the formatted schema representation.
        
        Returns:
            Formatted string showing schema structure and values
        """
        return self._format_schema()
    
    def _format_schema(self) -> str:
        """
        Format the schema in a user-friendly way.
        
        Returns:
            Formatted string showing schema structure and values
        """
        lines = []
        lines.append(f"Schema: {self.get_schema_name()}")
        lines.append("=" * (len(self.get_schema_name()) + 8))
        lines.append("")
        
        data = self.dict()
        if not data:
            lines.append("No fields")
            return "\n".join(lines)
        
        for key, value in data.items():
            if key.endswith('_numpy_bytes') or key.endswith('_numpy_shape') or key.endswith('_numpy_dtype'):
                continue
            
            value_str = self._format_value(value)
            lines.append(f"  {key}: {value_str}")
        
        return "\n".join(lines)
    
    def _format_value(self, value: Any) -> str:
        """
        Format a value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string representation
        """
        if value is None:
            return "None"
        elif isinstance(value, (int, float, str, bool)):
            return repr(value)
        elif isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            elif len(value) <= 3:
                return f"[{', '.join(repr(x) for x in value)}]"
            else:
                return f"[{', '.join(repr(x) for x in value[:3])}, ...] (len={len(value)})"
        elif isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            elif len(value) <= 3:
                items = [f"{k}: {repr(v)}" for k, v in list(value.items())[:3]]
                return f"{{{', '.join(items)}}}"
            else:
                items = [f"{k}: {repr(v)}" for k, v in list(value.items())[:3]]
                return f"{{{', '.join(items)}, ...}} (len={len(value)})"
        else:
            return str(value)