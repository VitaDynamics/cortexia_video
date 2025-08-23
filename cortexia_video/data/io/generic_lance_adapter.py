"""Generic Lance dataset adapter for BaseResult schemas.

This module provides a unified adapter that works with any schema inheriting
from BaseResult, enabling consistent Lance dataset operations across all
feature and gate results.
"""

import json
from typing import Any, Dict, List, Type, TypeVar

import pyarrow as pa

from ..models.base_result import BaseResult

T = TypeVar('T', bound=BaseResult)


class GenericLanceAdapter:
    """
    Generic adapter for converting BaseResult schemas to/from Lance datasets.
    
    This adapter works with any schema that inherits from BaseResult,
    providing a unified approach for all feature and gate results.
    """
    
    @staticmethod
    def result_to_lance_record(result: BaseResult) -> Dict[str, Any]:
        """
        Convert any BaseResult to Lance-compatible record.
        
        Args:
            result: Any BaseResult instance
            
        Returns:
            Dictionary suitable for Lance storage
        """
        # Use the result's own dict() method which handles serialization
        record = result.dict()
        
        # Add schema metadata
        record["_schema_name"] = result.get_schema_name()
        record["_schema_version"] = "1.0"  # For future schema evolution
        
        return record
    
    @staticmethod
    def lance_record_to_result(record: Dict[str, Any], result_class: Type[T]) -> T:
        """
        Convert Lance record back to BaseResult instance.
        
        Args:
            record: Dictionary from Lance dataset
            result_class: The specific BaseResult subclass to reconstruct
            
        Returns:
            Reconstructed result instance
        """
        # Remove metadata fields
        clean_record = {k: v for k, v in record.items() 
                       if not k.startswith("_schema")}
        
        # Use the class's from_dict method
        return result_class.from_dict(clean_record)
    
    @staticmethod 
    def infer_schema_from_result(result: BaseResult) -> pa.Schema:
        """
        Dynamically infer PyArrow schema from a BaseResult instance.
        
        Args:
            result: BaseResult instance to analyze
            
        Returns:
            PyArrow schema suitable for Lance dataset
        """
        record = result.dict()
        schema_fields = []
        
        # Add schema metadata fields
        schema_fields.append(pa.field("_schema_name", pa.string()))
        schema_fields.append(pa.field("_schema_version", pa.string()))
        
        # Infer fields from the record
        for key, value in record.items():
            if value is None:
                # For None values, use string as safe default
                schema_fields.append(pa.field(key, pa.string()))
            elif isinstance(value, bool):
                schema_fields.append(pa.field(key, pa.bool_()))
            elif isinstance(value, int):
                schema_fields.append(pa.field(key, pa.int64()))
            elif isinstance(value, float):
                schema_fields.append(pa.field(key, pa.float64()))
            elif isinstance(value, str):
                schema_fields.append(pa.field(key, pa.string()))
            elif isinstance(value, bytes):
                schema_fields.append(pa.field(key, pa.binary()))
            elif isinstance(value, list):
                if value and isinstance(value[0], (int, float)):
                    # List of numbers
                    if isinstance(value[0], int):
                        schema_fields.append(pa.field(key, pa.list_(pa.int64())))
                    else:
                        schema_fields.append(pa.field(key, pa.list_(pa.float64())))
                else:
                    # List of strings or mixed - use string list
                    schema_fields.append(pa.field(key, pa.list_(pa.string())))
            else:
                # Default to string for complex types
                schema_fields.append(pa.field(key, pa.string()))
        
        return pa.schema(schema_fields)
    
    @staticmethod
    def get_unified_schema_for_results(results: List[BaseResult]) -> pa.Schema:
        """
        Create a unified schema that can accommodate all provided results.
        
        This is useful when saving mixed result types to the same dataset.
        
        Args:
            results: List of BaseResult instances to analyze
            
        Returns:
            Unified PyArrow schema
        """
        if not results:
            raise ValueError("Cannot create schema from empty results list")
        
        # Start with schema from first result
        unified_fields = {}
        
        for result in results:
            result_schema = GenericLanceAdapter.infer_schema_from_result(result)
            
            # Merge fields, using most permissive type when conflicts occur
            for field in result_schema:
                if field.name in unified_fields:
                    # If types differ, use string as safe common type
                    existing_type = unified_fields[field.name].type
                    if existing_type != field.type:
                        unified_fields[field.name] = pa.field(field.name, pa.string())
                else:
                    unified_fields[field.name] = field
        
        return pa.schema(list(unified_fields.values()))
    
    @staticmethod
    def batch_results_to_lance_table(results: List[BaseResult]) -> pa.Table:
        """
        Convert a batch of BaseResult instances to PyArrow Table.
        
        Args:
            results: List of BaseResult instances
            
        Returns:
            PyArrow Table ready for Lance storage
        """
        if not results:
            raise ValueError("Cannot create table from empty results list")
        
        # Convert all results to records
        records = [GenericLanceAdapter.result_to_lance_record(result) for result in results]
        
        # Create unified schema
        schema = GenericLanceAdapter.get_unified_schema_for_results(results)
        
        # Create table
        return pa.Table.from_pylist(records, schema=schema)
    
    @staticmethod
    def lance_table_to_results(table: pa.Table, result_class: Type[T]) -> List[T]:
        """
        Convert PyArrow Table back to list of BaseResult instances.
        
        Args:
            table: PyArrow Table from Lance dataset
            result_class: The specific BaseResult subclass to reconstruct
            
        Returns:
            List of reconstructed result instances
        """
        records = table.to_pylist()
        return [GenericLanceAdapter.lance_record_to_result(record, result_class) 
                for record in records]


class VideoFrameLanceAdapter:
    """
    Specialized adapter for VideoFramePacket and related frame structures.
    
    This handles the core video data that doesn't fit the BaseResult pattern.
    """
    
    @staticmethod
    def get_video_frame_schema() -> pa.Schema:
        """Get PyArrow schema for VideoFramePacket data."""
        return pa.schema([
            pa.field("frame_number", pa.int64()),
            pa.field("timestamp_seconds", pa.float64()),
            pa.field("source_video_id", pa.string()),
            pa.field("frame_data_numpy_bytes", pa.binary()),
            pa.field("frame_data_numpy_shape", pa.list_(pa.int32())),
            pa.field("frame_data_numpy_dtype", pa.string()),
            pa.field("additional_metadata", pa.string()),
        ])
    
    @staticmethod
    def video_frame_to_lance_record(packet) -> Dict[str, Any]:
        """Convert VideoFramePacket to Lance record."""
        from ..models.base_result import BaseResult
        
        # Serialize frame data using BaseResult's method
        frame_data_bytes, frame_shape, frame_dtype = BaseResult._serialize_image_data(packet.frame_data)
        
        return {
            "frame_number": packet.frame_number,
            "timestamp_seconds": packet.timestamp.total_seconds(),
            "source_video_id": packet.source_video_id,
            "frame_data_numpy_bytes": frame_data_bytes,
            "frame_data_numpy_shape": frame_shape,
            "frame_data_numpy_dtype": frame_dtype,
            "additional_metadata": json.dumps(packet.additional_metadata),
        }
    
    @staticmethod
    def lance_record_to_video_frame(record: Dict[str, Any]):
        """Convert Lance record to VideoFramePacket."""
        import datetime
        from ..models.video import VideoFramePacket
        from ..models.base_result import BaseResult
        
        # Deserialize frame data
        frame_data = BaseResult._deserialize_image_data(
            record["frame_data_numpy_bytes"],
            record["frame_data_numpy_shape"],
            record["frame_data_numpy_dtype"]
        )
        
        timestamp = datetime.timedelta(seconds=record["timestamp_seconds"])
        additional_metadata = json.loads(record["additional_metadata"]) if record["additional_metadata"] else {}
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=record["frame_number"],
            timestamp=timestamp,
            source_video_id=record["source_video_id"],
            additional_metadata=additional_metadata,
        )