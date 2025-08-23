"""Generic Lance mixin for any BaseResult schema.

This module provides a mixin that can work with any schema inheriting from
BaseResult, enabling consistent Lance dataset operations across all components.
"""

from abc import ABC
from pathlib import Path
from typing import Iterator, List, Optional, Type, TypeVar, Union

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from .generic_lance_adapter import GenericLanceAdapter, VideoFrameLanceAdapter
from ..models.base_result import BaseResult
from ..models.video import VideoFramePacket

T = TypeVar('T', bound=BaseResult)


class GenericLanceMixin(ABC):
    """
    Generic mixin providing Lance dataset I/O for any BaseResult schema.
    
    This mixin can be inherited by features and gates to add standardized
    Lance dataset operations that work with any result schema type.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter = GenericLanceAdapter()
        self._frame_adapter = VideoFrameLanceAdapter()
    
    def _ensure_lance_available(self):
        """Check if Lance is available and raise error if not."""
        if not LANCE_AVAILABLE:
            raise ImportError(
                "Lance is not available. Install with: pip install pylance"
            )
    
    def save_results_to_lance(
        self,
        results: List[BaseResult],
        dataset_path: Union[str, Path],
        mode: str = "append"
    ) -> None:
        """
        Save any BaseResult instances to Lance dataset.
        
        Args:
            results: List of BaseResult instances (any schema type)
            dataset_path: Path to Lance dataset
            mode: Save mode - "append", "overwrite", or "create"
        """
        self._ensure_lance_available()
        
        if not results:
            return
        
        # Convert results to PyArrow table
        table = self._adapter.batch_results_to_lance_table(results)
        
        # Save to Lance dataset
        if mode == "create" or not Path(dataset_path).exists():
            lance.write_dataset(table, str(dataset_path))
        elif mode == "append":
            lance.write_dataset(table, str(dataset_path), mode="append")
        elif mode == "overwrite":
            lance.write_dataset(table, str(dataset_path), mode="overwrite")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def load_results_from_lance(
        self,
        dataset_path: Union[str, Path],
        result_class: Type[T],
        filter_expr: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Iterator[T]:
        """
        Load BaseResult instances from Lance dataset.
        
        Args:
            dataset_path: Path to Lance dataset
            result_class: The specific BaseResult subclass to load
            filter_expr: SQL-like filter expression for Lance
            limit: Maximum number of results to load
            
        Yields:
            BaseResult instances of the specified type
        """
        self._ensure_lance_available()
        
        dataset = lance.dataset(str(dataset_path))
        
        # Build query
        query = dataset.to_table()
        
        if filter_expr:
            query = query.filter(pa.compute.expression(filter_expr))
        
        if limit:
            query = query.limit(limit)
        
        # Convert table to results
        results = self._adapter.lance_table_to_results(query, result_class)
        
        for result in results:
            yield result
    
    def batch_load_results_from_lance(
        self,
        dataset_path: Union[str, Path],
        result_class: Type[T],
        batch_size: int = 32,
        filter_expr: Optional[str] = None
    ) -> Iterator[List[T]]:
        """
        Load results from Lance dataset in batches.
        
        Args:
            dataset_path: Path to Lance dataset
            result_class: The specific BaseResult subclass to load
            batch_size: Number of results per batch
            filter_expr: SQL-like filter expression
            
        Yields:
            Batches of BaseResult instances
        """
        batch = []
        for result in self.load_results_from_lance(dataset_path, result_class, filter_expr):
            batch.append(result)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining results
        if batch:
            yield batch
    
    def update_results_in_lance(
        self,
        results: List[BaseResult],
        dataset_path: Union[str, Path],
        key_field: str = "id"
    ) -> None:
        """
        Update existing results in Lance dataset.
        
        Note: This is a simplified implementation. In production, you'd want
        proper upsert operations.
        
        Args:
            results: List of BaseResult instances to update
            dataset_path: Path to Lance dataset
            key_field: Field to use as the update key
        """
        self._ensure_lance_available()
        
        if not results:
            return
        
        # For now, we'll do delete + append
        # In production, you'd want proper upsert logic
        dataset = lance.dataset(str(dataset_path))
        
        # Extract keys to delete
        keys_to_delete = []
        for result in results:
            record = result.dict()
            if key_field in record:
                keys_to_delete.append(record[key_field])
        
        # Delete existing records (simplified)
        for key in keys_to_delete:
            try:
                dataset.delete(f"{key_field} = '{key}'")
            except:
                pass  # Key might not exist, that's okay
        
        # Append new records
        self.save_results_to_lance(results, dataset_path, mode="append")
    
    def save_video_frames_to_lance(
        self,
        frames: List[VideoFramePacket],
        dataset_path: Union[str, Path],
        mode: str = "append"
    ) -> None:
        """
        Save VideoFramePacket instances to Lance dataset.
        
        Args:
            frames: List of video frame packets
            dataset_path: Path to Lance dataset
            mode: Save mode
        """
        self._ensure_lance_available()
        
        if not frames:
            return
        
        # Convert frames to records
        records = [self._frame_adapter.video_frame_to_lance_record(frame) for frame in frames]
        
        # Create table
        schema = self._frame_adapter.get_video_frame_schema()
        table = pa.Table.from_pylist(records, schema=schema)
        
        # Save to Lance
        if mode == "create" or not Path(dataset_path).exists():
            lance.write_dataset(table, str(dataset_path))
        elif mode == "append":
            lance.write_dataset(table, str(dataset_path), mode="append")
        elif mode == "overwrite":
            lance.write_dataset(table, str(dataset_path), mode="overwrite")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def load_video_frames_from_lance(
        self,
        dataset_path: Union[str, Path],
        filter_expr: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Iterator[VideoFramePacket]:
        """
        Load VideoFramePacket instances from Lance dataset.
        
        Args:
            dataset_path: Path to Lance dataset
            filter_expr: SQL-like filter expression
            limit: Maximum number of frames to load
            
        Yields:
            VideoFramePacket instances
        """
        self._ensure_lance_available()
        
        dataset = lance.dataset(str(dataset_path))
        
        query = dataset.to_table()
        
        if filter_expr:
            query = query.filter(pa.compute.expression(filter_expr))
        
        if limit:
            query = query.limit(limit)
        
        # Convert records to VideoFramePackets
        for record in query.to_pylist():
            yield self._frame_adapter.lance_record_to_video_frame(record)
    
    def get_dataset_info(self, dataset_path: Union[str, Path]) -> dict:
        """
        Get information about Lance dataset.
        
        Args:
            dataset_path: Path to Lance dataset
            
        Returns:
            Dictionary containing dataset information
        """
        self._ensure_lance_available()
        
        try:
            dataset = lance.dataset(str(dataset_path))
            schema = dataset.schema
            
            # Get basic stats
            table = dataset.to_table()
            num_rows = len(table)
            
            # Check for schema metadata
            schema_types = set()
            if "_schema_name" in [field.name for field in schema]:
                schema_names = table["_schema_name"].to_pylist()
                schema_types = set(name for name in schema_names if name)
            
            return {
                "path": str(dataset_path),
                "schema": schema,
                "num_rows": num_rows,
                "columns": [field.name for field in schema],
                "schema_types": list(schema_types),
                "version": dataset.version,
            }
        except Exception as e:
            return {
                "path": str(dataset_path),
                "error": str(e),
                "exists": False,
            }
    
    def filter_results_by_schema(
        self,
        dataset_path: Union[str, Path],
        schema_name: str,
        result_class: Type[T]
    ) -> Iterator[T]:
        """
        Load only results of a specific schema type.
        
        Args:
            dataset_path: Path to Lance dataset
            schema_name: Name of schema to filter by
            result_class: BaseResult subclass to reconstruct
            
        Yields:
            BaseResult instances of the specified schema type
        """
        filter_expr = f"_schema_name = '{schema_name}'"
        return self.load_results_from_lance(dataset_path, result_class, filter_expr)


# Backwards compatibility - alias to the old name
LanceMixin = GenericLanceMixin