"""PyArrow/Lance adapter for cortexia video processing.

This module provides a clean interface to work with pyarrow Tables and Lance datasets,
allowing users to process video frames stored in columnar format while maintaining
the existing VideoFramePacket abstraction.
"""

from typing import List, Optional, Dict, Any, Union, Callable
import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
from pathlib import Path
import io

from ..models.video import VideoFramePacket
from ..models.result.base_result import BaseResult
from ..models.result.detection_result import DetectionResult
from ..models.result.caption_result import CaptionResult
from ..models.result.segmentation_result import SegmentationResult
from ..models.result.depth_result import DepthResult
from ..models.result.description_result import DescriptionResult
from ..models.result.feature_extraction_result import FeatureExtractionResult
from ..models.result.tagging_result import TaggingResult
from ..models.result.gate_result import GateResult
from .batch_processor import BatchProcessor


# TODO: Can of mapping should be in common.py or other mapping file. 
# Predefined column names for cortexia results
RESULT_COLUMN_MAPPING = {
    "detection": "cortexia_detection",
    "caption": "cortexia_caption", 
    "segmentation": "cortexia_segmentation",
    "depth": "cortexia_depth",
    "description": "cortexia_description",
    "feature_extraction": "cortexia_features",
    "tagging": "cortexia_tags",
    "gate": "cortexia_gate_results"
}

# Mapping from result class to column name
RESULT_TYPE_TO_COLUMN = {
    DetectionResult: RESULT_COLUMN_MAPPING["detection"],
    CaptionResult: RESULT_COLUMN_MAPPING["caption"],
    SegmentationResult: RESULT_COLUMN_MAPPING["segmentation"],
    DepthResult: RESULT_COLUMN_MAPPING["depth"],
    DescriptionResult: RESULT_COLUMN_MAPPING["description"],
    FeatureExtractionResult: RESULT_COLUMN_MAPPING["feature_extraction"],
    TaggingResult: RESULT_COLUMN_MAPPING["tagging"],
    GateResult: RESULT_COLUMN_MAPPING["gate"]
}


class PyArrowVideoAdapter:
    """Adapter for processing video data from pyarrow Tables/Lance datasets."""
    
    def __init__(
        self,
        table: Union[pa.Table, ds.Dataset],
        frame_column: str = "frame_data",
        video_id_column: str = "video_id",
        frame_number_column: str = "frame_number",
        timestamp_column: Optional[str] = "timestamp",
        metadata_columns: Optional[List[str]] = None,
        result_columns: Optional[Dict[str, type]] = None
    ):
        """
        Initialize adapter for pyarrow/lance dataset.
        
        Args:
            table: PyArrow Table or Lance Dataset
            frame_column: Column name containing frame data (bytes or numpy arrays)
            video_id_column: Column name for video identifiers
            frame_number_column: Column name for frame numbers in one clip
            timestamp_column: Column name for timestamps (optional)
            metadata_columns: Additional columns to include as metadata
            result_columns: Mapping of column names to result types for existing results
        """
        self.table = table
        self.frame_column = frame_column
        self.video_id_column = video_id_column
        self.frame_number_column = frame_number_column
        self.timestamp_column = timestamp_column
        self.metadata_columns = metadata_columns or []
        self.result_columns = result_columns or {}
        
        # Validate columns exist
        self._validate_columns()
    
    def _validate_columns(self):
        """Validate that specified columns exist in the table."""
        if isinstance(self.table, ds.Dataset):
            schema = self.table.schema
        else:
            schema = self.table.schema
            
        required_cols = [
            self.frame_column, 
            self.video_id_column, 
            self.frame_number_column
        ]
        if self.timestamp_column:
            required_cols.append(self.timestamp_column)
            
        missing_cols = [col for col in required_cols if col not in schema.names]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def to_frame_packets(
        self,
        filter_expr: Optional[Union[str, ds.Expression]] = None,
        batch_size: Optional[int] = None
    ) -> List[VideoFramePacket]:
        """
        Convert pyarrow data to VideoFramePacket objects.
        
        Args:
            filter_expr: Optional filter expression for rows
            batch_size: Optional batch size for processing large datasets
            
        Returns:
            List of VideoFramePacket objects
        """
        if isinstance(self.table, ds.Dataset):
            if filter_expr:
                scanner = self.table.scan(filter=filter_expr)
            else:
                scanner = self.table.scan()
            
            batches = scanner.to_batches(batch_size=batch_size)
            table = pa.concat_tables(batches)
        else:
            table = self.table
            if filter_expr:
                # For regular tables, apply filter manually
                if isinstance(filter_expr, str):
                    table = table.filter(pa.dataset.field(filter_expr))
                else:
                    table = table.filter(filter_expr)
        
        return self._table_to_packets(table)
    
    def _table_to_packets(self, table: pa.Table) -> List[VideoFramePacket]:
        """Convert pyarrow table to VideoFramePacket objects."""
        packets = []
        
        for i in range(len(table)):
            row = table.slice(i, 1)
            
            # Extract frame data
            frame_data = self._extract_frame_data(row[self.frame_column][0])
            
            # Extract basic fields
            video_id = str(row[self.video_id_column][0])
            frame_number = int(row[self.frame_number_column][0])
            
            # Handle timestamp
            if self.timestamp_column and self.timestamp_column in row.column_names:
                timestamp_val = row[self.timestamp_column][0]
                if isinstance(timestamp_val, (int, float)):
                    timestamp = np.timedelta64(int(timestamp_val), 'ms')
                else:
                    timestamp = np.timedelta64(0, 'ms')
            else:
                timestamp = np.timedelta64(0, 'ms')
            
            # Build metadata
            metadata = {}
            for col in self.metadata_columns:
                if col in row.column_names:
                    metadata[col] = row[col][0]
            
            # Build annotations from result columns
            annotations = {}
            for col, result_type in self.result_columns.items():
                if col in row.column_names:
                    value = row[col][0]
                    if value is not None:
                        if isinstance(value, dict):
                            # Deserialize from dict
                            annotations[col] = result_type.from_dict(value)
                        else:
                            annotations[col] = value
            
            packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=frame_number,
                timestamp=timestamp,
                source_video_id=video_id,
                additional_metadata=metadata,
                annotations=annotations if annotations else None
            )
            
            packets.append(packet)
        
        return packets
    
    def _extract_frame_data(self, frame_value) -> np.ndarray:
        """Extract numpy array from frame data column."""
        if isinstance(frame_value, bytes):
            # Assume it's a serialized numpy array
            return np.load(io.BytesIO(frame_value))
        elif isinstance(frame_value, np.ndarray):
            return frame_value
        elif hasattr(frame_value, 'as_py'):
            # Handle pyarrow Array scalars
            return np.array(frame_value.as_py())
        else:
            return np.array(frame_value)
    
    def _get_output_column_name(self, feature) -> str:
        """Auto-detect output column name based on feature type."""
        feature_name = getattr(feature, 'name', None)
        if feature_name and feature_name in RESULT_COLUMN_MAPPING:
            return RESULT_COLUMN_MAPPING[feature_name]
        
        # Try to infer from output schema
        output_schema = getattr(feature, 'output_schema', None)
        if output_schema and output_schema in RESULT_TYPE_TO_COLUMN:
            return RESULT_TYPE_TO_COLUMN[output_schema]
        
        # Fallback to generic name
        return f"cortexia_{feature_name or 'results'}"
    
    def process_with_feature(
        self,
        feature_class,
        input_columns: Optional[Dict[str, str]] = None,
        output_column: Optional[str] = None,
        batch_size: int = 4,
        filter_expr: Optional[Union[str, ds.Expression]] = None
    ) -> pa.Table:
        """
        Process frames using a cortexia feature and return results as pyarrow table.
        
        Args:
            feature_class: Cortexia feature class (e.g., DetectionFeature)
            input_columns: Mapping of feature input names to column names
            output_column: Column name for results (auto-detected if None)
            batch_size: Batch size for processing
            filter_expr: Optional filter for rows to process
            
        Returns:
            PyArrow table with original data plus results
        """
        # Get frame packets
        packets = self.to_frame_packets(filter_expr)
        
        if not packets:
            return self.table
        
        # Initialize feature
        feature = feature_class()
        
        # Auto-detect output column name
        if output_column is None:
            output_column = self._get_output_column_name(feature)
        
        # Process in batches
        results = []
        for i in range(0, len(packets), batch_size):
            batch = packets[i:i+batch_size]
            
            # Prepare inputs
            inputs = {}
            if input_columns:
                for input_name, col_name in input_columns.items():
                    inputs[input_name] = [p.additional_metadata.get(col_name) for p in batch]
            
            # Process batch
            batch_results = feature.process_batch(batch, **inputs)
            results.extend(batch_results)
        
        # Convert results to pyarrow format
        result_dicts = [r.dict() if hasattr(r, 'dict') else r for r in results]
        
        # Add results to table
        if isinstance(self.table, ds.Dataset):
            # For datasets, we need to work with the underlying table
            table = self.table.to_table()
        else:
            table = self.table
            
        # Filter table to match processed packets
        if filter_expr:
            filtered_table = table.filter(filter_expr)
        else:
            filtered_table = table
        
        # Add results column
        result_array = pa.array(result_dicts)
        return filtered_table.append_column(output_column, result_array)
    
    def create_batch_processor(
        self,
        feature_class,
        input_columns: Optional[Dict[str, str]] = None,
        output_column: Optional[str] = None
    ) -> BatchProcessor:
        """
        Create a BatchProcessor configured for this dataset.
        
        Args:
            feature_class: Cortexia feature class
            input_columns: Mapping of input names to column names
            output_column: Column name for results (auto-detected if None)
            
        Returns:
            Configured BatchProcessor instance
        """
        # Auto-detect output column name
        if output_column is None:
            feature = feature_class()
            output_column = self._get_output_column_name(feature)
        
        # Create indices (row indices for pyarrow table)
        indices = list(range(len(self.table)))
        
        processor = BatchProcessor()
        processor.load_indices(indices)
        
        return processor


class LanceVideoDataset:
    """High-level interface for working with Lance datasets containing video data."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize Lance dataset interface.
        
        Args:
            dataset_path: Path to Lance dataset
        """
        self.dataset_path = Path(dataset_path)
        self.dataset = lance.dataset(str(self.dataset_path))
    
    def create_adapter(
        self,
        frame_column: str = "frame_data",
        video_id_column: str = "video_id",
        frame_number_column: str = "frame_number",
        timestamp_column: Optional[str] = "timestamp",
        metadata_columns: Optional[List[str]] = None,
        result_columns: Optional[Dict[str, type]] = None
    ) -> PyArrowVideoAdapter:
        """
        Create PyArrowVideoAdapter for this dataset.
        
        Args:
            frame_column: Column name for frame data
            video_id_column: Column name for video IDs
            frame_number_column: Column name for frame numbers
            timestamp_column: Column name for timestamps
            metadata_columns: Additional metadata columns
            result_columns: Mapping of result column names to types
            
        Returns:
            PyArrowVideoAdapter instance
        """
        return PyArrowVideoAdapter(
            table=self.dataset,
            frame_column=frame_column,
            video_id_column=video_id_column,
            frame_number_column=frame_number_column,
            timestamp_column=timestamp_column,
            metadata_columns=metadata_columns,
            result_columns=result_columns
        )
    
    def write_results(
        self,
        results: List[Dict[str, Any]],
        result_column: str = "cortexia_results",
        merge_strategy: str = "append"
    ):
        """
        Write processing results back to the Lance dataset.
        
        Args:
            results: List of result dictionaries
            result_column: Column name for results
            merge_strategy: How to merge with existing data ("append", "overwrite", "merge")
        """
        # Create pyarrow table from results
        result_table = pa.Table.from_pylist(results)
        
        # Write back to lance
        if merge_strategy == "append":
            self.dataset.merge(result_table, left_on="row_id", right_on="row_id")
        elif merge_strategy == "overwrite":
            # Create new version with updated column
            updated_dataset = self.dataset.merge_columns({result_column: result_table[result_column]})
            updated_dataset.write(str(self.dataset_path))
        elif merge_strategy == "merge":
            # Merge specific columns
            self.dataset.merge(result_table, left_on="row_id", right_on="row_id")


def from_pyarrow_table(
    table: pa.Table,
    frame_column: str = "frame_data",
    video_id_column: str = "video_id",
    frame_number_column: str = "frame_number",
    timestamp_column: Optional[str] = "timestamp",
    metadata_columns: Optional[List[str]] = None,
    result_columns: Optional[Dict[str, type]] = None
) -> PyArrowVideoAdapter:
    """
    Convenience function to create PyArrowVideoAdapter from pyarrow Table.
    
    Args:
        table: PyArrow table
        frame_column: Column name for frame data
        video_id_column: Column name for video IDs
        frame_number_column: Column name for frame numbers
        timestamp_column: Column name for timestamps
        metadata_columns: Additional metadata columns
        result_columns: Mapping of result column names to types
        
    Returns:
        PyArrowVideoAdapter instance
    """
    return PyArrowVideoAdapter(
        table=table,
        frame_column=frame_column,
        video_id_column=video_id_column,
        frame_number_column=frame_number_column,
        timestamp_column=timestamp_column,
        metadata_columns=metadata_columns,
        result_columns=result_columns
    )


def from_lance_dataset(
    dataset_path: str,
    frame_column: str = "frame_data",
    video_id_column: str = "video_id",
    frame_number_column: str = "frame_number",
    timestamp_column: Optional[str] = "timestamp",
    metadata_columns: Optional[List[str]] = None,
    result_columns: Optional[Dict[str, type]] = None
) -> PyArrowVideoAdapter:
    """
    Convenience function to create PyArrowVideoAdapter from Lance dataset.
    
    Args:
        dataset_path: Path to Lance dataset
        frame_column: Column name for frame data
        video_id_column: Column name for video IDs
        frame_number_column: Column name for frame numbers
        timestamp_column: Column name for timestamps
        metadata_columns: Additional metadata columns
        result_columns: Mapping of result column names to types
        
    Returns:
        PyArrowVideoAdapter instance
    """
    import lance
    dataset = lance.dataset(dataset_path)
    
    return PyArrowVideoAdapter(
        table=dataset,
        frame_column=frame_column,
        video_id_column=video_id_column,
        frame_number_column=frame_number_column,
        timestamp_column=timestamp_column,
        metadata_columns=metadata_columns,
        result_columns=result_columns
    )