"""Video data models"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import datetime
from io import BytesIO

import numpy as np
from pydantic import BaseModel, Field

from .result.detection_result import DetectionResult
from .result.segmentation_result import SegmentationResult
from .result.caption_result import CaptionResult
from .result.depth_result import DepthResult
from .result.description_result import DescriptionResult
from .result.feature_extraction_result import FeatureExtractionResult
from .result.gate_result import GateResult
from .result.tagging_result import TaggingResult
from .registry import schema_registry


    
# TODO: should support all optional of result in result folder. 
# TODO: we also need to support actions (future action seuquence in build on this frame. Or as a Windows)
# TODO: How does action look like? Numpy array is a good start point. 

@schema_registry.register("video.frame_packet")
@dataclass
class VideoFramePacket:
    """
    Standardized data packet for a single video frame and its metadata.
    This structure will be used for passing frame data between samplers,
    gates, buffers, and other processing modules.
    
    Extended to support unified data flow between features and gates with
    optional fields for computed embeddings, annotations, gate results,
    and derived metadata.
    """

    frame_data: np.ndarray  # The raw frame image data as a NumPy array (from decord)
    frame_number: (
        int  # Sequential frame number within the source video (0-indexed or 1-indexed)
    )
    timestamp: datetime.timedelta  # Timestamp of the frame relative to the video start
    source_video_id: str  # A unique identifier for the source video file or stream

    # Core metadata
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Extended fields for unified processing
    # TODO: We need to support Action.  
    annotations: Optional[Dict[str, Any]] = None  # Results from feature processing

    def __post_init__(self):
        if not isinstance(self.frame_data, np.ndarray):
            raise TypeError(
                f"frame_data must be a NumPy array, got {type(self.frame_data)}"
            )
        if not (
            self.frame_data.ndim == 3 and self.frame_data.shape[2] in [1, 3, 4]
        ) and not (self.frame_data.ndim == 2):  # (H, W, C) or (H, W)
            raise ValueError(
                "frame_data must be a 2D (grayscale) or 3D (e.g., RGB, RGBA) NumPy array. "
                f"Received shape: {self.frame_data.shape}"
            )
        if not isinstance(self.frame_number, int) or self.frame_number < 0:
            raise ValueError(
                f"frame_number must be a non-negative integer, got {self.frame_number}"
            )
        if not isinstance(self.timestamp, datetime.timedelta):
            raise TypeError(
                f"timestamp must be a datetime.timedelta, got {type(self.timestamp)}"
            )
        if (
            not isinstance(self.source_video_id, str)
            or not self.source_video_id.strip()
        ):
            raise ValueError(
                f"source_video_id must be a non-empty string, got '{self.source_video_id}'"
            )
        if not isinstance(self.additional_metadata, dict):
            raise TypeError(
                f"additional_metadata must be a dict, got {type(self.additional_metadata)}"
            )
        

    def __eq__(self, other):
        """
        Compare two VideoFramePacket objects for equality.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if objects are equal, False otherwise
        """
        if not isinstance(other, VideoFramePacket):
            return False

        # Compare all fields except frame_data which contains numpy arrays
        fields_equal = (
            self.frame_number == other.frame_number
            and self.timestamp == other.timestamp
            and self.source_video_id == other.source_video_id
            and self.additional_metadata == other.additional_metadata
            and self.annotations == other.annotations
            and self.gate_results == other.gate_results
        )

        # Compare the numpy arrays
        arrays_equal = np.array_equal(self.frame_data, other.frame_data)

        return fields_equal and arrays_equal
    
    # Convenience methods for unified data flow
    
    def add_annotation_result(self, result_data: Union[CaptionResult, DetectionResult, SegmentationResult, DepthResult, DescriptionResult, FeatureExtractionResult, TaggingResult, GateResult]) -> None:
        """
        Add annotation result to frame using schema classes.
        
        Args:
            result_data: The annotation result object (one of the result schema classes)
        """
        
        result_type = result_data.__class__.__name__
        self.annotations.result_type = result_data
    
    def has_annotations(self) -> bool:
        """Check if frame has any annotation results."""
        return self.annotations is not None
    
    # TODO: this dict is not good
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert frame packet to dictionary for serialization with actual frame data.
        
        This method handles serialization of complex types like numpy arrays
        by converting them to bytes or appropriate serializable formats,
        similar to BaseResult.dict() method.
        
        Args:
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Dictionary representation with serialized frame data
        """
        data = {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.total_seconds(),
            'source_video_id': self.source_video_id,
            'additional_metadata': self.additional_metadata.copy(),
            'annotations': self.annotations,
        }
        
        # Serialize numpy arrays using the same pattern as BaseResult
        serialized_data = self._serialize_special_types(data)
        return serialized_data
    
    def _serialize_special_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle serialization of special types like numpy arrays.
        
        Args:
            data: Dictionary with potentially unserializable values
            
        Returns:
            Dictionary with all values serialized appropriately
        """
        serialized = {}
        
        # Handle frame_data separately since it's not in the data dict
        if isinstance(self.frame_data, np.ndarray):
            buffer = BytesIO()
            np.save(buffer, self.frame_data)
            serialized['frame_data_numpy_bytes'] = buffer.getvalue()
            serialized['frame_data_numpy_shape'] = list(self.frame_data.shape)
            serialized['frame_data_numpy_dtype'] = str(self.frame_data.dtype)
        
        # Handle other fields in the data dict
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Serialize numpy arrays to bytes
                buffer = BytesIO()
                np.save(buffer, value)
                serialized[f"{key}_numpy_bytes"] = buffer.getvalue()
                serialized[f"{key}_numpy_shape"] = list(value.shape)
                serialized[f"{key}_numpy_dtype"] = str(value.dtype)
            elif value is None:
                serialized[key] = None
            else:
                serialized[key] = value
        return serialized
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoFramePacket':
        """
        Reconstruct VideoFramePacket from dictionary.
        
        Args:
            data: Dictionary with serialized frame data
            
        Returns:
            Reconstructed VideoFramePacket instance
        """
        deserialized_data = cls._deserialize_special_types(data)
        
        # Convert timestamp back to timedelta
        if 'timestamp' in deserialized_data:
            deserialized_data['timestamp'] = datetime.timedelta(seconds=deserialized_data['timestamp'])
        
        return cls(**deserialized_data)
    
    @classmethod
    def _deserialize_special_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle deserialization of special types like numpy arrays.
        
        Args:
            data: Dictionary with serialized values
            
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
    