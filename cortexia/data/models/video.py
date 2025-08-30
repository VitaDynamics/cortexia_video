"""Video data models"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
import datetime
from .result.tagging_result import TaggingResult
from io import BytesIO
import math

import numpy as np
from pydantic import BaseModel, Field

from .result.detection_result import DetectionResult
from .result.segmentation_result import SegmentationResult
from .result.caption_result import CaptionResult
from .result.depth_result import DepthResult
from .result.description_result import DescriptionResult
from .result.feature_extraction_result import FeatureExtractionResult
from .result.gate_result import GateResult
from .result.trajectory_result import TrajectoryResult



    
# TODO: should support all optional of result in result folder. 
# TODO: we also need to support actions (future action seuquence in build on this frame. Or as a Windows)
# TODO: How does action look like? Numpy array is a good start point. 


@dataclass
class TrajectoryPoint:
    """
    Represents a single point in the trajectory with state information.
    
    This class stores position, orientation, and state classification for
    a single trajectory point, supporting both 2D and 3D trajectory analysis.
    """
    
    x: float
    y: float
    z: float = 0.0  # Optional z-coordinate for 3D support
    qx: float = 0.0  # Quaternion x component
    qy: float = 0.0  # Quaternion y component
    qz: float = 0.0  # Quaternion z component
    qw: float = 1.0  # Quaternion w component
    yaw: Optional[float] = None  # Calculated yaw angle from quaternion
    state: Optional[str] = None  # Classification state
    future_yaw_diff: Optional[float] = None  # Orientation difference with future point
    velocity: Optional[float] = None  # Velocity magnitude
    
    def __post_init__(self):
        """Calculate yaw angle from quaternion if not provided."""
        if self.yaw is None:
            self.calculate_yaw()

    def calculate_yaw(self):
        """Calculate yaw angle from quaternion."""
        # Convert quaternion to yaw angle (rotation around z-axis)
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        self.yaw = math.atan2(2 * (self.qw * self.qz + self.qx * self.qy), 
                                1 - 2 * (self.qy**2 + self.qz**2))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'qx': self.qx,
            'qy': self.qy,
            'qz': self.qz,
            'qw': self.qw,
            'yaw': self.yaw,
            'state': self.state,
            'future_yaw_diff': self.future_yaw_diff,
            'velocity': self.velocity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryPoint':
        """Reconstruct TrajectoryPoint from dictionary."""
        return cls(**data)

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
    # Action Related 
    trajectory: List[TrajectoryPoint]
    current_traj_index: int # Indicate the current frame's position in the trajectory list.

    # Core metadata
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Extended fields for unified processing 
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
        )

        # Compare the numpy arrays
        arrays_equal = np.array_equal(self.frame_data, other.frame_data)

        return fields_equal and arrays_equal
    
    # Convenience methods for unified data flow
    
    def add_annotation_result(self, result_data: Union[CaptionResult, DetectionResult, SegmentationResult, DepthResult, DescriptionResult, FeatureExtractionResult, TaggingResult, GateResult, TrajectoryResult]) -> None:
        """
        Add annotation result to frame using schema classes.
        
        Args:
            result_data: The annotation result object (one of the result schema classes)
        """
        
        # Handle None result_data
        if result_data is None:
            return
            
        result_type = result_data.__class__.__name__
        
        # Initialize annotations dict if None
        if self.annotations is None:
            self.annotations = {}
        
        self.annotations[result_type] = result_data
    
    def has_annotations(self) -> bool:
        """Check if frame has any annotation results."""
        return self.annotations is not None
    
    # TODO: this dict is not good
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert frame packet to dictionary for serialization.
        
        This method handles basic type conversion for serialization,
        keeping numpy arrays as-is for further processing.
        
        Args:
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Dictionary representation with basic types
        """
        return {
            'frame_data': self.frame_data,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.total_seconds(),
            'source_video_id': self.source_video_id,
            'additional_metadata': self.additional_metadata.copy(),
            'annotations': self.annotations,
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize frame packet to dictionary with binary data for storage.
        
        This method handles serialization of complex types like numpy arrays
        by converting them to bytes for database or network storage.
        
        Returns:
            Dictionary representation with serialized binary data
        """
        data = self.dict()
        return self._serialize_special_types(data)
    
    def _serialize_special_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle serialization of special types like numpy arrays.
        
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
            data: Dictionary with frame data
            
        Returns:
            Reconstructed VideoFramePacket instance
        """
        # Convert timestamp back to timedelta
        if 'timestamp' in data:
            data['timestamp'] = datetime.timedelta(seconds=data['timestamp'])
        
        return cls(**data)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'VideoFramePacket':
        """
        Reconstruct VideoFramePacket from serialized dictionary.
        
        Args:
            data: Dictionary with serialized binary data
            
        Returns:
            Reconstructed VideoFramePacket instance
        """
        deserialized_data = cls._deserialize_special_types(data)
        return cls.from_dict(deserialized_data)
    
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
    