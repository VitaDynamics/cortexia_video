"""Video data models"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import datetime

import numpy as np
from pydantic import BaseModel, Field

from .detection import DetectionResult
from .segmentation import SegmentationResult


@dataclass
class VideoFramePacket:
    """
    Standardized data packet for a single video frame and its metadata.
    This structure will be used for passing frame data between samplers,
    gates, buffers, and other processing modules.
    """

    frame_data: np.ndarray  # The raw frame image data as a NumPy array (from decord)
    frame_number: (
        int  # Sequential frame number within the source video (0-indexed or 1-indexed)
    )
    timestamp: datetime.timedelta  # Timestamp of the frame relative to the video start
    source_video_id: str  # A unique identifier for the source video file or stream

    additional_metadata: Dict[str, Any] = field(default_factory=dict)

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

        # Compare all fields except frame_data which is a numpy array
        fields_equal = (
            self.frame_number == other.frame_number
            and self.timestamp == other.timestamp
            and self.source_video_id == other.source_video_id
            and self.additional_metadata == other.additional_metadata
        )

        # Compare the numpy arrays
        arrays_equal = np.array_equal(self.frame_data, other.frame_data)

        return fields_equal and arrays_equal


class AnnotationResults(BaseModel):
    """Container for all annotation results from features"""
    
    detections: list[DetectionResult] = Field(default_factory=list)
    segments: list[SegmentationResult] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    
    # Specific feature results
    lister_results: Optional[list[str]] = None
    dino_prompt: Optional[str] = None
    scene_clip_features: Optional[np.ndarray] = None
    caption: Optional[str] = None
    depth_map: Optional[np.ndarray] = None
    depth_statistics: Optional[Dict[str, float]] = None
    
    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Override dict method to handle numpy arrays"""
        data = super().dict(**kwargs)
        # Skip large data in serialization
        data["scene_clip_features"] = None
        data["depth_map"] = None
        return data


class GateResults(BaseModel):
    """Container for gate decision results and metadata"""
    
    passes: bool = True  # Whether frame passes all gates
    gate_decisions: Dict[str, bool] = Field(default_factory=dict)  # Individual gate results
    gate_scores: Dict[str, float] = Field(default_factory=dict)    # Gate confidence scores
    gate_metadata: Dict[str, Any] = Field(default_factory=dict)    # Gate-specific data
    
    def add_gate_result(self, gate_name: str, passes: bool, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add result from a specific gate"""
        self.gate_decisions[gate_name] = passes
        if score is not None:
            self.gate_scores[gate_name] = score
        if metadata:
            self.gate_metadata[gate_name] = metadata
        # Update overall pass/fail
        self.passes = self.passes and passes


class AnnotatedFramePacket(BaseModel):
    """Frame packet with annotation results - post-processing format"""
    
    base_frame: VideoFramePacket
    annotations: AnnotationResults = Field(default_factory=AnnotationResults)
    
    class Config:
        arbitrary_types_allowed = True
        
    @property
    def frame_number(self) -> int:
        return self.base_frame.frame_number
        
    @property
    def timestamp(self) -> datetime.timedelta:
        return self.base_frame.timestamp
        
    @property
    def frame_data(self) -> np.ndarray:
        return self.base_frame.frame_data


class TaggedFramePacket(BaseModel):
    """Frame packet with both annotations and gate results - complete processing format"""
    
    base_frame: VideoFramePacket
    annotations: AnnotationResults = Field(default_factory=AnnotationResults) 
    gate_results: GateResults = Field(default_factory=GateResults)
    
    class Config:
        arbitrary_types_allowed = True
        
    @property  
    def frame_number(self) -> int:
        return self.base_frame.frame_number
        
    @property
    def timestamp(self) -> datetime.timedelta:
        return self.base_frame.timestamp
        
    @property
    def frame_data(self) -> np.ndarray:
        return self.base_frame.frame_data
        
    @property
    def passes_gates(self) -> bool:
        """Check if frame passes all gate criteria"""
        return self.gate_results.passes


class VideoContent(BaseModel):
    """Complete video content structure - using new annotation format"""

    video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    frames: Dict[int, AnnotatedFramePacket] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_stats: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(cls, video_dict: Dict[str, Any]) -> "VideoContent":
        """Initialize from dictionary format"""
        frames = {}
        for k, v in video_dict.get("frames", {}).items():
            # Convert old FrameData format if needed
            if "base_frame" not in v:
                # Legacy conversion - create VideoFramePacket from old data
                base_frame = VideoFramePacket(
                    frame_data=v.get("rgb_image", np.array([])),
                    frame_number=v.get("frame_number", 0),
                    timestamp=datetime.timedelta(seconds=v.get("timestamp", 0.0)),
                    source_video_id=video_dict.get("video_path", ""),
                    additional_metadata={}
                )
                annotations = AnnotationResults(
                    detections=v.get("detections", []),
                    segments=v.get("segments", []),
                    features=v.get("features", {})
                )
                frames[int(k)] = AnnotatedFramePacket(
                    base_frame=base_frame,
                    annotations=annotations
                )
            else:
                frames[int(k)] = AnnotatedFramePacket(**v)
        
        return cls(
            video_path=video_dict["video_path"],
            total_frames=video_dict["total_frames"],
            fps=video_dict["fps"],
            width=video_dict.get("width", 0),
            height=video_dict.get("height", 0),
            frames=frames,
            metadata=video_dict.get("metadata", {}),
            processing_stats=video_dict.get("processing_stats", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frames": {str(k): v.dict() for k, v in self.frames.items()},
            "metadata": self.metadata,
            "processing_stats": self.processing_stats,
        }