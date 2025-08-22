"""Video data models"""

from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from .detection import DetectionResult
from .segmentation import SegmentationResult


class FrameData(BaseModel):
    """Enhanced frame data with detection and segmentation results"""

    frame_number: int
    timestamp: float
    rgb_image: Optional[np.ndarray] = None
    rgb_path: Optional[str] = None
    depth_image: Optional[np.ndarray] = None
    lister_results: Optional[list[str]] = None
    detections: list[DetectionResult] = Field(default_factory=list)
    segments: list[SegmentationResult] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    dino_prompt: Optional[str] = None

    # clip features
    scene_clip_features: Optional[np.ndarray] = None
    
    # image caption
    caption: Optional[str] = None
    
    # depth estimation results
    depth_map: Optional[np.ndarray] = None
    depth_statistics: Optional[Dict[str, float]] = None

    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Override dict method to handle numpy arrays"""
        data = super().dict(**kwargs)
        # Skip image data in serialization
        data["rgb_image"] = None
        data["depth_image"] = None
        data["depth_map"] = None
        return data


class VideoContent(BaseModel):
    """Complete video content structure with serialization support"""

    video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    frames: Dict[int, FrameData] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_stats: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(cls, video_dict: Dict[str, Any]) -> "VideoContent":
        """Initialize from dictionary format"""
        frames = {
            int(k): FrameData(**v) for k, v in video_dict.get("frames", {}).items()
        }
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