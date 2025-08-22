"""Data models for Cortexia Video SDK"""

from .video import FrameData, VideoContent
from .detection import BoundingBox, DetectionResult
from .segmentation import SegmentationResult

__all__ = [
    "FrameData",
    "VideoContent", 
    "BoundingBox",
    "DetectionResult",
    "SegmentationResult",
]