"""Data models for Cortexia Video SDK"""

from .video import FrameData, VideoContent
from .detection_result import BoundingBox, DetectionResult
from .segmentation_result import SegmentationResult

__all__ = [
    "FrameData",
    "VideoContent", 
    "BoundingBox",
    "DetectionResult",
    "SegmentationResult",
]