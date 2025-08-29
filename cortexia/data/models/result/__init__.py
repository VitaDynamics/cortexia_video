"""Result models for all feature outputs"""

from .trajectory_result import TrajectoryResult
from .detection_result import DetectionResult, SingleDetection, BoundingBox
from .segmentation_result import SegmentationResult
from .caption_result import CaptionResult
from .depth_result import DepthResult
from .description_result import DescriptionResult
from .feature_extraction_result import FeatureExtractionResult
from .gate_result import GateResult
from .tagging_result import TaggingResult

__all__ = [
    "TrajectoryResult",
    "DetectionResult",
    "SingleDetection",
    "BoundingBox",
    "SegmentationResult",
    "CaptionResult",
    "DepthResult",
    "DescriptionResult",
    "FeatureExtractionResult",
    "GateResult",
    "TaggingResult",
]