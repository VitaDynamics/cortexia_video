"""Data models for Cortexia Video SDK"""

# Import registry to make it available
from .registry import schema_registry

# Import all result classes to trigger decorator registration
from .result.base_result import BaseResult
from .result.caption_result import CaptionResult
from .result.depth_result import DepthResult
from .result.description_result import DescriptionResult
from .result.detection_result import DetectionResult
from .result.feature_extraction_result import FeatureExtractionResult
from .result.gate_result import GateResult
from .result.segmentation_result import SegmentationResult
from .result.tagging_result import TaggingResult

# Make key classes available at package level
__all__ = [
    "schema_registry",
    "BaseResult",
    "CaptionResult", 
    "DepthResult",
    "DescriptionResult",
    "DetectionResult",
    "FeatureExtractionResult",
    "GateResult",
    "SegmentationResult",
    "TaggingResult",
]

