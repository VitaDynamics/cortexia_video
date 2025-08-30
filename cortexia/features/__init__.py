"""
Cortexia Video Features Module

This module provides various feature implementations for video frame processing
including caption generation, object detection, depth estimation, feature extraction, etc.

Features are self-contained processors that add annotations and metadata to frames.

"""

from .registry import feature_registry

# Import all feature modules manually
from .caption.captioner import CaptionFeature
from .depth.estimator import DepthFeature  
from .description.describer import DescriptionFeature
from .detection.detector import DetectionFeature
from .listing.lister import ListingFeature
from .segmentation.segmenter import SegmentationFeature
from .feature_extraction.extractor import FeatureExtractionFeature
from .trajectory.analyzer import TrajectoryFeature

# Make key classes available at package level
__all__ = [
    "feature_registry",
    "CaptionFeature", 
    "DepthFeature",
    "DescriptionFeature", 
    "DetectionFeature",
    "ListingFeature",
    "SegmentationFeature", 
    "FeatureExtractionFeature",
    "TrajectoryFeature"
]