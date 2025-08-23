"""
Cortexia Video Features Module

This module provides various feature implementations for video frame processing
including caption generation, object detection, depth estimation, feature extraction, etc.

Features are self-contained processors that add annotations and metadata to frames.

Example usage:

    # Using individual feature classes
    from cortexia_video.features import CaptionFeature
    caption_feature = CaptionFeature({'model': 'vikhyatk/moondream2'})
    
    # Using the factory
    from cortexia_video.features import FeatureFactory
    caption_feature = FeatureFactory.create('caption', {'model': 'vikhyatk/moondream2'})
    
    # Using convenience functions
    from cortexia_video.features import create_caption_feature
    caption_feature = create_caption_feature(model='vikhyatk/moondream2')
"""

# Import base class
from .base import BaseFeature

# Import all feature classes
try:
    from .caption.captioner import CaptionFeature
except ImportError:
    CaptionFeature = None

try:
    from .detection.detector import DetectionFeature
except ImportError:
    DetectionFeature = None

try:
    from .depth.estimator import DepthFeature
except ImportError:
    DepthFeature = None

try:
    from .feature_extraction.extractor import FeatureExtractionFeature
except ImportError:
    FeatureExtractionFeature = None

try:
    from .listing.lister import ListingFeature
except ImportError:
    ListingFeature = None

try:
    from .segmentation.segmenter import SegmentationFeature
except ImportError:
    SegmentationFeature = None

try:
    from .description.describer import DescriptionFeature
except ImportError:
    DescriptionFeature = None

# Import factory and convenience functions
from .factory import (
    FeatureFactory,
    register_all_features,
    create_caption_feature,
    create_detection_feature,
    create_depth_feature,
    create_feature_extraction_feature,
    create_listing_feature,
    create_segmentation_feature,
    create_description_feature,
)

# Register all available features when the module is imported
register_all_features()

# Define what gets exported when using "from cortexia_video.features import *"
__all__ = [
    # Base class
    'BaseFeature',
    
    # Factory
    'FeatureFactory',
    'register_all_features',
    
    # Convenience functions
    'create_caption_feature',
    'create_detection_feature',
    'create_depth_feature',
    'create_feature_extraction_feature',
    'create_listing_feature',
    'create_segmentation_feature',
    'create_description_feature',
]

# Add available feature classes to exports
if CaptionFeature:
    __all__.append('CaptionFeature')
if DetectionFeature:
    __all__.append('DetectionFeature')
if DepthFeature:
    __all__.append('DepthFeature')
if FeatureExtractionFeature:
    __all__.append('FeatureExtractionFeature')
if ListingFeature:
    __all__.append('ListingFeature')
if SegmentationFeature:
    __all__.append('SegmentationFeature')
if DescriptionFeature:
    __all__.append('DescriptionFeature')