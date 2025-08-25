"""
Cortexia Video Features Module

This module provides various feature implementations for video frame processing
including caption generation, object detection, depth estimation, feature extraction, etc.

Features are self-contained processors that add annotations and metadata to frames.

"""

from .registry import feature_registry

# Auto-import all features
import os
import pkgutil
import importlib


# Need to think about remove this mapping for auto import
def _auto_import_features():
    """Automatically import all feature modules to trigger decorator registration"""
    current_dir = os.path.dirname(__file__)
    imported_features = []
    
    # Feature directory to module mapping
    feature_modules = {
        'caption': 'captioner',
        'depth': 'estimator', 
        'description': 'describer',
        'detection': 'detector',
        'listing': 'lister',
        'segmentation': 'segmenter',
        'feature_extraction': 'extractor'
    }
    
    # Import each feature module
    for feature_dir, module_name in feature_modules.items():
        item_path = os.path.join(current_dir, feature_dir)
        if os.path.isdir(item_path):
            try:
                # Import the module
                module = importlib.import_module(f".{feature_dir}.{module_name}", package=__name__)
                
                # Look for classes that end with 'Feature' but exclude 'BaseFeature'
                for attr_name in dir(module):
                    if attr_name.endswith('Feature') and attr_name != 'BaseFeature':
                        feature_class = getattr(module, attr_name)
                        if hasattr(feature_class, '__name__'):
                            imported_features.append(feature_class)
            except (ImportError, AttributeError) as e:
                # Skip modules that can't be imported
                continue
    
    return imported_features

# Auto-import all features
imported_feature_classes = _auto_import_features()

# Make feature classes available at module level
for feature_class in imported_feature_classes:
    globals()[feature_class.__name__] = feature_class

# Make key classes available at package level
__all__ = ["feature_registry"] + [cls.__name__ for cls in imported_feature_classes]

