"""
Cortexia Video - Video Annotation Framework

A modular, pipeline-free SDK for video annotation with independent features.
"""

from .api.cortexia import (
    Cortexia,
    feature_registry,
    gate_registry,
    get_feature,
    get_gate,
    list_features,
    list_gates,
    create_feature,
    create_gate,
)
from .api.exceptions import CortexiaError, ModelLoadError, ProcessingError

__version__ = "0.1.0"
__all__ = [
    "Cortexia",
    "feature_registry",
    "gate_registry",
    "get_feature",
    "get_gate",
    "list_features",
    "list_gates",
    "create_feature",
    "create_gate",
    "CortexiaError", 
    "ModelLoadError",
    "ProcessingError",
]