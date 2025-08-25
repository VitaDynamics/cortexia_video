"""
Cortexia Video - Video Annotation Framework

A modular, pipeline-free SDK for video annotation with independent features.
"""

from .api.cortexia import Cortexia
from .api.exceptions import CortexiaError, ModelLoadError, ProcessingError

__version__ = "0.1.0"
__all__ = [
    "Cortexia",
    "CortexiaError", 
    "ModelLoadError",
    "ProcessingError",
]