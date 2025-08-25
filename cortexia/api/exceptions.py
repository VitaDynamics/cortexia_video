"""Cortexia Video SDK exceptions"""


class CortexiaError(Exception):
    """Base exception for Cortexia SDK"""
    pass


class ModelLoadError(CortexiaError):
    """Failed to load model"""
    pass


class ProcessingError(CortexiaError):
    """Failed during processing"""
    pass


class ConfigurationError(CortexiaError):
    """Configuration error"""
    pass


class FeatureNotFoundError(CortexiaError):
    """Requested feature not found"""
    pass