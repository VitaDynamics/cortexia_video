"""Main Cortexia Video SDK API

This module provides a unified interface to access all features and gates
from the cortexia package through registries.
"""

from .cortexia_sdk import Cortexia
from ..features import feature_registry
from ..gates import gate_registry
from ..data.io.batch_processor import BatchProcessor
from typing import Any, Callable, List, Optional, Union
from ..data.models.video import VideoFramePacket
from ..data.models.result.base_result import BaseResult
from ..features.base import BaseFeature
from ..gates.base_gate import BaseGate

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
    "process_batch_with_processor",
]


def get_feature(name: str):
    """Get a feature class by name from the feature registry.
    
    Args:
        name: The feature name (e.g., "caption", "detection", "depth")
        
    Returns:
        The feature class
        
    Raises:
        KeyError: If the feature is not found
    """
    return feature_registry.require(name)


def get_gate(name: str):
    """Get a gate class by name from the gate registry.
    
    Args:
        name: The gate name (e.g., "blur", "entropy", "clip")
        
    Returns:
        The gate class
        
    Raises:
        KeyError: If the gate is not found
    """
    return gate_registry.require(name)


def list_features():
    """List all available feature names.
    
    Returns:
        List of feature names
    """
    return list(feature_registry.keys())


def list_gates():
    """List all available gate names.
    
    Returns:
        List of gate names
    """
    return list(gate_registry.keys())


def create_feature(name: str, **kwargs):
    """Create a feature instance by name.
    
    Args:
        name: The feature name
        **kwargs: Arguments to pass to the feature constructor
        
    Returns:
        Feature instance
    """
    feature_class = get_feature(name)
    return feature_class(**kwargs)


def create_gate(name: str, **kwargs):
    """Create a gate instance by name.
    
    Args:
        name: The gate name
        **kwargs: Arguments to pass to the gate constructor
        
    Returns:
        Gate instance
    """
    gate_class = get_gate(name)
    return gate_class(**kwargs)


def process_batch_with_processor(
    indices: List[Any],
    load_func: Callable[[List[Any]], List[VideoFramePacket]],
    processor: Union[BaseFeature, BaseGate],
    batch_size: int = 4,
    filter_func: Optional[Callable[[Any], bool]] = None,
) -> List[BaseResult]:
    """
    Process a batch of video frames using the BatchProcessor with a feature or gate.
    
    Args:
        indices: List of indices (file paths, database IDs, etc.) to process
        load_func: Function that takes indices and returns List[VideoFramePacket]
        processor: BaseFeature or BaseGate instance to use for processing
        batch_size: Number of items to process in each batch
        filter_func: Optional function to filter which indices to process
        
    Returns:
        List of BaseResult instances from processing
    """
    
    def inference_func(frames: List[VideoFramePacket], batch_indices: List[Any]) -> List[BaseResult]:
        """Inference function that uses the processor's process_batch method."""
        return processor.process_batch(frames)
    
    # Create and configure batch processor
    batch_processor = BatchProcessor(batch_size=batch_size)
    batch_processor.load_indices(indices)
    
    # Process batches without save_func - results will be returned
    results = batch_processor.process_batch(
        load_func=load_func,
        inference_func=inference_func,
        save_func=None,  # None to collect results instead of saving
        filter_func=filter_func
    )
    
    return results