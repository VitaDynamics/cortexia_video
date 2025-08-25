from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Type, TypeVar, Generic, Union, Callable
import math
from typing import Any

import numpy as np

from ..data.models.video import VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket, GateResults
from ..data.models.result.base_result import BaseResult
from ..data.models.result.gate_result import GateResult
from ..data.models.field_validation import FrameField, validate_frame_requirements
from ..data.models.registry import get_schema
from ..data.io.generic_lance_mixin import GenericLanceMixin

# Type variable for return type of process_frame
T = TypeVar('T', bound=BaseResult)

# Input types for gates - can handle raw, annotated, or tagged frames  
FrameInput = Union[VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket]


class BaseGate(GenericLanceMixin, Generic[T], ABC):
    """
    Abstract base class for all gate implementations.

    Gates are calculators: they compute per-frame measurements or features
    (e.g., blur score, hash, signature, embeddings) and do not make pass/fail
    decisions. Thresholding or comparison with stored context is delegated to
    downstream buffers/policies.
    """
    
    # Class attributes that subclasses should define
    output_schema: Type[BaseResult] = GateResult  # Default to GateResult
    required_inputs: List[str] = []  # List of required input schema names
    required_fields: List[Union[str, FrameField]] = []  # List of required VideoFramePacket fields
    
    def __init__(self):
        """Initialize the gate with I/O capabilities."""
        super().__init__()  # Initialize GenericLanceMixin
        
        # Validate that subclass defined required attributes appropriately
        if self.__class__.output_schema is None:
            raise ValueError(f"{self.__class__.__name__} must define output_schema class attribute")
    
    def get_output_schema(self) -> Type[BaseResult]:
        """Get the output schema class for this gate."""
        return self.__class__.output_schema
    
    def get_required_inputs(self) -> List[str]:
        """Get the list of required input schema names."""
        return self.__class__.required_inputs.copy()
    
    def get_required_fields(self) -> List[Union[str, FrameField]]:
        """Get the list of required VideoFramePacket fields."""
        return self.__class__.required_fields.copy()
    
    def validate_frame_inputs(self, frame: FrameInput) -> None:
        """
        Validate that all required fields are present in the frame packet.
        
        Args:
            frame: Frame to validate (VideoFramePacket, AnnotatedFramePacket, or TaggedFramePacket)
            
        Raises:
            ProcessingError: If required fields are missing or empty
        """
        if self.required_fields:
            # For tagged/annotated frames, validate against base frame
            base_frame = getattr(frame, 'base_frame', frame)
            validate_frame_requirements(base_frame, self.required_fields, self.name)
    
    def validate_inputs(self, **inputs) -> None:
        """
        Validate that all required inputs are provided.
        
        Args:
            **inputs: Input data with schema names as keys
            
        Raises:
            ProcessingError: If required inputs are missing
        """
        from ..api.exceptions import ProcessingError
        
        missing_inputs = []
        for required_input in self.required_inputs:
            if required_input not in inputs:
                missing_inputs.append(required_input)
        
        if missing_inputs:
            raise ProcessingError(
                f"Gate {self.name} missing required inputs: {missing_inputs}. "
                f"Provided: {list(inputs.keys())}"
            )
    
    @property
    def name(self) -> str:
        """Gate name for identification in results"""
        return self.__class__.__name__
    
    @abstractmethod
    def process_frame(self, frame: FrameInput, **inputs) -> T:
        """
        Compute a measurement/feature from a single frame.
        
        Args:
            frame: Frame to process (VideoFramePacket, AnnotatedFramePacket, or TaggedFramePacket)
            **inputs: Additional inputs keyed by schema name (e.g., CaptionResult="caption_result")
            
        Returns:
            BaseResult instance (typically GateResult) containing metric/feature and metadata
        """
        pass
    
    def process_batch(self, frames: List[FrameInput], **batch_inputs) -> List[T]:
        """
        Process multiple frames through the gate.
        
        Default implementation processes frames one by one.
        Override for batch-specific optimizations.
        
        Args:
            frames: List of frames to process
            **batch_inputs: Batch inputs keyed by schema name
            
        Returns:
            List of BaseResult instances
        """
        # Validate inputs once for the batch
        self.validate_inputs(**batch_inputs)
        
        results = []
        for frame in frames:
            # Validate frame fields for each frame
            self.validate_frame_inputs(frame)
            result = self.process_frame(frame, **batch_inputs)
            results.append(result)
        return results
    
    def process_from_dataset(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        input_datasets: Optional[Dict[str, Union[str, Path]]] = None,
        filter_expr: Optional[str] = None,
        batch_size: int = 32,
        save_mode: str = "append"
    ) -> None:
        """
        Process frames from dataset through gate to produce gate results.
        
        Args:
            input_dataset_path: Path to video frames dataset
            output_dataset_path: Path to save gate results dataset
            input_datasets: Dict mapping input schema names to dataset paths
            filter_expr: Filter expression for frame selection
            batch_size: Number of frames to process in each batch
            save_mode: Save mode for results ("append", "overwrite", "create")
        """
        from ..api.exceptions import ProcessingError
        
        # Load additional input data if required
        input_data = {}
        if self.required_inputs and input_datasets:
            for input_name in self.required_inputs:
                if input_name not in input_datasets:
                    raise ProcessingError(f"Required input dataset '{input_name}' not provided")
                
                # Load input data
                input_schema_class = self._resolve_schema_class(input_name)
                input_data[input_name] = list(self.load_results_from_lance(
                    input_datasets[input_name], 
                    input_schema_class
                ))
        
        # Process frames in batches
        batch = []
        results = []
        
        for frame in self.load_video_frames_from_lance(input_dataset_path, filter_expr):
            batch.append(frame)
            
            if len(batch) >= batch_size:
                try:
                    # Process batch
                    batch_results = self.process_batch(batch, **input_data)
                    results.extend(batch_results)
                    
                    # Save results periodically
                    if len(results) >= batch_size:
                        self.save_results_to_lance(results, output_dataset_path, "append" if results else save_mode)
                        results = []
                        
                except Exception as e:
                    print(f"Error processing batch through gate {self.name}: {e}")
                finally:
                    batch = []
        
        # Process remaining frames
        if batch:
            try:
                batch_results = self.process_batch(batch, **input_data)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing final batch through gate {self.name}: {e}")
        
        # Save remaining results
        if results:
            self.save_results_to_lance(results, output_dataset_path, "append" if Path(output_dataset_path).exists() else save_mode)
    
    def _resolve_schema_class(self, schema_name: str) -> Type[BaseResult]:
        """
        Resolve schema name to schema class using centralized registry.
        
        Args:
            schema_name: Name of the schema to resolve
            
        Returns:
            The schema class
            
        Raises:
            ValueError: If schema name is not registered
        """
        return get_schema(schema_name)
    
    def filter_dataset(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        predicate: Callable[[T], bool],
        save_mode: str = "create",
    ) -> Dict[str, int]:
        """
        Filter a results dataset using a predicate over result objects.

        Because gates no longer decide pass/fail, callers must provide a
        predicate that maps each result to True/False (e.g., thresholding).

        Args:
            input_dataset_path: Path to gate results dataset
            output_dataset_path: Path to filtered output dataset
            predicate: Callable that returns True to keep a result
            save_mode: Save mode for results ("append", "overwrite", "create")

        Returns:
            Dictionary with filtering statistics
        """
        total = 0
        kept = 0
        dropped = 0
        out = []

        for result in self.load_results_from_lance(input_dataset_path, self.output_schema):
            total += 1
            try:
                keep = bool(predicate(result))
            except Exception:
                keep = False
            if keep:
                kept += 1
                out.append(result)
            else:
                dropped += 1

        if out:
            self.save_results_to_lance(out, output_dataset_path, save_mode)

        return {"total": total, "kept": kept, "dropped": dropped}
    
    
# -----------------------------
# Comparator registry (decorator based)
# -----------------------------

_COMPARATOR_REGISTRY: Dict[str, Callable[[Any, Any], float]] = {}


def register_comparator(name: Optional[str] = None) -> Callable[[Callable[[Any, Any], float]], Callable[[Any, Any], float]]:
    """Decorator to register a comparator function by name.

    Usage:
        @register_comparator("cosine")
        def cosine(a, b): ...
    """

    def _decorator(fn: Callable[[Any, Any], float]) -> Callable[[Any, Any], float]:
        key = name or fn.__name__
        _COMPARATOR_REGISTRY[key] = fn
        return fn

    return _decorator


def get_comparator(name: str) -> Callable[[Any, Any], float]:
    if name not in _COMPARATOR_REGISTRY:
        raise KeyError(f"Comparator '{name}' not registered")
    return _COMPARATOR_REGISTRY[name]


# Built-in comparators (kept for backward compatibility)

@register_comparator("cosine")
def _cosine_similarity(a: Any, b: Any) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@register_comparator("hamming_hex")
def _hamming_distance_hex(a: str, b: str) -> float:
    if len(a) != len(b):
        return float(max(len(a), len(b)))
    return float(sum(1 for i in range(len(a)) if a[i] != b[i]))


@register_comparator("abs")
def _abs_difference(a: float, b: float) -> float:
    return float(abs(a - b))
