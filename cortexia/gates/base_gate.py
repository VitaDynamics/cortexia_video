from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Type, TypeVar, Generic, Union, Callable

from ..data.models.video import VideoFramePacket
from ..data.models.result.base_result import BaseResult
from ..data.models.result.gate_result import GateResult
from ..data.models.field_validation import FrameField, validate_frame_requirements

# Type variable for return type of process_frame
T = TypeVar('T', bound=BaseResult)


class BaseGate(Generic[T], ABC):
    """
    Abstract base class for all gate implementations.

    Gates are calculators: they compute per-frame measurements or features
    (e.g., blur score, hash, signature, embeddings) and do not make pass/fail
    decisions. Thresholding or comparison with stored context is delegated to
    downstream buffers/policies.
    """
    
    # Class attributes that subclasses should define
    output_schema: Type[BaseResult] = None  # Must be defined by subclasses
    required_inputs: List[str] = []  # List of required input schema names
    required_fields: List[Union[str, FrameField]] = []  # List of required VideoFramePacket fields
    
    def __init__(self):
        """Initialize the gate with I/O capabilities."""
        super().__init__() 
        
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
    
    def validate_frame_inputs(self, frame: VideoFramePacket) -> None:
        """
        Validate that all required fields are present in the frame packet.
        
        Args:
            frame: Frame to validate VideoFramePacket
            
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
    def process_frame(self, frame: VideoFramePacket, **inputs) -> T:
        """
        Compute a measurement/feature from a single frame.
        
        Args:
            frame: Frame to process VideoFramePacket
            **inputs: Additional inputs keyed by schema name (e.g., CaptionResult="caption_result")
            
        Returns:
            BaseResult instance (typically GateResult) containing metric/feature and metadata
        """
        pass
    
    def process_batch(self, frames: List[VideoFramePacket], **batch_inputs) -> List[T]:
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
    
    @staticmethod
    def compare(a, b) -> bool:
        """
        Comparable method for results
        """
        return a < b
