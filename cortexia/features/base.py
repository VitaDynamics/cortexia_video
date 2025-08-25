"""Base class for all annotation features"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ..api.exceptions import CortexiaError, ProcessingError
from ..data.models.video import VideoFramePacket
from ..data.models.result.base_result import BaseResult
from ..data.models.field_validation import FrameField, validate_frame_requirements
from ..core.registry import Registry
from .registry import feature_registry


@feature_registry.register("base")
class BaseFeature(ABC):
    """
    Base class for all independent annotation features.
    
    Each feature is self-contained and can be used independently
    without requiring other features or pipelines. Enhanced with schema-based
    I/O capabilities for modular dataset integration.
    """
    
    # Class attributes that subclasses should define
    output_schema: Type[BaseResult] = None  # The result schema this feature produces
    required_inputs: List[str] = []  # List of required input schema names (e.g., ["TaggingResult"])
    required_fields: List[Union[str, FrameField]] = []  # List of required VideoFramePacket fields
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature.
        
        Args:
            config: Feature-specific configuration dictionary
        """
        super().__init__()  
        self.config = config or {}
        self.initialized = False
        self._initialize()
        
        # Validate that subclass defined required attributes
        if self.__class__.output_schema is None:
            raise ValueError(f"{self.__class__.__name__} must define output_schema class attribute")
    
    def get_output_schema(self) -> Type[BaseResult]:
        """Get the output schema class for this feature."""
        return self.__class__.output_schema
    
    def get_required_fields(self) -> List[Union[str, FrameField]]:
        """Get the list of required VideoFramePacket fields."""
        return self.__class__.required_fields.copy()
    
    def validate_frame_inputs(self, frame: VideoFramePacket) -> None:
        """
        Validate that all required fields are present in the frame packet.
        
        Args:
            frame: VideoFramePacket to validate
            
        Raises:
            ProcessingError: If required fields are missing or empty
        """
        if self.required_fields:
            validate_frame_requirements(frame, self.required_fields, self.name)
    
    def validate_inputs(self, **inputs) -> None:
        """
        Validate that all required inputs are provided.
        
        Args:
            **inputs: Input data with schema names as keys
            
        Raises:
            ProcessingError: If required inputs are missing
        """
        missing_inputs = []
        for required_input in self.required_inputs:
            if required_input not in inputs:
                missing_inputs.append(required_input)
        
        if missing_inputs:
            raise ProcessingError(
                f"Feature {self.name} missing required inputs: {missing_inputs}. "
                f"Provided: {list(inputs.keys())}"
            )
    
    @abstractmethod
    def _initialize(self):
        """
        Initialize the feature with models and resources.
        
        This method is called during initialization and should:
        - Load models
        - Set up resources
        - Validate configuration
        - Set self.initialized = True on success
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: VideoFramePacket, **inputs) -> BaseResult:
        """
        Process a single frame with optional additional inputs.
        
        Args:
            frame: Video frame packet to process
            **inputs: Additional inputs keyed by schema name (e.g., TaggingResult="tagging_result")
            
        Returns:
            BaseResult instance of the feature's output_schema type
        """
        pass
    
    def process_batch(self, frames: List[VideoFramePacket], **batch_inputs) -> List[BaseResult]:
        """
        Process multiple frames with optional batch inputs.
        
        Default implementation processes frames one by one.
        Override for batch-specific optimizations.
        
        Args:
            frames: List of video frame packets to process
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
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Feature name for registry identification.
        
        Returns:
            Feature name string
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of the feature.
        
        Returns:
            Feature description
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate feature-specific configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            CortexiaError: If configuration is invalid
        """
        return True
    
    def get_config_param(self, key: str, default: Any = None) -> Any:
        """
        Get configuration parameter with default value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        current = self.config
        
        try:
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    return default
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_config_param(self, key: str, value: Any):
        """
        Set configuration parameter.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        current = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set value
        current[keys[-1]] = value
    
    def is_ready(self) -> bool:
        """
        Check if feature is ready for processing.
        
        Returns:
            True if feature is initialized and ready
        """
        return self.initialized

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.initialized})"
