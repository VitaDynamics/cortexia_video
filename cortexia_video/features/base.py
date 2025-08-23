"""Base class for all annotation features"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ..api.exceptions import CortexiaError, ProcessingError
from ..data.models.video import VideoFramePacket
from ..data.models.base_result import BaseResult
from ..data.io.generic_lance_mixin import GenericLanceMixin


class BaseFeature(GenericLanceMixin, ABC):
    """
    Base class for all independent annotation features.
    
    Each feature is self-contained and can be used independently
    without requiring other features or pipelines. Enhanced with schema-based
    I/O capabilities for modular dataset integration.
    """
    
    # Class attributes that subclasses should define
    output_schema: Type[BaseResult] = None  # The result schema this feature produces
    required_inputs: List[str] = []  # List of required input schema names (e.g., ["TaggingResult"])
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature.
        
        Args:
            config: Feature-specific configuration dictionary
        """
        super().__init__()  # Initialize GenericLanceMixin
        self.config = config or {}
        self.initialized = False
        self._initialize()
        
        # Validate that subclass defined required attributes
        if self.__class__.output_schema is None:
            raise ValueError(f"{self.__class__.__name__} must define output_schema class attribute")
    
    def get_output_schema(self) -> Type[BaseResult]:
        """Get the output schema class for this feature."""
        return self.__class__.output_schema
    
    def get_required_inputs(self) -> List[str]:
        """Get the list of required input schema names."""
        return self.__class__.required_inputs.copy()
    
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
    
    # Schema-based I/O Methods
    
    def process_from_dataset(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        input_datasets: Optional[Dict[str, Union[str, Path]]] = None,
        filter_expr: Optional[str] = None,
        batch_size: int = 4,
        save_mode: str = "append"
    ) -> None:
        """
        Process video frames from dataset with optional additional input datasets.
        
        Args:
            input_dataset_path: Path to video frames dataset
            output_dataset_path: Path to save results dataset
            input_datasets: Dict mapping input schema names to dataset paths
            filter_expr: Filter expression for frame selection
            batch_size: Number of frames to process in each batch
            save_mode: Save mode for results ("append", "overwrite", "create")
        """
        if not self.is_ready():
            raise ProcessingError(f"Feature {self.name} not initialized")
        
        # Load additional input data if required
        input_data = {}
        if self.required_inputs and input_datasets:
            for input_name in self.required_inputs:
                if input_name not in input_datasets:
                    raise ProcessingError(f"Required input dataset '{input_name}' not provided")
                
                # For now, we'll load all input data into memory
                # In production, you'd want streaming/batch loading
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
                    print(f"Error processing batch: {e}")
                finally:
                    batch = []
        
        # Process remaining frames
        if batch:
            try:
                batch_results = self.process_batch(batch, **input_data)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing final batch: {e}")
        
        # Save remaining results
        if results:
            self.save_results_to_lance(results, output_dataset_path, "append" if Path(output_dataset_path).exists() else save_mode)
    
    def _resolve_schema_class(self, schema_name: str) -> Type[BaseResult]:
        """
        Resolve schema name to schema class.
        
        This is a simplified implementation. In production, you'd want
        a proper registry system.
        """
        schema_mapping = {
            "CaptionResult": "caption_result.CaptionResult",
            "TaggingResult": "tagging_result.TaggingResult", 
            "DetectionResult": "detection.DetectionResult",
            "SegmentationResult": "segmentation.SegmentationResult",
            "DepthResult": "depth_result.DepthResult",
            "FeatureExtractionResult": "feature_extraction_result.FeatureExtractionResult",
            "DescriptionResult": "description_result.DescriptionResult",
            "GateResult": "gate_result.GateResult",
        }
        
        if schema_name not in schema_mapping:
            raise ValueError(f"Unknown schema name: {schema_name}")
        
        # Import and return the class
        module_path, class_name = schema_mapping[schema_name].rsplit(".", 1)
        module = __import__(f"..data.models.{module_path}", fromlist=[class_name])
        return getattr(module, class_name)
    
    def batch_process_iterator(
        self,
        frame_iterator: Iterator[VideoFramePacket],
        batch_size: int = 4,
        **inputs
    ) -> Iterator[List[BaseResult]]:
        """
        Process frames from an iterator in batches.
        
        Args:
            frame_iterator: Iterator of video frame packets
            batch_size: Number of frames per batch
            **inputs: Additional inputs for processing
            
        Yields:
            Lists of BaseResult instances (batches)
        """
        if not self.is_ready():
            raise ProcessingError(f"Feature {self.name} not initialized")
        
        # Validate inputs once
        self.validate_inputs(**inputs)
        
        batch = []
        for frame in frame_iterator:
            batch.append(frame)
            
            if len(batch) >= batch_size:
                try:
                    results = self.process_batch(batch, **inputs)
                    yield results
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Yield empty results on error
                    yield []
                finally:
                    batch = []
        
        # Process remaining frames
        if batch:
            try:
                results = self.process_batch(batch, **inputs)
                yield results
            except Exception as e:
                print(f"Error processing final batch: {e}")
                yield []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.initialized})"