"""Base class for all annotation features"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..api.exceptions import CortexiaError, ProcessingError


class BaseFeature(ABC):
    """
    Base class for all independent annotation features.
    
    Each feature is self-contained and can be used independently
    without requiring other features or pipelines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature.
        
        Args:
            config: Feature-specific configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self._initialize()
    
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
    def process_frame(self, frame_data: Any) -> Any:
        """
        Process a single frame.
        
        Args:
            frame_data: Frame data to process
            
        Returns:
            Processed frame data
        """
        pass
    
    def process_batch(self, frames: List[Any]) -> List[Any]:
        """
        Process multiple frames.
        
        Default implementation processes frames one by one.
        Override for batch-specific optimizations.
        
        Args:
            frames: List of frame data to process
            
        Returns:
            List of processed frame data
        """
        return [self.process_frame(frame) for frame in frames]
    
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