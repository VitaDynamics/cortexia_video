"""Field validation system for unified VideoFramePacket processing.

This module provides standardized field requirement checking for features and gates,
ensuring that required data is present before processing begins.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np

from ...api.exceptions import ProcessingError

# TODO: how to easily link between this and other filed. 
class FrameField(Enum):
    """Enumeration of all possible fields in VideoFramePacket that features/gates might require."""
    
    # Core fields (always present)
    FRAME_DATA = "frame_data"
    FRAME_NUMBER = "frame_number"
    TIMESTAMP = "timestamp"
    SOURCE_VIDEO_ID = "source_video_id"
    
    # Metadata fields
    ADDITIONAL_METADATA = "additional_metadata"
    
    # Annotation fields
    ANNOTATIONS = "annotations"
    ANNOTATIONS_DETECTIONS = "annotations.detections"
    ANNOTATIONS_SEGMENTS = "annotations.segments"
    ANNOTATIONS_FEATURES = "annotations.features"
    ANNOTATIONS_CAPTION = "annotations.caption"
    ANNOTATIONS_DEPTH_MAP = "annotations.depth_map"
    ANNOTATIONS_DEPTH_STATISTICS = "annotations.depth_statistics"
    ANNOTATIONS_LISTER_RESULTS = "annotations.lister_results"
    ANNOTATIONS_DINO_PROMPT = "annotations.dino_prompt"
    ANNOTATIONS_SCENE_CLIP_FEATURES = "annotations.scene_clip_features"
    
    # Embeddings fields
    EMBEDDINGS = "embeddings"
    EMBEDDINGS_CLIP = "embeddings.clip"
    EMBEDDINGS_DINO = "embeddings.dino"
    
    # Legacy metadata fields (for backwards compatibility)
    METADATA_LISTER_RESULTS = "additional_metadata.lister_results"
    METADATA_DINO_PROMPT = "additional_metadata.dino_prompt"
    METADATA_SCENE_CLIP_FEATURES = "additional_metadata.scene_clip_features"


class FieldValidator:
    """Utility class for validating VideoFramePacket field requirements."""
    
    @staticmethod
    def validate_frame_fields(frame: 'VideoFramePacket', required_fields: List[Union[str, FrameField]]) -> None:
        """
        Validate that all required fields are present and non-empty in the frame packet.
        
        Args:
            frame: VideoFramePacket to validate
            required_fields: List of field names or FrameField enums to check
            
        Raises:
            ProcessingError: If any required field is missing or empty
        """
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            field_path = field.value if isinstance(field, FrameField) else field
            
            try:
                value = FieldValidator._get_nested_field(frame, field_path)
                
                # Check if field is empty/None
                if FieldValidator._is_field_empty(value):
                    empty_fields.append(field_path)
                    
            except (AttributeError, KeyError, TypeError):
                missing_fields.append(field_path)
        
        # Report validation errors
        error_messages = []
        if missing_fields:
            error_messages.append(f"Missing fields: {missing_fields}")
        if empty_fields:
            error_messages.append(f"Empty fields: {empty_fields}")
        
        if error_messages:
            raise ProcessingError(f"Frame field validation failed: {'; '.join(error_messages)}")
    
    @staticmethod
    def _get_nested_field(obj: Any, field_path: str) -> Any:
        """
        Get a nested field value from an object using dot notation.
        
        Args:
            obj: Object to get field from
            field_path: Dot-separated field path (e.g., "annotations.caption")
            
        Returns:
            Field value
            
        Raises:
            AttributeError, KeyError, TypeError: If field path is invalid
        """
        parts = field_path.split(".")
        current = obj
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise AttributeError(f"Field '{part}' not found in path '{field_path}'")
        
        return current
    
    @staticmethod
    def _is_field_empty(value: Any) -> bool:
        """
        Check if a field value is considered empty.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is empty, False otherwise
        """
        if value is None:
            return True
        
        # Handle common empty cases
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            return True
        
        # Handle numpy arrays
        if isinstance(value, np.ndarray) and value.size == 0:
            return True
        
        return False
    
    @staticmethod
    def get_available_fields(frame: 'VideoFramePacket') -> Set[str]:
        """
        Get all available (non-empty) fields in a VideoFramePacket.
        
        Args:
            frame: VideoFramePacket to analyze
            
        Returns:
            Set of available field paths
        """
        available = set()
        
        # Check all defined fields
        for field in FrameField:
            try:
                value = FieldValidator._get_nested_field(frame, field.value)
                if not FieldValidator._is_field_empty(value):
                    available.add(field.value)
            except (AttributeError, KeyError, TypeError):
                continue
        
        return available
    
    @staticmethod
    def check_field_compatibility(required_fields: List[Union[str, FrameField]], 
                                available_fields: Set[str]) -> Dict[str, bool]:
        """
        Check which required fields are available.
        
        Args:
            required_fields: List of required field names or FrameField enums
            available_fields: Set of available field paths
            
        Returns:
            Dictionary mapping field names to availability status
        """
        compatibility = {}
        
        for field in required_fields:
            field_path = field.value if isinstance(field, FrameField) else field
            compatibility[field_path] = field_path in available_fields
        
        return compatibility
    
    @staticmethod
    def get_field_summary(frame: 'VideoFramePacket') -> Dict[str, Any]:
        """
        Get a summary of field availability and basic statistics.
        
        Args:
            frame: VideoFramePacket to summarize
            
        Returns:
            Dictionary with field summary information
        """
        summary = {
            "available_fields": FieldValidator.get_available_fields(frame),
            "frame_shape": frame.frame_data.shape,
            "embeddings_count": len(frame.embeddings) if frame.embeddings else 0,
            "has_annotations": frame.annotations is not None,
            "has_gate_results": frame.gate_results is not None,
            "additional_metadata_keys": list(frame.additional_metadata.keys()) if frame.additional_metadata else [],
        }
        
        return summary


# Convenience function for validation
def validate_frame_requirements(frame: 'VideoFramePacket', 
                              required_fields: List[Union[str, FrameField]],
                              component_name: str = "Component") -> None:
    """
    Validate frame requirements with component-specific error messaging.
    
    Args:
        frame: VideoFramePacket to validate
        required_fields: List of required fields
        component_name: Name of component for error messages
        
    Raises:
        ProcessingError: If validation fails
    """
    try:
        FieldValidator.validate_frame_fields(frame, required_fields)
    except ProcessingError as e:
        raise ProcessingError(f"{component_name} requirements not met: {e}")