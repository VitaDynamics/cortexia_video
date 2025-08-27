"""Depth estimation result schema."""

from typing import Any, Dict, Optional

import numpy as np

from .base_result import BaseResult

class DepthResult(BaseResult):
    """Result schema for depth estimation operations."""
    
    def __init__(self, depth_map: np.ndarray, depth_statistics: Optional[Dict[str, float]] = None, 
                 model_name: Optional[str] = None, focal_length: Optional[float] = None, 
                 processing_time_ms: Optional[float] = None):
        super().__init__(depth_map=depth_map, depth_statistics=depth_statistics, model_name=model_name, 
                        focal_length=focal_length, processing_time_ms=processing_time_ms)
    
    def _serialize_special_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle depth map serialization."""
        # Use the base class method which handles numpy arrays
        return super()._serialize_special_types(data)
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        if hasattr(self, 'depth_map') and self.depth_map is not None:
            fields.append(f"depth_map.shape={self.depth_map.shape}")
            if self.depth_statistics:
                fields.append(f"depth_range=[{self.depth_statistics.get('min', 'N/A'):.2f}, {self.depth_statistics.get('max', 'N/A'):.2f}]")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> "DepthResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)