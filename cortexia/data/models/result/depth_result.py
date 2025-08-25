"""Depth estimation result schema."""

from typing import Any, Dict, Optional

import numpy as np

from .base_result import BaseResult
from ..registry import schema_registry

@schema_registry.register("result.depth")
class DepthResult(BaseResult):
    """Result schema for depth estimation operations."""
    
    depth_map: np.ndarray  # The depth map as numpy array
    depth_statistics: Optional[Dict[str, float]] = None  # min, max, mean, std, etc.
    model_name: Optional[str] = None
    focal_length: Optional[float] = None  # If available from DepthPro
    processing_time_ms: Optional[float] = None
    
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
    def from_dict(cls, data: dict) -> "DepthResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)