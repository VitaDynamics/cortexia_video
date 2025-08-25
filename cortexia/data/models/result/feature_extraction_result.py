"""Feature extraction result schema for CLIP and other vision models."""

from typing import Optional

import numpy as np

from .base_result import BaseResult
from ..registry import schema_registry


@schema_registry.register("result.feature_extraction")
class FeatureExtractionResult(BaseResult):
    """Result schema for feature extraction operations."""
    
    features: np.ndarray  # The extracted feature vector
    feature_dim: Optional[int] = None  # Dimension of feature vector
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    normalization: Optional[str] = None  # "l2", "none", etc.
    processing_time_ms: Optional[float] = None
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        if hasattr(self, 'features') and self.features is not None:
            fields.append(f"features.shape={self.features.shape}")
            if self.feature_dim:
                fields.append(f"dim={self.feature_dim}")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureExtractionResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)