"""Feature extraction result schema for CLIP and other vision models."""

from typing import Optional, Dict, Any

import numpy as np

from .base_result import BaseResult

class FeatureExtractionResult(BaseResult):
    """Result schema for feature extraction operations."""
    
    def __init__(self, features: np.ndarray, feature_dim: Optional[int] = None, model_name: Optional[str] = None, 
                 model_version: Optional[str] = None, normalization: Optional[str] = None, 
                 processing_time_ms: Optional[float] = None):
        super().__init__(features=features, feature_dim=feature_dim, model_name=model_name, 
                        model_version=model_version, normalization=normalization, processing_time_ms=processing_time_ms)
    
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
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureExtractionResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)