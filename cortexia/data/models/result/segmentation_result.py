"""Segmentation data models"""

from typing import Optional, Dict, Any

import numpy as np

from .base_result import BaseResult
from .detection_result import BoundingBox
from ..registry import schema_registry

@schema_registry.register("result.segmentation")
class SegmentationResult(BaseResult):
    """Result of image segmentation with area and label information"""

    def __init__(self, mask: np.ndarray, score: float, label: str, area: int, bbox: BoundingBox, detection_id: Optional[str] = None):
        super().__init__(mask=mask, score=score, label=label, area=area, bbox=bbox, detection_id=detection_id)

    def _get_repr_fields(self) -> str:
        """Show key segmentation fields."""
        fields = []
        fields.append(f"label='{self.label}'")
        fields.append(f"score={self.score:.3f}")
        fields.append(f"area={self.area}")
        if hasattr(self, 'mask') and self.mask is not None:
            fields.append(f"mask.shape={self.mask.shape}")
        return ", ".join(fields)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationResult":
        """Reconstruct from dictionary."""
        # Handle bbox conversion
        bbox_data = data["bbox"]
        if isinstance(bbox_data, dict):
            bbox = BoundingBox(**bbox_data)
        else:
            # Assume list format
            bbox = BoundingBox(
                xmin=bbox_data[0],
                ymin=bbox_data[1],
                xmax=bbox_data[2], 
                ymax=bbox_data[3]
            )
        
        # Use base class deserialization for numpy arrays
        deserialized_data = cls._deserialize_special_types(data)
        
        # Override with specific conversions  
        deserialized_data.update({
            "score": data["score"],
            "label": data["label"],
            "area": data["area"],
            "bbox": bbox,
            "detection_id": data.get("detection_id"),
        })
        
        return cls(**deserialized_data)