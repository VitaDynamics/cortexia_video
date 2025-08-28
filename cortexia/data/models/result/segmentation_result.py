"""Segmentation data models"""

from typing import Optional, Dict, Any, List

import numpy as np

from .base_result import BaseResult
from .detection_result import BoundingBox


class SingleSegmentation:
    """Represents a single segmented object with mask, label, and area."""
    
    def __init__(self, mask: np.ndarray, label: str, area: int, detection_id: Optional[str] = None):
        self.mask = mask
        self.label = label
        self.area = area
        self.detection_id = detection_id
    
    def _get_repr_fields(self) -> str:
        """Show key segmentation fields for single segmentation."""
        fields = []
        fields.append(f"label='{self.label}'")
        fields.append(f"area={self.area}")
        if hasattr(self, 'mask') and self.mask is not None:
            fields.append(f"mask.shape={self.mask.shape}")
        if self.detection_id:
            fields.append(f"detection_id='{self.detection_id}'")
        return ", ".join(fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mask": self.mask.tolist() if hasattr(self.mask, 'tolist') else self.mask,
            "label": self.label,
            "area": self.area,
            "detection_id": self.detection_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleSegmentation":
        """Reconstruct SingleSegmentation from dictionary."""
        mask_data = data["mask"]
        if isinstance(mask_data, list):
            mask = np.array(mask_data)
        else:
            mask = mask_data
        
        return cls(
            mask=mask,
            label=data["label"],
            area=data["area"],
            detection_id=data.get("detection_id")
        )


class SegmentationResult(BaseResult):
    """Result schema for segmentation operations containing multiple segmentations."""
    
    def __init__(self, segmentations: Optional[List[SingleSegmentation]] = None, 
                 model_name: Optional[str] = None, processing_time_ms: Optional[float] = None):
        super().__init__(segmentations=segmentations or [], model_name=model_name, processing_time_ms=processing_time_ms)
        
        # Explicitly set attributes for type checking
        self.segmentations = segmentations or []
        self.model_name = model_name
        self.processing_time_ms = processing_time_ms
    
    def _get_repr_fields(self) -> str:
        """Show key segmentation fields for the result."""
        fields = []
        if self.segmentations:
            fields.append(f"segmentations={len(self.segmentations)}")
            # Show first few segmentations as preview
            preview_segmentations = self.segmentations[:2]
            preview_str = ", ".join([f"'{seg.label}'({seg.area})" for seg in preview_segmentations])
            if len(self.segmentations) > 2:
                preview_str += f"... (+{len(self.segmentations)-2} more)"
            fields.append(f"objects=[{preview_str}]")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @property
    def count(self) -> int:
        """Number of segmentations."""
        return len(self.segmentations)
    
    @property
    def has_segmentations(self) -> bool:
        """Whether there are any segmentations."""
        return len(self.segmentations) > 0
    
    def get_segmentations_by_label(self, label: str) -> List[SingleSegmentation]:
        """Get segmentations with matching label (case-insensitive)."""
        return [seg for seg in self.segmentations if seg.label.lower() == label.lower()]
    
    def get_segmentations_by_area(self, min_area: int = 0) -> List[SingleSegmentation]:
        """Get segmentations with area above threshold."""
        return [seg for seg in self.segmentations if seg.area >= min_area]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationResult":
        """Reconstruct SegmentationResult from dictionary."""
        # Handle segmentations list
        segmentations_data = data.get("segmentations", [])
        segmentations = []
        
        for seg_data in segmentations_data:
            if isinstance(seg_data, dict):
                segmentations.append(SingleSegmentation.from_dict(seg_data))
        
        # Use base class deserialization for other fields
        deserialized_data = cls._deserialize_special_types(data)
        
        # Override with specific conversions
        deserialized_data.update({
            "segmentations": segmentations,
            "model_name": data.get("model_name"),
            "processing_time_ms": data.get("processing_time_ms"),
        })
        
        return cls(**deserialized_data)