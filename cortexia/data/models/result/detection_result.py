"""Detection data models"""

import uuid
from typing import Any, Dict, Optional, List

import numpy as np

from .base_result import BaseResult


class BoundingBox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class SingleDetection:
    """Represents a single detected object with score, label, and bounding box."""
    
    def __init__(self, score: float, label: str, box: "BoundingBox"):
        self.score = score
        self.label = label
        self.box = box
    
    def _get_repr_fields(self) -> str:
        """Show key detection fields for single detection."""
        fields = []
        fields.append(f"label='{self.label}'")
        fields.append(f"score={self.score:.3f}")
        if self.box:
            fields.append(f"box=[{self.box.xmin:.1f},{self.box.ymin:.1f},{self.box.xmax:.1f},{self.box.ymax:.1f}]")
        return ", ".join(fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "score": self.score,
            "label": self.label,
            "box": self.box.xyxy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleDetection":
        """Reconstruct SingleDetection from dictionary."""
        box_data = data["box"]
        if isinstance(box_data, list):
            # Box as list format [xmin, ymin, xmax, ymax]
            box = BoundingBox(
                xmin=box_data[0],
                ymin=box_data[1], 
                xmax=box_data[2],
                ymax=box_data[3],
            )
        else:
            # Box as dict format
            box = BoundingBox(**box_data)
        
        return cls(
            score=data["score"],
            label=data["label"],
            box=box
        )


class DetectionResult(BaseResult):
    """Result schema for object detection operations containing multiple detections."""
    
    def __init__(self, detections: Optional[List[SingleDetection]] = None, 
                 model_name: Optional[str] = None, processing_time_ms: Optional[float] = None):
        super().__init__(detections=detections or [], model_name=model_name, processing_time_ms=processing_time_ms)
        
        # Explicitly set attributes for type checking
        self.detections = detections or []
        self.model_name = model_name
        self.processing_time_ms = processing_time_ms
    
    def _get_repr_fields(self) -> str:
        """Show key detection fields for the result."""
        fields = []
        if self.detections:
            fields.append(f"detections={len(self.detections)}")
            # Show first few detections as preview
            preview_detections = self.detections[:2]
            preview_str = ", ".join([f"'{det.label}'({det.score:.2f})" for det in preview_detections])
            if len(self.detections) > 2:
                preview_str += f"... (+{len(self.detections)-2} more)"
            fields.append(f"objects=[{preview_str}]")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)
    
    @property
    def has_detections(self) -> bool:
        """Whether there are any detections."""
        return len(self.detections) > 0
    
    def get_detections_by_score(self, threshold: float = 0.0) -> List[SingleDetection]:
        """Get detections with score above threshold."""
        return [det for det in self.detections if det.score >= threshold]
    
    def get_detections_by_label(self, label: str) -> List[SingleDetection]:
        """Get detections with matching label (case-insensitive)."""
        return [det for det in self.detections if det.label.lower() == label.lower()]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Reconstruct DetectionResult from dictionary."""
        # Handle detections list
        detections_data = data.get("detections", [])
        detections = []
        
        for det_data in detections_data:
            if isinstance(det_data, dict):
                detections.append(SingleDetection.from_dict(det_data))
        
        # Use base class deserialization for other fields
        deserialized_data = cls._deserialize_special_types(data)
        
        # Override with specific conversions
        deserialized_data.update({
            "detections": detections,
            "model_name": data.get("model_name"),
            "processing_time_ms": data.get("processing_time_ms"),
        })
        
        return cls(**deserialized_data)