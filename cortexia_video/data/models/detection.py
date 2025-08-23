"""Detection data models"""

import uuid
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from .base_result import BaseResult


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class DetectionResult(BaseResult):
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None
    description: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object_clip_features: Optional[np.ndarray] = None  # Changed to numpy array for consistency

    def _get_repr_fields(self) -> str:
        """Show key detection fields.""" 
        fields = []
        fields.append(f"label='{self.label}'")
        fields.append(f"score={self.score:.3f}")
        if self.box:
            fields.append(f"box=[{self.box.xmin:.1f},{self.box.ymin:.1f},{self.box.xmax:.1f},{self.box.ymax:.1f}]")
        return ", ".join(fields)

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        """Reconstruct DetectionResult from dictionary.""" 
        # Handle box conversion
        box_data = detection_dict["box"]
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
        
        # Use base class deserialization for numpy arrays
        deserialized_data = cls._deserialize_special_types(detection_dict)
        
        # Override with specific conversions
        deserialized_data.update({
            "score": detection_dict["score"],
            "label": detection_dict["label"],
            "box": box,
            "id": detection_dict.get("id", str(uuid.uuid4())),
            "description": detection_dict.get("description"),
        })
        
        return cls(**deserialized_data)