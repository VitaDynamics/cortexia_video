"""Detection data models"""

import uuid
from typing import Any, Dict, Optional

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


class DetectionResult(BaseResult):
    def __init__(self, score: float, label: str, box: "BoundingBox", mask: Optional[np.ndarray] = None, 
                 description: Optional[str] = None, id: str = None, object_clip_features: Optional[np.ndarray] = None):
        super().__init__(score=score, label=label, box=box, mask=mask, description=description, 
                        id=id or str(uuid.uuid4()), object_clip_features=object_clip_features)

    def _get_repr_fields(self) -> str:
        """Show key detection fields.""" 
        fields = []
        fields.append(f"label='{self.label}'")
        fields.append(f"score={self.score:.3f}")
        if self.box:
            fields.append(f"box=[{self.box.xmin:.1f},{self.box.ymin:.1f},{self.box.xmax:.1f},{self.box.ymax:.1f}]")
        return ", ".join(fields)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Reconstruct DetectionResult from dictionary.""" 
        # Handle box conversion
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
        
        # Use base class deserialization for numpy arrays
        deserialized_data = cls._deserialize_special_types(data)
        
        # Override with specific conversions
        deserialized_data.update({
            "score": data["score"],
            "label": data["label"],
            "box": box,
            "id": data.get("id", str(uuid.uuid4())),
            "description": data.get("description"),
        })
        
        return cls(**deserialized_data)