"""Detection data models"""

import uuid
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class DetectionResult(BaseModel):
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None
    description: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object_clip_features: Optional[list[float]] = None

    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Override dict method to handle features"""
        data = super().dict(**kwargs)
        # Keep mask and features in serialization
        return data

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"][0],
                ymin=detection_dict["box"][1],
                xmax=detection_dict["box"][2],
                ymax=detection_dict["box"][3],
            ),
            id=detection_dict.get("id", str(uuid.uuid4())),
            description=detection_dict.get("description"),
            mask=detection_dict.get("mask"),
            object_clip_features=detection_dict.get("object_clip_features"),
        )