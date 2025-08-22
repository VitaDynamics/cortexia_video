"""Segmentation data models"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .detection import BoundingBox


class SegmentationResult(BaseModel):
    """Result of image segmentation with area and label information"""

    mask: np.ndarray
    score: float
    label: str
    area: int
    bbox: BoundingBox
    detection_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Override dict method to handle numpy arrays"""
        data = super().dict(**kwargs)
        # Skip mask in serialization
        data["mask"] = None
        return data