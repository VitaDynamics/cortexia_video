"""Object description feature implementation (DAM-only path)"""

from typing import List

import numpy as np
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.detection import DetectionResult
from ...data.models.video import FrameData
from .models import ObjectDescriber


class DescriptionFeature(BaseFeature):
    """Object description feature using NVIDIA DAM via ObjectDescriber."""

    def __init__(self, config=None):
        super().__init__(config)
        # DAM engine from models.py (only path)
        self.engine = None

    def _initialize(self):
        """Initialize description engine (DAM-only)."""
        try:
            self.engine = ObjectDescriber(self.config)
            self.initialized = True
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize description engine: {e}")

    @property
    def name(self) -> str:
        return "description"

    @property
    def description(self) -> str:
        return "Object description using DAM model"

    def process_frame(self, frame_data: FrameData) -> FrameData:
        """Process a single frame for object description."""
        if not self.is_ready():
            raise ProcessingError("Description feature not initialized")

        if frame_data.rgb_image is None:
            return frame_data

        try:
            image = Image.fromarray(frame_data.rgb_image)
            descriptions = self._describe_objects(image, frame_data.detections)
            for i, detection in enumerate(frame_data.detections):
                if i < len(descriptions):
                    detection.description = descriptions[i]
            return frame_data
        except Exception as e:
            raise ProcessingError(f"Error in description processing: {e}")

    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """Process multiple frames for object description."""
        if not self.is_ready():
            raise ProcessingError("Description feature not initialized")

        valid_frames = [f for f in frames if f.rgb_image is not None]
        if not valid_frames:
            return frames

        try:
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            detections_list = [f.detections for f in valid_frames]
            batch_descriptions = self._describe_objects_batch(images, detections_list)

            for i, frame in enumerate(valid_frames):
                if i < len(batch_descriptions):
                    descriptions = batch_descriptions[i]
                    for j, detection in enumerate(frame.detections):
                        if j < len(descriptions):
                            detection.description = descriptions[j]
            return frames
        except Exception as e:
            raise ProcessingError(f"Error in batch description processing: {e}")

    def _describe_objects(self, image: Image.Image, detections: List[DetectionResult]) -> List[str]:
        """Describe objects in a single image using DAM."""
        if not detections:
            return []

        height, width = image.height, image.width
        descriptions: List[str] = []
        for det in detections:
            mask = self._bbox_to_mask(width, height, det.box)
            desc = self.engine.describe_object(image, mask, streaming=False)
            descriptions.append(desc)
        return descriptions

    def _describe_objects_batch(self, images: List[Image.Image], detections_list: List[List[DetectionResult]]) -> List[List[str]]:
        """Describe objects in a batch of images using DAM."""
        results: List[List[str]] = []
        for i, image in enumerate(images):
            detections = detections_list[i] if i < len(detections_list) else []
            results.append(self._describe_objects(image, detections))
        return results

    def _bbox_to_mask(self, width: int, height: int, bbox) -> np.ndarray:
        """Create a binary mask (H,W) from a bbox with attributes xmin,ymin,xmax,ymax."""
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

