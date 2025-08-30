"""Object segmentation feature implementation"""

from typing import List, Any

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...data.models.result.base_result import BaseResult
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.result.detection_result import BoundingBox, DetectionResult, SingleDetection
from ...data.models.video import VideoFramePacket
from ...data.models.result.segmentation_result import SegmentationResult, SingleSegmentation
from .models import ObjectSegmenter

from ..registry import feature_registry

@feature_registry.register("segmentation")
class SegmentationFeature(BaseFeature):
    """Object segmentation feature using SAM"""

    output_schema = SegmentationResult
        
    def __init__(self, config=None):
        super().__init__(config)
        self.segmenter = None
        self.device = None
    
    def _initialize(self):
        """Initialize segmentation model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create config dict for the segmenter
            segmenter_config = {
                "model": self.get_config_param("model", "facebook/sam-vit-base"),
                "batch_size": self.get_config_param("batch_size", 8)
            }
            
            # Initialize segmenter
            self.segmenter = ObjectSegmenter(segmenter_config)
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize segmentation model: {e}")

    def _release(self) -> None:
        """Release segmenter resources and clear device refs."""
        try:
            if self.segmenter is not None and hasattr(self.segmenter, "release"):
                try:
                    self.segmenter.release()
                except Exception:
                    pass
        finally:
            self.segmenter = None
            self.device = None
    
    @property
    def name(self) -> str:
        return "segmentation"
    
    @property
    def description(self) -> str:
        return "Object segmentation using Segment Anything Model (SAM)"
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> SegmentationResult:
        """
        Process a single frame for object segmentation.
        
        Args:
            frame: VideoFramePacket containing RGB frame data and detections
            **inputs: Additional inputs (not used by segmentation)
            
        Returns:
            SegmentationResult containing segmentation masks
        """
        if not self.is_ready():
            self._initialize()
        
        if frame.frame_data is None:
            return SegmentationResult(segmentations=[])
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Extract bounding boxes from detections (tolerant to annotation shape)
            boxes: List[List[float]] = []
            single_detections: List[SingleDetection] = []
            if hasattr(frame, 'annotations') and frame.annotations:
                detection_results = self._extract_detections(frame.annotations)
                # Flatten all SingleDetection objects from DetectionResult objects
                for detection_result in detection_results:
                    single_detections.extend(detection_result.detections)
                for det in single_detections:
                    if getattr(det, 'box', None):
                        box = [det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax]
                        boxes.append(box)
            
            # Run segmentation using ObjectSegmenter
            if boxes:
                masks = self.segmenter.segment_object(image, boxes)
                
                # Create segmentation results
                segments = self._create_segmentation_results(masks, boxes, single_detections)
                
                # Return all segments
                return SegmentationResult(segmentations=segments)
            
            # Return empty result if no segments
            return SegmentationResult(segmentations=[])
            
        except Exception as e:
            raise ProcessingError(f"Error in segmentation processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[BaseResult]:
        """
        Process multiple frames for object segmentation.
        
        Args:
            frames: List of VideoFramePacket objects
            **inputs: Additional inputs (not used by segmentation)
            
        Returns:
            List of SegmentationResult objects
        """
        if not self.is_ready():
            self._initialize()
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.frame_data is not None]
        
        if not valid_frames:
            return [SegmentationResult(segmentations=[]) for _ in frames]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_frames]
            
            # Extract bounding boxes from detections for each frame
            batch_boxes = []
            batch_single_detections = []
            for frame in valid_frames:
                frame_boxes = []
                frame_single_detections = []
                
                # Check for detections in annotations (tolerant to annotation shape)
                if hasattr(frame, 'annotations') and frame.annotations:
                    detection_results = self._extract_detections(frame.annotations)
                    # Flatten all SingleDetection objects from DetectionResult objects
                    for detection_result in detection_results:
                        frame_single_detections.extend(detection_result.detections)
                    for det in frame_single_detections:
                        if getattr(det, 'box', None):
                            box = [det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax]
                            frame_boxes.append(box)
                
                batch_boxes.append(frame_boxes)
                batch_single_detections.append(frame_single_detections)
            
            # Run batch segmentation
            batch_masks = self.segmenter.segment_object(images, batch_boxes)
            
            # Create SegmentationResult objects
            results = []
            valid_idx = 0
            for frame in frames:
                if frame.frame_data is not None and valid_idx < len(batch_masks):
                    masks = batch_masks[valid_idx]
                    if batch_boxes[valid_idx]:
                        segments = self._create_segmentation_results(masks, batch_boxes[valid_idx], batch_single_detections[valid_idx])
                        results.append(SegmentationResult(segmentations=segments))
                    else:
                        results.append(SegmentationResult(segmentations=[]))
                    valid_idx += 1
                else:
                    results.append(SegmentationResult(segmentations=[]))
            
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error in batch segmentation processing: {e}")
    
    def _create_segmentation_results(self, masks: List[np.ndarray], boxes: List[List[float]], 
                                    detections: List[SingleDetection]) -> List[SingleSegmentation]:
        """
        Create SingleSegmentation objects from masks.
        
        Args:
            masks: List of binary masks
            boxes: List of bounding boxes used for segmentation
            detections: List of single detection results for matching
            
        Returns:
            List of SingleSegmentation objects
        """
        segments = []
        
        for i, mask in enumerate(masks):
            if i < len(boxes):
                # Calculate mask area
                area = int(np.sum(mask))
                
                # Get label from detection if available
                label = f"segment_{i}"  # Generic label
                if i < len(detections):
                    label = detections[i].label
                
                # Create single segmentation result
                segment = SingleSegmentation(
                    mask=mask,
                    label=label,
                    area=area
                )
                
                segments.append(segment)
        
        return segments
    
    def _add_masks_to_detections(self, segments: List[SingleSegmentation], detections: List[SingleDetection]):
        """
        Add masks to detection objects for backward compatibility.
        
        Args:
            segments: List of single segmentation results
            detections: List of single detection results
        """
        # Note: This method is kept for backward compatibility but may need adjustment
        # since SingleDetection doesn't have an id attribute and SingleSegmentation
        # doesn't have a detection_id attribute in the new design
        pass
    # TODO: remove the list detections result usage. 
    def _extract_detections(self, annotations: Any) -> List[DetectionResult]:
        """
        Extract detections from annotations in a backward-compatible way.
        Supports:
        - An object with attribute `detections`
        - A dict with key 'detections' (list or single)
        - A dict with key 'DetectionResult' (list or single)
        """
        # Case 1: object with .detections list
        if hasattr(annotations, 'detections') and getattr(annotations, 'detections'):
            vals = getattr(annotations, 'detections')
            return list(vals) if isinstance(vals, list) else [vals]

        # Case 2: dict-like structures
        if isinstance(annotations, dict):
            if 'detections' in annotations and annotations['detections']:
                vals = annotations['detections']
                return list(vals) if isinstance(vals, list) else [vals]
            if 'DetectionResult' in annotations and annotations['DetectionResult']:
                vals = annotations['DetectionResult']
                return list(vals) if isinstance(vals, list) else [vals]

        return []
