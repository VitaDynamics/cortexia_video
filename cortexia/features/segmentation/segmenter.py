"""Object segmentation feature implementation"""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.result.detection_result import BoundingBox, DetectionResult
from ...data.models.video import VideoFramePacket
from ...data.models.result.segmentation_result import SegmentationResult
from .models import ObjectSegmenter

from ..registry import feature_registry

@feature_registry.register("segmentation")
class SegmentationFeature(BaseFeature):
    """Object segmentation feature using SAM"""
    
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
            return SegmentationResult(
                mask=np.array([]),
                score=0.0,
                label="",
                area=0,
                bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
            )
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Extract bounding boxes from detections
            boxes = []
            detections = []
            
            # Check for detections in annotations
            if hasattr(frame, 'annotations') and frame.annotations:
                if hasattr(frame.annotations, 'detections') and frame.annotations.detections:
                    detections = frame.annotations.detections
                    for detection in detections:
                        if detection.bbox:
                            box = [detection.bbox.xmin, detection.bbox.ymin, 
                                   detection.bbox.xmax, detection.bbox.ymax]
                            boxes.append(box)
            
            # Run segmentation using ObjectSegmenter
            if boxes:
                masks = self.segmenter.segment_object(image, boxes)
                
                # Create segmentation results
                segments = self._create_segmentation_results(masks, boxes, detections)
                
                # Return first segment or empty result
                if segments:
                    return segments[0]
            
            # Return empty result if no segments
            return SegmentationResult(
                mask=np.array([]),
                score=0.0,
                label="",
                area=0,
                bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
            )
            
        except Exception as e:
            raise ProcessingError(f"Error in segmentation processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[SegmentationResult]:
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
            return [SegmentationResult(
                mask=np.array([]),
                score=0.0,
                label="",
                area=0,
                bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
            ) for _ in frames]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_frames]
            
            # Extract bounding boxes from detections for each frame
            batch_boxes = []
            batch_detections = []
            for frame in valid_frames:
                frame_boxes = []
                frame_detections = []
                
                # Check for detections in annotations
                if hasattr(frame, 'annotations') and frame.annotations:
                    if hasattr(frame.annotations, 'detections') and frame.annotations.detections:
                        frame_detections = frame.annotations.detections
                        for detection in frame_detections:
                            if detection.bbox:
                                box = [detection.bbox.xmin, detection.bbox.ymin, 
                                       detection.bbox.xmax, detection.bbox.ymax]
                                frame_boxes.append(box)
                
                batch_boxes.append(frame_boxes)
                batch_detections.append(frame_detections)
            
            # Run batch segmentation
            batch_masks = self.segmenter.segment_object(images, batch_boxes)
            
            # Create SegmentationResult objects
            results = []
            valid_idx = 0
            for frame in frames:
                if frame.frame_data is not None and valid_idx < len(batch_masks):
                    masks = batch_masks[valid_idx]
                    if batch_boxes[valid_idx]:
                        segments = self._create_segmentation_results(masks, batch_boxes[valid_idx], batch_detections[valid_idx])
                        if segments:
                            results.append(segments[0])
                        else:
                            results.append(SegmentationResult(
                                mask=np.array([]),
                                score=0.0,
                                label="",
                                area=0,
                                bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
                            ))
                    else:
                        results.append(SegmentationResult(
                            mask=np.array([]),
                            score=0.0,
                            label="",
                            area=0,
                            bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
                        ))
                    valid_idx += 1
                else:
                    results.append(SegmentationResult(
                        mask=np.array([]),
                        score=0.0,
                        label="",
                        area=0,
                        bbox=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
                    ))
            
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error in batch segmentation processing: {e}")
    
    def _create_segmentation_results(self, masks: List[np.ndarray], boxes: List[List[float]], 
                                    detections: List[DetectionResult]) -> List[SegmentationResult]:
        """
        Create SegmentationResult objects from masks.
        
        Args:
            masks: List of binary masks
            boxes: List of bounding boxes used for segmentation
            detections: List of detection results for matching
            
        Returns:
            List of SegmentationResult objects
        """
        segments = []
        
        for i, mask in enumerate(masks):
            if i < len(boxes):
                bbox = boxes[i]
                
                # Calculate mask area
                area = int(np.sum(mask))
                
                # Convert bbox format to BoundingBox object
                bbox_obj = BoundingBox(xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3])
                
                # Match with detection if possible
                detection_id = None
                if i < len(detections):
                    detection_id = detections[i].id
                
                # Create segmentation result
                segment = SegmentationResult(
                    mask=mask,
                    score=1.0,  # Default score
                    label=f"segment_{i}",  # Generic label
                    area=area,
                    bbox=bbox_obj,
                    detection_id=detection_id
                )
                
                segments.append(segment)
        
        return segments
    
    def _add_masks_to_detections(self, segments: List[SegmentationResult], detections: List[DetectionResult]):
        """
        Add masks to detection objects for backward compatibility.
        
        Args:
            segments: List of segmentation results
            detections: List of detection results
        """
        # Create mapping from detection_id to mask
        mask_map = {}
        for segment in segments:
            if segment.detection_id:
                mask_map[segment.detection_id] = segment.mask
        
        # Add masks to detections
        for detection in detections:
            if detection.id in mask_map:
                detection.mask = mask_map[detection.id]