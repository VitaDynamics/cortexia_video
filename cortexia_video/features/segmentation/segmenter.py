"""Object segmentation feature implementation"""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.detection_result import BoundingBox, DetectionResult
from ...data.models.segmentation_result import SegmentationResult
from ...data.models.video import FrameData
from .models import ObjectSegmenter


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
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for object segmentation.
        
        Args:
            frame_data: Frame data containing RGB image and detections
            
        Returns:
            Frame data with segmentation results added
        """
        if not self.is_ready():
            raise ProcessingError("Segmentation feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Extract bounding boxes from detections
            boxes = []
            for detection in frame_data.detections:
                if detection.bbox:
                    box = [detection.bbox.xmin, detection.bbox.ymin, 
                           detection.bbox.xmax, detection.bbox.ymax]
                    boxes.append(box)
            
            # Run segmentation using ObjectSegmenter
            if boxes:
                masks = self.segmenter.segment_object(image, boxes)
                
                # Create segmentation results
                segments = self._create_segmentation_results(masks, boxes, frame_data.detections)
                
                # Add segments to frame data
                frame_data.segments = segments
                
                # Also add masks to detections for compatibility
                self._add_masks_to_detections(segments, frame_data.detections)
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in segmentation processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for object segmentation.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with segmentation results added
        """
        if not self.is_ready():
            raise ProcessingError("Segmentation feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Extract bounding boxes from detections for each frame
            batch_boxes = []
            for frame in valid_frames:
                frame_boxes = []
                for detection in frame.detections:
                    if detection.bbox:
                        box = [detection.bbox.xmin, detection.bbox.ymin, 
                               detection.bbox.xmax, detection.bbox.ymax]
                        frame_boxes.append(box)
                batch_boxes.append(frame_boxes)
            
            # Run batch segmentation
            batch_masks = self.segmenter.segment_object(images, batch_boxes)
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_masks) and batch_boxes[i]:
                    masks = batch_masks[i]
                    segments = self._create_segmentation_results(masks, batch_boxes[i], frame.detections)
                    frame.segments = segments
                    self._add_masks_to_detections(segments, frame.detections)
            
            return frames
            
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