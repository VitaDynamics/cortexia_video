"""Object detection feature implementation"""

from typing import List

import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.detection import BoundingBox, DetectionResult
from ...data.models.video import FrameData
from .models import ObjectDetector


class DetectionFeature(BaseFeature):
    """Object detection feature using Grounding DINO"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.detector = None
        self.device = None
    
    def _initialize(self):
        """Initialize detection model and processor"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create config dict for the detector
            detector_config = {
                "model": self.get_config_param("model", "IDEA-Research/grounding-dino-base"),
                "box_threshold": self.get_config_param("box_threshold", 0.3),
                "text_threshold": self.get_config_param("text_threshold", 0.3)
            }
            
            # Initialize detector
            self.detector = ObjectDetector(detector_config)
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize detection model: {e}")
    
    @property
    def name(self) -> str:
        return "detection"
    
    @property
    def description(self) -> str:
        return "Object detection using Grounding DINO model"
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for object detection.
        
        Args:
            frame_data: Frame data containing RGB image
            
        Returns:
            Frame data with detection results added
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Get detection prompts
            prompts = self._get_detection_prompts(frame_data)
            
            # Run detection using ObjectDetector
            detection_results = self.detector.detect_objects([image], [prompts])
            
            # Convert to DetectionResult objects
            detections = self._convert_to_detection_results(detection_results[0])
            
            # Add detections to frame data
            frame_data.detections = detections
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in detection processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for object detection.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with detection results added
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Get prompts for each frame
            all_prompts = [self._get_detection_prompts(f) for f in valid_frames]
            
            # Run batch detection
            batch_detection_results = self.detector.detect_objects(images, all_prompts)
            
            # Convert and add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_detection_results):
                    detections = self._convert_to_detection_results(batch_detection_results[i])
                    frame.detections = detections
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch detection processing: {e}")
    
    def _get_detection_prompts(self, frame_data: FrameData) -> List[str]:
        """
        Get detection prompts for frame.
        
        Args:
            frame_data: Frame data object
            
        Returns:
            List of detection prompts
        """
        # Use lister results if available
        if hasattr(frame_data, 'lister_results') and frame_data.lister_results:
            return frame_data.lister_results
        
        # Use dino prompt if available
        if hasattr(frame_data, 'dino_prompt') and frame_data.dino_prompt:
            return frame_data.dino_prompt.split(".")
        
        # Default prompts
        return self.get_config_param("default_prompts", ["object"])
    
    def _convert_to_detection_results(self, detection_results: List[dict]) -> List[DetectionResult]:
        """
        Convert ObjectDetector output to DetectionResult objects.
        
        Args:
            detection_results: List of detection dictionaries from ObjectDetector
            
        Returns:
            List of DetectionResult objects
        """
        detections = []
        
        for detection_dict in detection_results:
            box = detection_dict["box"]  # [x1, y1, x2, y2]
            score = detection_dict["score"]
            label = detection_dict["label"]
            
            detection = DetectionResult(
                score=score,
                label=label,
                bbox=BoundingBox(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
            )
            
            detections.append(detection)
        
        return detections