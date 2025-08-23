"""Object detection feature implementation"""

from typing import List

import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.detection import BoundingBox, DetectionResult
from ...data.models.video import VideoFramePacket, AnnotationResults
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
    
    def process_frame(self, frame_packet: VideoFramePacket) -> AnnotationResults:
        """
        Process a single frame for object detection.
        
        Args:
            frame_packet: VideoFramePacket containing RGB frame data
            
        Returns:
            AnnotationResults with detection results
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        if frame_packet.frame_data is None or frame_packet.frame_data.size == 0:
            return AnnotationResults()
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_packet.frame_data)
            
            # Get detection prompts from metadata
            prompts = self._get_detection_prompts(frame_packet)
            
            # Run detection using ObjectDetector
            detection_results = self.detector.detect_objects([image], [prompts])
            
            # Convert to DetectionResult objects
            detections = self._convert_to_detection_results(detection_results[0])
            
            # Return annotation results
            return AnnotationResults(detections=detections)
            
        except Exception as e:
            raise ProcessingError(f"Error in detection processing: {e}")
    
    def process_batch(self, frame_packets: List[VideoFramePacket]) -> List[AnnotationResults]:
        """
        Process multiple frames for object detection.
        
        Args:
            frame_packets: List of VideoFramePacket objects
            
        Returns:
            List of AnnotationResults with detection results
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        # Filter frames with valid data
        valid_packets = [f for f in frame_packets if f.frame_data is not None and f.frame_data.size > 0]
        
        if not valid_packets:
            return [AnnotationResults() for _ in frame_packets]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_packets]
            
            # Get prompts for each frame
            all_prompts = [self._get_detection_prompts(f) for f in valid_packets]
            
            # Run batch detection
            batch_detection_results = self.detector.detect_objects(images, all_prompts)
            
            # Convert results
            results = []
            valid_idx = 0
            for packet in frame_packets:
                if packet.frame_data is not None and packet.frame_data.size > 0:
                    if valid_idx < len(batch_detection_results):
                        detections = self._convert_to_detection_results(batch_detection_results[valid_idx])
                        results.append(AnnotationResults(detections=detections))
                        valid_idx += 1
                    else:
                        results.append(AnnotationResults())
                else:
                    results.append(AnnotationResults())
            
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error in batch detection processing: {e}")
    
    def _get_detection_prompts(self, frame_packet: VideoFramePacket) -> List[str]:
        """
        Get detection prompts for frame.
        
        Args:
            frame_packet: VideoFramePacket object
            
        Returns:
            List of detection prompts
        """
        # Check metadata for prompts
        metadata = frame_packet.additional_metadata
        
        # Use lister results if available
        if 'lister_results' in metadata and metadata['lister_results']:
            return metadata['lister_results']
        
        # Use dino prompt if available
        if 'dino_prompt' in metadata and metadata['dino_prompt']:
            return metadata['dino_prompt'].split(".")
        
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
                box=BoundingBox(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
            )
            
            detections.append(detection)
        
        return detections