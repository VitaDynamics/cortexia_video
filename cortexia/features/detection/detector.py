"""Object detection feature implementation"""

from typing import List

import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.result.base_result import BaseResult
from ...data.models.result.detection_result import BoundingBox, DetectionResult
from ...data.models.video import VideoFramePacket
from .models import ObjectDetector


class DetectionFeature(BaseFeature):
    """Object detection feature using Grounding DINO"""
    
    # Class attributes for BaseFeature
    output_schema = DetectionResult
    required_inputs = []
    required_fields = []
    
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
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> BaseResult:
        """
        Process a single frame for object detection.
        
        Args:
            frame: VideoFramePacket containing RGB frame data
            **inputs: Additional inputs (not used by detection)
            
        Returns:
            DetectionResult containing list of detections
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        if frame.frame_data is None or frame.frame_data.size == 0:
            return DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0))
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Get detection prompts from metadata
            prompts = self._get_detection_prompts(frame)
            
            # Run detection using ObjectDetector
            detection_results = self.detector.detect_objects([image], [prompts])
            
            # Convert to DetectionResult objects
            detections = self._convert_to_detection_results(detection_results[0])
            
            # Return first detection or empty result
            if detections:
                return detections[0]
            else:
                return DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0))
            
        except Exception as e:
            raise ProcessingError(f"Error in detection processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **batch_inputs) -> List[BaseResult]:
        """
        Process multiple frames for object detection.
        
        Args:
            frames: List of VideoFramePacket objects
            **batch_inputs: Additional batch inputs (not used by detection)
            
        Returns:
            List of DetectionResult objects
        """
        if not self.is_ready():
            raise ProcessingError("Detection feature not initialized")
        
        # Filter frames with valid data
        valid_packets = [f for f in frames if f.frame_data is not None and f.frame_data.size > 0]
        
        if not valid_packets:
            return [DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)) for _ in frames]
        
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
            for packet in frames:
                if packet.frame_data is not None and packet.frame_data.size > 0:
                    if valid_idx < len(batch_detection_results):
                        detections = self._convert_to_detection_results(batch_detection_results[valid_idx])
                        if detections:
                            results.append(detections[0])
                        else:
                            results.append(DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)))
                        valid_idx += 1
                    else:
                        results.append(DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)))
                else:
                    results.append(DetectionResult(score=0.0, label="", box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)))
            
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