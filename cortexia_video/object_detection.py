from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
from cortexia_video.config_manager import ConfigManager
from cortexia_video.schemes import DetectionResult, BoundingBox


# Add SAHI inference during detection.

class ObjectDetector:
    def __init__(self, config_manager: ConfigManager):
        """Initialize the object detector with model from config.
        
        Args:
            config_manager: ConfigManager instance with model settings
        """
        self.config = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize grounding dino model
        model_name = self.config.get_param("model_settings.object_detection_model", 
                                  "IDEA-Research/grounding-dino-base")
        
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load object detection model: {str(e)}")
        
        # SAM model is not directly integrated in this class to avoid compatibility issues
        # Use the standalone ObjectSegmenter class for mask generation

    def detect_objects(self, image_data: Union[Image.Image, np.ndarray], text_prompt: str, 
                       box_threshold: float = 0.3, text_threshold: float = 0.3):
        """Detect objects in image based on text prompt.
        
        Args:
            image_data: PIL Image or numpy array
            text_prompt: Text describing objects to detect (e.g. "dog. cat. person.")
                        IMPORTANT: text queries need to be lowercased + end with a dot
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            List of detection dictionaries with boxes, labels and scores
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image_data = Image.fromarray(image_data)
            
        inputs = self.processor(images=image_data, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert model outputs to usable format
        target_sizes = torch.tensor([image_data.size[::-1]]).to(self.device)
        #FIXME: there is a warning here, but it's not clear what it is about grounding dino label. 
        results = self.processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids, 
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        
        detections = []
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result.get("text_labels", result.get("labels", []))  # Handle both old and new versions of the API
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                
                detection = DetectionResult(
                    score=score.item(),
                    label=label,
                    box=BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
                )
                
                detections.append(detection)
        
        return results, detections