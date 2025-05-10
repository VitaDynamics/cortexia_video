from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from cortexia_video.config_manager import ConfigManager

class ObjectDetector:
    def __init__(self, config_manager: ConfigManager):
        """Initialize the object detector with model from config.
        
        Args:
            config_manager: ConfigManager instance with model settings
        """
        self.config = config_manager
        model_name = self.config.get_param("model_settings.object_detection_model", 
                                   "IDEA-Research/GroundingDINO-tiny")
        
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load object detection model: {str(e)}")

    def detect_objects(self, image_data: Any, text_prompt: str) -> List[Dict[str, Any]]:
        """Detect objects in image based on text prompt.
        
        Args:
            image_data: PIL Image or numpy array
            text_prompt: Text describing objects to detect (e.g. "a cat . a dog .")
            
        Returns:
            List of detections with boxes, labels and scores
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image_data = Image.fromarray(image_data)
            
        inputs = self.processor(images=image_data, text=text_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert model outputs to usable format
        target_sizes = torch.tensor([image_data.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids, 
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=target_sizes
        )
        
        detections = []
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                # Convert box from [x_center, y_center, width, height] to [x1, y1, x2, y2]
                x1 = box[0] - box[2]/2
                y1 = box[1] - box[3]/2
                x2 = box[0] + box[2]/2
                y2 = box[1] + box[3]/2
                
                detections.append({
                    "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                    "label": label,
                    "score": score.item()
                })
        
        return detections