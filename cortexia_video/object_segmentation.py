import torch
from typing import List, Any
from transformers import AutoModelForImageSegmentation, AutoProcessor
from typing import Optional, List, Tuple
from PIL import Image
from typing import Any

class ObjectSegmenter:
    def __init__(self, config_manager):
        """
        Initialize the ObjectSegmenter with SAM model and processor.
        
        Args:
            model_name (str): Name of the SAM model to load
        """
        self.config_manager = config_manager
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load SAM model and processor from config"""
        model_name = self.config_manager.get_param('model_settings.segmentation_model')
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {str(e)}")
    
    def segment_object(self, image_data: Any, input_boxes: List[List[int]]) -> List[Any]:
        """
        Segment objects in image using bounding boxes as prompts.
        
        Args:
            image_data: Input image (PIL Image or compatible)
            input_boxes: List of bounding boxes [x_min,  y_min, x_max, y_max]
            
        Returns:
            List of binary masks (one for each input box)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
            
        try:
            # Convert boxes to format expected by SAM (list of [x1, y1,  x2, y2] coordinates)
            boxes = torch.tensor(input_boxes, dtype=torch.float32)
            
            # Process image and boxes
            inputs = self.processor(images=image_data, input_boxes=boxes, return_tensors="pt")
            
            # Get segmentation masks
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process masks (convert logits to binary masks)
            masks = (outputs.logits > 0).squeeze(1).int().numpy()
            
            return masks.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Error during segmentation: {str(e)}")