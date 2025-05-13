import torch
import numpy as np
from typing import List, Any, Optional, Tuple, Union
from transformers import SamModel, SamProcessor
from PIL import Image


class ObjectSegmenter:
    def __init__(self, config_manager):
        """
        Initialize the ObjectSegmenter with SAM model and processor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load SAM model and processor from config"""
        model_name = self.config_manager.get_param('model_settings.segmentation_model', 'facebook/sam-vit-base')
        try:
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.processor = SamProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {str(e)}")
    
    def segment_object(self, image_data: Image.Image, input_boxes: List[List[float]], 
                       input_points: Optional[List[List[float]]] = None,
                       input_labels: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        Segment objects in image using bounding boxes or points as prompts.
        
        Args:
            image_data: Input image (PIL Image)
            input_boxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            input_points: Optional list of points [[x1, y1], [x2, y2], ...]
            input_labels: Optional list of labels for points [1, 1, ...] (1=foreground, 0=background)
            
        Returns:
            List of binary masks (one for each input box or points set)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
            
        try:
            # Process inputs for the model
            inputs = self.processor(
                images=image_data, 
                input_boxes=[input_boxes], # ne
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            
            # Get segmentation masks
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process masks
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0][0]
            
            # Select best mask using IoU scores
            scores = outputs.iou_scores[0, 0]
            
            mask_selection_index = scores.argmax()
            mask_np = masks[mask_selection_index].numpy()
            
            return mask_np
            
        except Exception as e:
            raise RuntimeError(f"Error during segmentation: {str(e)}")
    
    def generate_contour_visualization(self, image: Image.Image, mask: np.ndarray, 
                                       input_boxes: Optional[List[List[float]]] = None,
                                       input_points: Optional[List[List[float]]] = None) -> Image.Image:
        """
        Generate a visualization of segmentation contours on the image.
        
        Args:
            image: PIL Image object
            mask: Binary segmentation mask
            input_boxes: Optional bounding boxes used for segmentation
            input_points: Optional points used for segmentation
            
        Returns:
            PIL Image with visualization
        """
        import cv2
        
        # Convert PIL to numpy for OpenCV
        img_np = np.array(image).astype(float) / 255.0
        img = img_np.copy()
        
        # Create contours from mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=3)
        
        # Draw input points if provided
        if input_points is not None:
            for points in input_points:
                for x, y in points:
                    cv2.circle(img, (int(x), int(y)), radius=8, color=(1.0, 0.0, 0.0), thickness=-1)
                    cv2.circle(img, (int(x), int(y)), radius=8, color=(1.0, 1.0, 1.0), thickness=2)
        
        # Draw input boxes if provided
        if input_boxes is not None:
            for box in input_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=1)
        
        # Convert back to PIL
        img_with_contour_pil = Image.fromarray((img * 255.0).astype(np.uint8))
        return img_with_contour_pil