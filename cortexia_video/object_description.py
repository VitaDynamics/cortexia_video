import numpy as np
from PIL import Image
from typing import Optional, List
import torch
import cv2
from transformers import AutoModel

class ObjectDescriber:
    def __init__(self, config_manager):
        """Initialize the object describer with NVIDIA DAM model from config.
        
        Args:
            config_manager: Instance of ConfigManager to load model config
        """
        model_name = config_manager.model_settings.description_model
        try:
            # Default to NVIDIA DAM if not specified
            if not model_name:
                model_name = "nvidia/DAM-3B-Self-Contained"
                
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Initialize DAM with conversation mode and prompt mode
            self.dam = self.model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def _get_bounding_box_from_mask(self, mask: np.ndarray) -> Optional[List[int]]:
        """Get bounding box coordinates from segmentation mask.
        
        Args:
            mask: Binary numpy array representing segmentation mask
            
        Returns:
            List of [x, y, w, h] coordinates or None if no contour found
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, w, h]

    def describe_object(self, image: Image.Image, segmentation_mask: np.ndarray) -> str:
        """Generate description of object in image using segmentation mask with DAM model.
        
        Args:
            image: PIL Image object
            segmentation_mask: 2D numpy array (binary mask)
            
        Returns:
            Generated description string
        """
        if segmentation_mask.max() > 1:
            # Normalize mask if necessary
            segmentation_mask = (segmentation_mask > 0).astype(np.uint8)
        
        # Convert mask to PIL for the DAM model
        mask_pil = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
        
        # Generate description using DAM
        prompt = '<image>\nDescribe the masked region in detail.'
        description = ""
        
        # Call the DAM model
        tokens = self.dam.get_description(
            image,
            mask_pil,
            prompt,
            streaming=False,
            temperature=0.2,
            top_p=0.5,
            num_beams=1,
            max_new_tokens=512
        )
        
        if isinstance(tokens, list):
            description = "".join(tokens)
        else:
            description = tokens
            
        return description.strip()