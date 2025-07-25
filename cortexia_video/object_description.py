import numpy as np
from PIL import Image
from typing import Optional, List, Union, Callable
import torch
import cv2
from transformers import AutoModel
import logging

class ObjectDescriber:
    def __init__(self, config_manager):
        """Initialize the object describer with NVIDIA DAM model from config.
        
        Args:
            config_manager: Instance of ConfigManager to load model config
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        model_name = self.config_manager.get_param('model_settings.description_model', 'nvidia/DAM-3B-Self-Contained')
        
        try:
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model with proper settings
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Initialize DAM with conversation mode and prompt mode as in the self-contained example
            self.dam = self.model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
            self.logger.info(f"Successfully loaded DAM model from {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def _get_bounding_box_from_mask(self, mask: np.ndarray) -> Optional[List[int]]:
        """Get bounding box coordinates from segmentation mask.
        
        Args:
            mask: Binary numpy array representing segmentation mask
            
        Returns:
            List of [x, y, w, h] coordinates or None if no contour found
        """
        # Ensure mask is binary and of proper type
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, w, h]

    def describe_object(self, 
                       image: Image.Image, 
                       segmentation_mask: np.ndarray,
                       streaming: bool = False,
                       callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate description of object in image using segmentation mask with DAM model.
        
        Args:
            image: PIL Image object
            segmentation_mask: 2D numpy array (binary mask)
            streaming: Whether to stream the output tokens
            callback: Optional callback function for streaming mode
            
        Returns:
            Generated description string
        """
        # Ensure mask is binary and of proper type
        if segmentation_mask.max() > 1:
            segmentation_mask = (segmentation_mask > 0).astype(np.uint8)
        
        # Convert mask to PIL for the DAM model
        mask_pil = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
        
        # Generate description using DAM
        prompt = '<image>\nDescribe the masked region in detail.'
        
        # Load generation parameters from config or use defaults
        temperature = self.config_manager.get_param('description_settings.temperature', 0.2)
        top_p = self.config_manager.get_param('description_settings.top_p', 0.5)
        num_beams = self.config_manager.get_param('description_settings.num_beams', 1)
        max_new_tokens = self.config_manager.get_param('description_settings.max_tokens', 512)
        # Generate description
        try:
            if streaming:
                # Stream tokens and collect into description
                description = ""
                for token in self.dam.get_description(
                    image,
                    mask_pil,
                    prompt,
                    streaming=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens
                ):
                    description += token
                    if callback:
                        callback(token)
                return description.strip()
            else:
                # Get complete description at once
                tokens = self.dam.get_description(
                    image,
                    mask_pil,
                    prompt,
                    streaming=False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens
                )
                
                description = tokens
                    
                return description.strip()
        except Exception as e:
            self.logger.error(f"Error during description generation: {str(e)}")
            return f"Error generating description: {str(e)}"