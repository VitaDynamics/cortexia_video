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
    
    def create_visualization(self, 
                           image: Image.Image, 
                           mask: np.ndarray, 
                           description: str = None) -> Image.Image:
        """Create a visualization of the masked region with optional description.
        
        Args:
            image: PIL Image object
            mask: Binary segmentation mask
            description: Optional description text to overlay
            
        Returns:
            PIL Image with visualization
        """
        # Convert mask to proper format
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
            
        # Create contour visualization
        img_np = np.array(image).astype(float) / 255.0
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Create a copy for drawing
        img_vis = img_np.copy()
        
        # Draw contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_vis, contours, -1, (1.0, 1.0, 1.0), 2)
        
        # Create semi-transparent colored overlay for the mask
        overlay = img_np.copy()
        mask_area = (mask > 0)
        overlay[mask_area] = overlay[mask_area] * 0.7 + np.array([0.3, 0.0, 0.5])  # Purple tint
        
        # Blend overlay with contour image
        alpha = 0.5
        img_vis = cv2.addWeighted(img_vis, 1-alpha, overlay, alpha, 0)
        
        # Convert back to PIL
        result_img = Image.fromarray((img_vis * 255).astype(np.uint8))
        
        # Add description text if provided
        if description and len(description) > 0:
            # Get bounding box for text placement
            bbox = self._get_bounding_box_from_mask(mask)
            if bbox:
                # Use PIL for text
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(result_img)
                
                # Limit description length for visualization
                max_display_chars = 100
                display_text = description[:max_display_chars]
                if len(description) > max_display_chars:
                    display_text += "..."
                
                # Position text above the bounding box
                x, y, w, h = bbox
                try:
                    # Try to get a font, or use default
                    font = ImageFont.truetype("arial.ttf", 14)
                except IOError:
                    font = ImageFont.load_default()
                    
                # Draw white text with black outline for visibility
                text_position = (x, max(0, y - 20))
                draw.text((text_position[0]-1, text_position[1]-1), display_text, (0,0,0), font=font)
                draw.text((text_position[0]+1, text_position[1]-1), display_text, (0,0,0), font=font)
                draw.text((text_position[0]-1, text_position[1]+1), display_text, (0,0,0), font=font)
                draw.text((text_position[0]+1, text_position[1]+1), display_text, (0,0,0), font=font)
                draw.text(text_position, display_text, (255,255,255), font=font)
                
        return result_img