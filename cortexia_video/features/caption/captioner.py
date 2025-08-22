"""Image captioning feature implementation
"""

from typing import List

import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import FrameData
from .models import IMAGE_CAPTIONER_REGISTRY


class CaptionFeature(BaseFeature):
    """Image captioning feature using vision-language models"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.captioner = None
        self.device = None
    
    def _initialize(self):
        """Initialize captioning model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get model configuration
            model_name = self.get_config_param("model", "vikhyatk/moondream2")
            
            # Get captioner class from registry
            captioner_class = IMAGE_CAPTIONER_REGISTRY.get(model_name)
            if not captioner_class:
                raise ModelLoadError(f"Model {model_name} not found in registry")
            
            # Create config dict for the captioner
            captioner_config = {
                "model": model_name,
                "revision": self.get_config_param("revision", "2025-06-21"),
                "device": self.device,
                "caption_length": self.get_config_param("caption_length", "long")
            }
            
            # Initialize captioner
            self.captioner = captioner_class(captioner_config)
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize captioning model: {e}")
    
    @property
    def name(self) -> str:
        return "caption"
    
    @property
    def description(self) -> str:
        return "Image captioning using vision-language models"
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for image captioning.
        
        Args:
            frame_data: Frame data containing RGB image
            
        Returns:
            Frame data with caption added
        """
        if not self.is_ready():
            raise ProcessingError("Captioning feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Generate caption
            caption = self.captioner.caption_image(image)
            
            # Add caption to frame data
            frame_data.caption = caption
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in captioning processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for image captioning.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with captions added
        """
        if not self.is_ready():
            raise ProcessingError("Captioning feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Generate batch captions
            batch_captions = [self.captioner.caption_image(img) for img in images]
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_captions):
                    frame.caption = batch_captions[i]
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch captioning processing: {e}")