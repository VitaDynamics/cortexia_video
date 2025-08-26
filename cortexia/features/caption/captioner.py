"""Image captioning feature implementation
"""

from typing import List

import torch
from PIL import Image

from ..base import BaseFeature 
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import VideoFramePacket
from ...data.models.result.caption_result import CaptionResult
from .models import MoonDreamCaptioner
from ..registry import feature_registry

@feature_registry.register("caption")
class CaptionFeature(BaseFeature):
    """Image captioning feature using vision-language models"""
    
    output_schema = CaptionResult
    
    def __init__(self, config=None):
        super().__init__(config)
        self.captioner = None
        self.device = None
    
    def _initialize(self):
        """Initialize captioning model"""
        if self.initialized:
            return
            
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get model configuration
            model_name = self.get_config_param("model", "vikhyatk/moondream2")
            
            # Create config dict for the captioner
            captioner_config = {
                "model": model_name,
                "revision": self.get_config_param("revision", "2025-06-21"),
                "device": self.device,
                "caption_length": self.get_config_param("caption_length", "long")
            }
            
            # Initialize captioner
            self.captioner = MoonDreamCaptioner(captioner_config)

            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize captioning model: {e}")
    
    @property
    def name(self) -> str:
        return "caption"
    
    @property
    def description(self) -> str:
        return "Image captioning using vision-language models"
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> CaptionResult:
        """
        Process a single frame for image captioning.
        
        Args:
            frame: VideoFramePacket containing RGB frame data
            **inputs: Additional inputs (not used by captioning)
            
        Returns:
            CaptionResult containing the generated caption
        """
        if not self.is_ready():
            self._initialize()
        
        if not self.is_ready():
            raise ProcessingError("Captioning feature not initialized")
        
        if frame.frame_data is None:
            return CaptionResult(caption="")
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Generate caption
            caption = self.captioner.caption_image(image)
            
            # Create and return CaptionResult
            return CaptionResult(
                caption=caption,
                model_name=self.get_config_param("model", "vikhyatk/moondream2"),
                caption_length=self.get_config_param("caption_length", "long")
            )
            
        except Exception as e:
            raise ProcessingError(f"Error in captioning processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[CaptionResult]:
        """
        Process multiple frames for image captioning.
        
        Args:
            frames: List of VideoFramePacket objects
            **inputs: Additional inputs (not used by captioning)
            
        Returns:
            List of CaptionResult objects
        """
        if not self.is_ready():
            self._initialize()
        
        if not self.is_ready():
            raise ProcessingError("Captioning feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.frame_data is not None]
        
        if not valid_frames:
            return [CaptionResult(caption="") for _ in frames]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_frames]
            
            # Generate batch captions
            batch_captions = [self.captioner.caption_image(img) for img in images]
            
            # Create CaptionResult objects
            results = []
            valid_idx = 0
            for frame in frames:
                if frame.frame_data is not None and valid_idx < len(batch_captions):
                    results.append(CaptionResult(
                        caption=batch_captions[valid_idx],
                        model_name=self.get_config_param("model", "vikhyatk/moondream2"),
                        caption_length=self.get_config_param("caption_length", "long")
                    ))
                    valid_idx += 1
                else:
                    results.append(CaptionResult(caption=""))
            
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error in batch captioning processing: {e}")