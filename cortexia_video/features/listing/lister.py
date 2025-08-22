"""Object listing feature implementation"""

from typing import List

from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import FrameData
from .models import OBJECT_LISTER_REGISTRY


class ListingFeature(BaseFeature):
    """Object listing feature using various models"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.lister = None
    
    def _initialize(self):
        """Initialize listing model using ObjectLister registry"""
        try:
            # Get model configuration
            model_name = self.get_config_param("model", "vikhyatk/moondream2")
            
            # Find appropriate ObjectLister class
            lister_class = None
            for registry_key, lister_cls in OBJECT_LISTER_REGISTRY.items():
                if registry_key in model_name:
                    lister_class = lister_cls
                    break
            
            if lister_class is None:
                raise ValueError(f"Unsupported listing model: {model_name}. Available models: {list(OBJECT_LISTER_REGISTRY.keys())}")
            
            # Initialize ObjectLister with the same config dict
            self.lister = lister_class(self.config)
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize listing model: {e}")
    
    
    @property
    def name(self) -> str:
        return "listing"
    
    @property
    def description(self) -> str:
        return "Object listing using various vision-language models"
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for object listing.
        
        Args:
            frame_data: Frame data containing RGB image
            
        Returns:
            Frame data with listing results added
        """
        if not self.is_ready():
            raise ProcessingError("Listing feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Run listing
            objects = self._list_objects(image)
            
            # Add listing results to frame data
            frame_data.lister_results = objects
            
            # Generate DINO prompt for detection
            if objects:
                frame_data.dino_prompt = ". ".join(objects) + "."
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in listing processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for object listing.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with listing results added
        """
        if not self.is_ready():
            raise ProcessingError("Listing feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Run batch listing
            batch_objects = self._list_objects_batch(images)
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_objects):
                    objects = batch_objects[i]
                    frame.lister_results = objects
                    
                    # Generate DINO prompt
                    if objects:
                        frame.dino_prompt = ". ".join(objects) + "."
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch listing processing: {e}")
    
    def _list_objects(self, image: Image.Image) -> List[str]:
        """
        List objects in a single image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of detected object names
        """
        return self.lister.list_objects_in_image(image)
    
    def _list_objects_batch(self, images: List[Image.Image]) -> List[List[str]]:
        """
        List objects in batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of object lists for each image
        """
        return self.lister.list_objects_in_image_batched(images)
    
