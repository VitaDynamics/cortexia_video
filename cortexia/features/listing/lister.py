"""Object listing feature implementation"""

from typing import List

from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import VideoFramePacket
from ...data.models.result.tagging_result import TaggingResult
from ..registry import feature_registry
from .models import MoonDreamLister

@feature_registry.register("listing")
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

            # Initialize ObjectLister with the same config dict
            self.lister = MoonDreamLister(self.config)
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize listing model: {e}")
    
    
    @property
    def name(self) -> str:
        return "listing"
    
    @property
    def description(self) -> str:
        return "Object listing using various vision-language models"
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> TaggingResult:
        """
        Process a single frame for object listing.
        
        Args:
            frame: VideoFramePacket containing RGB frame data
            **inputs: Additional inputs (not used by listing)
            
        Returns:
            TaggingResult containing list of detected objects
        """
        if not self.is_ready():
            self._initialize()
        
        if frame.frame_data is None:
            return TaggingResult(tags=[])
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Run listing
            objects = self._list_objects(image)
            
            # Create and return TaggingResult
            return TaggingResult(
                tags=objects,
                model_name=self.get_config_param("model", "vikhyatk/moondream2"),
                raw_response=". ".join(objects) + "." if objects else ""
            )
            
        except Exception as e:
            raise ProcessingError(f"Error in listing processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[TaggingResult]:
        """
        Process multiple frames for object listing.
        
        Args:
            frames: List of VideoFramePacket objects
            **inputs: Additional inputs (not used by listing)
            
        Returns:
            List of TaggingResult objects
        """
        if not self.is_ready():
            self._initialize()
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.frame_data is not None]
        
        if not valid_frames:
            return [TaggingResult(tags=[]) for _ in frames]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_frames]
            
            # Run batch listing
            batch_objects = self._list_objects_batch(images)
            
            # Create TaggingResult objects
            results = []
            valid_idx = 0
            for frame in frames:
                if frame.frame_data is not None and valid_idx < len(batch_objects):
                    objects = batch_objects[valid_idx]
                    results.append(TaggingResult(
                        tags=objects,
                        model_name=self.get_config_param("model", "vikhyatk/moondream2"),
                        raw_response=". ".join(objects) + "." if objects else ""
                    ))
                    valid_idx += 1
                else:
                    results.append(TaggingResult(tags=[]))
            
            return results
            
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
        results = []
        for image in images:
            results.append(self.lister.list_objects_in_image(image))
        return results
