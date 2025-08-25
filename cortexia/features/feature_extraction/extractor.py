"""Feature extraction implementation"""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import VideoFramePacket
from ...data.models.result.feature_extraction_result import FeatureExtractionResult
from .models import CLIPFeatureExtractor

from ..registry import feature_registry

@feature_registry.register("feature_extraction")
class FeatureExtractionFeature(BaseFeature):
    """Feature extraction using CLIP and other models"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.engine = None  # Backend from models.py (PE-only path)
    
    def _initialize(self):
        """Initialize feature extraction model"""
        try:
            # Initialize PE engine (no fallback)
            self.engine = CLIPFeatureExtractor(self.config)
            self.initialized = True
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize PE feature extraction engine: {e}")
    
    @property
    def name(self) -> str:
        return "feature_extraction"
    
    @property
    def description(self) -> str:
        return "Feature extraction using CLIP and other vision models"
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> FeatureExtractionResult:
        """
        Process a single frame for feature extraction.
        
        Args:
            frame_data: Frame data containing RGB image
            
        Returns:
            Frame data with features added
        """
        if not self.is_ready():
            raise ProcessingError("Feature extraction feature not initialized")
        
        if frame.frame_data is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Extract scene features
            scene_features = self._extract_scene_features(image)
            frame.add_annotation_result('scene_clip_features', scene_features)
            
            # Extract object features if detections are available
            if frame.annotations and frame.annotations.detections:
                object_features = self._extract_object_features(image, frame.annotations.detections)
                self._add_object_features_to_detections(object_features, frame.annotations.detections)
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in feature extraction processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[FeatureExtractionResult]:
        """
        Process multiple frames for feature extraction.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with features added
        """
        if not self.is_ready():
            raise ProcessingError("Feature extraction feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Extract scene features in batch
            batch_scene_features = self._extract_scene_features_batch(images)
            
            # Extract object features
            batch_object_features = []
            for i, frame in enumerate(valid_frames):
                if frame.detections:
                    object_features = self._extract_object_features(images[i], frame.detections)
                    batch_object_features.append(object_features)
                else:
                    batch_object_features.append([])
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_scene_features):
                    frame.scene_clip_features = batch_scene_features[i]
                
                if i < len(batch_object_features):
                    self._add_object_features_to_detections(
                        batch_object_features[i], frame.detections
                    )
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch feature extraction processing: {e}")
    
    def _extract_scene_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract scene-level features from image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Scene feature vector
        """
        # Use PE engine only
        feats = self.engine.extract_image_features([image])  # Tensor [1, D]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        return np.asarray(feats)[0].flatten()
    
    def _extract_scene_features_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Extract scene-level features from batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of scene feature vectors
        """
        # Use PE engine only
        feats = self.engine.extract_image_features(images)  # Tensor [N, D]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        return [np.asarray(f).flatten() for f in feats]
    
    def _extract_object_features(self, image: Image.Image, detections: List) -> List[np.ndarray]:
        """
        Extract object-level features from image detections.
        
        Args:
            image: PIL Image to process
            detections: List of detection results
            
        Returns:
            List of object feature vectors
        """
        if not detections:
            return []
        
        object_features = []
        
        for detection in detections:
            # Crop image to detection bounding box
            cropped_image = self._crop_detection(image, detection.box)
            
            # Extract features for cropped object
            features = self._extract_scene_features(cropped_image)
            object_features.append(features)
        
        return object_features
    
    def _extract_object_features_batch(self, images: List[Image.Image], detections_list: List[List]) -> List[List[np.ndarray]]:
        """
        Extract object-level features from batch of images.
        
        Args:
            images: List of PIL Images
            detections_list: List of detection lists for each image
            
        Returns:
            List of object feature vector lists
        """
        results = []
        
        for i, image in enumerate(images):
            detections = detections_list[i] if i < len(detections_list) else []
            object_features = self._extract_object_features(image, detections)
            results.append(object_features)
        
        return results
    
    def _crop_detection(self, image: Image.Image, bbox) -> Image.Image:
        """
        Crop image to detection bounding box.
        
        Args:
            image: PIL Image to crop
            bbox: Bounding box coordinates
            
        Returns:
            Cropped PIL Image
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        
        # Add padding
        padding = self.get_config_param("crop_padding", 10)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Crop image
        cropped = image.crop((x1, y1, x2, y2))
        
        return cropped
    
    def _add_object_features_to_detections(self, object_features: List[np.ndarray], detections: List):
        """
        Add object features to detection objects.
        
        Args:
            object_features: List of object feature vectors
            detections: List of detection results
        """
        for i, features in enumerate(object_features):
            if i < len(detections):
                detections[i].object_clip_features = features.tolist()
