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

    def _release(self) -> None:
        """Release feature extraction engine and free memory."""
        try:
            if self.engine is not None and hasattr(self.engine, "release"):
                try:
                    self.engine.release()
                except Exception:
                    pass
        finally:
            self.engine = None
    
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
            frame: Frame data containing RGB image
            
        Returns:
            FeatureExtractionResult containing extracted features
        """
        if not self.is_ready():
            self._initialize()
        
        if frame.frame_data is None:
            return FeatureExtractionResult(features=np.array([]))
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame.frame_data)
            
            # Extract scene features
            scene_features = self._extract_scene_features(image)
            
            # Extract object features if detections are available
            object_features = []
            if frame.annotations and frame.annotations.detections:
                object_features = self._extract_object_features(image, frame.annotations.detections)
            
            # Combine scene and object features into a single result
            # For now, we'll prioritize scene features as the main result
            return FeatureExtractionResult(
                features=scene_features,
                feature_dim=len(scene_features),
                model_name="CLIP",
                model_version="1.0"
            )
            
        except Exception as e:
            raise ProcessingError(f"Error in feature extraction processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **inputs) -> List[FeatureExtractionResult]:
        """
        Process multiple frames for feature extraction.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of FeatureExtractionResult objects
        """
        if not self.is_ready():
            self._initialize()
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.frame_data is not None]
        
        if not valid_frames:
            return [FeatureExtractionResult(features=np.array([])) for _ in frames]
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.frame_data) for f in valid_frames]
            
            # Extract scene features in batch
            batch_scene_features = self._extract_scene_features_batch(images)
            
            # Extract object features
            batch_object_features = []
            for i, frame in enumerate(valid_frames):
                if frame.annotations and frame.annotations.detections:
                    object_features = self._extract_object_features(images[i], frame.annotations.detections)
                    batch_object_features.append(object_features)
                else:
                    batch_object_features.append([])
            
            # Create results
            results = []
            for i, frame in enumerate(frames):
                if frame.frame_data is None:
                    results.append(FeatureExtractionResult(features=np.array([])))
                elif i < len(batch_scene_features):
                    scene_features = batch_scene_features[i]
                    results.append(FeatureExtractionResult(
                        features=scene_features,
                        feature_dim=len(scene_features),
                        model_name="CLIP",
                        model_version="1.0"
                    ))
                else:
                    results.append(FeatureExtractionResult(features=np.array([])))
            
            return results
            
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
