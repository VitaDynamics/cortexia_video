import os, sys
import torch
import numpy as np
from PIL import Image
import decord
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import open_clip


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from images and videos.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeatureExtractor with a configuration dict.

        Args:
            config: Plain dict of configuration values
        """
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def extract_image_features(self, images_data: List[Image.Image]) -> torch.Tensor:
        """
        Extract features from one or more images.
        
        Args:
            images_data: List of input images (PIL Images)
            
        Returns:
            torch.Tensor: Extracted features for all images
        """
        pass

    @abstractmethod
    def extract_video_features(self, video_path: str, num_frames: int) -> torch.Tensor:
        """
        Extract features from a video.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            
        Returns:
            torch.Tensor: Extracted features
        """
        pass

    @abstractmethod
    def calculate_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> np.ndarray:
        """
        Calculate similarity between two feature sets.
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            np.ndarray: Similarity scores
        """
        pass


class CLIPFeatureExtractor(FeatureExtractor):
    """
    CLIP-based feature extractor for images and videos using open_clip.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CLIPFeatureExtractor with configuration dict.

        Expected config keys (all optional):
        - 'model_settings.clip_feature_model' or 'clip_feature_model' or 'model': model identifier for open_clip
        - 'device': explicit device string
        """
        super().__init__(config)
        # Resolve model name from multiple possible keys to preserve compatibility
        model_name = (
            self.config.get('model_settings', {}).get('clip_feature_model')
            or self.config.get('clip_feature_model')
            or self.config.get('model')
            or "ViT-B-32"
        )
        device_str = self.config.get("device")
        self.device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def extract_image_features(self, images_data: List[Image.Image]) -> torch.Tensor:
        """
        Extract features from one or more images using CLIP.
        
        Args:
            images_data: List of input images (PIL Images)
            
        Returns:
            torch.Tensor: Extracted image features for all images
        """
        # Handle empty input
        if not images_data:
            # Get feature dimension for CLIP model (typically 512 for ViT-B-32)
            feature_dim = 512
            return torch.empty((0, feature_dim), device=self.device)
            
        # Preprocess all images in the batch
        processed_images_list = []
        for img in images_data:
            # The preprocess transforms PIL images to tensors
            processed_tensor = self.preprocess(img)
            processed_images_list.append(processed_tensor)
        
        # Stack the tensors into a batch
        batch_image_tensors = torch.stack(processed_images_list).to(self.device)
        
        # Encode images
        with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image_features = self.model.encode_image(batch_image_tensors)
            
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features


    def extract_text_features(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Extract features from text prompts using CLIP.
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            torch.Tensor: Extracted text features
        """
        text_inputs = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            text_features = self.model.encode_text(text_inputs)
        # Normalize features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def calculate_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> np.ndarray:
        """
        Calculate similarity between two feature sets.
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            np.ndarray: Similarity scores
        """
        # Normalize features
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * features1 @ features2.T).softmax(dim=-1)
        
        # Return probabilities
        # If the input was for a single feature (shape [1, D]), return similarity[0]
        if features1.shape[0] == 1:
            return similarity.cpu().numpy()[0]
        return similarity.cpu().numpy()


