import os, sys
import torch
import numpy as np
from PIL import Image
import decord
from abc import ABC, abstractmethod
from typing import Any, List, Optional

# Add path from perception_models to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception_models"))

import perception_models.core.vision_encoder.pe as pe
import perception_models.core.vision_encoder.transforms as pe_transforms

# TODO: need to be integrated with extractor.py 


def preprocess_video(video_path, num_frames=8, transform=None, return_first_frame_for_demo=False):
    """
    Uniformly samples a specified number of frames from a video and preprocesses them.
    
    Parameters:
    - video_path: str, path to the video file.
    - num_frames: int, number of frames to sample. Defaults to 8.
    - transform: torchvision.transforms, a transform function to preprocess frames.
    - return_first_frame_for_demo: bool, whether to return the first frame. Defaults to False.
    
    Returns:
    - Video Tensor: a tensor of shape (num_frames, 3, H, W) where H and W are the height and width of the frames.
    - First frame (optional): the first frame as a numpy array if return_first_frame_for_demo is True.
    """
    # Load the video
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    # Uniformly sample frame indices
    frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    # Preprocess frames
    # Preprocess frames if transform is provided
    if transform is not None:
        preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    else:
        # Handle case where no transform is provided (convert to tensor)
        preprocessed_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]

    first_frame = None
    if return_first_frame_for_demo:
        first_frame = frames[0]
    return torch.stack(preprocessed_frames, dim=0), first_frame


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from images and videos.
    """
    def __init__(self, config_manager: Any):
        """
        Initialize the FeatureExtractor with configuration manager.
        
        Args:
            config_manager: Instance of ConfigManager to get model settings
        """
        self.config_manager = config_manager

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
    CLIP-based feature extractor for images and videos.
    """
    def __init__(self, config_manager: Any):
        """
        Initialize the CLIPFeatureExtractor with configuration manager.
        
        Args:
            config_manager: Instance of ConfigManager to get model settings
        """
        super().__init__(config_manager)
        model_name = config_manager.get_param('model_settings.clip_feature_model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model.to(self.device)
        self.image_preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)

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
            # Get feature dimension for CLIP visual model
            feature_dim = self.model.visual.output_dim if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'output_dim') else 512
            return torch.empty((0, feature_dim), device=self.device)
            
        # Preprocess all images in the batch
        processed_images_list = []
        for img in images_data:
            # The image_preprocess transforms PIL images to tensors
            processed_tensor = self.image_preprocess(img)
            # Ensure it's a tensor (this helps with type checking)
            if isinstance(processed_tensor, torch.Tensor):
                processed_images_list.append(processed_tensor)
            else:
                # In case it's not already a tensor, convert it (shouldn't happen with proper preprocess)
                processed_images_list.append(torch.tensor(processed_tensor))
        
        # Stack the tensors into a batch
        batch_image_tensors = torch.stack(processed_images_list).to(self.device)
        
        # Encode images
        with torch.no_grad():
            batch_features = self.model.encode_image(batch_image_tensors)
            
        # Normalize features
        batch_features /= batch_features.norm(dim=-1, keepdim=True)
        
        return batch_features

    def extract_video_features(self, video_path: str, num_frames: int = 8) -> torch.Tensor:
        """
        Extract features from a video using CLIP.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample. Defaults to 8.
            
        Returns:
            torch.Tensor: Extracted video features
        """
        video_frames, _ = preprocess_video(
            video_path, 
            num_frames=num_frames, 
            transform=self.image_preprocess
        )
        video_inputs = video_frames.unsqueeze(0).to(self.device)
        with torch.no_grad():
            video_features = self.model.encode_video(video_inputs)
        return video_features

    def extract_text_features(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Extract features from text prompts using CLIP.
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            torch.Tensor: Extracted text features
        """
        text_inputs = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        # Ensure we return a Tensor, not a tuple
        if isinstance(text_features, tuple):
            text_features = text_features[0]
        return text_features

    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> np.ndarray:
        """
        Calculate similarity between image/video features and text features.
        
        Args:
            image_features: Image or video features
            text_features: Text features
            
        Returns:
            np.ndarray: Similarity scores
        """
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Return probabilities
        # If the input was for a single image (shape [1, D]), return text_probs[0]
        if image_features.shape[0] == 1:
            return text_probs.cpu().numpy()[0]
        return text_probs.cpu().numpy()


# Registry for feature extractors
FEATURE_EXTRACTOR_REGISTRY = {
    "clip_pe": CLIPFeatureExtractor,
}
