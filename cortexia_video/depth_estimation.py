from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import torch
from PIL import Image

import depth_pro


class BaseDepthEstimator(ABC):
    @abstractmethod
    def estimate_batch_depth(self, image_paths: List[str]) -> List[Dict]:
        """Estimate depth for a batch of images.
        
        Args:
            image_paths: List of paths to images to process
            
        Returns:
            List of dictionaries containing at least 'depth' and 'focallength_px' for each image
        """
        pass


class DepthEstimator(BaseDepthEstimator):
    def __init__(self):
        """Initialize the depth estimator with model and transform."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image_path: Union[str, Path]) -> Tuple[Image.Image, Optional[float]]:
        """Load an image and extract its focal length.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (PIL Image, focal length in pixels or None)
        """
        try:
            # Convert string path to Path object if needed
            path = Path(image_path) if isinstance(image_path, str) else image_path
            
            # Load image and get focal length
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Try to get focal length from image metadata if available
            # This is a simplified version - you might want to adapt depth_pro.load_rgb's
            # full implementation if needed
            f_px = None  # Default to None, letting the model estimate it
            
            return img, f_px
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

    def estimate_batch_depth(self, image_paths: List[str]) -> List[Dict]:
        """Estimate depth for a batch of images.
        
        Args:
            image_paths: List of paths to images to process
            
        Returns:
            List of dictionaries containing 'depth' and 'focallength_px' for each image
        """
        # Lists to store processed images and their focal lengths
        processed_images = []
        focal_lengths = []
        
        # Load and preprocess each image
        for image_path in image_paths:
            # Load image and get focal length
            img, f_px = self._load_image(image_path)
            
            # Apply transform
            processed_img = self.transform(img)
            
            processed_images.append(processed_img)
            focal_lengths.append(f_px)
        
        # Stack images into a batch
        batch_images = torch.stack(processed_images)
        
        # Prepare focal lengths tensor if any focal lengths are available
        batch_f_px = None
        if any(f is not None for f in focal_lengths):
            batch_f_px = torch.tensor([f if f is not None else 0.0 for f in focal_lengths],
                                    dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            # Move batch to device
            batch_images = batch_images.to(self.device)
            if batch_f_px is not None:
                batch_f_px = batch_f_px.to(self.device)
            
            predictions = self.model.infer(batch_images, f_px=batch_f_px)
        
        # Process results
        results = []
        depths = predictions['depth']  # Should be a batch of depth maps
        pred_focal_lengths = predictions['focallength_px']  # Should be a batch of focal lengths
        
        # Convert predictions to numpy if they're on GPU
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(pred_focal_lengths, torch.Tensor):
            pred_focal_lengths = pred_focal_lengths.cpu().numpy()
        
        # Create result dictionaries
        for i in range(len(image_paths)):
            results.append({
                'depth': depths[i],  # Individual depth map
                'focallength_px': pred_focal_lengths[i]  # Individual focal length
            })
        
        return results