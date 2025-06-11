from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import depth_pro
import torch
from PIL import Image


class BaseDepthEstimator(ABC):
    @abstractmethod
    def estimate_batch_depth(
        self, inputs: Union[List[str], List[Image.Image]]
    ) -> List[Dict]:
        """Estimate depth for a batch of images.

        Args:
            inputs: List of image paths (str) or PIL Images to process

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

    def _load_image(
        self, image_path: Union[str, Path]
    ) -> Tuple[Image.Image, Optional[float]]:
        """Load an image and extract its focal length.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (PIL Image, focal length in pixels or None)
        """
        try:
            img, _, f_px = depth_pro.load_rgb(image_path)
            return img, f_px

        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

    def estimate_batch_depth(
        self, inputs: Union[List[str], List[Image.Image]]
    ) -> List[Dict]:
        """Estimate depth for a batch of images.

        Args:
            inputs: List of image paths (str) or PIL Images to process

        Returns:
            List of dictionaries containing 'depth' and 'focallength_px' for each image
        """
        if not inputs:
            return []

        # Detect input type by checking the first element
        is_path_input = isinstance(inputs[0], (str, Path))

        # Lists to store processed images and their focal lengths
        processed_images = []
        focal_lengths = []

        if is_path_input:
            # Handle list of file paths
            for image_path in inputs:
                # Load image and get focal length
                img, f_px = self._load_image(image_path)

                # Apply transform
                processed_img = self.transform(img)

                processed_images.append(processed_img)
                focal_lengths.append(f_px)
        else:
            # Handle list of PIL Images
            for img in inputs:
                # Apply transform to the PIL image
                processed_img = self.transform(img)
                processed_images.append(processed_img)
                # No focal length info available from PIL images
                focal_lengths.append(None)

        # Stack images into a batch
        batch_images = torch.stack(processed_images)

        # Prepare focal lengths tensor if any focal lengths are available
        batch_f_px = None
        if any(f is not None for f in focal_lengths):
            batch_f_px = torch.tensor(
                [f if f is not None else 0.0 for f in focal_lengths],
                dtype=torch.float32,
            )

        # Run inference
        with torch.no_grad():
            # Move batch to device
            batch_images = batch_images.to(self.device)
            if batch_f_px is not None:
                batch_f_px = batch_f_px.to(self.device)

            predictions = self.model.infer(batch_images, f_px=batch_f_px)

        # Process results
        results = []
        depths = predictions["depth"]  # Should be a batch of depth maps
        pred_focal_lengths = predictions[
            "focallength_px"
        ]  # Should be a batch of focal lengths

        # Convert predictions to numpy if they're on GPU
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(pred_focal_lengths, torch.Tensor):
            pred_focal_lengths = pred_focal_lengths.cpu().numpy()

        # Create result dictionaries
        for i in range(len(inputs)):
            results.append(
                {
                    "depth": depths[i],  # Individual depth map
                    "focallength_px": pred_focal_lengths[i],  # Individual focal length
                }
            )

            return results