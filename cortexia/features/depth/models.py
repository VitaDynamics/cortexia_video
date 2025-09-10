from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gc
import torch
from PIL import Image
from transformers import (
    DepthProImageProcessorFast,
    DepthProForDepthEstimation,
)


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


class DepthProEstimator(BaseDepthEstimator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the depth estimator with model and transform.

        Args:
            config: Optional configuration with ``device_map`` controlling model
                placement. Defaults to "auto" when multiple GPUs are available,
                otherwise the current device ("cuda" or "cpu").
        """
        self.config = config or {}
        model_name = self.config.get("model", "apple/DepthPro-hf")
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = self.config.get(
            "device_map",
            "auto" if torch.cuda.device_count() > 1 else default_device,
        )
        self.processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.device = torch.device(next(iter(self.model.hf_device_map.values())))
        self.model.eval()

    def release(self) -> None:
        """Release model, processor, and free GPU/CPU memory."""
        try:
            if getattr(self, "model", None) is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                del self.model
                self.model = None
            # Processor is CPU-side; drop reference
            if getattr(self, "processor", None) is not None:
                del self.processor
                self.processor = None
        finally:
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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
            img = Image.open(image_path)
            return img, None

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

        is_path_input = isinstance(inputs[0], (str, Path))

        images = []
        target_sizes = []
        for item in inputs:
            if is_path_input:
                img, _ = self._load_image(item)
            else:
                img = item
            images.append(img)
            target_sizes.append((img.height, img.width))

        model_inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        post_processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=target_sizes
        )

        results = []
        for out in post_processed:
            depth_array = out["predicted_depth"].detach().cpu().numpy()
            results.append({"depth": depth_array, "focallength_px": out["focal_length"]})

        return results
