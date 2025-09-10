from abc import ABC, abstractmethod
from typing import Any

import gc
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
import torch


class ImageCaptioner(ABC):
    """Abstract base class for generating captions for images."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def caption_image(self, image_data: Any) -> str:
        """Return a caption describing the given image."""
        pass

    def release(self) -> None:
        """Release any resources held by the captioner (no-op by default)."""
        return None

class MoonDreamCaptioner(ImageCaptioner):
    """Captioner that uses the MoonDream2 VLM."""

    def __init__(self, config: dict = None):
        """Initialize the captioner.

        Args:
            config: Optional configuration with keys ``model``, ``revision`` and
                ``device_map``. ``device_map`` defaults to "auto" when multiple
                GPUs are available, otherwise the current device ("cuda" or
                "cpu"). The ``device`` key is used as the single-device default
                for backward compatibility.
        """
        super().__init__(config)
        model_name = self.config.get("model", "vikhyatk/moondream2")
        revision = self.config.get("revision", "2025-06-21")
        default_device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        device_map = self.config.get(
            "device_map",
            "auto" if torch.cuda.device_count() > 1 else default_device,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.device = torch.device(next(iter(self.model.hf_device_map.values())))

    def caption_image(self, image_data: Any) -> str:
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        try:
            # Get caption length from config
            length = self.config.get("caption_length", "long")
            result = self.model.caption(image_data, length=length, stream=False)
            if isinstance(result, dict):
                return str(result.get("caption", ""))
            return str(result)
        except Exception as e:
            print(f"Error in image captioning: {e}")
            return ""

    def release(self) -> None:
        """Release model and free GPU/CPU memory where possible."""
        try:
            if getattr(self, "model", None) is not None:
                try:
                    # Move to CPU before delete to help free GPU VRAM
                    self.model.to("cpu")
                except Exception:
                    pass
                # Drop reference and collect
                del self.model
                self.model = None
        finally:
            # Trigger Python GC and clear CUDA cache if available
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
