from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM


class ImageCaptioner(ABC):
    """Abstract base class for generating captions for images."""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    @abstractmethod
    def caption_image(self, image_data: Any) -> str:
        """Return a caption describing the given image."""
        pass


class MoonDreamCaptioner(ImageCaptioner):
    """Captioner that uses the MoonDream2 VLM."""

    def __init__(self, config_manager):
        super().__init__(config_manager)
        # This is a 2B Model in huggingface
        model_name = config_manager.get_param(
            "model_settings.image_captioning_model", "vikhyatk/moondream2"
        )
        revision = "2025-06-21"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device_map={"": "cuda"},
        )

    def caption_image(self, image_data: Any) -> str:
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        try:
            # we need long caption here.
            result = self.model.caption(image_data, length="long", stream=False)
            if isinstance(result, dict):
                return str(result.get("caption", ""))
            return str(result)
        except Exception as e:
            print(f"Error in image captioning: {e}")
            return ""


# Registry for captioners
IMAGE_CAPTIONER_REGISTRY = {
    "vikhyatk/moondream2": MoonDreamCaptioner,
}
