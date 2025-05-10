from typing import Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor



class ObjectLister(ABC):
    def __init__(self, config_manager):
        """
        Initialize the ObjectLister with configuration manager.
        
        Args:
            config_manager: Instance of ConfigManager to get model settings
        """
        self.config_manager = config_manager

    @abstractmethod
    def list_objects_in_image(self, image_data: Any) -> List[str]:
        """
        Extract a list of objects from an image.
        
        Args:
            image_data: Input image (PIL Image or numpy array)
            
        Returns:
            List of detected objects as strings
        """
        pass

class MoonDreamLister(ObjectLister):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        model_name = config_manager.get_param('model_settings.object_listing_model')
        revision = "2025-04-14"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device_map={"": "cuda"}
        )

    def list_objects_in_image(self, image_data: Any) -> List[str]:
        """
        Extract a list of objects from an image using VQA.
        
        Args:
            image_data: Input image (PIL Image or numpy array)
            
        Returns:
            List of detected objects as strings
        """
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        try:
            prompt = (
            "Analyze the following image and list the labels of objects in the image. "
            "Provide the output as a list of strings, where each string in the array is the label of a object. Any label is acceptable."
            )
            # Use the model's query method for VQA instead of caption
            result = self.model.query(image_data, prompt)
            objects_text = result.get("answer", "")
            if not objects_text:
                return []
            
            # Process the comma-separated list
            objects = [obj.strip().lower() for obj in objects_text.split(',')]
            return [obj for obj in objects if obj]  # Remove any empty strings
        except Exception as e:
            print(f"Error in object listing: {e}")
            return []

class Qwen2_5VLLister(ObjectLister):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        model_name = config_manager.get_param('model_settings.object_listing_model')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def list_objects_in_image(self, image_data: Any) -> List[str]:
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        prompt = (
            "Analyze the following image and list the labels of objects in the image. "
            "Provide the output as a list of strings, where each string in the array is the label of a object. Any label is acceptable."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = [image_data]
            video_inputs = None
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            answer = output_text[0]
            import ast
            try:
                labels = ast.literal_eval(answer)
                if isinstance(labels, list):
                    return [str(label) for label in labels]
                return []  # Return empty list if labels is not a list
            except Exception:
                return [x.strip() for x in answer.split(",") if x.strip()]
        except Exception as e:
            print(f"Error in Qwen2_5VLLister object listing: {e}")
            return []

# Registry for object listers
OBJECT_LISTER_REGISTRY = {
    "vikhyatk/moondream2": MoonDreamLister,
    "qwen/qwen2.5-vl": Qwen2_5VLLister,
}