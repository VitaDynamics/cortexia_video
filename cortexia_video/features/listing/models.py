import os

# add path from recognize_anything to sys.path
import sys
from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from ...core.registry import create_registry

# for RAM++
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "recognize-anything"))


class ObjectLister(ABC):
    def __init__(self, config: dict):
        """
        Initialize the ObjectLister with configuration dict.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
    
    def get_config_param(self, key: str, default=None):
        """
        Get configuration parameter with default value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        current = self.config
        
        try:
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    return default
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    @abstractmethod
    def list_objects_in_image(self, image_data: List[Any]) -> List[str]:
        """
        Extract a list of objects from an image.

        Args:
            image_data: List of Input image (PIL Image or numpy array). It should be a list of Torch

        Returns:
            List of detected objects as strings
        """
        pass

    @abstractmethod
    def list_objects_in_image_batched(
        self, images_batch: List[Image.Image]
    ) -> List[List[str]]:
        """
        Extract lists of objects from a batch of images.

        Args:
            images_batch: List of input images (PIL Images)

        Returns:
            List of lists of detected objects as strings, one list per image
        """
        pass


# module-level registry
OBJECT_LISTER_REGISTRY = create_registry("object_lister")


@OBJECT_LISTER_REGISTRY.decorator("vikhyatk/moondream2", aliases=["moondream2"]) 
class MoonDreamLister(ObjectLister):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = self.get_config_param("model", "vikhyatk/moondream2")
        revision = "2025-04-14"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device_map={"": "cuda"},
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
            objects = [obj.strip().lower() for obj in objects_text.split(",")]
            return [obj for obj in objects if obj]  # Remove any empty strings
        except Exception as e:
            print(f"Error in object listing: {e}")
            return []


@OBJECT_LISTER_REGISTRY.decorator(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    aliases=[
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-72B-Instruct",
    ],
)
class Qwen2_5VLLister(ObjectLister):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = self.get_config_param("model", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.task_prompt = self.get_config_param("task_prompt", "List all objects in this image.")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def list_objects_in_image(self, image_data: Any) -> List[str]:
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        format_prompt = "Provide the output as a list of strings, where each string in the array is the label of a object."
        prompt = f"{self.task_prompt} {format_prompt}"
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
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer = output_text[0]
            return answer
            
            # TODO: We need to extend this class instead of list objects.
            # import ast
            # try:
            #     labels = ast.literal_eval(answer)
            #     if isinstance(labels, list):
            #         return [str(label) for label in labels]
            #     return []  # Return empty list if labels is not a list
            # except Exception:
            #     return [x.strip() for x in answer.split(",") if x.strip()]
        except Exception as e:
            print(f"Error in Qwen2_5VLLister object listing: {e}")
            return []

    def list_objects_in_image_batched(
        self, images_batch: List[Image.Image]
    ) -> List[List[str]]:
        """
        Extract lists of objects from a batch of images.

        Args:
            images_batch: List of input images (PIL Images)

        Returns:
            List of lists of detected objects as strings, one list per image
        """
        if isinstance(images_batch[0], np.ndarray):
            images_batch = [Image.fromarray(image) for image in images_batch]
        format_prompt = "Provide the output as a list of strings, where each string in the array is the label of a object."
        prompt = f"{self.task_prompt} {format_prompt}"
        texts = []
        for image_data in images_batch:
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
        try:
            image_inputs = images_batch
            video_inputs = None
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answers = output_text
            import ast

            try:
                final_labels = [] 
                for answer in answers:
                    labels = ast.literal_eval(answer)
                    if isinstance(labels, list):
                        final_labels.append([str(label) for label in labels])
                return final_labels
            except Exception:
                final_labels = [x.strip() for x in answer.split(",") if x.strip()]
                return final_labels
        except Exception as e:
            print(f"Error in Qwen2_5VLLister object listing: {e}")
            return []


@OBJECT_LISTER_REGISTRY.decorator("recognize_anything/ram", aliases=["ram", "recognize-anything"])
class RAMLister(ObjectLister):
    """
    Uses Recognize Anything Model (RAM) to identify objects in images.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.image_size = 384

        # Define paths
        self.repo_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "recognize-anything"
        )
        self.weights_path = os.path.join(
            self.repo_path, "pretrained", "ram_plus_swin_large_14m.pth"
        )

        # We'll lazily initialize these
        self.model = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.openset_mode = False
        self.llm_tag_des_path = os.path.join(
            self.repo_path,
            "datasets",
            "openimages_rare_200",
            "openimages_rare_200_llm_tag_descriptions.json",
        )

        print(f"RAMLister initialized, will use weights at: {self.weights_path}")

    def _initialize_model(self):
        """Initialize the model and transform pipeline."""
        if self.initialized:
            return

        try:
            # Ensure the recognize_anything module is in the path
            if self.repo_path not in sys.path:
                sys.path.append(self.repo_path)

            import json

            from ram import get_transform
            from ram.models import ram_plus
            from ram.utils import build_openset_llm_label_embedding

            # Ensure pretrained directory exists and weights file is downloaded
            pretrained_dir = os.path.dirname(self.weights_path)
            if not os.path.exists(pretrained_dir):
                os.makedirs(pretrained_dir)

            if not os.path.exists(self.weights_path):
                print(f"Downloading RAM model weights to {self.weights_path}...")
                try:
                    from tqdm import tqdm

                    url = "https://hf-mirror.com/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        total_size = int(response.headers.get("content-length", 0))
                        with (
                            open(self.weights_path, "wb") as f,
                            tqdm(
                                desc=self.weights_path,
                                total=total_size,
                                unit="iB",
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as bar,
                        ):
                            for chunk in response.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                        print("Download completed!")
                    else:
                        raise Exception(
                            f"Failed to download model weights: HTTP {response.status_code}"
                        )
                except Exception as e:
                    print(f"Error downloading weights: {e}")
                    raise

            # Set up transform
            self.transform = get_transform(image_size=self.image_size)

            # Load the model
            self.model = ram_plus(
                pretrained=self.weights_path, image_size=self.image_size, vit="swin_l"
            )

            # Set up openset mode if enabled
            if self.openset_mode and os.path.exists(self.llm_tag_des_path):
                print("Building tag embedding for openset mode:")
                with open(self.llm_tag_des_path, "rb") as fo:
                    llm_tag_des = json.load(fo)
                openset_label_embedding, openset_categories = (
                    build_openset_llm_label_embedding(llm_tag_des)
                )

                self.model.tag_list = np.array(openset_categories)
                self.model.label_embed = torch.nn.Parameter(
                    openset_label_embedding.float()
                )
                self.model.num_class = len(openset_categories)
                # the threshold for unseen categories is often lower
                self.model.class_threshold = torch.ones(self.model.num_class) * 0.5

            self.model.eval()
            self.model = self.model.to(self.device)

            self.initialized = True
            print("RAMLister model initialized successfully")

        except Exception as e:
            print(f"Error initializing RAM model: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize RAM model: {e}")

    def list_objects_in_image_batched(
        self, images_batch: List[Image.Image]
    ) -> List[List[str]]:
        """
        Extract lists of objects from a batch of images using RAM.

        Args:
            images_batch: List of input images (PIL Images)

        Returns:
            List of lists of detected objects as strings, one list per image
        """
        try:
            # Lazily initialize model
            self._initialize_model()

            # Convert numpy arrays to PIL Images if needed
            if len(images_batch) > 0 and isinstance(images_batch[0], np.ndarray):
                images_batch = [Image.fromarray(image) for image in images_batch]

            # Ensure all images are in RGB format
            images_batch = [img.convert("RGB") for img in images_batch]

            # Process all images in batch
            all_results = []

            # Transform and stack all images into a batch tensor
            batch_tensor = torch.stack(
                [self.transform(img).to(self.device) for img in images_batch]
            )

            # Run batch inference
            # with torch.no_grad():
            #     # Note: We're calling inference for each image since RAM doesn't have a native batch inference function
            #     # If RAM adds batch support in the future, this could be optimized further
            #     for i in range(len(images_batch)):
            #         single_image_tensor = batch_tensor[i:i+1]  # Keep batch dimension
            #         tags_string = inference(single_image_tensor, self.model)

            #         # Parse tags
            #         tags = [tag.strip().lower() for tag in tags_string.split("|") if tag.strip()]
            #         all_results.append(tags)

            # return all_results

            with torch.no_grad():
                batch_tags_string = self.model.generate_tag_openset(batch_tensor)
                tags = [
                    [
                        tag.strip().lower()
                        for tag in tags_string.split("|")
                        if tag.strip()
                    ]
                    for tags_string in batch_tags_string
                ]

            return tags

        except Exception as e:
            print(f"Error in RAMLister batch object listing: {e}")
            import traceback

            traceback.print_exc()

            # Return empty lists for all images in the batch
            return [[] for _ in range(len(images_batch))]

    def list_objects_in_image(
        self, image_data: Union[Image.Image, np.ndarray]
    ) -> List[str]:
        """
        Extract a list of objects from a single image using RAM.

        Args:
            image_data: Input image (PIL Image or numpy array)

        Returns:
            List of detected objects as strings
        """
        try:
            # Lazily initialize model.
            self._initialize_model()

            # Convert numpy array to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                image_data = Image.fromarray(image_data)

            # Handle case where image_data is already a list (backward compatibility)
            if isinstance(image_data, list):
                if len(image_data) == 1:
                    # Single image in a list - extract it
                    image_data = image_data[0]
                else:
                    # Multiple images - use the batched method
                    results = self.list_objects_in_image_batched(image_data)
                    return results[0] if results else []

            # TODO: After add batch inference, remove this is ok now.
            image_data = image_data.convert("RGB")

            # Preprocess image
            image_tensor = self.transform(image_data).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                tags_string = self.model.generate_tag_openset(image_tensor)[0]

            # Parse tags
            tags = [
                tag.strip().lower() for tag in tags_string.split("|") if tag.strip()
            ]
            return tags

        except Exception as e:
            print(f"Error in RAMLister object listing: {e}")
            import traceback

            traceback.print_exc()
            return []


# End: registry entries added via decorators above
