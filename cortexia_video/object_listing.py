import os
from typing import Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import requests

# add path from recognize_anything to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "recognize_anything"))

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

class RAMLister(ObjectLister):
    # TODO: need integrate with more robust way. 
    """
    Uses Recognize Anything Model (RAM) to identify objects in images.
    """
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.config_manager = config_manager
        self.image_size = 384
        
        # Define paths
        self.repo_path = os.path.join(os.path.dirname(__file__), "..", "recognize_anything")
        self.weights_path = os.path.join(self.repo_path, "pretrained", "ram_swin_large_14m.pth")
        
        # We'll lazily initialize these
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        
        print(f"RAMLister initialized, will use weights at: {self.weights_path}")
    
    def _initialize_model(self):
        """Initialize the model and transform pipeline."""
        if self.initialized:
            return
        
        # Ensure pretrained directory exists and weights file is downloaded
        pretrained_dir = os.path.dirname(self.weights_path)
        if not os.path.exists(pretrained_dir):
            os.makedirs(pretrained_dir)
            
        if not os.path.exists(self.weights_path):
             # TODO: use huggingface api to download it 
            print(f"Downloading RAM model weights to {self.weights_path}...")
            try:
                from tqdm import tqdm
                url = "https://hf-mirror.com/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    with open(self.weights_path, 'wb') as f, tqdm(
                        desc=self.weights_path,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)
                    print("Download completed!")
                else:
                    raise Exception(f"Failed to download model weights: HTTP {response.status_code}")
            except Exception as e:
                print(f"Error downloading weights: {e}")
                raise
                
        # Set up transform
        try:
            # We'll skip setting up the transform since we're using the subprocess approach
            pass
            
            # We'll use inference_ram.py to perform inference directly as a fallback
            # if we can't properly load the model
            self.inference_script = os.path.join(self.repo_path, "inference_ram.py")
            
            self.initialized = True
            print("RAMLister model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing transform: {e}")
            raise

    def list_objects_in_image(self, image_data: Any) -> List[str]:
        """
        Extract a list of objects from an image using RAM.
        
        Args:
            image_data: Input image (PIL Image or numpy array)
            
        Returns:
            List of detected objects as strings
        """
        try:
            # Make sure model is initialized
            self._initialize_model()
            
            # Convert numpy array to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                image_data = Image.fromarray(image_data)
            
            # Ensure image is in RGB format
            image_data = image_data.convert("RGB")
            
            # Save image to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp_path = temp.name
                image_data.save(temp_path)
            
            # Use subprocess to run inference_ram.py directly
            import subprocess
            import json
            
            cmd = [
                sys.executable,
                self.inference_script,
                '--image', temp_path,
                '--pretrained', self.weights_path,
                '--image-size', str(self.image_size)
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.repo_path
            )
            
            stdout, stderr = process.communicate()
            
            if stderr:
                print(f"Error from subprocess: {stderr.decode()}")
            
            if process.returncode != 0:
                print(f"Process exited with code {process.returncode}")
                return []
            
            output = stdout.decode()
            print(f"Raw output: {output}")
            
            # Parse the output to extract tags
            # The format should be: Image Tags: tag1 | tag2 | tag3 ...
            for line in output.split('\n'):
                if "Image Tags:" in line:
                    tags_str = line.split("Image Tags:")[1].strip()
                    tags = [tag.strip().lower() for tag in tags_str.split('|') if tag.strip()]
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    return tags
            
            # If we couldn't parse the output
            print(f"Could not parse RAM output")
            return []
            
        except Exception as e:
            print(f"Error in RAMLister object listing: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Clean up temp file if it exists
            if 'temp_path' in locals():
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass

# Registry for object listers
OBJECT_LISTER_REGISTRY = {
    "vikhyatk/moondream2": MoonDreamLister,
    "qwen/qwen2.5-vl": Qwen2_5VLLister,
    "recognize_anything/ram": RAMLister,
}