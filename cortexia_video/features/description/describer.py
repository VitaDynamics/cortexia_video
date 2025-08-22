"""Object description feature implementation"""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.detection import DetectionResult
from ...data.models.video import FrameData


class DescriptionFeature(BaseFeature):
    """Object description feature using vision-language models"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = None
    
    def _initialize(self):
        """Initialize description model"""
        try:
            # Get model configuration
            model_name = self.get_config_param(
                "model", 
                "Salesforce/blip2-opt-2.7b"
            )
            
            # Initialize model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize based on model type
            if "blip" in model_name.lower():
                self._initialize_blip(model_name)
            elif "qwen" in model_name.lower():
                self._initialize_qwen(model_name)
            else:
                raise ValueError(f"Unsupported description model: {model_name}")
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize description model: {e}")
    
    def _initialize_blip(self, model_name: str):
        """Initialize BLIP model"""
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = Blip2Processor.from_pretrained(model_name)
    
    def _initialize_qwen(self, model_name: str):
        """Initialize Qwen model"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    @property
    def name(self) -> str:
        return "description"
    
    @property
    def description(self) -> str:
        return "Object description using vision-language models"
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for object description.
        
        Args:
            frame_data: Frame data containing RGB image and detections
            
        Returns:
            Frame data with descriptions added to detections
        """
        if not self.is_ready():
            raise ProcessingError("Description feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Generate descriptions for detections
            descriptions = self._describe_objects(image, frame_data.detections)
            
            # Add descriptions to detections
            for i, detection in enumerate(frame_data.detections):
                if i < len(descriptions):
                    detection.description = descriptions[i]
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in description processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for object description.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with descriptions added to detections
        """
        if not self.is_ready():
            raise ProcessingError("Description feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Get detections for each frame
            detections_list = [f.detections for f in valid_frames]
            
            # Run batch description
            batch_descriptions = self._describe_objects_batch(images, detections_list)
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(batch_descriptions):
                    descriptions = batch_descriptions[i]
                    for j, detection in enumerate(frame.detections):
                        if j < len(descriptions):
                            detection.description = descriptions[j]
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch description processing: {e}")
    
    def _describe_objects(self, image: Image.Image, detections: List[DetectionResult]) -> List[str]:
        """
        Describe objects in a single image.
        
        Args:
            image: PIL Image to process
            detections: List of detection results
            
        Returns:
            List of object descriptions
        """
        if not detections:
            return []
        
        model_name = self.get_config_param("model", "Salesforce/blip2-opt-2.7b")
        
        if "blip" in model_name.lower():
            return self._describe_objects_blip(image, detections)
        elif "qwen" in model_name.lower():
            return self._describe_objects_qwen(image, detections)
        else:
            raise ValueError(f"Unsupported description model: {model_name}")
    
    def _describe_objects_batch(self, images: List[Image.Image], detections_list: List[List[DetectionResult]]) -> List[List[str]]:
        """
        Describe objects in batch of images.
        
        Args:
            images: List of PIL Images
            detections_list: List of detection lists for each image
            
        Returns:
            List of description lists for each image
        """
        model_name = self.get_config_param("model", "Salesforce/blip2-opt-2.7b")
        
        if "blip" in model_name.lower():
            # BLIP doesn't support batch processing for multiple objects
            results = []
            for i, image in enumerate(images):
                detections = detections_list[i] if i < len(detections_list) else []
                descriptions = self._describe_objects_blip(image, detections)
                results.append(descriptions)
            return results
        elif "qwen" in model_name.lower():
            return self._describe_objects_qwen_batch(images, detections_list)
        else:
            raise ValueError(f"Unsupported description model: {model_name}")
    
    def _describe_objects_blip(self, image: Image.Image, detections: List[DetectionResult]) -> List[str]:
        """Describe objects using BLIP model"""
        descriptions = []
        
        prompt = self.get_config_param("prompt", "Describe this object in detail:")
        
        for detection in detections:
            try:
                # Crop image to detection bounding box
                cropped_image = self._crop_detection(image, detection.box)
                
                # Generate description
                inputs = self.processor(cropped_image, prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=50)
                
                description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                descriptions.append(description)
                
            except Exception:
                # Fallback description
                descriptions.append(f"A {detection.label}")
        
        return descriptions
    
    def _describe_objects_qwen(self, image: Image.Image, detections: List[DetectionResult]) -> List[str]:
        """Describe objects using Qwen model"""
        descriptions = []
        
        prompt = self.get_config_param("prompt", "Describe this object in detail:")
        
        for detection in detections:
            try:
                # Crop image to detection bounding box
                cropped_image = self._crop_detection(image, detection.box)
                
                # Create message
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": cropped_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[cropped_image],
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                description = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                
                descriptions.append(description)
                
            except Exception:
                # Fallback description
                descriptions.append(f"A {detection.label}")
        
        return descriptions
    
    def _describe_objects_qwen_batch(self, images: List[Image.Image], detections_list: List[List[DetectionResult]]) -> List[List[str]]:
        """Describe objects using Qwen model in batch"""
        # For batch processing, we'll process each image individually
        # as Qwen doesn't easily support multiple objects per image in batch
        results = []
        
        for i, image in enumerate(images):
            detections = detections_list[i] if i < len(detections_list) else []
            descriptions = self._describe_objects_qwen(image, detections)
            results.append(descriptions)
        
        return results
    
    def _crop_detection(self, image: Image.Image, bbox) -> Image.Image:
        """
        Crop image to detection bounding box.
        
        Args:
            image: PIL Image to crop
            bbox: Bounding box coordinates
            
        Returns:
            Cropped PIL Image
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        
        # Add padding
        padding = self.get_config_param("crop_padding", 10)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Crop image
        cropped = image.crop((x1, y1, x2, y2))
        
        return cropped