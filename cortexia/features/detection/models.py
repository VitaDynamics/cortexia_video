from typing import List
import gc
import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForZeroShotObjectDetection
from transformers.models.auto.processing_auto import AutoProcessor


# TODO: Add SAHI inference during detection.

class ObjectDetector:
    def __init__(self, config: dict = None):
        """Initialize the object detector with model from config.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize grounding dino model
        model_name = self.config.get("model", "IDEA-Research/grounding-dino-base")
        

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        
        # SAM model is not directly integrated in this class to avoid compatibility issues
        # Use the standalone ObjectSegmenter class for mask generation

    def release(self) -> None:
        """Release model and processor resources, free GPU memory if any."""
        try:
            if getattr(self, "model", None) is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                del self.model
                self.model = None
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

    def detect_objects(self, images_data: List[Image.Image], text_prompts: List[List[str]]) -> List[List[dict]]:
        """Detect objects in batched images based on text prompts.
        
        Args:
            images_data: List of PIL Images
            text_prompts: List of text prompts for each image (List[List[str]])
                        IMPORTANT: text queries need to be lowercased + end with a dot
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            List[List[dict]]: List of detection lists for each image. Each detection is
            a dictionary with 'score', 'label', and 'box' keys.
        """
        # Convert text_prompts to strings if needed (handling both formats)
        string_prompts = []
        for prompts in text_prompts:
            if isinstance(prompts, list):
                string_prompts.append('. '.join(prompts) + '.')
            else:
                string_prompts.append(prompts)
                
        # Process batch using the model's processor
        inputs = self.processor(
            images=images_data,
            text=string_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        
        # Move inputs to device
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs_on_device)
        
        # Prepare target sizes for post-processing (height, width)
        target_sizes_list = [img.size[::-1] for img in images_data]  # List of (height, width) tuples
        target_sizes_tensor = torch.tensor(target_sizes_list, device=self.device)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config.get("box_threshold", 0.3),
            text_threshold=self.config.get("text_threshold", 0.3),
            target_sizes=target_sizes_tensor
        )
        
        
        # Convert the model outputs to the required List[List[dict]] format
        results_final = []
        
        for image_result in results:  # Iterate through each image result
            detections_for_image = []
            
            # Get boxes, scores, and labels for this image
            boxes = image_result["boxes"]
            scores = image_result["scores"]
            labels = image_result.get("text_labels", image_result.get("labels", []))  # Handle both API versions
            
            # For each detection in this image
            for i in range(len(scores)):
                box = boxes[i].tolist()  # [x1, y1, x2, y2]
                score = scores[i].item()
                label = labels[i] if isinstance(labels[i], str) else self.model.config.id2label[labels[i].item()]
                
                detection_dict = {
                    'score': score,
                    'label': label,
                    'box': box
                }
                
                detections_for_image.append(detection_dict)
            
            results_final.append(detections_for_image)
        
        return results_final
