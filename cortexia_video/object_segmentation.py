import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers.models.sam import SamModel, SamProcessor


class ObjectSegmenter:
    def __init__(self, config_manager):
        """
        Initialize the ObjectSegmenter with SAM model and processor.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        """Load SAM model and processor from config"""
        model_name = self.config_manager.get_param(
            "model_settings.segmentation_model", "facebook/sam-vit-base"
        )
        try:
            # Load model and place it on the correct device
            model = SamModel.from_pretrained(model_name)
            self.model = model.to(self.device)  # Correct way to move model to device
            self.processor = SamProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {str(e)}")

    def segment_object(
        self,
        images_data,  # Can be List[Image.Image] or Image.Image
        batch_input_boxes,  # Can be nested list structure depending on input
        batch_input_points: Optional[List[Optional[List[List[float]]]]] = None,
        batch_input_labels: Optional[List[Optional[List[int]]]] = None,
    ) -> List[List[np.ndarray]]:
        """
        Segment objects in a batch of images using bounding boxes or points as prompts.

        Args:
            images_data: List of input images (PIL Images) or single PIL Image
            batch_input_boxes: List of lists of list of bounding boxes per image [[[x_min, y_min, x_max, y_max], ...], ...]
                             or for single image: [[x_min, y_min, x_max, y_max], ...]
            batch_input_points: Optional list of points per image [[[x1, y1], [x2, y2], ...], None, ...]
                              or for single image: [[x1, y1], [x2, y2], ...]
            batch_input_labels: Optional list of labels for points per image [[1, 1, ...], None, ...]
                              or for single image: [1, 1, ...]

        Returns:
            List of lists of binary masks (one list per image, each containing masks for each box)
            or for single image: list of masks [mask1, mask2, ...]
        """
        # Support for old API: allows image_data parameter to be a single image
        # This enables backward compatibility with existing code
        if isinstance(images_data, Image.Image):
            self.logger.debug(
                "Called segment_object with a single image, converting to batch format"
            )

            # Convert to batch format
            images_list = [images_data]

            # Handle input_boxes format
            if isinstance(batch_input_boxes, list) and all(
                isinstance(item, list) for item in batch_input_boxes
            ):
                if len(batch_input_boxes) > 0 and all(
                    isinstance(item, (int, float)) for item in batch_input_boxes[0]
                ):
                    # Format is [[x1, y1, x2, y2], ...] - single image multiple boxes
                    batch_boxes = [batch_input_boxes]
                else:
                    # Format is already correct
                    batch_boxes = batch_input_boxes
            else:
                # Invalid format
                raise ValueError("Input boxes must be a list of lists for single image")

            # Handle input_points format
            batch_points = None
            if batch_input_points is not None:
                if (
                    isinstance(batch_input_points[0], list)
                    and len(batch_input_points[0]) == 2
                ):
                    # Format is [[x1, y1], [x2, y2], ...] - single image multiple points
                    batch_points = [[batch_input_points]]
                else:
                    # Format is already correct
                    batch_points = batch_input_points

            # Handle input_labels format
            batch_labels = None
            if batch_input_labels is not None:
                if isinstance(batch_input_labels[0], int):
                    # Format is [1, 0, ...] - single image multiple labels
                    batch_labels = [[batch_input_labels]]
                else:
                    # Format is already correct
                    batch_labels = batch_input_labels

            # Continue with batched processing
            mask_results = self._batch_segment_objects(
                images_list, batch_boxes, batch_points, batch_labels
            )

            return mask_results

        else:
            # Handle regular batch processing
            return self._batch_segment_objects(
                images_data, batch_input_boxes, batch_input_points, batch_input_labels
            )

    def _batch_segment_objects(
        self,
        images_data: List[Image.Image],
        batch_input_boxes: List[List[List[float]]],
        batch_input_points: Optional[List[Optional[List[List[float]]]]] = None,
        batch_input_labels: Optional[List[Optional[List[int]]]] = None,
    ) -> List[List[np.ndarray]]:
        """
        Internal method for batch segmentation of objects.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        # Handle empty inputs
        if not images_data:
            return []

        # Check batch size consistency
        if not batch_input_boxes or len(images_data) != len(batch_input_boxes):
            self.logger.warning(
                "Mismatch between images_data and batch_input_boxes lengths or empty boxes."
            )
            return [[] for _ in images_data]

        try:
            final_batched_masks_np = []

            # Process images in batch
            batch_size = self.config_manager.get_param("processing.batch_size", 8)
            for batch_idx in range(0, len(images_data), batch_size):
                batch_end = min(batch_idx + batch_size, len(images_data))

                # Current batch slices
                current_images = images_data[batch_idx:batch_end]
                current_boxes = batch_input_boxes[batch_idx:batch_end]

                # Handle optional inputs for current batch
                current_points = None
                if batch_input_points:
                    current_points = batch_input_points[batch_idx:batch_end]

                current_labels = None
                if batch_input_labels:
                    current_labels = batch_input_labels[batch_idx:batch_end]

                # Process inputs for the current batch
                inputs = self.processor(
                    images=current_images,
                    input_boxes=current_boxes,
                    input_points=current_points,
                    input_labels=current_labels,
                    return_tensors="pt",
                )

                # Move inputs to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)

                # Get segmentation masks
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Prepare parameters for post-processing
                original_sizes = [
                    img.size[::-1] for img in current_images
                ]  # List of (H, W) tuples

                # Access tensor data correctly
                reshaped_input_sizes = []
                if "reshaped_input_sizes" in inputs and hasattr(
                    inputs["reshaped_input_sizes"], "cpu"
                ):
                    reshaped_input_sizes = inputs["reshaped_input_sizes"].cpu().tolist()

                # Get prediction masks and scores
                all_pred_masks_tensor = outputs.pred_masks
                batched_iou_scores = (
                    outputs.iou_scores if hasattr(outputs, "iou_scores") else None
                )

                # Post-process masks
                masks_list = []
                for i in range(len(current_images)):
                    # Get masks for this image
                    image_masks = (
                        all_pred_masks_tensor[i]
                        if i < all_pred_masks_tensor.shape[0]
                        else None
                    )

                    if image_masks is None:
                        masks_list.append([])
                        continue

                    # Get original image size
                    original_size = original_sizes[i]

                    # Get reshaped size
                    reshaped_size = None
                    if reshaped_input_sizes and i < len(reshaped_input_sizes):
                        reshaped_size = reshaped_input_sizes[i]

                    # Get the iou scores for this image
                    iou_scores = None
                    if (
                        batched_iou_scores is not None
                        and i < batched_iou_scores.shape[0]
                    ):
                        iou_scores = batched_iou_scores[i]

                    # Process masks for this image
                    masks_for_image = []
                    for j in range(image_masks.shape[0]):
                        mask = image_masks[j].cpu()

                        # Resize mask to original image size
                        # TODO: Need check the efficiency of this operation
                        if reshaped_size:
                            # Check dimensionality of the mask and reshape appropriately
                            if len(mask.shape) == 3:
                                # If mask has shape [3, height, width]
                                mask = torch.nn.functional.interpolate(
                                    mask.unsqueeze(
                                        0
                                    ),  # Add batch dimension [1, 3, height, width]
                                    size=original_size,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                mask = mask.squeeze(0)  # Remove batch dimension
                            else:
                                # Original approach for shape [height, width]
                                mask = torch.nn.functional.interpolate(
                                    mask.unsqueeze(0).unsqueeze(
                                        0
                                    ),  # Add batch and channel dimensions
                                    size=original_size,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                mask = mask.squeeze()

                        # Threshold the mask
                        binary_mask = (mask > 0.0).numpy().astype(np.uint8)
                        masks_for_image.append(binary_mask)

                    masks_list.append(masks_for_image)

                # Add masks for this batch to the final result
                final_batched_masks_np.extend(masks_list)

            return final_batched_masks_np

        except Exception as e:
            self.logger.error(f"Error during segmentation: {str(e)}")
            raise RuntimeError(f"Error during segmentation: {str(e)}")

    # For complete backward compatibility with older code
    def segment_object_single(
        self,
        image_data: Image.Image,
        input_boxes: List[List[float]],
        input_points: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Backward compatibility method for segmenting objects in a single image.

        Args:
            image_data: Single PIL Image
            input_boxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
            input_points: Optional list of points [[x1, y1], [x2, y2], ...]
            input_labels: Optional list of labels [1, 1, ...] (1=foreground, 0=background)

        Returns:
            Segmentation mask (numpy array)
        """
        # Format for batch processing
        images_list = [image_data]
        batch_boxes = [input_boxes]

        # Format points and labels
        batch_points = None
        if input_points is not None:
            batch_points = [[input_points]]

        batch_labels = None
        if input_labels is not None:
            batch_labels = [[input_labels]]

        # Call internal batch method
        batch_results = self._batch_segment_objects(
            images_list, batch_boxes, batch_points, batch_labels
        )

        # Return first mask from first image result
        if batch_results and batch_results[0]:
            return batch_results[0][0]
        return np.zeros((image_data.height, image_data.width), dtype=np.uint8)

    def generate_contour_visualization(
        self,
        image: Image.Image,
        mask: np.ndarray,
        input_boxes: Optional[List[List[float]]] = None,
        input_points: Optional[List[List[float]]] = None,
    ) -> Image.Image:
        """
        Generate a visualization of segmentation contours on the image.

        Args:
            image: PIL Image object
            mask: Binary segmentation mask
            input_boxes: Optional bounding boxes used for segmentation
            input_points: Optional points used for segmentation

        Returns:
            PIL Image with visualization
        """
        import cv2

        # Convert PIL to numpy for OpenCV
        img_np = np.array(image).astype(float) / 255.0
        img = img_np.copy()

        # Create contours from mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=3)

        # Draw input points if provided
        if input_points is not None:
            for point_group in input_points:
                if (
                    isinstance(point_group, list)
                    and len(point_group) == 2
                    and all(isinstance(p, (int, float)) for p in point_group)
                ):
                    # This is a single point as [x, y]
                    x, y = point_group
                    cv2.circle(
                        img,
                        (int(x), int(y)),
                        radius=8,
                        color=(1.0, 0.0, 0.0),
                        thickness=-1,
                    )
                    cv2.circle(
                        img,
                        (int(x), int(y)),
                        radius=8,
                        color=(1.0, 1.0, 1.0),
                        thickness=2,
                    )
                else:
                    # This is a list of points
                    for point in point_group:
                        if isinstance(point, list) and len(point) == 2:
                            x, y = point
                            cv2.circle(
                                img,
                                (int(x), int(y)),
                                radius=8,
                                color=(1.0, 0.0, 0.0),
                                thickness=-1,
                            )
                            cv2.circle(
                                img,
                                (int(x), int(y)),
                                radius=8,
                                color=(1.0, 1.0, 1.0),
                                thickness=2,
                            )

        # Draw input boxes if provided
        if input_boxes is not None:
            for box in input_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=2
                )
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=1
                )

        # Convert back to PIL
        img_with_contour_pil = Image.fromarray((img * 255.0).astype(np.uint8))
        return img_with_contour_pil
