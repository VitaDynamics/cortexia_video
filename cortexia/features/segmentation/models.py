import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers.models.sam import SamModel, SamProcessor

from typing import Tuple
import cv2

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


# NOTE: A label of 1 means the point is a positive prompt-the user wants the segmented mask to include the region around this point.
# NOTE: A label of 0 means the point is a negative prompt-the user wants the segmented mask to exclude the region around this point.
class ObjectSegmenter:
    def __init__(self, config: dict = None):
        """
        Initialize the ObjectSegmenter with SAM model and processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.processor = None
        # Support different devices for different model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        """Load SAM model and processor from config"""
        model_name = self.config.get(
            "model", "facebook/sam-vit-base"
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
    ) -> List[np.ndarray]:
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

            # Initialize batch_points to None by default
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
                images_data,
                batch_input_boxes,
                batch_input_points=batch_input_points,
                batch_input_labels=batch_input_labels,
            )

    def _pad_input_points(
        self, batch_input_points: List[List[List[float]]], pad_point_value=-10
    ):
        """
        Pad the input points to the maximum number of points in the batch.
        """
        max_points = max(len(points) for points in batch_input_points)
        padded_points = []
        for points in batch_input_points:
            if len(points) < max_points:
                for _ in range(max_points - len(points)):
                    points.append([pad_point_value, pad_point_value])
                padded_points.append(points)
            else:
                padded_points.append(points)
        return padded_points

    def _convert_box_to_point(
        self, batched_boxes: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """Convert bounding box to point format by calculating its center point.

        Args:
            box: List of bounding boxes in format [[x_min, y_min, x_max, y_max], ...]

        Returns:
            List containing [x_center, y_center] coordinates
        """
        batched_points = []
        for frame_boxes in batched_boxes:
            frame_points = []
            for object_box in frame_boxes:
                x_min, y_min, x_max, y_max = object_box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                object_point = [x_center, y_center]
                frame_points.append(object_point)
            batched_points.append(frame_points)
        return batched_points

    def _generate_input_labels(
        self, batch_input_points: List[List[List[float]]]
    ) -> List[List[List[int]]]:
        """
        Generate input labels for the model.
        """
        batched_labels = []
        for frame_points in batch_input_points:
            frame_labels = []
            for object_point in frame_points:
                frame_labels.append([1])
            batched_labels.append(frame_labels)
        return batched_labels

    def _pad_bbox(
        self, batched_boxes: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """
        Pad the bounding boxes to the maximum number of boxes in the batch.
        Returns padded boxes and a mask indicating which boxes are padded.
        """
        max_boxes = max(len(frame_boxes) for frame_boxes in batched_boxes)
        padded_boxes = []
        padding_mask = []  # Track which boxes are padded (True = real box, False = padded)

        for frame_boxes in batched_boxes:
            frame_padding_mask = [True] * len(frame_boxes)
            if len(frame_boxes) < max_boxes:
                frame_padded_boxes = []
                for _ in range(max_boxes - len(frame_boxes)):
                    frame_padded_boxes.append([-10, -10, -10, -10])
                    frame_padding_mask.append(False)  # Mark as padded
                frame_boxes.extend(frame_padded_boxes)
                padded_boxes.append(frame_boxes)
            else:
                padded_boxes.append(frame_boxes)
            padding_mask.append(frame_padding_mask)

        return padded_boxes, padding_mask

    def _batch_segment_objects(
        self,
        images_data: List[Image.Image],
        batch_input_boxes: Optional[List[List[List[float]]]] = None,
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
            raise ValueError(
                "Mismatch between images_data and batch_input_boxes lengths or empty boxes."
            )

        final_batched_masks_np = []

        # Process images in batch
        batch_size = self.config.get("batch_size", 8)
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

            # Pad boxes and track which ones are real vs padded
            padded_current_boxes, padding_mask = self._pad_bbox(current_boxes)

            # Process inputs for the current batch
            inputs = self.processor(
                images=current_images,
                input_boxes=padded_current_boxes,
                # input_points=current_points,
                # input_labels=current_labels,
                return_tensors="pt",
            )

            # Move inputs to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)

            # Get segmentation masks
            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes,
            )
            masks_tensor = torch.stack(masks, dim=0)
            # Only squeeze dimensions of size 1, but be careful not to squeeze too much
            if masks_tensor.dim() > 4:
                # If we have 5D tensor [B, 1, N, H, W], squeeze the singleton dimension
                masks_tensor = masks_tensor.squeeze(1)
            refined_masks = self._refine_masks(masks_tensor)
            # Remove padded and empty masks
            final_batched_masks_np.extend(
                self._remove_padded_masks(refined_masks, padding_mask)
            )

        return final_batched_masks_np

    def _remove_padded_masks(
        self, masks: List[List[np.ndarray]], padding_mask: List[List[bool]]
    ) -> List[List[np.ndarray]]:
        """
        Remove masks corresponding to padded boxes and empty masks.

        Args:
            masks: List of lists of masks, one list per image
            padding_mask: List of lists of booleans indicating which boxes are real (True) vs padded (False)

        Returns:
            Cleaned list of lists of masks with only real, non-empty masks
        """
        empty_count = 0
        padded_count = 0
        cleaned_masks = []

        for i, frame_masks in enumerate(masks):
            # Create a new list with only non-padded, non-empty masks
            frame_cleaned_masks = []
            frame_padding = padding_mask[i] if i < len(padding_mask) else []

            for j, mask in enumerate(frame_masks):
                # Skip if mask corresponds to a padded box
                if j < len(frame_padding) and not frame_padding[j]:
                    padded_count += 1
                    continue

                # Skip empty masks (could still be from real boxes, but empty)
                if mask.sum() == 0:
                    empty_count += 1
                    continue

                # Keep only real, non-empty masks
                frame_cleaned_masks.append(mask)

            cleaned_masks.append(frame_cleaned_masks)

        self.logger.info(
            f"Removed {padded_count} padded masks and {empty_count} empty masks"
        )
        return cleaned_masks

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

    def _refine_masks(
        self, masks, polygon_refinement: bool = False
    ) -> List[List[np.ndarray]]:
        # Handle input with shape [B, N, C, H, W] where N is number of objects per frame
        # B: batch size, N: number of objects, C: channels, H: height, W: width
        masks = masks.cpu().float()

        self.logger.debug(f"Refining masks with shape: {masks.shape}")

        # Check if we have the object dimension (4D or 5D tensor)
        if len(masks.shape) == 4:
            # Shape is [B, N, H, W] - already processed by model
            B, N, H, W = masks.shape
            masks_list = []

            # Process each batch
            for b in range(B):
                frame_masks = []
                # Process each object in the frame
                for n in range(N):
                    # Extract single object mask
                    mask = masks[b, n]  # [H, W]
                    mask = (mask > 0).int()
                    mask_np = mask.numpy().astype(np.uint8)

                    # Apply polygon refinement if requested
                    if polygon_refinement:
                        shape = mask_np.shape
                        polygon = mask_to_polygon(mask_np)
                        mask_np = polygon_to_mask(polygon, shape)

                    frame_masks.append(mask_np)
                masks_list.append(frame_masks)
            return masks_list

        elif len(masks.shape) == 5:
            # Shape is [B, N, C, H, W] - need to process channels
            B, N, C, H, W = masks.shape
            masks_list = []

            # Process each batch
            for b in range(B):
                frame_masks = []
                # Process each object in the frame
                for n in range(N):
                    # Extract single object mask and process it
                    mask = masks[b, n]  # [C, H, W]
                    if C > 1:
                        mask = mask.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                        mask = mask.mean(axis=-1)  # Average across channels
                    else:
                        mask = mask.squeeze(0)  # Remove channel dimension if C=1
                    mask = (mask > 0).int()
                    mask_np = mask.numpy().astype(np.uint8)

                    # Apply polygon refinement if requested
                    if polygon_refinement:
                        shape = mask_np.shape
                        polygon = mask_to_polygon(mask_np)
                        mask_np = polygon_to_mask(polygon, shape)

                    frame_masks.append(mask_np)
                masks_list.append(frame_masks)
            return masks_list

        elif len(masks.shape) == 3:
            # Shape is [B, H, W] - single mask per batch
            B, H, W = masks.shape
            masks_list = []

            # Process each batch
            for b in range(B):
                mask = masks[b]  # [H, W]
                mask = (mask > 0).int()
                mask_np = mask.numpy().astype(np.uint8)

                # Apply polygon refinement if requested
                if polygon_refinement:
                    shape = mask_np.shape
                    polygon = mask_to_polygon(mask_np)
                    mask_np = polygon_to_mask(polygon, shape)

                masks_list.append([mask_np])  # Wrap in list for consistency
            return masks_list

        else:
            # Fallback: try to handle as original [B, C, H, W] format
            if len(masks.shape) == 2:
                # Single mask [H, W]
                mask = (masks > 0).int()
                mask_np = mask.numpy().astype(np.uint8)
                return [[mask_np]]
            else:
                # Try original approach but with more robust dimension handling
                self.logger.warning(
                    f"Unexpected mask tensor shape: {masks.shape}, attempting fallback processing"
                )

                # Ensure we have at least 4 dimensions
                while len(masks.shape) < 4:
                    masks = masks.unsqueeze(0)

                # Assume last two dimensions are H, W
                *batch_dims, H, W = masks.shape

                # Flatten all batch dimensions except last two
                masks_flat = masks.view(-1, H, W)
                masks_list = []

                for i in range(masks_flat.shape[0]):
                    mask = masks_flat[i]
                    mask = (mask > 0).int()
                    mask_np = mask.numpy().astype(np.uint8)
                    masks_list.append([mask_np])

                return masks_list

    def _remove_empty_masks(
        self, masks: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """
        Remove empty masks from the list of masks.
        """
        count = 0
        cleaned_masks = []

        for frame_masks in masks:
            # Create a new list with only non-empty masks
            non_empty_masks = []
            for mask in frame_masks:
                if mask.sum() > 0:  # Keep only non-empty masks
                    non_empty_masks.append(mask)
                else:
                    count += 1

            cleaned_masks.append(non_empty_masks)

        self.logger.info(f"Removed {count} empty masks")
        return cleaned_masks
