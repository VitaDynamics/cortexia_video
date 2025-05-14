from PIL import Image
import numpy as np
from typing import List, Optional

def crop_pil_image_from_mask(original_pil_image: Image.Image, mask_array: np.ndarray, box_xyxy: Optional[List[float]] = None) -> Optional[Image.Image]:
    """
    Crop a PIL image using a segmentation mask and optionally a bounding box.
    
    Args:
        original_pil_image: The original PIL image
        mask_array: Numpy array of the segmentation mask
        box_xyxy: Optional bounding box [xmin, ymin, xmax, ymax]
        
    Returns:
        Cropped image with transparency mask applied, or None if invalid
    """
    if mask_array.dtype != np.uint8:
        mask_array = mask_array.astype(np.uint8)
    
    if mask_array.max() == 0:  # Empty mask
        return None

    # Ensure mask is 2D
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:  # (1, H, W)
        mask_array = mask_array.squeeze(0)
    elif mask_array.ndim != 2:  # Not (H,W)
        # Try to find a valid 2D slice if it's like (H, W, 1)
        if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
            mask_array = mask_array.squeeze(-1)
        else:
            return None  # Cannot process this mask shape

    # Get bounding box from mask contours or provided box
    if box_xyxy:
        x_min, y_min, x_max, y_max = map(int, box_xyxy)
    else:
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None  # Empty mask
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        y_max += 1
        x_max += 1

    # Ensure box is within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(original_pil_image.width, x_max)
    y_max = min(original_pil_image.height, y_max)

    if x_min >= x_max or y_min >= y_max:
        return None  # Invalid box

    cropped_image_from_box = original_pil_image.crop((x_min, y_min, x_max, y_max))
    
    # Create an alpha mask from the relevant part of the mask_array
    mask_for_crop = mask_array[y_min:y_max, x_min:x_max]
    alpha_mask_pil = Image.fromarray(mask_for_crop * 255, mode='L')

    # Ensure cropped_image_from_box is RGBA to apply alpha
    if cropped_image_from_box.mode != 'RGBA':
        cropped_image_from_box = cropped_image_from_box.convert('RGBA')
    
    cropped_image_from_box.putalpha(alpha_mask_pil)
    return cropped_image_from_box


class UtilManager:
    """
    Utility manager class for various helper functions
    """
    
    def __init__(self, logger=None):
        """
        Initialize the utility manager
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        
    def crop_pil_image_from_mask(self, original_pil_image: Image.Image, 
                                mask_array: np.ndarray, 
                                box_xyxy: Optional[List[float]] = None) -> Optional[Image.Image]:
        """
        Class method wrapper for crop_pil_image_from_mask
        """
        try:
            return crop_pil_image_from_mask(original_pil_image, mask_array, box_xyxy)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cropping image from mask: {e}")
            return None
