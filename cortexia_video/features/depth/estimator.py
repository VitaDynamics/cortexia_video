"""Depth estimation feature implementation"""

from typing import List

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeature
from ...api.exceptions import ModelLoadError, ProcessingError
from ...data.models.video import FrameData
from .models import DepthProEstimator


class DepthFeature(BaseFeature):
    """Depth estimation feature using transformer-based models"""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.estimator = None
        self.device = None
    
    def _initialize(self):
        """Initialize depth estimation model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize DepthPro estimator
            self.estimator = DepthProEstimator()
            
            self.initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize depth estimation model: {e}")
    
    @property
    def name(self) -> str:
        return "depth"
    
    @property
    def description(self) -> str:
        return "Depth estimation using transformer-based models"
    
    def process_frame(self, frame_data: FrameData) -> FrameData:
        """
        Process a single frame for depth estimation.
        
        Args:
            frame_data: Frame data containing RGB image
            
        Returns:
            Frame data with depth map added
        """
        if not self.is_ready():
            raise ProcessingError("Depth feature not initialized")
        
        if frame_data.rgb_image is None:
            return frame_data
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame_data.rgb_image)
            
            # Estimate depth using batch method for single image
            results = self.estimator.estimate_batch_depth([image])
            
            if results:
                result = results[0]
                depth_map = result["depth"]
                focal_length = result.get("focallength_px")
                
                # Add depth data to frame
                frame_data.depth_map = depth_map
                
                # Add depth statistics
                depth_stats = {
                    "mean_depth": float(np.mean(depth_map)),
                    "min_depth": float(np.min(depth_map)),
                    "max_depth": float(np.max(depth_map)),
                    "std_depth": float(np.std(depth_map))
                }
                if focal_length is not None:
                    depth_stats["focal_length_px"] = float(focal_length)
                
                frame_data.depth_statistics = depth_stats
            
            return frame_data
            
        except Exception as e:
            raise ProcessingError(f"Error in depth estimation processing: {e}")
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process multiple frames for depth estimation.
        
        Args:
            frames: List of frame data objects
            
        Returns:
            List of frame data with depth maps added
        """
        if not self.is_ready():
            raise ProcessingError("Depth feature not initialized")
        
        # Filter frames with RGB images
        valid_frames = [f for f in frames if f.rgb_image is not None]
        
        if not valid_frames:
            return frames
        
        try:
            # Convert to PIL Images
            images = [Image.fromarray(f.rgb_image) for f in valid_frames]
            
            # Estimate depth for batch
            results = self.estimator.estimate_batch_depth(images)
            
            # Add results back to frames
            for i, frame in enumerate(valid_frames):
                if i < len(results):
                    result = results[i]
                    depth_map = result["depth"]
                    focal_length = result.get("focallength_px")
                    
                    frame.depth_map = depth_map
                    
                    # Add depth statistics
                    depth_stats = {
                        "mean_depth": float(np.mean(depth_map)),
                        "min_depth": float(np.min(depth_map)),
                        "max_depth": float(np.max(depth_map)),
                        "std_depth": float(np.std(depth_map))
                    }
                    if focal_length is not None:
                        depth_stats["focal_length_px"] = float(focal_length)
                    
                    frame.depth_statistics = depth_stats
            
            return frames
            
        except Exception as e:
            raise ProcessingError(f"Error in batch depth estimation processing: {e}")
    
    def get_depth_visualization(self, depth_map: np.ndarray, colormap: str = "viridis") -> np.ndarray:
        """
        Create a visualization of the depth map.
        
        Args:
            depth_map: Depth map as numpy array
            colormap: Matplotlib colormap name
            
        Returns:
            RGB visualization of depth map
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # Normalize depth map to 0-1 if not already
            if depth_map.max() > 1.0:
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            else:
                depth_normalized = depth_map
            
            # Apply colormap
            cmap = cm.get_cmap(colormap)
            colored_depth = cmap(depth_normalized)
            
            # Convert to 0-255 RGB
            rgb_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            
            return rgb_depth
            
        except ImportError:
            # Fallback: convert to grayscale if matplotlib not available
            depth_gray = (depth_map * 255).astype(np.uint8)
            return np.stack([depth_gray] * 3, axis=-1)
    
    def estimate_object_distances(self, depth_map: np.ndarray, detections: List = None) -> List[dict]:
        """
        Estimate distances to detected objects using the depth map.
        
        Args:
            depth_map: Depth map as numpy array
            detections: List of detection results with bounding boxes
            
        Returns:
            List of distance estimations for each detection
        """
        if not detections:
            return []
        
        distances = []
        
        for detection in detections:
            if hasattr(detection, 'bbox') and detection.bbox:
                bbox = detection.bbox
                
                # Extract depth values within bounding box
                y1, y2 = int(bbox.ymin), int(bbox.ymax)
                x1, x2 = int(bbox.xmin), int(bbox.xmax)
                
                # Ensure bounds are within depth map
                y1 = max(0, min(y1, depth_map.shape[0] - 1))
                y2 = max(0, min(y2, depth_map.shape[0]))
                x1 = max(0, min(x1, depth_map.shape[1] - 1))
                x2 = max(0, min(x2, depth_map.shape[1]))
                
                if y2 > y1 and x2 > x1:
                    object_depth = depth_map[y1:y2, x1:x2]
                    
                    # Calculate statistics for the object region
                    distance_info = {
                        "detection_id": getattr(detection, 'id', None),
                        "label": getattr(detection, 'label', 'unknown'),
                        "mean_distance": float(np.mean(object_depth)),
                        "min_distance": float(np.min(object_depth)),
                        "max_distance": float(np.max(object_depth)),
                        "median_distance": float(np.median(object_depth)),
                        "bbox": [x1, y1, x2, y2]
                    }
                    
                    distances.append(distance_info)
        
        return distances