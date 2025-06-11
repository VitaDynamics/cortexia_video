#!/usr/bin/env python3
"""
Example: Batch depth estimation using BatchProcessor
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images_recursive
from cortexia_video.depth_estimation import DepthEstimator


def depth_inference_func(images: List, paths: List[Path]) -> List[Any]:
    """Inference function for depth estimation.
    
    Args:
        images: List of PIL Images
        paths: List of corresponding paths
        
    Returns:
        List of depth estimation results
    """
    # Convert PIL images to path strings for the estimator
    image_paths = [str(path) for path in paths]
    estimator = DepthEstimator()
    results = estimator.estimate_batch_depth(image_paths)
    return results


def depth_save_func(path: Path, result: Any) -> None:
    """Save function for depth estimation results.
    
    Args:
        path: Original image path
        result: Depth estimation result
    """
    out_path = path.with_name(path.stem + "_depth.npy")
    np.save(out_path, result["depth"])
    print(f"Saved {out_path}")


def main():
    target_dir = Path("your_image_directory")  # Replace with actual directory
    
    # Collect all images
    images = collect_images_recursive(target_dir)
    if not images:
        print("No images found for depth estimation.")
        return
    
    print(f"Found {len(images)} images for processing")
    
    # Initialize batch processor
    processor = BatchProcessor(batch_size=4)
    
    # Load images into buffer
    print("Loading images into buffer...")
    processor.load_images(images)
    print(f"Loaded {len(processor)} images")
    
    # Process batch
    print("Starting batch processing...")
    processor.process_batch(
        inference_func=depth_inference_func,
        save_func=depth_save_func
    )
    
    # Clean up
    processor.clear_buffer()
    print("Batch processing completed!")


if __name__ == "__main__":
    main() 