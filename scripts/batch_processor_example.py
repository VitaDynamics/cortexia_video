#!/usr/bin/env python3
"""
Example: Generic usage of BatchProcessor for any inference task

This demonstrates the basic pattern for using BatchProcessor:
1. Define an inference function that takes (images, paths) and returns results
2. Define a save function that takes (path, result) and saves the result
3. Use BatchProcessor to handle batching, buffering, and processing
"""

from pathlib import Path
from typing import Any, List

import numpy as np
from batch_processor import BatchProcessor, collect_images
from PIL import Image


def example_inference_func(images: List[Image.Image], paths: List[Path]) -> List[Any]:
    """Example inference function - just returns image sizes.

    Args:
        images: List of PIL Images
        paths: List of corresponding paths

    Returns:
        List of results (image sizes in this example)
    """
    results = []
    for img in images:
        # Example processing - just get image size
        result = {"width": img.width, "height": img.height, "mode": img.mode}
        results.append(result)
    return results


def example_save_func(path: Path, result: Any) -> None:
    """Example save function - saves image info as numpy file.

    Args:
        path: Original image path
        result: Result to save
    """
    out_path = path.with_name(path.stem + "_info.npy")
    np.save(out_path, result, allow_pickle=True)
    print(f"Saved info for {path} -> {out_path}")


def main():
    """Example usage of BatchProcessor."""
    # Replace with your actual image directory
    image_dir = Path("sample_data")

    if not image_dir.exists():
        print(f"Directory {image_dir} does not exist. Please update the path.")
        return

    # Collect images (use collect_images for single folder, collect_images_recursive for subfolders)
    images = collect_images(image_dir)
    # images = collect_images_recursive(image_dir)  # For recursive search

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")

    # Initialize batch processor
    processor = BatchProcessor(batch_size=8)  # Adjust batch size based on your memory

    try:
        # Load images into buffer
        print("Loading images...")
        processor.load_images(images)
        print(f"Loaded {len(processor)} images into buffer")

        # Process in batches
        print("Processing batches...")
        processor.process_batch(
            inference_func=example_inference_func, save_func=example_save_func
        )

    finally:
        # Always clean up
        processor.clear_buffer()
        print("Processing completed and buffer cleared!")


if __name__ == "__main__":
    main()
