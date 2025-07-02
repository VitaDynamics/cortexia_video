import argparse
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images

from cortexia_video.depth_estimation import DepthProEstimator

# Global estimator instance to avoid reloading the model
estimator = None


def get_estimator() -> DepthProEstimator:
    """Get or create the depth estimator instance."""
    global estimator
    if estimator is None:
        estimator = DepthProEstimator()
    return estimator


def depth_inference_func(images: List[Image.Image], paths: List[Path]) -> List[Any]:
    """Inference function for depth estimation.

    Args:
        images: List of PIL Images
        paths: List of corresponding paths

    Returns:
        List of depth estimation results
    """
    try:
        # Convert paths to strings for the estimator
        image_paths = [str(path) for path in paths]
        estimator = get_estimator()
        results = estimator.estimate_batch_depth(image_paths)
        return results
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        # Return empty results for each image in case of failure
        return [{"depth": None, "focallength_px": None} for _ in paths]


def depth_save_func(path: Path, result: Any) -> None:
    """Save function for depth estimation results.

    Args:
        path: Original image path
        result: Depth estimation result
    """
    try:
        if result["depth"] is not None:
            out_path = path.with_name(path.stem + "_depth_new.npy")
            np.save(out_path, result["depth"])
            print(f"Saved {out_path}")
        else:
            print(f"Skipped saving {path} due to estimation failure")
    except Exception as e:
        print(f"Error saving depth for {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch depth estimation for images")
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder to process",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories instead of the folder directly",
    )
    args = parser.parse_args()

    try:
        if args.recursive:
            # Original behavior: process subdirectories
            subdirs_processed = 0
            for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
                images = collect_images(sub)
                if not images:
                    print(f"No images found in {sub}, skipping...")
                    continue

                print(f"Found {len(images)} images for processing in {sub}")
                process_folder_images(images, f"subdirectory {sub.name}")
                subdirs_processed += 1

            if subdirs_processed == 0:
                print(f"No subdirectories with images found in {args.folder}")
        else:
            # New behavior: process images directly in the specified folder
            images = collect_images(args.folder)
            if not images:
                print(f"No images found in {args.folder}")
                return

            print(f"Found {len(images)} images for processing in {args.folder}")
            process_folder_images(images, f"folder {args.folder.name}")

        print("Batch processing completed!")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise


def process_folder_images(images: List[Path], folder_description: str) -> None:
    """Process images from a folder using BatchProcessor."""
    # Initialize batch processor
    processor = BatchProcessor(batch_size=4)

    try:
        # Load images into buffer
        print("Loading images into buffer...")
        processor.load_images(images)
        print(f"Loaded {len(processor)} images")

        # Process batch
        print(f"Starting batch processing for {folder_description}...")
        processor.process_batch(
            inference_func=depth_inference_func, save_func=depth_save_func
        )
        print(f"Completed processing {folder_description}")
    finally:
        # Always clean up, even if an error occurs
        processor.clear_buffer()


if __name__ == "__main__":
    main()
