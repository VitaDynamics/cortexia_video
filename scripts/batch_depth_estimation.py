import argparse
import functools
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


def depth_save_func(path: Path, result: Any, save_format: str) -> None:
    """Save function for depth estimation results.

    Args:
        path: Original image path
        result: Depth estimation result
        save_format: Format to save the depth map ('npy' or 'png')
    """
    try:
        if result["depth"] is not None:
            depth_array = result["depth"]
            if save_format == "npy":
                out_path = path.with_name(path.stem + "_depth_new.npy")
                np.save(out_path, depth_array)
                print(f"Saved {out_path}")
            elif save_format == "png":
                out_path = path.with_name(path.stem + "_depth_new.png")
                # Normalize depth map to 0-255 and convert to uint8
                depth_normalized = (
                    255.0
                    * (depth_array - np.min(depth_array))
                    / (np.max(depth_array) - np.min(depth_array) + 1e-6)
                )
                depth_uint8 = depth_normalized.astype(np.uint8)
                img_pil = Image.fromarray(depth_uint8)
                img_pil.save(out_path)
                print(f"Saved {out_path}")
            else:
                print(f"Unsupported save format: {save_format}")
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
    parser.add_argument(
        "--save_format",
        type=str,
        default="npy",
        choices=["npy", "png"],
        help="Format to save the depth map (npy or png). Default is npy.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to process in a batch. Default is 4.",
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
                process_folder_images(
                    images,
                    f"subdirectory {sub.name}",
                    args.batch_size,
                    args.save_format,
                )
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
            process_folder_images(
                images, f"folder {args.folder.name}", args.batch_size, args.save_format
            )

        print("Batch processing completed!")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise


def process_folder_images(
    images: List[Path], folder_description: str, batch_size: int, save_format: str
) -> None:
    """Process images from a folder using BatchProcessor."""
    # Initialize batch processor
    processor = BatchProcessor(batch_size=batch_size)

    # Create a save function with the save_format argument pre-filled
    bound_save_func = functools.partial(depth_save_func, save_format=save_format)

    try:
        # Load images into buffer
        print("Loading images into buffer...")
        processor.load_images(images)
        print(f"Loaded {len(processor)} images")

        # Process batch
        print(f"Starting batch processing for {folder_description}...")
        processor.process_batch(
            inference_func=depth_inference_func, save_func=bound_save_func
        )
        print(f"Completed processing {folder_description}")
    finally:
        # Always clean up, even if an error occurs
        processor.clear_buffer()


if __name__ == "__main__":
    main()
