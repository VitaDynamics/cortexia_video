import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images_recursive

from cortexia_video.depth_estimation import DepthEstimator


def unzip_archives(source_dir: Path, target_dir: Path, force: bool = False) -> None:
    """Unzip all zip files from source_dir into subfolders of target_dir.

    Args:
        source_dir: Directory containing zip files
        target_dir: Directory to extract archives to
        force: If True, overwrite existing extracted folders
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in source_dir.glob("*.zip"):
        dest = target_dir / zip_path.stem
        if dest.exists() and not force:
            print(f"Skipping {zip_path.name} - already extracted at {dest}")
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            dest.mkdir(parents=True, exist_ok=True)
            archive.extractall(dest)


def depth_inference_func(images: List, paths: List[Path]) -> List[Any]:
    """Inference function for depth estimation.

    Args:
        images: List of PIL Images
        paths: List of corresponding paths

    Returns:
        List of depth estimation results
    """
    # Convert paths to strings for the estimator
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch depth estimation for images")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing zip files or images",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory to extract archives and store results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted folders",
    )
    args = parser.parse_args()

    unzip_archives(args.source_dir, args.target_dir, args.force)
    images = collect_images_recursive(args.target_dir)

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
        inference_func=depth_inference_func, save_func=depth_save_func
    )

    # Clean up
    processor.clear_buffer()
    print("Batch processing completed!")


if __name__ == "__main__":
    main()
