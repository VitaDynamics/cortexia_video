import argparse
from pathlib import Path
import zipfile
import numpy as np

from cortexia_video.depth_estimation import DepthEstimator


def unzip_archives(source_dir: Path, target_dir: Path) -> None:
    """Unzip all zip files from source_dir into subfolders of target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in source_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            dest = target_dir / zip_path.stem
            dest.mkdir(parents=True, exist_ok=True)
            archive.extractall(dest)


def collect_images(root_dir: Path) -> list[Path]:
    """Recursively collect all JPG images from root_dir."""
    images = []
    for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        images.extend(root_dir.rglob(pattern))
    return images


def save_depth_results(image_paths: list[Path], estimator: DepthEstimator) -> None:
    """Estimate depth for images and save .npy files next to them."""
    if not image_paths:
        return

    # Estimate depth in batches to avoid memory issues
    batch_size = 4
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        results = estimator.estimate_batch_depth([str(p) for p in batch])
        for path, res in zip(batch, results):
            out_path = path.with_name(path.stem + "_depth.npy")
            np.save(out_path, res["depth"])
            print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch depth estimation for images")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing zip files or images")
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory to extract archives and store results",
    )
    args = parser.parse_args()

    unzip_archives(args.source_dir, args.target_dir)
    images = collect_images(args.target_dir)

    if not images:
        print("No images found for depth estimation.")
        return

    estimator = DepthEstimator()
    save_depth_results(images, estimator)


if __name__ == "__main__":
    main()
