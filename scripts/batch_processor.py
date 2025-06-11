import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image


class BatchProcessor:
    """Generic batch processor for image inference tasks.

    Maintains a buffer mapping paths to PIL Images and provides batch processing
    capabilities with automatic memory management.
    """

    def __init__(self, batch_size: int = 4):
        """Initialize batch processor.

        Args:
            batch_size: Number of images to process in each batch
        """
        self.batch_size = batch_size
        self.path_to_image: Dict[Path, Image.Image] = {}

    def load_images(self, image_paths: List[Path]) -> None:
        """Load images into the buffer.

        Args:
            image_paths: List of image file paths to load
        """
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                self.path_to_image[path] = img
            except Exception as e:
                print(f"Failed to load {path}: {e}")

    def process_batch(
        self,
        inference_func: Callable[[List[Image.Image], List[Path]], List[Any]],
        save_func: Callable[[Path, Any], None],
        filter_func: Optional[Callable[[Path], bool]] = None,
    ) -> None:
        """Process images in batches and save results.

        Args:
            inference_func: Function that takes (images, paths) and returns inference results
            save_func: Function that takes (path, result) and saves the result
            filter_func: Optional function to filter which paths to process
        """
        # Filter paths if filter function provided
        paths_to_process = list(self.path_to_image.keys())
        if filter_func:
            paths_to_process = [p for p in paths_to_process if filter_func(p)]

        # Process in batches
        for i in range(0, len(paths_to_process), self.batch_size):
            batch_paths = paths_to_process[i : i + self.batch_size]
            batch_images = [self.path_to_image[path] for path in batch_paths]

            print(
                f"Processing batch {i // self.batch_size + 1}/{(len(paths_to_process) + self.batch_size - 1) // self.batch_size}"
            )

            try:
                # Run inference on batch
                results = inference_func(batch_images, batch_paths)

                # Save results for each image
                for path, result in zip(batch_paths, results):
                    save_func(path, result)

            except Exception as e:
                print(f"Error processing batch starting at {batch_paths[0]}: {e}")
                # Continue with next batch
                continue

            # Optional: Force garbage collection after each batch
            gc.collect()

    def clear_buffer(self) -> None:
        """Clear the image buffer to free memory."""
        for img in self.path_to_image.values():
            img.close()
        self.path_to_image.clear()
        gc.collect()

    def get_loaded_paths(self) -> List[Path]:
        """Get list of currently loaded image paths."""
        return list(self.path_to_image.keys())

    def __len__(self) -> int:
        """Return number of loaded images."""
        return len(self.path_to_image)


def collect_images(folder: Path) -> List[Path]:
    """Collect JPG/JPEG images from a folder.

    Args:
        folder: Folder to search for images

    Returns:
        List of image file paths
    """
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    images = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))
    return sorted(images)


def collect_images_recursive(root_dir: Path) -> List[Path]:
    """Recursively collect all JPG images from root_dir.

    Args:
        root_dir: Root directory to search recursively

    Returns:
        List of image file paths
    """
    images = []
    for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        images.extend(root_dir.rglob(pattern))
    return sorted(images)
