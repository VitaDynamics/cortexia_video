import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from PIL import Image

T = TypeVar('T')


class BatchProcessor:
    """Generic batch processor for inference tasks.

    Stores indices (paths, Lance indices, etc.) and loads objects on-demand during batch processing
    for memory-efficient processing of large datasets.
    """

    def __init__(self, batch_size: int = 4):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
        """
        self.batch_size = batch_size
        self.indices: List[Any] = []

    def load_indices(self, indices: List[Any]) -> None:
        """Store indices for batch processing.

        Args:
            indices: List of indices (paths, Lance indices, etc.) to store
        """
        self.indices = indices.copy()
    
    def load_images(self, image_paths: List[Path]) -> None:
        """Store image paths for batch processing (backward compatibility).

        Args:
            image_paths: List of image file paths to store
        """
        self.load_indices(image_paths)

    def process_batch_generic(
        self,
        load_func: Callable[[List[T]], List[Any]],
        inference_func: Callable[[List[Any], List[T]], List[Any]],
        save_func: Callable[[T, Any], None],
        filter_func: Optional[Callable[[T], bool]] = None,
    ) -> None:
        """Process items in batches using generic load function.

        Args:
            load_func: Function that takes indices and returns loaded objects
            inference_func: Function that takes (objects, indices) and returns inference results
            save_func: Function that takes (index, result) and saves the result
            filter_func: Optional function to filter which indices to process
        """
        # Filter indices if filter function provided
        indices_to_process = self.indices.copy()
        if filter_func:
            indices_to_process = [idx for idx in indices_to_process if filter_func(idx)]

        # Process in batches
        for i in range(0, len(indices_to_process), self.batch_size):
            batch_indices = indices_to_process[i : i + self.batch_size]
            
            print(
                f"Processing batch {i // self.batch_size + 1}/{(len(indices_to_process) + self.batch_size - 1) // self.batch_size}"
            )

            # Load objects for this batch only
            try:
                batch_objects = load_func(batch_indices)
                if not batch_objects:
                    print(f"No valid objects loaded for batch starting at {batch_indices[0]}")
                    continue
                    
                print(f"Loaded {len(batch_objects)} objects")
                
            except Exception as e:
                print(f"Failed to load batch starting at {batch_indices[0]}: {e}")
                continue

            try:
                # Run inference on batch
                print(f"Infering {len(batch_objects)} objects")
                results = inference_func(batch_objects, batch_indices)

                print(f"Infered {len(results)} objects")

                # Save results for each object
                for idx, result in zip(batch_indices, results):
                    save_func(idx, result)

            except Exception as e:
                print(f"Error processing batch starting at {batch_indices[0]}: {e}")
                # Continue with next batch
            finally:
                # Clean up batch objects from memory if they have close method
                if hasattr(batch_objects, '__iter__'):
                    for obj in batch_objects:
                        if hasattr(obj, 'close'):
                            obj.close()
                batch_objects = None
                
                # Force garbage collection after each batch
                gc.collect()
    
    def process_batch(
        self,
        inference_func: Callable[[List[Image.Image], List[Path]], List[Any]],
        save_func: Callable[[Path, Any], None],
        filter_func: Optional[Callable[[Path], bool]] = None,
    ) -> None:
        """Process images in batches and save results (backward compatibility).

        Args:
            inference_func: Function that takes (images, paths) and returns inference results
            save_func: Function that takes (path, result) and saves the result
            filter_func: Optional function to filter which paths to process
        """
        def default_image_load_func(paths: List[Path]) -> List[Image.Image]:
            """Default function to load PIL Images from paths."""
            loaded_images = []
            for path in paths:
                try:
                    img = Image.open(path).convert("RGB")
                    loaded_images.append(img)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    # Skip this image
                    continue
            return loaded_images
        
        self.process_batch_generic(
            load_func=default_image_load_func,
            inference_func=inference_func,
            save_func=save_func,
            filter_func=filter_func
        )

    def clear_buffer(self) -> None:
        """Clear the index buffer."""
        self.indices.clear()
        gc.collect()

    def get_loaded_indices(self) -> List[Any]:
        """Get list of stored indices."""
        return self.indices.copy()
    
    def get_loaded_paths(self) -> List[Path]:
        """Get list of stored image paths (backward compatibility)."""
        return self.indices.copy()

    def __len__(self) -> int:
        """Return number of stored indices."""
        return len(self.indices)

def collect_images(folder: Path, image_format: str = "jpg") -> List[Path]:
    """Collect images of a specific format from a folder.

    Args:
        folder: Folder to search for images.
        image_format: Desired image file extension (e.g. ``"jpg"``).

    Returns:
        List of image file paths.
    """
    image_format = image_format.lower()
    if image_format in {"jpg", "jpeg"}:
        patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    else:
        patterns = [f"*.{image_format}", f"*.{image_format.upper()}"]

    images: List[Path] = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))
    return sorted(images)


def collect_images_recursive(root_dir: Path, image_format: str = "jpg") -> List[Path]:
    """Recursively collect images of a specific format from ``root_dir``.

    Args:
        root_dir: Root directory to search recursively.
        image_format: Desired image file extension.

    Returns:
        List of image file paths.
    """
    image_format = image_format.lower()
    if image_format in {"jpg", "jpeg"}:
        patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    else:
        patterns = [f"*.{image_format}", f"*.{image_format.upper()}"]

    images: List[Path] = []
    for pattern in patterns:
        images.extend(root_dir.rglob(pattern))
    return sorted(images)


def create_lance_load_func(lance_dataset, image_column: str = "image"):
    """Create a load function for Lance dataset.
    
    Args:
        lance_dataset: Lance dataset object
        image_column: Name of column containing image data
        
    Returns:
        Load function that takes indices and returns loaded images
    """
    def lance_load_func(indices: List[int]) -> List[Any]:
        """Load images from Lance dataset by indices."""
        try:
            # Take specific rows by indices
            batch_data = lance_dataset.take(indices)
            images = []
            for row in batch_data.to_pylist():
                # Assuming image data is stored as bytes or PIL-compatible format
                img_data = row[image_column]
                if isinstance(img_data, bytes):
                    from io import BytesIO
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                else:
                    # Handle other formats as needed
                    img = img_data
                images.append(img)
            return images
        except Exception as e:
            print(f"Failed to load Lance batch with indices {indices}: {e}")
            return []
    
    return lance_load_func
