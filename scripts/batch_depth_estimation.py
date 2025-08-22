import argparse
import functools
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images, create_lance_load_func

from cortexia_video.depth_estimation import DepthProEstimator

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

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


def create_lance_depth_save_func(lance_dataset_path: Path):
    """Create a save function that updates Lance dataset with depth estimation results.
    
    Args:
        lance_dataset_path: Path to Lance dataset
        
    Returns:
        Save function for Lance dataset updates with depth data
    """
    def lance_depth_save_func(index: int, result: Any) -> None:
        """Save depth estimation results to Lance dataset.
        
        Args:
            index: Row index in Lance dataset
            result: Depth estimation result
        """
        try:
            # Open the dataset
            dataset = lance.dataset(str(lance_dataset_path))
            
            if result["depth"] is None:
                print(f"No depth data for index {index}, skipping")
                return
            
            depth_array = result["depth"]
            focal_length = result.get("focallength_px", None)
            
            # Convert depth array to bytes for storage
            depth_bytes = depth_array.tobytes()
            depth_shape = list(depth_array.shape)
            depth_dtype = str(depth_array.dtype)
            
            # Calculate min/max for normalization metadata
            min_depth = float(np.min(depth_array))
            max_depth = float(np.max(depth_array))
            
            # Create the depth data structure
            depth_data = {
                "depth_array": depth_bytes,
                "shape": depth_shape,
                "dtype": depth_dtype,
                "focal_length_px": float(focal_length) if focal_length is not None else None,
                "min_depth": min_depth,
                "max_depth": max_depth
            }
            
            # Create update data
            update_data = pa.table({
                "_rowid": [index],
                "depth_data": [depth_data]
            })
            
            # Merge the depth data into the dataset
            dataset = dataset.merge(
                update_data, 
                left_on="_rowid", 
                right_on="_rowid"
            )
            
            depth_info = f"shape={depth_shape}, range=[{min_depth:.3f}, {max_depth:.3f}]"
            print(f"Saved depth data for index {index}: {depth_info}")
            
        except Exception as e:
            print(f"Failed to save depth data to Lance dataset for index {index}: {e}")
    
    return lance_depth_save_func


def create_lance_depth_schema() -> pa.Schema:
    """Create PyArrow schema for depth estimation storage.
    
    Returns:
        PyArrow schema with nested depth structure
    """
    # Define the depth data structure
    depth_struct = pa.struct([
        pa.field("depth_array", pa.binary()),        # Serialized numpy array
        pa.field("shape", pa.list_(pa.int32())),     # Array dimensions
        pa.field("dtype", pa.string()),              # Numpy data type
        pa.field("focal_length_px", pa.float64()),   # Focal length in pixels (nullable)
        pa.field("min_depth", pa.float64()),         # Minimum depth value
        pa.field("max_depth", pa.float64())          # Maximum depth value
    ])
    
    # Create the full schema
    schema = pa.schema([
        pa.field("_rowid", pa.int64()),              # Row identifier
        pa.field("image", pa.binary()),              # Image data (if storing images)
        pa.field("depth_data", depth_struct)         # Nested depth structure
    ])
    
    return schema


def reconstruct_depth_array(depth_data: dict) -> np.ndarray:
    """Reconstruct depth array from Lance storage data.
    
    Args:
        depth_data: Dictionary containing depth array metadata
        
    Returns:
        Reconstructed numpy depth array
    """
    depth_bytes = depth_data["depth_array"]
    shape = tuple(depth_data["shape"])
    dtype = depth_data["dtype"]
    
    # Reconstruct the numpy array
    depth_array = np.frombuffer(depth_bytes, dtype=dtype).reshape(shape)
    return depth_array


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
    parser.add_argument(
        "--lance-dataset",
        type=Path,
        help="Path to Lance dataset for depth storage (optional, defaults to NPY/PNG files)",
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
                    args.lance_dataset,
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
                images, f"folder {args.folder.name}", args.batch_size, args.save_format, args.lance_dataset
            )

        output_mode = "Lance dataset" if args.lance_dataset else f"{args.save_format.upper()} files"
        print("Batch processing completed!")
        print(f"Output mode: {output_mode}")
        
        if args.lance_dataset and not LANCE_AVAILABLE:
            print("Warning: Lance dataset path provided but lance/pyarrow not available. Used file mode instead.")
        
        if args.lance_dataset and LANCE_AVAILABLE:
            print("\nLance schema for reference:")
            schema = create_lance_depth_schema()
            print(schema)

    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise


def process_folder_images(
    images: List[Path], 
    folder_description: str, 
    batch_size: int, 
    save_format: str,
    lance_dataset_path: Optional[Path] = None
) -> None:
    """Process images from a folder using BatchProcessor.
    
    Args:
        images: List of image paths to process
        folder_description: Description for logging
        batch_size: Batch size for processing
        save_format: Format for file-based saving ('npy' or 'png')
        lance_dataset_path: Optional path to Lance dataset for storage
    """
    # Initialize batch processor
    processor = BatchProcessor(batch_size=batch_size)

    try:
        if lance_dataset_path and LANCE_AVAILABLE:
            # Use Lance dataset mode
            print("Using Lance dataset mode for depth storage...")
            
            # Assume images are indexed 0, 1, 2, ... in the Lance dataset
            indices = list(range(len(images)))
            processor.load_indices(indices)
            
            # Create Lance-specific functions
            lance_dataset = lance.dataset(str(lance_dataset_path))
            load_func = create_lance_load_func(lance_dataset, image_column="image")
            save_func = create_lance_depth_save_func(lance_dataset_path)
            
            def inference_func(imgs, indices):
                return depth_inference_func(imgs, indices)
            
            print(f"Starting Lance batch processing for {folder_description}...")
            processor.process_batch_generic(
                load_func=load_func,
                inference_func=inference_func, 
                save_func=save_func
            )
        else:
            # Use traditional file-based mode
            print(f"Using {save_format.upper()} file mode for depth storage...")
            
            # Create a save function with the save_format argument pre-filled
            bound_save_func = functools.partial(depth_save_func, save_format=save_format)
            
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
