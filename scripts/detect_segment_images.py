import argparse
import json
import uuid
from pathlib import Path
from typing import Any, List, Optional
import sys
import numpy as np
from batch_processor import BatchProcessor, collect_images, create_lance_load_func
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortexia_video.config_manager import ConfigManager
from cortexia_video.features.detection.models import ObjectDetector
from cortexia_video.features.segmentation.models import ObjectSegmenter

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

# collect_images is now imported from batch_processor


def parse_tags_string(tags_string: str) -> List[str]:
    """Parse a tags string in the format '- tag1\n- tag2\n- tag3' into a list of individual tags.

    Args:
        tags_string: String containing tags separated by newlines and prefixed with '- '

    Returns:
        List of individual tag strings
    """
    if not tags_string.strip():
        return []

    # Split by newlines and clean up each tag
    lines = tags_string.strip().split("\n")
    tags = []

    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            # Remove the '- ' prefix and any extra whitespace
            tag = line[2:].strip()
            if tag:  # Only add non-empty tags
                tags.append(tag)
        elif line:  # Handle tags without '- ' prefix
            tags.append(line)

    return tags


def _extract_tags(obj: Any) -> List[str]:
    """Recursively extract tag strings from nested dict/list structures."""
    tags: List[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            tags.extend(_extract_tags(value))
    elif isinstance(obj, list):
        for item in obj:
            tags.extend(_extract_tags(item))
    elif isinstance(obj, str):
        tags.append(obj)
    return tags


def load_extra_tags(extra_file: Path) -> List[str]:
    """Load additional tags from a JSON file."""
    if not extra_file.exists():
        return []
    with open(extra_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return list(dict.fromkeys(_extract_tags(data)))


def load_tags(tag_file: Path, extra_tags: List[str] | None = None) -> List[str]:
    """Load detectable tags from a tag file and merge with extra tags."""
    if not tag_file.exists():
        base_tags: List[str] = []
    else:
        with open(tag_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_tags = data.get("detectable_tags", [])

    if extra_tags:
        base_tags.extend(extra_tags)
    # Deduplicate while preserving order
    seen = set()
    merged: List[str] = []
    for t in base_tags:
        if t not in seen:
            merged.append(t)
            seen.add(t)
    return merged


def detect_segment_inference_func(
    images: List[Image.Image],
    paths: List[Path],
    detector: ObjectDetector,
    segmenter: ObjectSegmenter,
    extra_tags: List[str] | None = None,
) -> List[Any]:
    """Inference function for detection and segmentation.

    Args:
        images: List of PIL Images
        paths: List of corresponding paths
        detector: Object detector instance
        segmenter: Object segmenter instance

    Returns:
        List of detection and segmentation results
    """
    results = []

    # Load tags for all images
    tags_list = []
    for path in paths:
        tag_file = path.with_name(f"{path.stem}_tag.json")
        tags = load_tags(tag_file, extra_tags)
        tags_list.append(tags)

    # Filter out images without tags
    valid_indices = [i for i, tags in enumerate(tags_list) if tags]
    if not valid_indices:
        return [None] * len(images)

    valid_images = [images[i] for i in valid_indices]
    valid_tags = [tags_list[i] for i in valid_indices]

    # Batch detection
    detections_batch = detector.detect_objects(valid_images, valid_tags)

    # Process each image's detections and get masks
    valid_results = []
    for img, detections in zip(valid_images, detections_batch):
        boxes = [d["box"] for d in detections]
        masks = []
        if boxes:
            masks_batch = segmenter.segment_object(img, boxes)
            masks = masks_batch[0] if masks_batch else []

        valid_results.append({"detections": detections, "masks": masks})

    # Map results back to original order
    result_idx = 0
    for i in range(len(images)):
        if i in valid_indices:
            results.append(valid_results[result_idx])
            result_idx += 1
        else:
            results.append(None)

    return results


def detect_segment_save_func(path: Path, result: Any) -> None:
    """Save function for detection and segmentation results.

    Args:
        path: Original image path
        result: Detection and segmentation result
    """
    tag_file = path.with_name(f"{path.stem}_tag.json")

    if result is None:
        print(f"No tags found for {path}, skipping")
        return

    detections = result["detections"]
    masks = result["masks"]

    # Load original tags
    if not tag_file.exists():
        print(f"Tag file {tag_file} not found, skipping")
        return

    with open(tag_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get detected labels
    detected_labels = [det["label"] for det in detections]
    data["detectable_tags"] = detected_labels

    # Remove the old 'tags' key if it exists
    if "tags" in data:
        del data["tags"]

    obj_entries = []
    masks_dict = {}
    for idx, det in enumerate(detections):
        obj_id = str(uuid.uuid4())
        det_entry = {
            "id": obj_id,
            "label": det["label"],
            "score": det["score"],
            "box": det["box"],
        }
        obj_entries.append(det_entry)
        if idx < len(masks):
            masks_dict[obj_id] = masks[idx]

    # Add objects to the data
    data["objects"] = obj_entries

    # Save updated json
    with open(tag_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save mask npy
    if masks_dict:
        mask_path = path.with_name(f"{path.stem}_masks.npy")
        np.save(mask_path, masks_dict, allow_pickle=True)
        print(f"Saved masks for {path} -> {mask_path}")
    else:
        print(f"No detections for {path}")


def create_lance_detection_save_func(lance_dataset_path: Path):
    """Create a save function that updates Lance dataset with detection/segmentation results.
    
    Args:
        lance_dataset_path: Path to Lance dataset
        
    Returns:
        Save function for Lance dataset updates with detection data
    """
    def lance_detection_save_func(index: int, result: Any) -> None:
        """Save detection and segmentation results to Lance dataset.
        
        Args:
            index: Row index in Lance dataset
            result: Detection and segmentation result
        """
        try:
            # Open the dataset
            dataset = lance.dataset(str(lance_dataset_path))
            
            if result is None:
                print(f"No detections for index {index}, skipping")
                return
            
            detections = result["detections"]
            masks = result["masks"]
            
            # Process detections and masks
            detected_labels = [det["label"] for det in detections]
            
            # Create object entries with UUIDs
            obj_entries = []
            masks_data = {}
            
            for idx, det in enumerate(detections):
                obj_id = str(uuid.uuid4())
                obj_entry = {
                    "id": obj_id,
                    "label": det["label"],
                    "score": float(det["score"]),  # Ensure float type
                    "box": det["box"],  # [x1, y1, x2, y2]
                }
                obj_entries.append(obj_entry)
                
                # Store mask data if available
                if idx < len(masks):
                    # Convert numpy array to bytes for storage
                    mask_bytes = masks[idx].tobytes()
                    mask_shape = masks[idx].shape
                    masks_data[obj_id] = {
                        "mask_bytes": mask_bytes,
                        "shape": list(mask_shape),
                        "dtype": str(masks[idx].dtype)
                    }
            
            # Create the nested detection structure
            detection_data = {
                "detectable_tags": detected_labels,
                "objects": obj_entries,
                "masks": masks_data
            }
            
            # Create update data
            update_data = pa.table({
                "_rowid": [index],
                "detections": [detection_data]
            })
            
            # Merge the detection data into the dataset
            dataset = dataset.merge(
                update_data, 
                left_on="_rowid", 
                right_on="_rowid"
            )
            
            print(f"Saved {len(detections)} detections for index {index}")
            
        except Exception as e:
            print(f"Failed to save detections to Lance dataset for index {index}: {e}")
    
    return lance_detection_save_func


def create_lance_detection_schema() -> pa.Schema:
    """Create PyArrow schema for detection and segmentation storage.
    
    Returns:
        PyArrow schema with nested detection structure
    """
    # Define the mask data structure
    mask_struct = pa.struct([
        pa.field("mask_bytes", pa.binary()),
        pa.field("shape", pa.list_(pa.int32())),
        pa.field("dtype", pa.string())
    ])
    
    # Define the object structure
    object_struct = pa.struct([
        pa.field("id", pa.string()),
        pa.field("label", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("box", pa.list_(pa.float64()))  # [x1, y1, x2, y2]
    ])
    
    # Define the detection structure
    detection_struct = pa.struct([
        pa.field("detectable_tags", pa.list_(pa.string())),
        pa.field("objects", pa.list_(object_struct)),
        pa.field("masks", pa.map_(pa.string(), mask_struct))  # object_id -> mask_data
    ])
    
    # Create the full schema
    schema = pa.schema([
        pa.field("_rowid", pa.int64()),  # Row identifier
        pa.field("image", pa.binary()),  # Image data (if storing images)
        pa.field("detections", detection_struct)  # Nested detection structure
    ])
    
    return schema


def process_video_folder(
    folder: Path,
    detector: ObjectDetector,
    segmenter: ObjectSegmenter,
    extra_tags: List[str] | None = None,
    image_format: str = "jpg",
    lance_dataset_path: Optional[Path] = None,
) -> None:
    """Process video folder for detection and segmentation.
    
    Args:
        folder: Folder containing images
        detector: Object detector instance
        segmenter: Object segmenter instance
        extra_tags: Additional tags to detect
        image_format: Image file format
        lance_dataset_path: Optional path to Lance dataset for storage
    """
    images = collect_images(folder, image_format=image_format)
    if not images:
        return

    # Initialize batch processor
    processor = BatchProcessor(batch_size=8)

    if lance_dataset_path and LANCE_AVAILABLE:
        # Use Lance dataset mode
        # Assume images are indexed 0, 1, 2, ... in the Lance dataset
        indices = list(range(len(images)))
        processor.load_indices(indices)
        
        # Create Lance-specific functions
        lance_dataset = lance.dataset(str(lance_dataset_path))
        load_func = create_lance_load_func(lance_dataset, image_column="image")
        save_func = create_lance_detection_save_func(lance_dataset_path)
        
        def inference_func(imgs, indices):
            return detect_segment_inference_func(
                imgs, indices, detector, segmenter, extra_tags
            )
        
        processor.process_batch_generic(
            load_func=load_func,
            inference_func=inference_func, 
            save_func=save_func
        )
    else:
        # Use traditional path-based mode
        processor.load_images(images)
        
        # Create inference function with detector and segmenter
        def inference_func(imgs, paths):
            return detect_segment_inference_func(
                imgs, paths, detector, segmenter, extra_tags
            )
        
        # Process batch
        processor.process_batch(
            inference_func=inference_func, save_func=detect_segment_save_func
        )

    # Clean up
    processor.clear_buffer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and segment objects for tagged images"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Root folder containing video subfolders",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/heavy_mode.toml"),
        help="Path to configuration file specifying models",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum number of JPEG images required for a folder to be considered a video folder",
    )
    parser.add_argument(
        "--extra-tags-file",
        type=Path,
        default=Path("config/default_extra_tags.json"),
        help="Path to JSON file containing additional tags to detect",
    )
    parser.add_argument(
        "--lance-dataset",
        type=Path,
        help="Path to Lance dataset for detection storage (optional, defaults to JSON/NPY files)",
    )
    args = parser.parse_args()

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    image_format = cfg.get_param("processing.image_format", "jpg")
    detector = ObjectDetector(cfg)
    segmenter = ObjectSegmenter(cfg)

    extra_tags = load_extra_tags(args.extra_tags_file)

    print(f"Processing root folder: {args.folder}")
    print(f"Minimum images required per subfolder: {args.min_images}")

    processed_count = 0
    skipped_count = 0

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        images = collect_images(sub, image_format=image_format)
        if len(images) >= args.min_images:
            print(f"Processing video folder '{sub.name}' with {len(images)} images...")
            process_video_folder(sub, detector, segmenter, extra_tags, image_format, args.lance_dataset)
            processed_count += 1
        else:
            print(
                f"Skipping folder '{sub.name}' - only {len(images)} images (minimum {args.min_images} required)"
            )
            skipped_count += 1

    output_mode = "Lance dataset (nested struct)" if args.lance_dataset else "JSON + NPY files"
    print(
        f"\nSummary: Processed {processed_count} folders, skipped {skipped_count} folders"
    )
    print(f"Output mode: {output_mode}")
    
    if args.lance_dataset and not LANCE_AVAILABLE:
        print("Warning: Lance dataset path provided but lance/pyarrow not available. Used JSON mode instead.")
    
    if args.lance_dataset and LANCE_AVAILABLE:
        print("\nLance schema for reference:")
        schema = create_lance_detection_schema()
        print(schema)


if __name__ == "__main__":
    main()
