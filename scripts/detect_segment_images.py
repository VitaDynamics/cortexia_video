import argparse
import json
import uuid
from pathlib import Path
from typing import Any, List
import sys
import numpy as np
from batch_processor import BatchProcessor, collect_images
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.object_segmentation import ObjectSegmenter

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


def load_tags(tag_file: Path) -> List[str]:
    """Load and parse tags from a JSON file.

    Args:
        tag_file: Path to the JSON file containing tags

    Returns:
        List of individual tag strings
    """
    if not tag_file.exists():
        return []

    with open(tag_file, "r") as f:
        data = json.load(f)

    # New mapping structure
    if "category_map" in data:
        mapping = data.get("category_map", {})
        return list(mapping.keys())

    raw_tags = data.get("tags", [])
    if not raw_tags:
        return []

    all_tags = []
    # Old categorized dict format
    if isinstance(raw_tags, dict):
        for tag_list in raw_tags.values():
            if isinstance(tag_list, list):
                all_tags.extend(tag_list)
    else:
        for tag_item in raw_tags:
            if isinstance(tag_item, str):
                if "\n" in tag_item and "- " in tag_item:
                    parsed_tags = parse_tags_string(tag_item)
                    all_tags.extend(parsed_tags)
                else:
                    all_tags.append(tag_item.strip())
            elif isinstance(tag_item, list):
                all_tags.extend(tag_item)

    return all_tags


def detect_segment_inference_func(
    images: List[Image.Image],
    paths: List[Path],
    detector: ObjectDetector,
    segmenter: ObjectSegmenter,
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
        tags = load_tags(tag_file)
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

    # Load original tags and category mapping
    original_data = {}
    if tag_file.exists():
        with open(tag_file, "r") as f:
            original_data = json.load(f)

    tags = load_tags(tag_file)
    category_map = original_data.get("category_map")

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

    # Save updated json
    out_data = {"tags": tags, "objects": obj_entries}
    if category_map is not None:
        out_data["category_map"] = category_map
    with open(tag_file, "w") as f:
        json.dump(out_data, f, indent=2)

    # Save mask npy
    if masks_dict:
        mask_path = path.with_name(f"{path.stem}_masks.npy")
        np.save(mask_path, masks_dict, allow_pickle=True)
        print(f"Saved masks for {path} -> {mask_path}")
    else:
        print(f"No detections for {path}")


def process_video_folder(
    folder: Path, detector: ObjectDetector, segmenter: ObjectSegmenter
) -> None:
    images = collect_images(folder)
    if not images:
        return

    # Initialize batch processor
    processor = BatchProcessor(batch_size=8)

    # Load images into buffer
    processor.load_images(images)

    # Create inference function with detector and segmenter
    def inference_func(imgs, paths):
        return detect_segment_inference_func(imgs, paths, detector, segmenter)

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
    args = parser.parse_args()

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    detector = ObjectDetector(cfg)
    segmenter = ObjectSegmenter(cfg)

    print(f"Processing root folder: {args.folder}")
    print(f"Minimum images required per subfolder: {args.min_images}")

    processed_count = 0
    skipped_count = 0

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        images = collect_images(sub)
        if len(images) >= args.min_images:
            print(f"Processing video folder '{sub.name}' with {len(images)} images...")
            process_video_folder(sub, detector, segmenter)
            processed_count += 1
        else:
            print(
                f"Skipping folder '{sub.name}' - only {len(images)} images (minimum {args.min_images} required)"
            )
            skipped_count += 1

    print(
        f"\nSummary: Processed {processed_count} folders, skipped {skipped_count} folders"
    )


if __name__ == "__main__":
    main()
