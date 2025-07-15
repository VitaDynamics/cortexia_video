import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Dict
import os

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images

from cortexia_video.config_manager import ConfigManager
from cortexia_video.parser import parse_comma_separated_string



# collect_images is now imported from batch_processor


def tagging_inference_func(
    images: List[Image.Image], paths: List[Path], lister
) -> List[Any]:
    """Inference function for object listing/tagging.

    Args:
        images: List of PIL Images
        paths: List of corresponding paths
        lister: Object lister instance

    Returns:
        List of tagging results
    """
    # Batch process images for tagging
    results = []
    for img in images:
        tags = lister.list_objects_in_image(img)
        results.append(tags)
    return results


# Mapping of keywords to their high level categories
CATEGORY_KEYWORDS: Dict[str, set[str]] = {
    "accessible_area": {
        "road",
        "street",
        "sidewalk",
        "path",
        "trail",
        "lane",
        "bridge",
        "crosswalk",
    },
    "traffic_participants": {
        "car",
        "vehicle",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "bike",
        "pedestrian",
        "person",
        "cyclist",
    },
    "environmental_markers": {
        "traffic light",
        "traffic sign",
        "lane marking",
        "stop sign",
        "road sign",
        "traffic cone",
    },
    "light_weather_conditions": {
        "sunny",
        "rain",
        "snow",
        "fog",
        "cloudy",
        "overcast",
        "night",
        "daytime",
    },
}


def categorize_tags(tags: List[str]) -> Dict[str, str]:
    """Map each tag to its high level category."""
    mapping: Dict[str, str] = {}
    for tag in tags:
        tag_lower = tag.lower()
        category = "other"
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if tag_lower in keywords:
                category = cat
                break
        mapping[tag] = category
    return mapping


def tagging_save_func(path: Path, result: Any) -> None:
    """Save function for tagging results.

    Args:
        path: Original image path
        result: Tagging result (list of tags)
    """
    mapping = categorize_tags(result)
    out_path = path.with_name(f"{path.stem}_tag.json")
    with open(out_path, "w") as f:
        json.dump({"tags": result, "category_map": mapping}, f, indent=2)
    print(f"Saved tags for {path} -> {out_path}")


def process_video_folder(folder: Path, lister) -> None:
    """Annotate all images in a video folder with drivable area tags."""
    images = collect_images(folder)
    if not images:
        return

    # Initialize batch processor
    processor = BatchProcessor(batch_size=4)

    # Load images into buffer
    processor.load_images(images)

    # Create inference function with lister
    def inference_func(imgs, paths):
        return tagging_inference_func(imgs, paths, lister)

    # Process batch
    processor.process_batch(inference_func=inference_func, save_func=tagging_save_func)

    # Clean up
    processor.clear_buffer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate images in subfolders with drivable area tags"
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
        help="Path to configuration file specifying the VLM",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum number of JPEG images required for a folder to be considered a video folder",
    )
    
    parser.add_argument(
        "--drivable-keywords",
        type=str,
        default="",
        help="Comma-separated list of drivable area keywords to use for tagging",
    )
    
    args = parser.parse_args()

    DRIVABLE_KEYWORDS = [
        "road",
        "street",
    ]

    TASK_PROMPT = f"""Identify and list all objects in this image that represent passable areas or surfaces where a person, vehicle, or entity could travel, walk, or move through. Focus on identifying any areas that allow movement or transportation.

    Examples of passable areas to look for include (but are not limited to): {", ".join(DRIVABLE_KEYWORDS)}.

    Please list all visible passable areas, surfaces, and pathways in the image, even if they are partially visible or in the background. Be comprehensive and include any area that could be used for movement or transportation."""

    # merge drivable keywords
    DRIVABLE_KEYWORDS.extend(parse_comma_separated_string(args.drivable_keywords))
    
    # print current config for accessible area
    print(f"Current drivable area keywords: {DRIVABLE_KEYWORDS}, you can add more with --drivable-keywords")

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    # set custom task prompt for this task

    cfg.set_param("object_listing_settings.task_prompt", TASK_PROMPT)

    lister = cfg.get_object_lister()

    print(f"Processing root folder: {args.folder}")
    print(f"Minimum images required per subfolder: {args.min_images}")

    processed_count = 0
    skipped_count = 0

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        images = collect_images(sub)
        if len(images) >= args.min_images:
            print(f"Processing video folder '{sub.name}' with {len(images)} images...")
            process_video_folder(sub, lister)
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
