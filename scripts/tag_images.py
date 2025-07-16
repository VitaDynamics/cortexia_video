import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Dict
import os

import numpy as np
from PIL import Image
from json_repair import repair_json

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
        response_text = lister.list_objects_in_image(img)
        try:
            # Attempt to repair and parse the JSON output from the VLM
            repaired_json = repair_json(response_text)
            parsed_json = json.loads(repaired_json)
            results.append(parsed_json)
        # FIXME: if json repare failed, it will return a empty json. Never have except.
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {paths}: {response_text}")
            results.append({})  # Append an empty dict on failure
    return results




def tagging_save_func(path: Path, result: Any, detectable_tags_keys: List[str]) -> None:
    """Save function for tagging results.

    Args:
        path: Original image path
        result: Tagging result (parsed JSON from VLM)
        detectable_tags_keys: List of keys for detectable tags
    """
    out_path = path.with_name(f"{path.stem}_tag.json")
    
    # Extract detectable tags from the result
    detectable_tags = []
    for key in detectable_tags_keys:
        if key in result:
            detectable_tags.extend(result[key])
            
    result_with_detectable = result.copy()
    result_with_detectable["detectable_tags"] = detectable_tags

    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(result_with_detectable, f, indent=2, ensure_ascii=False)
    print(f"Saved tags for {path} -> {out_path}")


def process_video_folder(folder: Path, lister, detectable_tags_keys: List[str]) -> None:
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
    # Create a save function with the detectable tags
    def save_func(path, result):
        return tagging_save_func(path, result, detectable_tags_keys)

    processor.process_batch(inference_func=inference_func, save_func=save_func)

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
        "--categories",
        type=str,
        default=json.dumps({
            "Detectable tags": ["Drivable Area", "Traffic Participants", "Environmental Landmarks"],
            "Undetectable tags": ["Light and Weather"]
        }),
        help="JSON string of categories for tagging",
    )
    
    args = parser.parse_args()
    
    categories = json.loads(args.categories)
    detectable_tags_keys = categories.get("Detectable tags", [])
    
    all_category_keys = []
    for key in categories:
        all_category_keys.extend(categories[key])
    
    category_json_structure = {cat: [] for cat in all_category_keys}
    
    TASK_PROMPT = f"""Please analyze the provided image and generate a comprehensive list of relevant tags. The tags should be categorized according to the following structure. Please output a JSON object with the specified keys.

    For example:
    ```json
    {{
      "Drivable Area": ["Asphalt road", "Sidewalk"],
      "Traffic Participants": ["Car", "Pedestrian"],
      "Environmental Landmarks": ["Traffic light", "Road sign"],
      "Light and Weather": ["Daytime", "Sunny"]
    }}
    ```

    Here is the required structure for your response:
    {json.dumps(category_json_structure, indent=2, ensure_ascii=False)}

    Instructions:
    - For each category, provide a list of descriptive tags that accurately represent the objects, scenes, and conditions in the image.
    - If a category is not applicable, provide an empty list.
    - Ensure the output is a valid JSON object.
    - The language of the tags should be English.
    - Only return the JSON object, without any other text or explanations.
    """

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
            process_video_folder(sub, lister, detectable_tags_keys)
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
