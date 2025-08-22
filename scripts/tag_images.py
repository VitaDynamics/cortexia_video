import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional
import os

import numpy as np
from PIL import Image
from json_repair import repair_json

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images, create_lance_load_func

from cortexia_video.config_manager import ConfigManager
from cortexia_video.parser import parse_comma_separated_string

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False



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


def create_lance_tagging_save_func(lance_dataset_path: Path, all_category_keys: List[str]):
    """Create a save function that updates Lance dataset with nested tag structure.
    
    Args:
        lance_dataset_path: Path to Lance dataset
        all_category_keys: List of all possible category keys for schema
        
    Returns:
        Save function for Lance dataset updates with nested struct
    """
    def lance_tagging_save_func(index: int, result: Dict, detectable_tags_keys: List[str]) -> None:
        """Save tags to Lance dataset with nested struct format.
        
        Args:
            index: Row index in Lance dataset
            result: Tagging result (parsed JSON from VLM)
            detectable_tags_keys: List of keys for detectable tags
        """
        try:
            # Open the dataset
            dataset = lance.dataset(str(lance_dataset_path))
            
            # Extract detectable tags
            detectable_tags = []
            for key in detectable_tags_keys:
                if key in result:
                    detectable_tags.extend(result[key])
            
            # Create the nested struct with normalized field names
            # Convert category names to valid Arrow field names (replace spaces with underscores)
            tags_struct = {}
            for category in all_category_keys:
                field_name = category.replace(" ", "_").replace("-", "_")
                tags_struct[field_name] = result.get(category, [])
            
            # Add detectable tags
            tags_struct["detectable_tags"] = detectable_tags
            
            # Create the update data with nested struct
            update_data = pa.table({
                "_rowid": [index],
                "tags": [tags_struct]
            })
            
            # Merge the tags into the dataset
            dataset = dataset.merge(
                update_data, 
                left_on="_rowid", 
                right_on="_rowid"
            )
            
            print(f"Saved tags for index {index}: {len(detectable_tags)} detectable tags")
            
        except Exception as e:
            print(f"Failed to save tags to Lance dataset for index {index}: {e}")
    
    return lance_tagging_save_func


def create_lance_tag_schema(all_category_keys: List[str]) -> pa.Schema:
    """Create PyArrow schema for tag storage with nested struct.
    
    Args:
        all_category_keys: List of all category names
        
    Returns:
        PyArrow schema with nested tag structure
    """
    # Create struct fields for each category
    tag_fields = []
    for category in all_category_keys:
        field_name = category.replace(" ", "_").replace("-", "_")
        tag_fields.append(pa.field(field_name, pa.list_(pa.string())))
    
    # Add detectable_tags field
    tag_fields.append(pa.field("detectable_tags", pa.list_(pa.string())))
    
    # Create the full schema
    schema = pa.schema([
        pa.field("_rowid", pa.int64()),  # Row identifier
        pa.field("image", pa.binary()),  # Image data (if storing images)
        pa.field("tags", pa.struct(tag_fields))  # Nested tag structure
    ])
    
    return schema


def process_video_folder(
    folder: Path, 
    lister, 
    detectable_tags_keys: List[str], 
    image_format: str,
    lance_dataset_path: Optional[Path] = None,
    all_category_keys: Optional[List[str]] = None
) -> None:
    """Annotate all images in a video folder with tags.
    
    Args:
        folder: Folder containing images
        lister: Object lister instance
        detectable_tags_keys: List of keys for detectable tags
        image_format: Image file format to process
        lance_dataset_path: Optional path to Lance dataset for tag storage
        all_category_keys: List of all category keys (required for Lance mode)
    """
    images = collect_images(folder, image_format=image_format)
    if not images:
        return

    # Initialize batch processor
    processor = BatchProcessor(batch_size=4)

    if lance_dataset_path and LANCE_AVAILABLE and all_category_keys:
        # Use Lance dataset mode
        indices = list(range(len(images)))
        processor.load_indices(indices)
        
        # Create Lance-specific functions
        lance_dataset = lance.dataset(str(lance_dataset_path))
        load_func = create_lance_load_func(lance_dataset, image_column="image")
        save_func = create_lance_tagging_save_func(lance_dataset_path, all_category_keys)
        
        def inference_func(imgs, indices):
            return tagging_inference_func(imgs, indices, lister)
        
        # Create a save function wrapper with detectable tags
        def wrapped_save_func(index, result):
            return save_func(index, result, detectable_tags_keys)
        
        processor.process_batch_generic(
            load_func=load_func,
            inference_func=inference_func, 
            save_func=wrapped_save_func
        )
    else:
        # Use traditional path-based mode
        processor.load_images(images)
        
        # Create inference function with lister
        def inference_func(imgs, paths):
            return tagging_inference_func(imgs, paths, lister)
        
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
    parser.add_argument(
        "--lance-dataset",
        type=Path,
        help="Path to Lance dataset for tag storage (optional, defaults to JSON files)",
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
    image_format = cfg.get_param("processing.image_format", "jpg")
    # set custom task prompt for this task

    cfg.set_param("object_listing_settings.task_prompt", TASK_PROMPT)

    lister = cfg.get_object_lister()

    print(f"Processing root folder: {args.folder}")
    print(f"Minimum images required per subfolder: {args.min_images}")

    processed_count = 0
    skipped_count = 0

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        images = collect_images(sub, image_format=image_format)
        if len(images) >= args.min_images:
            print(f"Processing video folder '{sub.name}' with {len(images)} images...")
            process_video_folder(sub, lister, detectable_tags_keys, image_format, args.lance_dataset, all_category_keys)
            processed_count += 1
        else:
            print(
                f"Skipping folder '{sub.name}' - only {len(images)} images (minimum {args.min_images} required)"
            )
            skipped_count += 1

    output_mode = "Lance dataset (nested struct)" if args.lance_dataset else "JSON files"
    print(
        f"\nSummary: Processed {processed_count} folders, skipped {skipped_count} folders"
    )
    print(f"Output mode: {output_mode}")
    
    if args.lance_dataset and not LANCE_AVAILABLE:
        print("Warning: Lance dataset path provided but lance/pyarrow not available. Used JSON mode instead.")
    
    if args.lance_dataset and LANCE_AVAILABLE:
        print("\nLance schema for reference:")
        schema = create_lance_tag_schema(all_category_keys)
        print(schema)


if __name__ == "__main__":
    main()
