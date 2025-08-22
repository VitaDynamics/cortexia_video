import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images, create_lance_load_func
from cortexia_video.config_manager import ConfigManager

try:
    # TODO: Fix this
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False


def caption_inference_func(images: List[Image.Image], paths: List[Path], captioner) -> List[str]:
    """Generate captions for a batch of images."""
    results = []
    for img in images:
        caption = captioner.caption_image(img)
        results.append(caption)
    return results


def caption_save_func(path: Path, caption: str) -> None:
    """Save caption to the image's tag JSON file."""
    tag_file = path.with_name(f"{path.stem}_tag.json")
    if tag_file.exists():
        with open(tag_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data["caption"] = caption

    with open(tag_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved caption for {path} -> {tag_file}")


def create_lance_caption_save_func(lance_dataset_path: Path):
    """Create a save function that updates Lance dataset with captions.
    
    Args:
        lance_dataset_path: Path to Lance dataset
        
    Returns:
        Save function for Lance dataset updates
    """
    def lance_caption_save_func(index: int, caption: str) -> None:
        """Save caption to Lance dataset by index."""
        try:
            # Open the dataset
            dataset = lance.dataset(str(lance_dataset_path))
            
            # Create update data with the caption
            update_data = pa.table({
                "_rowid": [index],
                "caption": [caption]
            })
            
            # Merge the caption into the dataset
            # This will add the caption column if it doesn't exist or update existing values
            dataset = dataset.merge(
                update_data, 
                left_on="_rowid", 
                right_on="_rowid"
            )
            
            print(f"Saved caption for index {index}: {caption[:50]}...")
            
        except Exception as e:
            print(f"Failed to save caption to Lance dataset for index {index}: {e}")
    
    return lance_caption_save_func


def process_video_folder(folder: Path, captioner, lance_dataset_path: Optional[Path] = None) -> None:
    """Process video folder with either JSON or Lance output.
    
    Args:
        folder: Folder containing images
        captioner: Image captioning model
        lance_dataset_path: Optional path to Lance dataset for caption storage
    """
    images = collect_images(folder)
    if not images:
        return

    processor = BatchProcessor(batch_size=4)
    
    if lance_dataset_path and LANCE_AVAILABLE:
        # Use Lance dataset mode
        # Assume images are indexed 0, 1, 2, ... in the Lance dataset
        # You may need to adjust this mapping based on your Lance dataset structure
        indices = list(range(len(images)))
        processor.load_indices(indices)
        
        # Create Lance-specific functions
        lance_dataset = lance.dataset(str(lance_dataset_path))
        load_func = create_lance_load_func(lance_dataset, image_column="image")
        save_func = create_lance_caption_save_func(lance_dataset_path)
        
        def inference_func(imgs, indices):
            return caption_inference_func(imgs, indices, captioner)
        
        processor.process_batch_generic(
            load_func=load_func,
            inference_func=inference_func, 
            save_func=save_func
        )
    else:
        # Use traditional path-based mode
        processor.load_images(images)
        
        def inference_func(imgs, paths):
            return caption_inference_func(imgs, paths, captioner)
        
        processor.process_batch(inference_func=inference_func, save_func=caption_save_func)
    
    processor.clear_buffer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--folder", type=Path, required=True, help="Root folder containing video subfolders")
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
        help="Minimum number of JPEG images required for a folder to be processed",
    )
    parser.add_argument(
        "--lance-dataset",
        type=Path,
        help="Path to Lance dataset for caption storage (optional, defaults to JSON files)",
    )
    args = parser.parse_args()

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    captioner = cfg.get_image_captioner()

    processed_count = 0
    skipped_count = 0
    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        images = collect_images(sub)
        if len(images) >= args.min_images:
            print(f"Processing video folder '{sub.name}' with {len(images)} images...")
            process_video_folder(sub, captioner, args.lance_dataset)
            processed_count += 1
        else:
            print(
                f"Skipping folder '{sub.name}' - only {len(images)} images (minimum {args.min_images} required)"
            )
            skipped_count += 1

    output_mode = "Lance dataset" if args.lance_dataset else "JSON files"
    print(f"\nSummary: Processed {processed_count} folders, skipped {skipped_count} folders")
    print(f"Output mode: {output_mode}")
    
    if args.lance_dataset and not LANCE_AVAILABLE:
        print("Warning: Lance dataset path provided but lance/pyarrow not available. Used JSON mode instead.")


if __name__ == "__main__":
    main()
