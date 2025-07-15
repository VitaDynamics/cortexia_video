import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchProcessor, collect_images
from cortexia_video.config_manager import ConfigManager


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


def process_video_folder(folder: Path, captioner) -> None:
    images = collect_images(folder)
    if not images:
        return

    processor = BatchProcessor(batch_size=4)
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
            process_video_folder(sub, captioner)
            processed_count += 1
        else:
            print(
                f"Skipping folder '{sub.name}' - only {len(images)} images (minimum {args.min_images} required)"
            )
            skipped_count += 1

    print(f"\nSummary: Processed {processed_count} folders, skipped {skipped_count} folders")


if __name__ == "__main__":
    main()
