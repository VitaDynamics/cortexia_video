import argparse
import json
from pathlib import Path

from PIL import Image

from cortexia_video.config_manager import ConfigManager

DRIVABLE_KEYWORDS = [
    "road",
    "street",
    "lane",
    "highway",
    "sidewalk",
    "path",
    "trail",
    "stairs",
    "staircase",
    "stairway",
    "elevator",
    "escalator",
    "turnstile",
    "parking",
    "driveway",
    "ramp",
    "bridge",
    "tunnel",
    "floor",
    "pavement",
    "ground",
    "passage",
    "corridor",
    "hallway",
]


def filter_drivable_tags(tags: list[str]) -> list[str]:
    """Filter tags to keep only drivable area related keywords."""
    filtered = []
    for tag in tags:
        lower = tag.lower()
        if any(keyword in lower for keyword in DRIVABLE_KEYWORDS):
            filtered.append(tag)
    return filtered


def collect_images(folder: Path) -> list[Path]:
    """Collect JPG/JPEG images from a folder."""
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    all_images = []
    for pattern in patterns:
        all_images.extend(folder.glob(pattern))
    return sorted(all_images)


def process_video_folder(folder: Path, lister) -> None:
    """Annotate all images in a video folder with drivable area tags."""
    for image_path in collect_images(folder):
        with Image.open(image_path) as img:
            tags = lister.list_objects_in_image(img.convert("RGB"))
        drivable_tags = filter_drivable_tags(tags)
        out_path = image_path.with_name(f"{image_path.stem}_tag.json")
        with open(out_path, "w") as f:
            json.dump({"tags": drivable_tags}, f, indent=2)
        print(f"Saved tags for {image_path} -> {out_path}")


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
        default=Path("config/example_config.toml"),
        help="Path to configuration file specifying the VLM",
    )
    args = parser.parse_args()

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    lister = cfg.get_object_lister()

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        process_video_folder(sub, lister)


if __name__ == "__main__":
    main()
