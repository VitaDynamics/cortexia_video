import argparse
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.object_segmentation import ObjectSegmenter


def collect_images(folder: Path) -> list[Path]:
    """Collect JPG/JPEG images from a folder."""
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    images = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))
    return sorted(images)


def load_tags(tag_file: Path) -> list[str]:
    if not tag_file.exists():
        return []
    with open(tag_file, "r") as f:
        data = json.load(f)
    return data.get("tags", [])


def process_image(image_path: Path, detector: ObjectDetector, segmenter: ObjectSegmenter) -> None:
    tag_file = image_path.with_name(f"{image_path.stem}_tag.json")
    tags = load_tags(tag_file)
    if not tags:
        print(f"No tags found for {image_path}, skipping")
        return

    with Image.open(image_path) as img:
        pil_img = img.convert("RGB")
        detections_batch = detector.detect_objects([pil_img], [tags])
    detections = detections_batch[0] if detections_batch else []
    boxes = [d["box"] for d in detections]
    masks = []
    if boxes:
        masks_batch = segmenter.segment_object(pil_img, boxes)
        masks = masks_batch[0] if masks_batch else []

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
    with open(tag_file, "w") as f:
        json.dump(out_data, f, indent=2)

    # Save mask npy
    if masks_dict:
        mask_path = image_path.with_name(f"{image_path.stem}_masks.npy")
        np.save(mask_path, masks_dict, allow_pickle=True)
        print(f"Saved masks for {image_path} -> {mask_path}")
    else:
        print(f"No detections for {image_path}")


def process_video_folder(folder: Path, detector: ObjectDetector, segmenter: ObjectSegmenter) -> None:
    for image_path in collect_images(folder):
        process_image(image_path, detector, segmenter)


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
        default=Path("config/example_config.toml"),
        help="Path to configuration file specifying models",
    )
    args = parser.parse_args()

    cfg = ConfigManager(config_file_path=str(args.config))
    cfg.load_config()
    detector = ObjectDetector(cfg)
    segmenter = ObjectSegmenter(cfg)

    for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        process_video_folder(sub, detector, segmenter)


if __name__ == "__main__":
    main()
