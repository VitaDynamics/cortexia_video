import json
import os
import pickle
from pathlib import Path
from typing import Any, Generator, Tuple

from decord import VideoReader, cpu

from .schemes import VideoContent


def load_video_frames(
    video_path: str, frame_interval: int, batch_size: int = 1
) -> Generator[Tuple[int, float, Any], None, None]:
    try:
        # Create VideoReader object from decord
        vr = VideoReader(video_path, ctx=cpu(0))

        # Get FPS information from VideoReader
        fps = vr.get_avg_fps()

        # Total number of frames
        total_frames = len(vr)

        # Generate frame indices based on interval
        frame_indices = range(0, total_frames, frame_interval)
        if batch_size == 1:
            for frame_count in frame_indices:
                # Read the frame at the specified index
                frame = vr[frame_count].asnumpy()

            # Calculate timestamp
                timestamp = frame_count / fps

                # Yield frame with metadata
                yield frame_count, timestamp, frame
        elif batch_size > 1:
            for i in range(0, total_frames, batch_size):

    except Exception as e:
        raise RuntimeError(f"Error processing video: {video_path}") from e


def save_annotations(video_content: VideoContent, output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Determine save location based on video_path
    video_path = Path(video_content.video_path)

    # Check if video_path is a directory or file
    if video_path.is_dir():
        # If video_path is a directory, save in that directory
        save_dir = video_path
        prefix = ""
    else:
        # If video_path is a file, use file name as prefix and save in its parent directory
        save_dir = video_path.parent
        prefix = f"{video_path.stem}_"

    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. Create metadata JSON file
    meta_json_path = save_dir / f"{prefix}metadata.json"

    # Extract metadata that doesn't need numpy serialization
    meta_json_label = {
        "video_path": str(video_content.video_path),
        "total_frames": video_content.total_frames,
        "fps": video_content.fps,
        "width": video_content.width,
        "height": video_content.height,
        "processing_stats": video_content.processing_stats,
        "frames": {},
    }

    # Process frames to extract metadata
    detections_by_id = {}
    segmentations_by_id = {}

    for frame_num, frame_data in video_content.frames.items():
        frame_meta = {
            "frame_number": frame_data.frame_number,
            "timestamp": frame_data.timestamp,
            "lister_results": frame_data.lister_results,
            "detection_ids": [],
        }

        # Extract detection metadata
        for detection in frame_data.detections:
            detection_dict = {
                "id": detection.id,
                "label": detection.label,
                "score": float(detection.score),
                "box": {
                    "xmin": float(detection.box.xmin),
                    "ymin": float(detection.box.ymin),
                    "xmax": float(detection.box.xmax),
                    "ymax": float(detection.box.ymax),
                },
            }

            # Include description if available
            if hasattr(detection, "description") and detection.description:
                detection_dict["description"] = detection.description

            # Store detection ID in frame metadata
            frame_meta["detection_ids"].append(detection.id)

            # Store full detection object (with numpy arrays) for pickle
            detections_by_id[detection.id] = detection

        # Extract segmentation metadata and full objects for pickle
        for segment in frame_data.segments:
            if segment.detection_id:
                segmentations_by_id[segment.detection_id] = segment

        meta_json_label["frames"][str(frame_num)] = frame_meta

    # Save metadata JSON
    with open(meta_json_path, "w") as f:
        json.dump(meta_json_label, f, indent=2)

    # 2. Save detections pickle file
    detections_path = save_dir / f"{prefix}detections.pkl"
    with open(detections_path, "wb") as f:
        pickle.dump(detections_by_id, f)

    # 3. Save segmentations pickle file
    segmentations_path = save_dir / f"{prefix}segmentations.pkl"
    with open(segmentations_path, "wb") as f:
        pickle.dump(segmentations_by_id, f)

    return str(meta_json_path)
