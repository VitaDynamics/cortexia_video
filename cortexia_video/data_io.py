import os
import json
from typing import Generator, Tuple, Any
import cv2
from pathlib import Path
from .schemes import VideoContent

def load_video_frames(video_path: str, frame_interval: int) -> Generator[Tuple[int, float, Any], None, None]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                yield frame_count, timestamp, frame

            frame_count += 1

        cap.release()
    except Exception as e:
        raise RuntimeError(f"Error processing video: {video_path}") from e

def save_annotations(video_content: VideoContent, output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(video_content.to_dict(), f, indent=2)