import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from cortexia_video import data_io
from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_description import ObjectDescriber
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.object_segmentation import ObjectSegmenter
from cortexia_video.parser import generate_formatted_string_dino
from cortexia_video.schemes import (
    DetectionResult,
    FrameData,
    SegmentationResult,
    VideoContent,
)
from cortexia_video.visualization import generate_annotated_frame


class ProcessingManager:
    def __init__(
        self,
        config_path: str = "config",
        config_manager: Optional[ConfigManager] = None,
    ):
        self.config_manager = (
            config_manager if config_manager else ConfigManager(config_path)
        )
        self.logger = logging.getLogger(__name__)

        # Initialize component variables to None
        self.lister = None
        self.detector = None
        self.segmenter = None
        self.describer = None

    def load_components(self, processing_mode: str) -> None:
        """
        Dynamically load components based on the processing mode.

        Args:
            processing_mode (str): Space-separated string of processing modes (e.g., "list detect segment describe")
        """
        # Reset all components to None
        self.lister = None
        self.detector = None
        self.segmenter = None
        self.describer = None

        # Load only the components needed for the processing modes
        if "list" in processing_mode:
            self.logger.info("Loading object lister component")
            self.lister = self.config_manager.get_object_lister()

        if "detect" in processing_mode:
            self.logger.info("Loading object detector component")
            self.detector = ObjectDetector(self.config_manager)

        if "segment" in processing_mode:
            self.logger.info("Loading object segmenter component")
            self.segmenter = ObjectSegmenter(self.config_manager)

        if "describe" in processing_mode:
            self.logger.info("Loading object describer component")
            self.describer = ObjectDescriber(self.config_manager)

    def process_video(self, video_path: str, processing_mode: str) -> str:
        # Dynamically load components based on processing mode
        self.load_components(processing_mode)

        frame_interval = self.config_manager.get_param("processing.frame_interval", 1)
        output_dir = self.config_manager.get_param(
            "processing.output_directory", "output"
        )

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_content_instance = VideoContent(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
        )

        processing_stats = {}

        for frame_number, timestamp, frame_np_array in data_io.load_video_frames(
            video_path, frame_interval
        ):
            # Convert BGR to RGB
            rgb_array = cv2.cvtColor(frame_np_array, cv2.COLOR_BGR2RGB)

            frame_data = FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                rgb_image=rgb_array,  # Store RGB numpy array
            )

            pil_image = Image.fromarray(rgb_array)
            detections: List[DetectionResult] = []
            segments: List[SegmentationResult] = []
            object_names = []
            dino_prompt = ""

            if "list" in processing_mode and self.lister is not None:
                try:
                    object_names = self.lister.list_objects_in_image(pil_image)
                    # Save lister's results directly in frame_data
                    frame_data.lister_results = object_names
                    dino_prompt = generate_formatted_string_dino(object_names)
                except Exception as e:
                    self.logger.error(
                        f"Error listing objects in frame {frame_number}: {e}",
                        exc_info=True,
                    )
                    frame_data.lister_results = ["error_processing_frame"]
                    exit()  # TODO: Correctly handle this error.

            if "detect" in processing_mode and self.detector is not None:
                text_prompt_for_detection = self.config_manager.get_param(
                    "model_settings.detection_prompt", dino_prompt
                )
                try:
                    results, detections = self.detector.detect_objects(
                        pil_image, text_prompt_for_detection
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error detecting objects in frame {frame_number}: {e}",
                        exc_info=True,
                    )

            # Create a detection lookup by ID for efficient access
            detection_by_id = {detection.id: detection for detection in detections}

            if (
                "segment" in processing_mode
                and self.segmenter is not None
                and detections
            ):
                for detection in detections:
                    try:
                        mask_np = self.segmenter.segment_object(
                            image_data=pil_image,
                            input_boxes=[
                                [
                                    detection.box.xmin,
                                    detection.box.ymin,
                                    detection.box.xmax,
                                    detection.box.ymax,
                                ]
                            ],
                        )
                        # Calculate area of mask (number of non-zero pixels)
                        area = np.count_nonzero(mask_np)

                        segment = SegmentationResult(
                            mask=mask_np,
                            score=detection.score,
                            label=detection.label,
                            area=area,
                            bbox=detection.box,
                            detection_id=detection.id,  # Link segment to detection by ID
                        )
                        segments.append(segment)
                        # Also add the mask to the detection
                        detection.mask = mask_np

                        # Optionally generate a visualization with contours if enabled
                        contour_viz_enabled = self.config_manager.get_param(
                            "visualization.contour_enabled", False
                        )
                        if contour_viz_enabled:
                            contour_img = self.segmenter.generate_contour_visualization(
                                pil_image,
                                mask_np,
                                [
                                    [
                                        detection.box.xmin,
                                        detection.box.ymin,
                                        detection.box.xmax,
                                        detection.box.ymax,
                                    ]
                                ],
                            )
                            # Save contour visualization
                            contour_output_dir = (
                                Path(output_dir) / "contours" / Path(video_path).stem
                            )
                            contour_output_dir.mkdir(parents=True, exist_ok=True)
                            contour_img_path = (
                                contour_output_dir
                                / f"frame_{frame_number:05d}_obj_{detection.label}.jpg"
                            )
                            contour_img.save(contour_img_path)
                    except Exception as e:
                        self.logger.error(
                            f"Error segmenting object in frame {frame_number}: {e}",
                            exc_info=True,
                        )

            if (
                "describe" in processing_mode
                and self.describer is not None
                and segments
            ):
                for segment in segments:
                    try:
                        # Use the detection_id to directly access the corresponding detection
                        if (
                            segment.detection_id
                            and segment.detection_id in detection_by_id
                        ):
                            detection = detection_by_id[segment.detection_id]

                            # Generate description with non-streaming option
                            description = self.describer.describe_object(
                                pil_image, segment.mask
                            )
                            # Save description directly in detection object
                            detection.description = description

                            # Generate visualization if enabled
                            description_viz_enabled = self.config_manager.get_param(
                                "visualization.description_viz_enabled", False
                            )
                            if description_viz_enabled:
                                # Create visualization with description overlay
                                desc_viz_img = self.describer.create_visualization(
                                    pil_image, segment.mask, description
                                )

                                # Save visualization
                                desc_viz_dir = (
                                    Path(output_dir)
                                    / "descriptions"
                                    / Path(video_path).stem
                                )
                                desc_viz_dir.mkdir(parents=True, exist_ok=True)
                                desc_img_path = (
                                    desc_viz_dir
                                    / f"frame_{frame_number:05d}_obj_{detection.label}.jpg"
                                )
                                desc_viz_img.save(desc_img_path)
                                self.logger.info(
                                    f"Saved description visualization: {desc_img_path}"
                                )
                    except Exception as e:
                        self.logger.error(
                            f"Error describing object in frame {frame_number}: {e}",
                            exc_info=True,
                        )

            # Update frame data - descriptions are already saved in detection objects
            frame_data.detections = detections
            frame_data.segments = segments

            # Handle visualization if enabled
            viz_enabled = self.config_manager.get_param("visualization.enabled", False)
            if viz_enabled:
                try:
                    annotated_pil_image = generate_annotated_frame(
                        pil_image, frame_data
                    )
                    output_img_dir = (
                        Path(output_dir) / "annotated_frames" / Path(video_path).stem
                    )
                    output_img_dir.mkdir(parents=True, exist_ok=True)
                    img_format = self.config_manager.get_param(
                        "visualization.annotated_image_format", "jpg"
                    )
                    output_img_path = (
                        output_img_dir / f"frame_{frame_number:05d}.{img_format}"
                    )
                    annotated_pil_image.save(output_img_path)
                    self.logger.info(f"Saved annotated frame: {output_img_path}")
                except Exception as e:
                    self.logger.error(
                        f"Error creating visualization for frame {frame_number}: {e}",
                        exc_info=True,
                    )

            # Store frame by number in dictionary
            video_content_instance.frames[frame_number] = frame_data

        # Update processing stats
        processing_stats["frames_processed"] = len(video_content_instance.frames)
        video_content_instance.processing_stats = processing_stats

        output_filename = f"{Path(video_path).stem}_annotations.json"
        output_path = Path(output_dir) / output_filename
        data_io.save_annotations(video_content_instance, str(output_path))

        return str(output_path)
