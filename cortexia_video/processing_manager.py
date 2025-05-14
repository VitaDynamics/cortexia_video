import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from cortexia_video import data_io
from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_description import ObjectDescriber
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.object_segmentation import ObjectSegmenter
from cortexia_video.parser import generate_formatted_string_dino
from cortexia_video.schemes import (
    BoundingBox,
    DetectionResult,
    FrameData,
    SegmentationResult,
    VideoContent,
)
from cortexia_video.utils import UtilManager
from cortexia_video.visualization import generate_annotated_frame

# TODO: Add a better config manger with mode pipeline.


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

        # Get batch processing configuration
        self.batch_size = self.config_manager.get_param("processing.batch_size", 4)
        self.logger.info(f"Using batch size: {self.batch_size}")

        # Initialize component variables to None
        self.lister = None
        self.detector = None
        self.segmenter = None
        self.describer = None
        self.feature_extractor = None
        # Initialize utility manager
        self.util_manager = UtilManager(logger=self.logger)
        self.object_describer = None  # Object describer for batch processing

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
        self.feature_extractor = None
        # Keep util_manager initialized

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
            self.object_describer = ObjectDescriber(
                self.config_manager
            )  # For batch processing

        if "extract_scene" in processing_mode or "extract_object" in processing_mode:
            self.logger.info("Loading feature extractor component")
            self.feature_extractor = self.config_manager.get_feature_extractor()

    def process_video(self, video_path: str, processing_mode: str) -> str:
        # Dynamically load components based on processing mode
        self.load_components(processing_mode)

        frame_interval = self.config_manager.get_param("processing.frame_interval", 1)
        output_dir = self.config_manager.get_param(
            "processing.output_directory", "output"
        )

        # Get video properties using Decord
        self.video_reader = VideoReader(video_path, ctx=cpu(0))
        fps = self.video_reader.get_avg_fps()
        total_frames = len(self.video_reader)
        # Get dimensions from the first frame
        first_frame = self.video_reader[0].asnumpy()
        height, width = first_frame.shape[:2]

        video_content_instance = VideoContent(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
        )

        processing_stats = {}

        # Generate frame indices based on interval
        frame_indices = range(0, total_frames, frame_interval)

        # Iterate over batches from load_video_frames
        for batch_data in data_io.load_video_frames(
            video_path, frame_interval, self.batch_size
        ):
            # Handle unpacking based on batch_size
            if self.batch_size == 1:
                # Single frame mode: unpacking (frame_number, timestamp, frame_np)
                frame_number, timestamp, frame_np = batch_data
                batch_frames_meta = [
                    {"frame_number": frame_number, "timestamp": timestamp}
                ]
                batch_frames_np = [frame_np]
            else:
                # Batch mode: unpacking (loaded_batch_meta, loaded_batch_np, _)
                loaded_batch_meta, loaded_batch_np, _ = batch_data
                batch_frames_meta = loaded_batch_meta
                batch_frames_np = loaded_batch_np

            # Initialize batch containers for current iteration
            current_batch_frame_data_objects = []
            current_batch_pil_images = []

            # Process each frame in the batch
            for i in range(len(batch_frames_np)):
                meta_item = batch_frames_meta[i]
                np_image = batch_frames_np[i]

                # Convert BGR to RGB if coming from cv2 (not needed with Decord, but keeping for compatibility)
                if (
                    np_image.shape[2] == 3
                    and np.sum(np.abs(np_image[:, :, 0] - np_image[:, :, 2])) > 0
                ):
                    rgb_array = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_array = np_image  # already in RGB format from Decord

                # Create FrameData object
                frame_data_obj = FrameData(
                    frame_number=meta_item["frame_number"],
                    timestamp=meta_item["timestamp"],
                    rgb_image=rgb_array,  # Store RGB numpy array
                )

                # Convert to PIL for processing
                pil_image = Image.fromarray(rgb_array)

                # Add to batch lists
                current_batch_frame_data_objects.append(frame_data_obj)
                current_batch_pil_images.append(pil_image)

            # Extract scene-level features if enabled (batched processing)
            if "extract_scene" in processing_mode and self.feature_extractor:
                try:
                    if current_batch_pil_images:
                        batch_scene_features_tensor = (
                            self.feature_extractor.extract_image_features(
                                current_batch_pil_images
                            )
                        )
                        for i, frame_data_obj in enumerate(
                            current_batch_frame_data_objects
                        ):
                            if i < batch_scene_features_tensor.shape[0]:
                                frame_data_obj.scene_clip_features = (
                                    batch_scene_features_tensor[i].cpu().numpy()
                                )
                    else:
                        self.logger.info(
                            "No PIL images in current batch for scene feature extraction."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error extracting batch scene features: {e}",
                        exc_info=True,
                    )

            segments: List[SegmentationResult] = []
            object_names = []
            dino_prompt = ""

            # Process object listing in batches if enabled
            if "list" in processing_mode and self.lister is not None:
                try:
                    # Call the batched method
                    batched_lister_outputs = self.lister.list_objects_in_image_batched(
                        current_batch_pil_images
                    )

                    # Map results back to individual frame data objects
                    for i, frame_data_obj in enumerate(
                        current_batch_frame_data_objects
                    ):
                        object_names_for_frame = batched_lister_outputs[i]
                        frame_data_obj.lister_results = object_names_for_frame

                        # Generate DINO prompt
                        dino_prompt = generate_formatted_string_dino(
                            object_names_for_frame
                        )
                        frame_data_obj.dino_prompt = dino_prompt
                except Exception as e:
                    self.logger.error(
                        f"Error in batch object listing: {e}",
                        exc_info=True,
                    )
                    # Set error results for all frames in this batch
                    for frame_data_obj in current_batch_frame_data_objects:
                        frame_data_obj.lister_results = ["error_processing_frame"]
                    exit()  # TODO: Correctly handle this error.

            # Process object detection in batches if enabled
            if "detect" in processing_mode and self.detector is not None:
                try:
                    # Prepare batch inputs for detector
                    batch_dino_prompts = [
                        fd.dino_prompt.split(".")
                        if hasattr(fd, "dino_prompt") and fd.dino_prompt
                        else []
                        for fd in current_batch_frame_data_objects
                    ]

                    # Call the batched detection method
                    batched_detector_outputs = self.detector.detect_objects(
                        current_batch_pil_images,
                        batch_dino_prompts,
                        confidence_threshold=self.config_manager.get_param(
                            "model_settings.confidence_threshold", 0.5
                        ),
                    )

                    # Map results back to individual frame data objects
                    for i, frame_data_obj in enumerate(
                        current_batch_frame_data_objects
                    ):
                        detections_for_frame_dicts = batched_detector_outputs[i]

                        # Convert dictionary results to DetectionResult objects
                        frame_data_obj.detections = []
                        for det_dict in detections_for_frame_dicts:
                            x1, y1, x2, y2 = det_dict["box"]

                            detection_result = DetectionResult(
                                score=det_dict["score"],
                                label=det_dict["label"],
                                box=BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2),
                            )

                            frame_data_obj.detections.append(detection_result)

                except Exception as e:
                    self.logger.error(
                        f"Error in batch object detection: {e}",
                        exc_info=True,
                    )
                    # Set empty detections for all frames in this batch
                    for frame_data_obj in current_batch_frame_data_objects:
                        frame_data_obj.detections = []

            # Process the rest of the pipeline for each frame individually
            for i, frame_data_obj in enumerate(current_batch_frame_data_objects):
                pil_image = current_batch_pil_images[i]
                meta_item = batch_frames_meta[i]

                # Get detections that were set during batch processing
                detections = (
                    frame_data_obj.detections
                    if hasattr(frame_data_obj, "detections")
                    else []
                )

                # Initialize for this frame
                segments = []

                # Get the object names and dino prompt set during batch processing
                object_names = (
                    frame_data_obj.lister_results
                    if hasattr(frame_data_obj, "lister_results")
                    else []
                )
                dino_prompt = (
                    frame_data_obj.dino_prompt
                    if hasattr(frame_data_obj, "dino_prompt")
                    else ""
                )

                # Create a detection lookup by ID for efficient access
                detection_by_id = {detection.id: detection for detection in detections}

                if (
                    "segment" in processing_mode
                    and self.segmenter is not None
                    and detections
                ):
                    # We'll do segmentation in batches
                    # First collect all frames, detections, and bounding boxes for batch processing
                    batch_frames_for_segmentation = []
                    batch_boxes_for_segmentation = []
                    detection_mapping = []  # To map results back to frame_data_obj

                    for i, fd in enumerate(current_batch_frame_data_objects):
                        if hasattr(fd, "detections") and fd.detections:
                            current_frame_pil = current_batch_pil_images[i]
                            batch_frames_for_segmentation.append(current_frame_pil)

                            # Collect boxes for this frame
                            boxes_for_frame = []
                            for det in fd.detections:
                                boxes_for_frame.append(
                                    [
                                        det.box.xmin,
                                        det.box.ymin,
                                        det.box.xmax,
                                        det.box.ymax,
                                    ]
                                )
                                # Store mapping to trace back to original detection
                                detection_mapping.append(
                                    {"frame_idx": i, "detection": det}
                                )

                            batch_boxes_for_segmentation.append(boxes_for_frame)

                    # Call batch segmentation if we have frames to process
                    if batch_frames_for_segmentation:
                        try:
                            batch_masks_results = self.segmenter.segment_object(
                                images_data=batch_frames_for_segmentation,
                                batch_input_boxes=batch_boxes_for_segmentation,
                            )

                            # Process the results
                            mask_idx = 0
                            for frame_idx, frame_masks in enumerate(
                                batch_masks_results
                            ):
                                for mask_in_frame in frame_masks:
                                    if mask_idx < len(detection_mapping):
                                        mapping = detection_mapping[mask_idx]
                                        detection = mapping["detection"]

                                        # Calculate area of mask (number of non-zero pixels)
                                        area = np.count_nonzero(mask_in_frame)

                                        # Create SegmentationResult
                                        segment = SegmentationResult(
                                            mask=mask_in_frame,
                                            score=detection.score,
                                            label=detection.label,
                                            area=area,
                                            bbox=detection.box,
                                            detection_id=detection.id,  # Link segment to detection by ID
                                        )

                                        # Add to the current frame's FrameData object
                                        frame_data = current_batch_frame_data_objects[
                                            mapping["frame_idx"]
                                        ]
                                        if not hasattr(frame_data, "segments"):
                                            frame_data.segments = []
                                        frame_data.segments.append(segment)

                                        # Also add the mask to the detection
                                        detection.mask = mask_in_frame

                                        # Optionally generate visualization with contours if enabled
                                        contour_viz_enabled = (
                                            self.config_manager.get_param(
                                                "visualization.contour_enabled", False
                                            )
                                        )

                                        if contour_viz_enabled:
                                            frame_pil = batch_frames_for_segmentation[
                                                frame_idx
                                            ]
                                            contour_img = self.segmenter.generate_contour_visualization(
                                                frame_pil,
                                                mask_in_frame,
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
                                                Path(output_dir)
                                                / "contours"
                                                / Path(video_path).stem
                                            )
                                            contour_output_dir.mkdir(
                                                parents=True, exist_ok=True
                                            )
                                            frame_number = frame_data.frame_number
                                            contour_img_path = (
                                                contour_output_dir
                                                / f"frame_{frame_number:05d}_obj_{detection.label}.jpg"
                                            )
                                            contour_img.save(contour_img_path)

                                    mask_idx += 1

                            # Get segments for the current frame
                            segments = (
                                frame_data_obj.segments
                                if hasattr(frame_data_obj, "segments")
                                else []
                            )

                        except Exception as e:
                            self.logger.error(
                                f"Error in batch segmentation: {e}",
                                exc_info=True,
                            )

                    # The old per-frame segmentation code is no longer needed
                    # We processed everything in batch

                # Collect cropped object images for batched feature extraction (moved from segmentation loop)
                if (
                    "extract_object" in processing_mode
                    and self.feature_extractor
                    and detections
                ):
                    all_cropped_object_pil_images_in_batch = []
                    object_references_for_mapping = []  # Stores {'frame_data_idx': int, 'detection_idx': int}

                    # Get the frame index in the current batch
                    frame_data_idx = current_batch_frame_data_objects.index(
                        frame_data_obj
                    )
                    original_pil_image = current_batch_pil_images[frame_data_idx]

                    for det_idx, detection in enumerate(detections):
                        if (
                            hasattr(detection, "segmentation")
                            and detection.segmentation
                            and hasattr(detection.segmentation, "mask_array")
                            and detection.segmentation.mask_array is not None
                        ):
                            try:
                                cropped_pil = (
                                    self.util_manager.crop_pil_image_from_mask(
                                        original_pil_image,
                                        detection.segmentation.mask_array,
                                        detection.box.xyxy if detection.box else None,
                                    )
                                )
                                if cropped_pil:
                                    all_cropped_object_pil_images_in_batch.append(
                                        cropped_pil
                                    )
                                    object_references_for_mapping.append(
                                        {
                                            "frame_data_idx": frame_data_idx,
                                            "detection_idx": det_idx,
                                        }
                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Error cropping object for frame {frame_data_obj.frame_number}, detection {det_idx}: {e}"
                                )
                                continue  # Skip this problematic crop

                    # Extract features for all cropped objects in this frame in one batch
                    if all_cropped_object_pil_images_in_batch:
                        try:
                            feature_dim = (
                                self.feature_extractor.model.visual.output_dim
                                if hasattr(self.feature_extractor.model, "visual")
                                else 512
                            )
                            batch_object_features_tensor = (
                                self.feature_extractor.extract_image_features(
                                    all_cropped_object_pil_images_in_batch
                                )
                            )

                            # Map object features back to their detections
                            for i, ref in enumerate(object_references_for_mapping):
                                if i < batch_object_features_tensor.shape[0]:
                                    target_det = detections[ref["detection_idx"]]
                                    target_det.object_clip_features = (
                                        batch_object_features_tensor[i].cpu().numpy()
                                    )
                        except Exception as e:
                            self.logger.error(
                                f"Error extracting batch object features for frame {frame_data_obj.frame_number}: {e}"
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
                                        / f"frame_{meta_item['frame_number']:05d}_obj_{detection.label}.jpg"
                                    )
                                    desc_viz_img.save(desc_img_path)
                                    self.logger.info(
                                        f"Saved description visualization: {desc_img_path}"
                                    )
                        except Exception as e:
                            self.logger.error(
                                f"Error describing object in frame {meta_item['frame_number']}: {e}",
                                exc_info=True,
                            )

                # Update frame data - descriptions are already saved in detection objects
                frame_data_obj.detections = detections
                frame_data_obj.segments = segments

                # Handle visualization if enabled
                viz_enabled = self.config_manager.get_param(
                    "visualization.enabled", False
                )
                if viz_enabled:
                    try:
                        annotated_pil_image = generate_annotated_frame(
                            pil_image, frame_data_obj
                        )
                        output_img_dir = (
                            Path(output_dir)
                            / "annotated_frames"
                            / Path(video_path).stem
                        )
                        output_img_dir.mkdir(parents=True, exist_ok=True)
                        img_format = self.config_manager.get_param(
                            "visualization.annotated_image_format", "jpg"
                        )
                        output_img_path = (
                            output_img_dir
                            / f"frame_{meta_item['frame_number']:05d}.{img_format}"
                        )
                        annotated_pil_image.save(output_img_path)
                        self.logger.info(f"Saved annotated frame: {output_img_path}")
                    except Exception as e:
                        self.logger.error(
                            f"Error creating visualization for frame {meta_item['frame_number']}: {e}",
                            exc_info=True,
                        )

            # Process object descriptions in batch context if enabled
            if "describe" in processing_mode and self.object_describer is not None:
                for frame_data_obj in current_batch_frame_data_objects:
                    frame_pil_image = None
                    try:
                        idx_in_batch = current_batch_frame_data_objects.index(
                            frame_data_obj
                        )
                        frame_pil_image = current_batch_pil_images[idx_in_batch]
                    except ValueError:
                        self.logger.error(
                            f"Could not find frame_data_obj for description in current_batch_pil_images for frame {frame_data_obj.frame_number}"
                        )
                        continue

                    if not frame_pil_image:
                        continue

                    for detection_idx, detection_result in enumerate(
                        frame_data_obj.detections
                    ):
                        if (
                            hasattr(detection_result, "segmentation")
                            and detection_result.segmentation
                            and hasattr(detection_result.segmentation, "mask_array")
                            and detection_result.segmentation.mask_array is not None
                        ):
                            try:
                                cropped_object_for_description = (
                                    self.util_manager.crop_pil_image_from_mask(
                                        frame_pil_image,
                                        detection_result.segmentation.mask_array,
                                        detection_result.box.xyxy
                                        if detection_result.box
                                        else None,
                                    )
                                )
                                if cropped_object_for_description:
                                    description = self.object_describer.describe_object(
                                        cropped_object_for_description, detection_result
                                    )
                                    detection_result.description = description
                            except Exception as e:
                                self.logger.error(
                                    f"Error describing object for frame {frame_data_obj.frame_number}, detection {detection_idx}: {e}"
                                )
                                continue

            # Aggregate frame data objects into video content instance for this batch
            for fd_obj_to_add in current_batch_frame_data_objects:
                video_content_instance.frames[fd_obj_to_add.frame_number] = (
                    fd_obj_to_add
                )

        # Update processing stats
        processing_stats["frames_processed"] = len(video_content_instance.frames)
        video_content_instance.processing_stats = processing_stats

        output_filename = f"{Path(video_path).stem}_annotations.json"
        output_path = Path(output_dir) / output_filename
        data_io.save_annotations(video_content_instance, str(output_path))

        return str(output_path)
