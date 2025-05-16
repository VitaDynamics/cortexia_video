import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from cortexia_video.utils import crop_pil_image_from_mask
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
        self.feature_extractor = None
        # Initialize utility manager
        self.object_describer = None  # Object describer for batch processing
        self.initialized = False

    def _load_components(self, processing_mode: str) -> None:
        """
        Dynamically load components based on the processing mode.

        Args:
            processing_mode (str): Space-separated string of processing modes (e.g., "list detect segment describe")
        """
        processing_modes = processing_mode.lower().replace(" ", "").split("|")
        self.logger.info(f"Loading components for processing modes: {processing_modes}")

        # Load only the components needed for the processing modes
        try:
            if "list" in processing_modes:
                self.logger.info("Loading object lister component")
                self.lister = self.config_manager.get_object_lister()
                if self.lister is None:
                    self.logger.warning("Failed to initialize object lister component")

            if "detect" in processing_modes:
                self.logger.info("Loading object detector component")
                self.detector = ObjectDetector(self.config_manager)
                if self.detector is None:
                    self.logger.warning(
                        "Failed to initialize object detector component"
                    )

            if "segment" in processing_modes:
                self.logger.info("Loading object segmenter component")
                self.segmenter = ObjectSegmenter(self.config_manager)
                if self.segmenter is None:
                    self.logger.warning(
                        "Failed to initialize object segmenter component"
                    )

            if "describe" in processing_modes:
                self.logger.info("Loading object describer component")
                self.object_describer = ObjectDescriber(
                    self.config_manager
                )  # For batch processing
                if self.object_describer is None:
                    self.logger.warning(
                        "Failed to initialize object describer component"
                    )

            if (
                "extract_scene" in processing_modes
                or "extract_object" in processing_modes
            ):
                self.logger.info("Loading feature extractor component")
                self.feature_extractor = self.config_manager.get_feature_extractor()
                if self.feature_extractor is None:
                    self.logger.warning(
                        "Failed to initialize feature extractor component"
                    )

            # Mark initialization as complete
            self.initialized = True
            self.logger.info("All components loaded successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}", exc_info=True)
            self.initialized = False

    def load_components(self, processing_mode: str) -> None:
        self._load_components(processing_mode)

    def process_video(self, video_path: str, processing_mode: str) -> str:
        """
        Process a video using the specified processing modes.

        Args:
            video_path (str): Path to the video file
            processing_mode (str): Space-separated string of processing modes
                                 (e.g., "list detect segment describe")

        Returns:
            str: Path to the saved annotation file
        """
        from pathlib import Path

        from tqdm import tqdm

        # Show overall process steps
        steps = ["initialize", "process_frames", "verify_results", "save_results"]
        with tqdm(total=len(steps), desc="Video processing pipeline") as progress:
            # Initialize video and get video content object
            progress.set_description("Initializing video")
            video_content_instance = self._initialize_video_processing(video_path)
            progress.update(1)

            # Process frames in batches
            progress.set_description("Processing video frames")
            video_content_instance = self._process_frames_in_batches(
                video_path, video_content_instance, processing_mode
            )
            progress.update(1)

            # Verify results
            progress.set_description("Verifying processing results")
            verification_stats = self.verify_processing_results(
                video_content_instance, processing_mode
            )
            # Add verification stats to processing stats
            video_content_instance.processing_stats.update(
                {"verification": verification_stats}
            )
            progress.update(1)

            # Save results
            progress.set_description("Saving results")
            output_path = self._save_results(video_path, video_content_instance)
            progress.update(1)

            progress.set_description(f"Completed processing: {Path(video_path).name}")

        return output_path

    def _initialize_video_processing(self, video_path: str) -> VideoContent:
        """
        Initialize video processing by setting up video reader and creating video content object.

        Args:
            video_path (str): Path to the video file

        Returns:
            VideoContent: Initialized video content object
        """
        # TODO: Ready to merge with the other init method with only one call for Video Reader.
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

        return video_content_instance

    def _process_frames_in_batches(
        self, video_path: str, video_content: VideoContent, processing_mode: str
    ) -> VideoContent:
        # NOTE: we only batch process each video file once.
        """
        Process frames in batches according to the specified processing mode.

        Args:
            video_path (str): Path to the video file
            video_content (VideoContent): Video content object to populate
            processing_mode (str): Space-separated string of processing modes

        Returns:
            VideoContent: Updated video content with processed frames
        """
        frame_interval = self.config_manager.get_param("processing.frame_interval", 1)
        processing_stats = {}

        # Import tqdm for progress tracking
        from tqdm import tqdm

        # Get total number of frames to process (accounting for frame_interval)
        total_frames = video_content.total_frames
        estimated_batches = (total_frames // frame_interval) // self.batch_size + 1

        # Create a master progress bar for all batches
        with tqdm(
            total=estimated_batches,
            desc=f"Processing video ({video_content.width}x{video_content.height})",
        ) as master_pbar:
            # Iterate over batches from load_video_frames
            for batch_data in data_io.load_video_frames(
                video_path, frame_interval, self.batch_size
            ):
                # Process this batch
                current_batch_frame_data_objects = self._process_batch(
                    batch_data, video_path, processing_mode
                )

                # Add processed frames to video content
                for fd_obj_to_add in current_batch_frame_data_objects:
                    video_content.frames[fd_obj_to_add.frame_number] = fd_obj_to_add

                # Update master progress bar
                master_pbar.update(1)
                # Update description with current progress
                frames_processed = len(video_content.frames)
                master_pbar.set_postfix(
                    {"frames": frames_processed, "mode": processing_mode}
                )

        # Update processing stats
        processing_stats["frames_processed"] = len(video_content.frames)
        video_content.processing_stats = processing_stats

        return video_content

    def _save_results(self, video_path: str, video_content: VideoContent) -> str:
        """
        Save processing results to disk.

        Args:
            video_path (str): Path to the video file
            video_content (VideoContent): Video content with processing results

        Returns:
            str: Path to the saved annotation file
        """
        output_dir = self.config_manager.get_param(
            "processing.output_directory", "output"
        )
        output_filename = f"{Path(video_path).stem}_annotations.json"
        output_path = Path(output_dir) / output_filename
        data_io.save_annotations(video_content, output_path=str(output_path))

        return str(output_path)

    def _process_batch(
        self, batch_data, video_path: str, processing_mode: str
    ) -> List[FrameData]:
        """
        Process a single batch of frames.

        Args:
            batch_data: Batch data from data_io.load_video_frames
            video_path (str): Path to the video file
            processing_mode (str): Space-separated string of processing modes

        Returns:
            List[FrameData]: List of processed frame data objects
        """
        # Verify components are initialized for requested processing modes
        processing_modes = processing_mode.lower().split()

        # Batch mode: unpacking (loaded_batch_meta, loaded_batch_np, _)
        loaded_batch_meta, loaded_batch_np, _ = batch_data
        batch_frames_meta = loaded_batch_meta
        batch_frames_np = loaded_batch_np

        batch_size = len(batch_frames_np)
        self.logger.info(
            f"Processing batch of {batch_size} frames with modes: {processing_modes}"
        )

        # Initialize batch containers
        current_batch_frame_data_objects = []
        current_batch_pil_images = []

        # Import tqdm once
        from tqdm import tqdm

        # Count total operations to perform based on processing_mode
        total_operations = 1  # Start with 1 for frame preparation
        if "extract_scene" in processing_mode and self.feature_extractor:
            total_operations += 1
        if "list" in processing_mode and self.lister is not None:
            total_operations += 1
        if "detect" in processing_mode and self.detector is not None:
            total_operations += 1
        if "segment" in processing_mode and self.segmenter is not None:
            total_operations += 1
        if "extract_object" in processing_mode and self.feature_extractor:
            total_operations += 1
        if "describe" in processing_mode and self.object_describer is not None:
            total_operations += 1
        viz_enabled = self.config_manager.get_param("visualization.enabled", False)
        if viz_enabled:
            total_operations += 1

        # Create one progress bar for the whole batch processing
        with tqdm(
            total=total_operations, desc=f"Processing batch of {batch_size} frames"
        ) as pbar:
            # Process each frame in the batch
            pbar.set_description("Preparing frames")
            for i in range(len(batch_frames_np)):
                meta_item = batch_frames_meta[i]
                rgb_array = batch_frames_np[i]

                # Create FrameData object
                frame_data_obj = FrameData(
                    frame_number=meta_item["frame_number"],
                    timestamp=meta_item["timestamp"],
                    rgb_image=rgb_array,  # Store RGB numpy array
                )

                # Convert to PIL for processing
                # TODO: update the batch process to use only np array instead of pil image as input.
                pil_image = Image.fromarray(rgb_array)

                # Add to batch lists
                current_batch_frame_data_objects.append(frame_data_obj)
                current_batch_pil_images.append(pil_image)

            pbar.update(1)  # Update progress after frame preparation

            # Extract scene-level features if enabled
            if "extract_scene" in processing_mode and self.feature_extractor:
                pbar.set_description("Extracting scene features")
                self._extract_scene_features(
                    current_batch_frame_data_objects,
                    batch_pil_images=current_batch_pil_images,
                )
                pbar.update(1)

            # Process object listing in batches if enabled
            if "list" in processing_mode and self.lister is not None:
                pbar.set_description("Processing object listing")
                self._process_object_listing(
                    current_batch_frame_data_objects, current_batch_pil_images
                )
                pbar.update(1)

            # Process object detection in batches if enabled
            if "detect" in processing_mode and self.detector is not None:
                pbar.set_description("Processing object detection")
                self._process_object_detection(
                    current_batch_frame_data_objects, current_batch_pil_images
                )
                pbar.update(1)

            # Process object segmentation in batches if enabled
            if "segment" in processing_mode and self.segmenter is not None:
                pbar.set_description("Processing object segmentation")
                self._process_object_segmentation(
                    video_path,
                    current_batch_frame_data_objects,
                    current_batch_pil_images,
                )
                pbar.update(1)

            # Extract object features if enabled
            if "extract_object" in processing_mode and self.feature_extractor:
                pbar.set_description("Extracting object features")
                self._extract_object_features(
                    current_batch_frame_data_objects, current_batch_pil_images
                )
                pbar.update(1)

            # Process object descriptions if enabled
            if "describe" in processing_mode and self.object_describer is not None:
                pbar.set_description("Processing object descriptions")
                self._process_object_descriptions(
                    current_batch_frame_data_objects, current_batch_pil_images
                )
                pbar.update(1)

            # Handle visualization if enabled
            if viz_enabled:
                pbar.set_description("Handling visualization")
                self._handle_visualization(
                    video_path,
                    current_batch_frame_data_objects,
                    current_batch_pil_images,
                )
                pbar.update(1)

        return current_batch_frame_data_objects

    def _extract_scene_features(
        self,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Extract scene-level features for a batch of frames.

        Args:
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        if not self.feature_extractor:
            self.logger.warning(
                "Feature extractor not initialized for scene feature extraction"
            )
            return

        if batch_pil_images:
            batch_scene_features_tensor = self.feature_extractor.extract_image_features(
                batch_pil_images
            )
            for i, frame_data_obj in enumerate(batch_frame_data_objects):
                if i < batch_scene_features_tensor.shape[0]:
                    frame_data_obj.scene_clip_features = (
                        batch_scene_features_tensor[i].cpu().numpy()
                    )
        else:
            self.logger.info(
                "No PIL images in current batch for scene feature extraction."
            )

    def _process_object_listing(
        self,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Process object listing for a batch of frames.

        Args:
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        try:
            # Call the batched method
            batched_lister_outputs = self.lister.list_objects_in_image_batched(
                batch_pil_images
            )

            # Map results back to individual frame data objects
            for i, frame_data_obj in enumerate(batch_frame_data_objects):
                object_names_for_frame = batched_lister_outputs[i]
                frame_data_obj.lister_results = object_names_for_frame

                # Generate DINO prompt
                dino_prompt = generate_formatted_string_dino(object_names_for_frame)
                frame_data_obj.dino_prompt = dino_prompt

        except Exception as e:
            self.logger.error(
                f"Error in batch object listing: {e}",
                exc_info=True,
            )
            # Set error results for all frames in this batch
            for frame_data_obj in batch_frame_data_objects:
                frame_data_obj.lister_results = ["error_processing_frame"]

    def _process_object_detection(
        self,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Process object detection for a batch of frames.

        Args:
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        try:
            # Prepare batch inputs for detector
            # TODO: remove the split the dino prompt. Because in detect objects we recover it again.
            batch_dino_prompts = [
                fd.dino_prompt.split(".")
                if hasattr(fd, "dino_prompt") and fd.dino_prompt
                else []
                for fd in batch_frame_data_objects
            ]

            # Call the batched detection method
            batched_detector_outputs = self.detector.detect_objects(
                batch_pil_images,
                batch_dino_prompts,
            )

            # Map results back to individual frame data objects
            for i, frame_data_obj in enumerate(batch_frame_data_objects):
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
            for frame_data_obj in batch_frame_data_objects:
                frame_data_obj.detections = []

    def _process_object_segmentation(
        self,
        video_path: str,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Process object segmentation for detected objects in a batch of frames.

        Args:
            video_path: Path to the video file
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        # Check if segmenter is initialized
        if not self.segmenter:
            self.logger.warning("Object segmenter not initialized for segmentation")
            return

        # First collect all frames, detections, and bounding boxes for batch processing
        batch_frames_for_segmentation = []
        batch_boxes_for_segmentation = []
        detection_mapping = []  # To map results back to frame_data_obj

        for i, fd in enumerate(batch_frame_data_objects):
            if hasattr(fd, "detections") and fd.detections:
                current_frame_pil = batch_pil_images[i]
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
                    detection_mapping.append({"frame_idx": i, "detection": det})

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
                for frame_idx, frame_masks in enumerate(batch_masks_results):
                    for frame_mask in frame_masks:
                        if mask_idx < len(detection_mapping):
                            mapping = detection_mapping[mask_idx]
                            detection = mapping["detection"]

                            # Calculate area of mask (number of non-zero pixels)
                            area = np.count_nonzero(frame_masks)

                            # Create SegmentationResult
                            segment = SegmentationResult(
                                mask=frame_mask,
                                score=detection.score,
                                label=detection.label,
                                area=area,
                                bbox=detection.box,
                                detection_id=detection.id,  # Link segment to detection by ID
                            )

                            # Add to the current frame's FrameData object
                            frame_data = batch_frame_data_objects[mapping["frame_idx"]]
                            if (
                                not hasattr(frame_data, "segments")
                                or frame_data.segments is None
                            ):
                                frame_data.segments = []
                            frame_data.segments.append(segment)

                            # Explicitly add the mask to the detection object
                            # This ensures the mask is available for later processing steps
                            detection.mask = (
                                frame_mask.copy()
                                if frame_mask is not None
                                else None
                            )

                            # Log successful mask assignment
                            self.logger.debug(
                                f"Added mask to detection {detection.id} with label {detection.label}"
                            )

                            # Optionally generate visualization with contours if enabled
                            contour_viz_enabled = self.config_manager.get_param(
                                "visualization.contour_enabled", False
                            )
                            if contour_viz_enabled:
                                # Contour visualization code would go here
                                pass

                        mask_idx += 1

            except Exception as e:
                self.logger.error(
                    f"Error in batch segmentation: {e}",
                    exc_info=True,
                )
                # Set empty segments for all frames in this batch
                for frame_data_obj in batch_frame_data_objects:
                    if (
                        not hasattr(frame_data_obj, "segments")
                        or frame_data_obj.segments is None
                    ):
                        frame_data_obj.segments = []

    def _extract_object_features(
        self,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Extract features from detected objects in a batch of frames.

        Args:
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        # Check if feature extractor is initialized
        if not self.feature_extractor:
            self.logger.warning(
                "Feature extractor not initialized for object feature extraction"
            )
            return

        for frame_data_obj in batch_frame_data_objects:
            detections = (
                frame_data_obj.detections
                if hasattr(frame_data_obj, "detections")
                and frame_data_obj.detections is not None
                else []
            )
            if not detections:
                continue

            all_cropped_object_pil_images_in_batch = []
            object_references_for_mapping = []  # Stores {'frame_data_idx': int, 'detection_idx': int}

            # Get the frame index in the current batch
            frame_data_idx = batch_frame_data_objects.index(frame_data_obj)
            original_pil_image = batch_pil_images[frame_data_idx]

            for det_idx, detection in enumerate(detections):
                # Check if mask is available (from segmentation)
                if hasattr(detection, "mask") and detection.mask is not None:
                    try:
                        cropped_pil = crop_pil_image_from_mask(
                            original_pil_image,
                            detection.mask,
                            detection.box.xyxy if detection.box else None,
                        )
                        if cropped_pil:
                            all_cropped_object_pil_images_in_batch.append(cropped_pil)
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
                else:
                    self.logger.debug(
                        f"Skipping feature extraction for detection {det_idx} in frame {frame_data_obj.frame_number} - no mask available"
                    )

            # Extract features for all cropped objects in this frame in one batch
            if all_cropped_object_pil_images_in_batch:
                try:
                    feature_dim = 512  # Default value
                    if hasattr(self.feature_extractor, "model"):
                        if hasattr(self.feature_extractor.model, "visual"):
                            feature_dim = self.feature_extractor.model.visual.output_dim

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
                            self.logger.debug(
                                f"Added object features to detection {target_det.id} with label {target_det.label}"
                            )
                except Exception as e:
                    self.logger.error(
                        f"Error extracting batch object features for frame {frame_data_obj.frame_number}: {e}",
                        exc_info=True,
                    )

    def _process_object_descriptions(
        self,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Generate descriptions for detected objects in a batch of frames.

        Args:
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        if not self.object_describer:
            self.logger.warning(
                "Object describer not initialized for description generation"
            )
            return

        for frame_data_obj in batch_frame_data_objects:
            frame_pil_image = None
            try:
                idx_in_batch = batch_frame_data_objects.index(frame_data_obj)
                frame_pil_image = batch_pil_images[idx_in_batch]
            except ValueError:
                self.logger.error(
                    f"Could not find frame_data_obj for description in current_batch_pil_images for frame {frame_data_obj.frame_number}"
                )
                continue

            if not frame_pil_image:
                continue

            # Make sure detections attribute exists and is not None
            if (
                not hasattr(frame_data_obj, "detections")
                or frame_data_obj.detections is None
            ):
                self.logger.debug(
                    f"No detections found for frame {frame_data_obj.frame_number}"
                )
                continue

            for detection_idx, detection_result in enumerate(frame_data_obj.detections):
                # Check if mask is available directly in detection
                if (
                    hasattr(detection_result, "mask")
                    and detection_result.mask is not None
                ):
                    try:
                        # cropped_object_for_description = (
                        #     crop_pil_image_from_mask(
                        #         frame_pil_image,
                        #         detection_result.mask,
                        #         detection_result.box.xyxy
                        #         if detection_result.box
                        #         else None,
                        #     )
                        # )
                        # Pass the cropped image and the segmentation mask
                        description = self.object_describer.describe_object(
                            frame_pil_image,
                            segmentation_mask=detection_result.mask,
                        )
                        # Set the description directly on the detection object
                        detection_result.description = description
                        self.logger.info(
                            f"Added description to object {detection_result.label}: {description[:50]}..."
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error describing object for frame {frame_data_obj.frame_number}, detection {detection_idx}: {e}",
                            exc_info=True,
                        )
                        continue
                else:
                    self.logger.debug(
                        f"Skipping description for detection {detection_idx} in frame {frame_data_obj.frame_number} - no mask available"
                    )

    def _handle_visualization(
        self,
        video_path: str,
        batch_frame_data_objects: List[FrameData],
        batch_pil_images: List[Image.Image],
    ) -> None:
        """
        Create and save visualizations for processed frames.

        Args:
            video_path: Path to the video file
            batch_frame_data_objects: List of FrameData objects
            batch_pil_images: List of PIL images
        """
        output_dir = self.config_manager.get_param(
            "processing.output_directory", "output"
        )

        for i, frame_data_obj in enumerate(iterable=batch_frame_data_objects):
            # visualize the frame after all frame object processing is done. 
            try:
                pil_image = frame_data_obj.rgb_image
                frame_number = frame_data_obj.frame_number

                # Generate annotated frame
                annotated_pil_image = generate_annotated_frame(
                 frame_data=frame_data_obj
                )

                # Save annotated frame
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
                    f"Error creating visualization for frame {frame_data_obj.frame_number}: {e}",
                    exc_info=True,
                )

    def verify_processing_results(
        self, video_content: VideoContent, processing_mode: str
    ) -> Dict[str, Any]:
        """
        Verify that all expected processing results exist in the video content.
        This is useful for debugging and ensuring data integrity.

        Args:
            video_content: The VideoContent object to verify
            processing_mode: The processing mode string used

        Returns:
            Dict[str, Any]: Statistics about the verification
        """
        processing_modes = processing_mode.lower().split()
        stats = {
            "total_frames": len(video_content.frames),
            "total_detections": 0,
            "total_segments": 0,
            "detections_with_masks": 0,
            "detections_with_features": 0,
            "detections_with_descriptions": 0,
            "frames_with_scene_features": 0,
            "issues": [],
        }

        # Check each frame for expected components
        for frame_num, frame_data in video_content.frames.items():
            # Check detections
            if "detect" in processing_modes:
                if (
                    not hasattr(frame_data, "detections")
                    or frame_data.detections is None
                ):
                    stats["issues"].append(f"Frame {frame_num} has no detections")
                else:
                    stats["total_detections"] += len(frame_data.detections)

                    # Check each detection for expected attributes
                    for det_idx, detection in enumerate(frame_data.detections):
                        # Check for mask if segmentation was requested
                        if "segment" in processing_modes:
                            if (
                                hasattr(detection, "mask")
                                and detection.mask is not None
                            ):
                                stats["detections_with_masks"] += 1
                            else:
                                stats["issues"].append(
                                    f"Detection {det_idx} in frame {frame_num} has no mask"
                                )

                        # Check for object features if feature extraction was requested
                        if "extract_object" in processing_modes:
                            if (
                                hasattr(detection, "object_clip_features")
                                and detection.object_clip_features is not None
                            ):
                                stats["detections_with_features"] += 1
                            else:
                                stats["issues"].append(
                                    f"Detection {det_idx} in frame {frame_num} has no object features"
                                )

                        # Check for description if description was requested
                        if "describe" in processing_modes:
                            if (
                                hasattr(detection, "description")
                                and detection.description is not None
                            ):
                                stats["detections_with_descriptions"] += 1
                            else:
                                stats["issues"].append(
                                    f"Detection {det_idx} in frame {frame_num} has no description"
                                )

            # Check segments
            if "segment" in processing_modes:
                if not hasattr(frame_data, "segments") or frame_data.segments is None:
                    stats["issues"].append(
                        f"Frame {frame_num} has no segments despite segmentation mode"
                    )
                else:
                    stats["total_segments"] += len(frame_data.segments)

            # Check scene features
            if "extract_scene" in processing_modes:
                if (
                    hasattr(frame_data, "scene_clip_features")
                    and frame_data.scene_clip_features is not None
                ):
                    stats["frames_with_scene_features"] += 1
                else:
                    stats["issues"].append(f"Frame {frame_num} has no scene features")

        # Calculate percentages for summary
        if stats["total_detections"] > 0:
            if "segment" in processing_modes:
                stats["mask_coverage_percent"] = (
                    stats["detections_with_masks"] / stats["total_detections"]
                ) * 100
            if "extract_object" in processing_modes:
                stats["feature_coverage_percent"] = (
                    stats["detections_with_features"] / stats["total_detections"]
                ) * 100
            if "describe" in processing_modes:
                stats["description_coverage_percent"] = (
                    stats["detections_with_descriptions"] / stats["total_detections"]
                ) * 100

        if stats["total_frames"] > 0 and "extract_scene" in processing_modes:
            stats["scene_feature_coverage_percent"] = (
                stats["frames_with_scene_features"] / stats["total_frames"]
            ) * 100

        # Log summary
        self.logger.info(
            f"Processing verification complete: {len(stats['issues'])} issues found"
        )
        if len(stats["issues"]) > 0:
            self.logger.warning(f"First 5 issues: {stats['issues'][:5]}")

        return stats
