from typing import List, Optional, Dict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_listing import ObjectLister
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.object_segmentation import ObjectSegmenter
from cortexia_video.object_description import ObjectDescriber
from cortexia_video import data_io
from cortexia_video.schemes import VideoContent, FrameData, DetectionResult, SegmentationResult, BoundingBox
from cortexia_video.visualization import generate_annotated_frame


class ProcessingManager:
    def __init__(self, config_path: str = "config", config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager if config_manager else ConfigManager(config_path)
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
        
        frame_interval = self.config_manager.get_param('processing.frame_interval', 1)
        output_dir = self.config_manager.get_param('processing.output_dir', 'output')
        
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
            height=height
        )
        
        processing_stats = {}
        
        for frame_number, timestamp, frame_np_array in data_io.load_video_frames(video_path, frame_interval):
            # Convert BGR to RGB
            rgb_array = cv2.cvtColor(frame_np_array, cv2.COLOR_BGR2RGB)
            
            frame_data = FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                rgb_image=rgb_array  # Store RGB numpy array
            )
            
            pil_image = Image.fromarray(rgb_array)
            detections: List[DetectionResult] = []
            segments: List[SegmentationResult] = []
            frame_features = {}
            
            if "list" in processing_mode and self.lister is not None:
                try:
                    object_names = self.lister.list_objects_in_image(pil_image)
                    frame_features["detected_objects"] = object_names
                except Exception as e:
                    self.logger.error(f"Error listing objects in frame {frame_number}: {e}", exc_info=True)
                    frame_features["detected_objects"] = ["error_processing_frame"]
            
            if "detect" in processing_mode and self.detector is not None:
                text_prompt_for_detection = self.config_manager.get_param(
                    'model_settings.detection_prompt', "various objects .")
                try:
                    detected_object_data = self.detector.detect_objects(pil_image, text_prompt_for_detection)
                    for det_obj in detected_object_data:
                        bbox = BoundingBox(
                            x_min=int(det_obj['box'][0]),
                            y_min=int(det_obj['box'][1]),
                            x_max=int(det_obj['box'][2]),
                            y_max=int(det_obj['box'][3])
                        )
                        detection = DetectionResult(
                            score=float(det_obj['score']),
                            label=det_obj['label'],
                            box=bbox
                        )
                        detections.append(detection)
                except Exception as e:
                    self.logger.error(f"Error detecting objects in frame {frame_number}: {e}", exc_info=True)
            
            if "segment" in processing_mode and self.segmenter is not None and detections:
                for detection in detections:
                    try:
                        mask_list = self.segmenter.segment_object(
                            pil_image,
                            [[detection.box.x_min, detection.box.y_min,
                              detection.box.x_max, detection.box.y_max]]
                        )
                        if mask_list and mask_list[0] is not None:
                            # Create segmentation mask
                            mask_np = np.array(mask_list[0], dtype=np.uint8)
                            # Calculate area of mask (number of non-zero pixels)
                            area = np.count_nonzero(mask_np)
                            
                            segment = SegmentationResult(
                                mask=mask_np,
                                score=detection.score,
                                label=detection.label,
                                area=area,
                                bbox=detection.box
                            )
                            segments.append(segment)
                            # Also add the mask to the detection
                            detection.mask = mask_np
                    except Exception as e:
                        self.logger.error(f"Error segmenting object in frame {frame_number}: {e}", exc_info=True)
            
            if "describe" in processing_mode and self.describer is not None and segments:
                for segment in segments:
                    try:
                        # Find the corresponding detection to update its description
                        for detection in detections:
                            if (detection.label == segment.label and
                                detection.box.x_min == segment.bbox.x_min and
                                detection.box.y_min == segment.bbox.y_min):
                                description = self.describer.describe_object(pil_image, segment.mask)
                                detection.description = description
                                break
                    except Exception as e:
                        self.logger.error(f"Error describing object in frame {frame_number}: {e}", exc_info=True)
            
            # Update frame data
            frame_data.detections = detections
            frame_data.segments = segments
            frame_data.features = frame_features
            
            # Handle visualization if enabled
            viz_enabled = self.config_manager.get_param('visualization.enabled', False)
            if viz_enabled:
                try:
                    annotated_pil_image = generate_annotated_frame(pil_image, frame_data)
                    output_img_dir = Path(output_dir) / "annotated_frames" / Path(video_path).stem
                    output_img_dir.mkdir(parents=True, exist_ok=True)
                    img_format = self.config_manager.get_param('visualization.annotated_image_format', 'jpg')
                    output_img_path = output_img_dir / f"frame_{frame_number:05d}.{img_format}"
                    annotated_pil_image.save(output_img_path)
                    self.logger.info(f"Saved annotated frame: {output_img_path}")
                except Exception as e:
                    self.logger.error(f"Error creating visualization for frame {frame_number}: {e}", exc_info=True)
            
            # Store frame by number in dictionary
            video_content_instance.frames[frame_number] = frame_data
        
        # Update processing stats
        processing_stats["frames_processed"] = len(video_content_instance.frames)
        video_content_instance.processing_stats = processing_stats
        
        output_filename = f"{Path(video_path).stem}_annotations.json"
        output_path = Path(output_dir) / output_filename
        data_io.save_annotations(video_content_instance, str(output_path))
        
        return str(output_path)