"""Enhanced SDK API methods for video processing"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from .cortexia import Cortexia
from ..api.exceptions import CortexiaError, ProcessingError
from ..data.io.video import VideoLoader, VideoSaver, extract_frames_from_video
from ..data.models.video import FrameData, VideoContent


class EnhancedCortexia(Cortexia):
    """
    Enhanced Cortexia SDK with video processing capabilities.
    
    Extends the base Cortexia class with video processing methods
    and advanced API functionality.
    """
    
    def process_video(
        self,
        video_path: Union[str, Path],
        features: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        frame_range: Optional[tuple[int, int]] = None,
        frame_step: int = 1,
        save_intermediate: bool = False,
        **kwargs
    ) -> VideoContent:
        """
        Process video with specified features.
        
        Args:
            video_path: Path to video file
            features: List of feature names to apply
            output_dir: Directory for output files
            frame_range: Tuple of (start_frame, end_frame) or None for full video
            frame_step: Process every Nth frame
            save_intermediate: Save intermediate results
            **kwargs: Additional processing options
            
        Returns:
            VideoContent with processing results
            
        Raises:
            CortexiaError: If processing fails
        """
        try:
            # Initialize output directory
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load video
            with VideoLoader(video_path) as loader:
                video_content = loader.load_video_content()
                
                # Set up frame range
                start_frame, end_frame = frame_range if frame_range else (0, None)
                
                # Process frames
                processed_frames = []
                for frame_data in loader.frame_generator(start_frame, end_frame, frame_step):
                    # Apply features
                    for feature_name in features:
                        feature = self.get_feature(feature_name)
                        frame_data = feature.process_frame(frame_data)
                    
                    processed_frames.append(frame_data)
                    
                    # Save intermediate results if requested
                    if save_intermediate and output_dir:
                        self._save_intermediate_frame(frame_data, output_dir, feature_name)
                
                # Update video content with processed frames
                video_content.frames = {f.frame_number: f for f in processed_frames}
                
                # Save final results
                if output_dir:
                    self._save_video_content(video_content, output_dir)
                
                return video_content
                
        except Exception as e:
            raise CortexiaError(f"Video processing failed: {e}")
    
    def process_video_streaming(
        self,
        video_path: Union[str, Path],
        features: List[str],
        frame_callback=None,
        frame_range: Optional[tuple[int, int]] = None,
        frame_step: int = 1,
        **kwargs
    ) -> VideoContent:
        """
        Process video with streaming frame processing.
        
        Args:
            video_path: Path to video file
            features: List of feature names to apply
            frame_callback: Callback function for each processed frame
            frame_range: Tuple of (start_frame, end_frame) or None for full video
            frame_step: Process every Nth frame
            **kwargs: Additional processing options
            
        Returns:
            VideoContent with processing results
            
        Raises:
            CortexiaError: If processing fails
        """
        try:
            # Load video
            with VideoLoader(video_path) as loader:
                video_content = VideoContent(
                    video_path=str(video_path),
                    **loader.get_video_info()
                )
                
                # Set up frame range
                start_frame, end_frame = frame_range if frame_range else (0, None)
                
                # Process frames with streaming
                for frame_data in loader.frame_generator(start_frame, end_frame, frame_step):
                    # Apply features
                    for feature_name in features:
                        feature = self.get_feature(feature_name)
                        frame_data = feature.process_frame(frame_data)
                    
                    # Add to video content
                    video_content.frames[frame_data.frame_number] = frame_data
                    
                    # Call callback if provided
                    if frame_callback:
                        frame_callback(frame_data)
                
                return video_content
                
        except Exception as e:
            raise CortexiaError(f"Video streaming processing failed: {e}")
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        frame_range: Optional[tuple[int, int]] = None,
        frame_step: int = 1,
        frame_format: str = "jpg",
        quality: int = 95
    ) -> List[str]:
        """
        Extract frames from video and save as images.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for output frames
            frame_range: Tuple of (start_frame, end_frame) or None for full video
            frame_step: Extract every Nth frame
            frame_format: Image format ("jpg", "png", etc.)
            quality: Image quality (1-100)
            
        Returns:
            List of saved frame file paths
            
        Raises:
            CortexiaError: If frame extraction fails
        """
        try:
            start_frame, end_frame = frame_range if frame_range else (0, None)
            
            frame_paths = extract_frames_from_video(
                video_path=video_path,
                output_dir=output_dir,
                frame_format=frame_format,
                quality=quality,
                start_frame=start_frame,
                end_frame=end_frame,
                step=frame_step
            )
            
            return frame_paths
            
        except Exception as e:
            raise CortexiaError(f"Frame extraction failed: {e}")
    
    def process_frames(
        self,
        frame_paths: List[Union[str, Path]],
        features: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> List[FrameData]:
        """
        Process list of frame images with specified features.
        
        Args:
            frame_paths: List of frame image paths
            features: List of feature names to apply
            output_dir: Directory for output files
            **kwargs: Additional processing options
            
        Returns:
            List of processed frame data
            
        Raises:
            CortexiaError: If processing fails
        """
        try:
            processed_frames = []
            
            for i, frame_path in enumerate(frame_paths):
                # Load frame
                frame_data = self._load_frame_from_path(frame_path, frame_number=i)
                
                # Apply features
                for feature_name in features:
                    feature = self.get_feature(feature_name)
                    frame_data = feature.process_frame(frame_data)
                
                processed_frames.append(frame_data)
            
            # Save results if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self._save_processed_frames(processed_frames, output_dir)
            
            return processed_frames
            
        except Exception as e:
            raise CortexiaError(f"Frame processing failed: {e}")
    
    def create_video_from_frames(
        self,
        frame_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        fps: float = 30.0,
        codec: str = "mp4v"
    ) -> str:
        """
        Create video from list of frame images.
        
        Args:
            frame_paths: List of frame image paths
            output_path: Path for output video
            fps: Frames per second for output video
            codec: Video codec
            
        Returns:
            Path to created video file
            
        Raises:
            CortexiaError: If video creation fails
        """
        try:
            from ..data.io.video import create_video_from_frames
            
            video_path = create_video_from_frames(
                frame_paths=frame_paths,
                output_path=output_path,
                fps=fps,
                codec=codec
            )
            
            return video_path
            
        except Exception as e:
            raise CortexiaError(f"Video creation failed: {e}")
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature information
            
        Raises:
            FeatureNotFoundError: If feature not found
        """
        return self.registry.get_feature_info(feature_name)
    
    def get_all_features_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered features.
        
        Returns:
            Dictionary mapping feature names to their information
        """
        features_info = {}
        
        for feature_name in self.registry.list_features():
            features_info[feature_name] = self.registry.get_feature_info(feature_name)
        
        return features_info
    
    def _load_frame_from_path(self, frame_path: Union[str, Path], frame_number: int) -> FrameData:
        """
        Load frame from image path.
        
        Args:
            frame_path: Path to frame image
            frame_number: Frame number for the frame
            
        Returns:
            FrameData object
        """
        import cv2
        from PIL import Image
        
        # Load image
        image = Image.open(frame_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        rgb_image = np.array(image)
        
        return FrameData(
            frame_number=frame_number,
            timestamp=0.0,  # No timestamp for individual frames
            rgb_image=rgb_image,
            rgb_path=str(frame_path)
        )
    
    def _save_intermediate_frame(self, frame_data: FrameData, output_dir: Path, feature_name: str):
        """
        Save intermediate processing results.
        
        Args:
            frame_data: Processed frame data
            output_dir: Output directory
            feature_name: Name of the feature being processed
        """
        # Create feature-specific subdirectory
        feature_dir = output_dir / feature_name
        feature_dir.mkdir(exist_ok=True)
        
        # Save frame image
        frame_filename = f"frame_{frame_data.frame_number:06d}.jpg"
        frame_path = feature_dir / frame_filename
        
        saver = VideoSaver(str(frame_path))
        saver.save_frame_as_image(
            frame_data.rgb_image,
            str(frame_path),
            quality=95
        )
        
        # Save frame data as JSON
        data_filename = f"frame_{frame_data.frame_number:06d}.json"
        data_path = feature_dir / data_filename
        
        import json
        with open(data_path, 'w') as f:
            json.dump(frame_data.dict(), f, indent=2)
    
    def _save_video_content(self, video_content: VideoContent, output_dir: Path):
        """
        Save video content results.
        
        Args:
            video_content: Processed video content
            output_dir: Output directory
        """
        # Save video content as JSON
        content_path = output_dir / "video_content.json"
        
        import json
        with open(content_path, 'w') as f:
            json.dump(video_content.to_dict(), f, indent=2)
        
        # Save summary statistics
        self._save_processing_stats(video_content, output_dir)
    
    def _save_processed_frames(self, frames: List[FrameData], output_dir: Path):
        """
        Save processed frames.
        
        Args:
            frames: List of processed frames
            output_dir: Output directory
        """
        for i, frame_data in enumerate(frames):
            # Save frame image
            frame_filename = f"processed_frame_{i:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            VideoSaver.save_frame_as_image(
                frame_data.rgb_image,
                str(frame_path),
                quality=95
            )
    
    def _save_processing_stats(self, video_content: VideoContent, output_dir: Path):
        """
        Save processing statistics.
        
        Args:
            video_content: Processed video content
            output_dir: Output directory
        """
        stats_path = output_dir / "processing_stats.json"
        
        # Calculate statistics
        total_frames = len(video_content.frames)
        total_detections = sum(len(f.detections) for f in video_content.frames.values())
        total_segments = sum(len(f.segments) for f in video_content.frames.values())
        
        stats = {
            "total_frames_processed": total_frames,
            "total_detections": total_detections,
            "total_segments": total_segments,
            "average_detections_per_frame": total_detections / total_frames if total_frames > 0 else 0,
            "average_segments_per_frame": total_segments / total_frames if total_frames > 0 else 0,
            "video_info": {
                "fps": video_content.fps,
                "width": video_content.width,
                "height": video_content.height,
                "total_frames": video_content.total_frames
            }
        }
        
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)