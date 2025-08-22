"""Video I/O utilities for processing video files"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

from ..data.models.video import FrameData, VideoContent


class VideoLoader:
    """Utility class for loading video files and extracting frames"""
    
    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video loader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = str(video_path)
        self.cap = None
        self.video_info = None
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def open(self):
        """Open video file"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            self._extract_video_info()
    
    def close(self):
        """Close video file"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _extract_video_info(self):
        """Extract video metadata"""
        if self.cap is None:
            raise RuntimeError("Video not opened")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.video_info = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        }
    
    def get_video_info(self) -> dict:
        """Get video metadata"""
        if self.video_info is None:
            self._extract_video_info()
        return self.video_info.copy()
    
    def read_frame(self, frame_number: Optional[int] = None) -> Optional[FrameData]:
        """
        Read a specific frame or next frame.
        
        Args:
            frame_number: Specific frame number to read, or None for next frame
            
        Returns:
            FrameData object or None if no more frames
        """
        if self.cap is None:
            raise RuntimeError("Video not opened")
        
        if frame_number is not None:
            # Seek to specific frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Get current frame number
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # Calculate timestamp
        fps = self.video_info["fps"]
        timestamp = current_frame / fps if fps > 0 else 0.0
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return FrameData(
            frame_number=current_frame,
            timestamp=timestamp,
            rgb_image=rgb_frame
        )
    
    def frame_generator(
        self, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Generator[FrameData, None, None]:
        """
        Generate frames from video.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of video)
            step: Frame step (1 for every frame, 2 for every other, etc.)
            
        Yields:
            FrameData objects
        """
        if self.cap is None:
            raise RuntimeError("Video not opened")
        
        # Seek to start frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        while True:
            # Check end frame
            if end_frame is not None and frame_count >= end_frame:
                break
            
            # Read frame
            frame_data = self.read_frame()
            if frame_data is None:
                break
            
            yield frame_data
            frame_count += step
            
            # Skip frames if step > 1
            for _ in range(step - 1):
                if not self.cap.grab():
                    break
                frame_count += 1
    
    def load_video_content(self) -> VideoContent:
        """
        Load complete video content structure.
        
        Returns:
            VideoContent object with all frames
        """
        video_info = self.get_video_info()
        
        video_content = VideoContent(
            video_path=self.video_path,
            total_frames=video_info["total_frames"],
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"]
        )
        
        # Load all frames
        for frame_data in self.frame_generator():
            video_content.frames[frame_data.frame_number] = frame_data
        
        return video_content


class VideoSaver:
    """Utility class for saving video files and frames"""
    
    def __init__(
        self, 
        output_path: Union[str, Path],
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = "mp4v"
    ):
        """
        Initialize video saver.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second for output video
            frame_size: Frame size as (width, height), None for auto-detection
            codec: Video codec (e.g., "mp4v", "XVID")
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
        self.frame_count = 0
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def _initialize_writer(self, frame_size: Tuple[int, int]):
        """Initialize video writer with frame size"""
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                frame_size
            )
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to video.
        
        Args:
            frame: Frame as numpy array (RGB format)
        """
        if self.writer is None:
            height, width = frame.shape[:2]
            frame_size = (width, height)
            if self.frame_size is not None:
                frame_size = self.frame_size
            self._initialize_writer(frame_size)
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr_frame)
        self.frame_count += 1
    
    def write_frames(self, frames: list[np.ndarray]):
        """
        Write multiple frames to video.
        
        Args:
            frames: List of frames as numpy arrays
        """
        for frame in frames:
            self.write_frame(frame)
    
    def close(self):
        """Close video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def save_frame_as_image(
        self, 
        frame: np.ndarray, 
        output_path: Union[str, Path],
        quality: int = 95
    ):
        """
        Save a single frame as image file.
        
        Args:
            frame: Frame as numpy array (RGB format)
            output_path: Path for output image
            quality: JPEG quality (1-100)
        """
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Determine file format from extension
        output_path = str(output_path)
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif output_path.lower().endswith('.png'):
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - min(quality // 10, 9)]
        else:
            params = []
        
        cv2.imwrite(output_path, bgr_frame, params)


def extract_frames_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_format: str = "jpg",
    quality: int = 95,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1
) -> list[str]:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for output frames
        frame_format: Image format ("jpg", "png", etc.)
        quality: Image quality (1-100)
        start_frame: Starting frame number
        end_frame: Ending frame number (None for end of video)
        step: Frame step
        
    Returns:
        List of saved frame file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    with VideoLoader(video_path) as loader:
        for frame_data in loader.frame_generator(start_frame, end_frame, step):
            # Generate output filename
            frame_filename = f"frame_{frame_data.frame_number:06d}.{frame_format}"
            frame_path = output_dir / frame_filename
            
            # Save frame
            VideoSaver.save_frame_as_image(
                frame_data.rgb_image,
                frame_path,
                quality
            )
            
            saved_paths.append(str(frame_path))
    
    return saved_paths


def create_video_from_frames(
    frame_paths: list[Union[str, Path]],
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
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not load first frame: {frame_paths[0]}")
    
    height, width = first_frame.shape[:2]
    frame_size = (width, height)
    
    with VideoSaver(output_path, fps, frame_size, codec) as saver:
        for frame_path in frame_paths:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saver.write_frame(rgb_frame)
    
    return str(output_path)