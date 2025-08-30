"""Trajectory analysis feature implementation"""

import time
from typing import List, Dict, Any, Optional

from ..base import BaseFeature
from ...api.exceptions import ProcessingError, ModelLoadError
from ...data.models.result.trajectory_result import TrajectoryResult
from ...data.models.video import VideoFramePacket
from .helpers import classify_state
from ..registry import feature_registry


@feature_registry.register("trajectory")
class TrajectoryFeature(BaseFeature):
    """Trajectory state classification feature"""
    
    # Class attributes for BaseFeature
    output_schema = TrajectoryResult
    required_inputs = []
    required_fields = ['trajectory', 'current_traj_index']  # Note: typo in field name
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def _initialize(self):
        """Initialize trajectory analyzer"""
        self.initialized = True
    
    def _release(self) -> None:
        """Release analyzer resources"""
        pass
    
    @property
    def name(self) -> str:
        return "trajectory"
    
    @property
    def description(self) -> str:
        return "Trajectory state classification and analysis"
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> TrajectoryResult:
        """
        Process a single frame using trajectory analysis.
        
        Args:
            frame: VideoFramePacket containing frame data with trajectory field
            **inputs: Additional inputs (not used by trajectory analysis)
            
        Returns:
            TrajectoryResult containing analysis results
        """
        if not self.is_ready():
            self._initialize()
        
        start_time = time.time()
        
        try:
            # Validate frame has required trajectory data
            if not hasattr(frame, 'trajectory') or not frame.trajectory:
                raise ProcessingError("Frame missing required trajectory data")
            
            if not hasattr(frame, 'current_traj_index') or frame.current_traj_index < 0:
                raise ProcessingError("Frame missing valid current_traj_index")
            
            # Get trajectory points and current index
            trajectory_points = frame.trajectory
            current_index = frame.current_traj_index
            
            # Run trajectory analysis
            result = classify_state(trajectory_points)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Return result
            return TrajectoryResult(
                trajectory_points=trajectory_points,
                current_index=current_index,
                state_distribution=result['state_distribution'],
                total_points=len(trajectory_points),
                processing_time_ms=processing_time_ms,
                model_name="trajectory_analyzer"
            )
            
        except Exception as e:
            raise ProcessingError(f"Error in trajectory analysis processing: {e}")
    
    def process_batch(self, frames: List[VideoFramePacket], **batch_inputs) -> List[TrajectoryResult]:
        """
        Process multiple frames using trajectory analysis.
        
        Args:
            frames: List of VideoFramePacket objects
            **batch_inputs: Additional batch inputs (not used by trajectory analysis)
            
        Returns:
            List of TrajectoryResult objects
        """
        if not self.is_ready():
            self._initialize()
        
        results = []
        
        try:
            # Process each frame individually since trajectory data is per-frame
            for frame in frames:
                result = self.process_frame(frame, **batch_inputs)
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error in batch trajectory analysis processing: {e}")