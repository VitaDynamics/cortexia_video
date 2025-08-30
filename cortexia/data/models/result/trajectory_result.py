"""Trajectory analysis data models"""

from typing import Any, Dict, List, Optional
from .base_result import BaseResult 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..video import TrajectoryPoint


class TrajectoryResult(BaseResult):
    """
    Result schema for trajectory state analysis.
    
    Contains trajectory analysis results including state classification.
    """
    
    def __init__(
        self,
        trajectory_points: List['TrajectoryPoint'],
        current_index: int,
        state_distribution: Dict[str, int],
        total_points: int,
        processing_time_ms: float,
        model_name: str = "trajectory_analyzer"
    ):
        """
        Initialize trajectory state result.
        
        Args:
            trajectory_points: List of trajectory points
            current_index: Current trajectory index
            state_distribution: Distribution of trajectory states
            total_points: Total number of trajectory points
            processing_time_ms: Processing time in milliseconds
            model_name: Name of the model used for analysis
        """
        self.trajectory_points = trajectory_points
        self.current_index = current_index
        self.state_distribution = state_distribution
        self.total_points = total_points
        self.processing_time_ms = processing_time_ms
        self.model_name = model_name
    
    def get_current_state(self) -> Optional[str]:
        """Get the state of the current trajectory point."""
        if 0 <= self.current_index < len(self.trajectory_points):
            return self.trajectory_points[self.current_index].state
        return None
    
    def get_state_percentage(self, state: str) -> float:
        """Get the percentage of points in a specific state."""
        if self.total_points == 0:
            return 0.0
        state_count = self.state_distribution.get(state, 0)
        return (state_count / self.total_points) * 100.0 
    