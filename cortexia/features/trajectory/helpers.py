import math 
from ...data.models.video import TrajectoryPoint
from typing import List, Dict, Any

# Default constants
STATE_THRESHOLD_DEG = 15
VELOCITY_THRESHOLD = 0.1
FUTURE_POINTS = 3
CURVATURE_WINDOW = 3

def classify_state(points: List[TrajectoryPoint], future_points: int = FUTURE_POINTS, **kwargs) -> Dict[str, Any]:
    """
    Classify trajectory states based on orientation differences.
    
    Args:
        points: List of trajectory points to classify
        future_points: Number of future points to use for orientation comparison
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing states and state distribution
    """
    states = []
    state_distribution = {"forward": 0, "backward": 0, "stop": 0, "forward_left": 0, "forward_right": 0, "backward_left": 0, "backward_right": 0}
    
    for i, point in enumerate(points):
        if i >= len(points) - future_points:
            point.state = "stop"
            states.append("stop")
            state_distribution["stop"] += 1
            continue
        
        current_yaw = point.yaw if point.yaw is not None else 0.0
        future_point = points[i + future_points]
        future_yaw = future_point.yaw if future_point.yaw is not None else 0.0
        
        yaw_diff = future_yaw - current_yaw
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        
        yaw_diff_deg = math.degrees(yaw_diff)
        point.future_yaw_diff = yaw_diff_deg
        
        dx = points[i+1].x - point.x if i < len(points) - 1 else 0
        dy = points[i+1].y - point.y if i < len(points) - 1 else 0
        velocity = math.sqrt(dx**2 + dy**2)
        
        if velocity < VELOCITY_THRESHOLD:
            point.state = "stop"
            states.append("stop")
            state_distribution["stop"] += 1
        else:
            movement_angle = math.atan2(dy, dx)
            current_yaw_for_calc = current_yaw if current_yaw is not None else 0.0
            orientation_diff = abs(movement_angle - current_yaw_for_calc)
            if orientation_diff > math.pi:
                orientation_diff = 2 * math.pi - orientation_diff
            
            is_forward = orientation_diff < math.pi / 2
            
            if abs(yaw_diff_deg) <= STATE_THRESHOLD_DEG:
                state = "forward" if is_forward else "backward"
            elif yaw_diff_deg > STATE_THRESHOLD_DEG:
                state = "forward_left" if is_forward else "backward_left"
            else:
                state = "forward_right" if is_forward else "backward_right"
            
            point.state = state
            states.append(state)
            state_distribution[state] += 1
    
    return {"states": states, "state_distribution": state_distribution}
