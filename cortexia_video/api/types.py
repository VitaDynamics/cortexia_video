"""Common types for Cortexia Video SDK"""

# These will be imported from the data.models module
# For now, we'll create placeholders and move the actual types later
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.models.video import FrameData, VideoContent
    from ..data.models.detection_result import DetectionResult
    from ..data.models.segmentation_result import SegmentationResult