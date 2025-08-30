import logging
import uuid
from typing import Optional

import cv2
import numpy as np

from .base_gate import BaseGate
from ..data.models.video import VideoFramePacket
from ..data.models.result.gate_result import GateResult
from .registry import gate_registry

@gate_registry.register("blur")
class BlurGate(BaseGate[GateResult]):
    output_schema = GateResult
    
    def __init__(self, session_id: str = "default_session"):
        """Initialize the BlurGate as a pure metric calculator."""
        self.logger = logging.getLogger(f"cortexia.{self.__class__.__name__}")
        self.session_id = session_id

    def _calculate_blur_score(self, frame_data: np.ndarray) -> float:
        """
        Calculates the blur score (variance of Laplacian) for an image.
        check: https://medium.com/@sagardhungel/laplacian-and-its-use-in-blur-detection-fbac689f0f88


        Args:
            frame_data (np.ndarray): The input frame data as a NumPy array.

        Returns:
            float: The variance of the Laplacian.
        """
        # Check if image is empty (has 0 width or height)
        if frame_data.size == 0 or 0 in frame_data.shape[:2]:
            return 0.0

        # Convert to grayscale if needed
        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            gray_image = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = frame_data

        # Compute the Laplacian of the grayscale image
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        # Calculate the variance of the Laplacian
        return laplacian.var()

    def process_frame(self, packet: VideoFramePacket) -> GateResult:
        """
        Process a single video frame packet to determine if it contains blur.

        Args:
            packet (VideoFramePacket): The frame packet containing the frame data and metadata

        Returns:
            GateResult: Result containing blur score and metadata
        """
        blur_score = self._calculate_blur_score(packet.frame_data)
        # Return GateResult with measured score only; policy is external
        return GateResult(
            gate_name="blur_gate",
            score=float(blur_score),
            metadata={
                "metric": "variance_of_laplacian",
                "frame_number": packet.frame_number,
                "timestamp": packet.timestamp.total_seconds(),
                "source_video_id": packet.source_video_id,
            },
        )
    
