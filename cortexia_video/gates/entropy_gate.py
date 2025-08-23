import logging
import uuid
from typing import Optional

import numpy as np

from .base_gate import BaseGate
from ..data.models.video import VideoFramePacket
from ..data.models.gate_result import GateResult

# TODO: we need to refactor logging schemas. 

class EntropyGate(BaseGate[GateResult]):
    DEFAULT_THRESHOLD = 4.0  # Class attribute for default entropy threshold

    def __init__(
        self, threshold: Optional[float] = None, session_id: str = "default_session"
    ):
        """
        Initializes the EntropyGate component.

        Args:
            threshold (Optional[float]): The entropy threshold.
                           Images with entropy below this are considered low-information.
                           If None, uses DEFAULT_THRESHOLD.
            session_id (str): Identifier for the current session, used in logging.
        """
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id

    def _calculate_entropy(self, frame_data: np.ndarray) -> float:
        """
        Calculates the Shannon entropy of an image.

        Args:
            frame_data (np.ndarray): The input frame data as a NumPy array.

        Returns:
            float: The calculated entropy value.
        """
        # Convert to grayscale if needed
        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            # Convert RGB to grayscale using standard RGB to grayscale conversion
            gray_image = np.dot(frame_data[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = frame_data

        if gray_image.size == 0:  # Check for empty image
            return 0.0

        # Calculate histogram
        hist = np.histogram(gray_image.ravel(), bins=256, range=(0, 255))[0]

        # Normalize histogram to get probability distribution
        probabilities = hist / (hist.sum() + np.finfo(float).eps)

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

        return entropy

    def process_frame(self, packet: VideoFramePacket) -> GateResult:
        """
        Process a single video frame packet to determine if it contains sufficient information
        based on its entropy.

        Args:
            packet (VideoFramePacket): The frame packet containing the frame data and metadata

        Returns:
            GateResult: Result containing gate decision and entropy information
        """
        entropy_score = self._calculate_entropy(packet.frame_data)
        is_low_entropy = entropy_score < self.threshold
        passes = not is_low_entropy  # Frame passes if high entropy

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "entropy_score_calculated": float(entropy_score),
            "threshold_used": self.threshold,
            "decision_is_low_entropy": bool(is_low_entropy),
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "process_frame_entropy_check",
            "outcome": "low_entropy" if is_low_entropy else "high_entropy",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed for entropy.", extra=log_dict
        )

        # Return GateResult with detailed information
        return GateResult(
            passes=passes,
            gate_name="entropy_gate",
            score=float(entropy_score),
            threshold=self.threshold,
            metadata={
                "is_low_entropy": is_low_entropy,
                "entropy_score": float(entropy_score),
                "processing_metadata": log_metadata
            }
        )
    
    def legacy_process_frame(self, packet: VideoFramePacket) -> bool:
        """
        Legacy method that returns boolean for backwards compatibility.
        
        Args:
            packet: The VideoFramePacket containing the frame data and metadata
            
        Returns:
            bool: True if frame passes (high entropy), False if filtered (low entropy)
        """
        result = self.process_frame(packet)
        return result.passes
