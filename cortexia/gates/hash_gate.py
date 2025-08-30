import logging
import uuid
from typing import Any

from PIL import Image

from .base_gate import BaseGate
from .image_hash import ImageHasher
from ..data.models.video import VideoFramePacket
from ..data.models.result.gate_result import GateResult
from .registry import gate_registry

@gate_registry.register("hash")
class HashGate(BaseGate[GateResult]):
    output_schema = GateResult
    DEFAULT_HASH_TYPE = "phash"
    DEFAULT_HASH_SIZE = 8
    DEFAULT_HIGHFREQ_FACTOR = 4

    def __init__(
        self,
        hash_type: str = DEFAULT_HASH_TYPE,
        hash_size: int = DEFAULT_HASH_SIZE,
        highfreq_factor: int = DEFAULT_HIGHFREQ_FACTOR,
        session_id: str = "default_session",
    ):
        """Initialize HashGate as a pure hash calculator."""
        self.logger = logging.getLogger(f"cortexia.{self.__class__.__name__}")
        self.session_id = session_id
        self.hash_type = hash_type
        # Create image hasher
        self.hasher = ImageHasher(hash_size=hash_size, highfreq_factor=highfreq_factor)

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert frame data to PIL Image for hashing.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    # TODO: Accelerate this with a calcualted hash function
    def _calculate_hash(self, packet: VideoFramePacket):
        """
        Calculates the perceptual hash for a frame.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            imagehash.ImageHash: The perceptual hash object.
        """
        # Use compute_hash_from_array directly with frame_data
        return self.hasher.compute_hash_from_array(packet.frame_data, self.hash_type)

    def process_frame(self, packet: VideoFramePacket) -> GateResult:
        """
        Process a single video frame packet to determine if it's a duplicate based on perceptual hashing.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            GateResult: Result containing computed hash (as string) and metadata
        """
        frame_hash = self._calculate_hash(packet)
        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "frame_hash_calculated": str(frame_hash),
            "hash_type": self.hash_type,
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "compute_hash",
            "outcome": "computed",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed for hash.", extra=log_dict
        )

        # Return GateResult with calculated hash; policy is external
        return GateResult(
            gate_name="hash_gate",
            score=None,
            metadata={
                "hash": str(frame_hash),
                "hash_type": self.hash_type,
                "hash_config": {
                    "hash_size": self.hasher.hash_size,
                    "highfreq_factor": getattr(self.hasher, 'highfreq_factor', None),
                },
                "processing_metadata": log_metadata,
            },
        )
