# ClipGate implementation for using CLIP models to analyze frame similarity

import logging
import uuid
from typing import Optional, List

import numpy as np
from PIL import Image

from .base_gate import BaseGate
from ..data.models.video import VideoFramePacket
from ..data.models.result.gate_result import GateResult
from .clip_utils import get_image_embeddings


class ClipGate(BaseGate[GateResult]):
    """
    Gate that uses a CLIP-like model to compute image embeddings.

    This calculator returns the embedding vector per frame; any similarity
    comparison/thresholding is external.

    Note: This implementation depends on an external CLIP package.
    """

    def __init__(
        self,
        session_id: str = "default_session",
        clip_model_name: str = "ViT-B/32",
        clip_pretrained_weights: str = "openai",
        device: Optional[str] = None
    ):
        """
        Initialize the ClipGate

        Args:
            session_id: Identifier for the current session, used in logging
            clip_model_name: CLIP model architecture to use
            clip_pretrained_weights: Pre-trained weights to use
            device: Device to perform inference on. Auto-detects if None.
        """
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id
        self.clip_model_name = clip_model_name
        self.clip_pretrained_weights = clip_pretrained_weights
        self.device = device

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert the frame data from a VideoFramePacket to a PIL Image.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    def _compute_embedding(self, packet: VideoFramePacket) -> np.ndarray:
        """
        Compute the CLIP embedding for a frame using utility function.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            np.ndarray: The embedding vector
        """
        pil_image = self._get_pil_image(packet)
        
        # get_image_embeddings returns a batch of embeddings (even for a single image)
        # so we take the first (and only) one.
        embedding_batch = get_image_embeddings(
            images=pil_image,
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained_weights,
            device=self.device
        )
        return embedding_batch[0]

    def process_frame(self, packet: VideoFramePacket) -> GateResult:
        """
        Compute a CLIP embedding for the frame.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            GateResult: Result containing embedding vector and metadata
        """
        # Compute embedding for the current frame
        embedding = self._compute_embedding(packet)

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "model": self.clip_model_name,
            "weights": self.clip_pretrained_weights,
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "compute_clip_embedding",
            "outcome": "computed",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed for CLIP embedding.",
            extra=log_dict,
        )

        # Return GateResult with embedding vector; policy is external
        return GateResult(
            gate_name="clip_gate",
            vector=embedding,
            metadata={
                "model_name": self.clip_model_name,
                "pretrained": self.clip_pretrained_weights,
                "processing_metadata": log_metadata,
            },
        )