import logging
import uuid

import imagehash
from PIL import Image

from .base_gate import BaseGate
from ..data.models.video import VideoFramePacket
from ..data.models.result.gate_result import GateResult


class GridGate(BaseGate[GateResult]):
    DEFAULT_GRID_ROWS = 4
    DEFAULT_GRID_COLS = 4
    DEFAULT_CELL_HASH_SIZE = 8

    def __init__(
        self,
        grid_rows: int = DEFAULT_GRID_ROWS,
        grid_cols: int = DEFAULT_GRID_COLS,
        cell_hash_size: int = DEFAULT_CELL_HASH_SIZE,
        session_id: str = "default_session",
    ):
        """Initialize GridGate as a grid signature calculator."""
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_hash_size = cell_hash_size
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert the frame data to a PIL Image.

        Args:
            packet (VideoFramePacket): The packet containing the frame data

        Returns:
            Image.Image: The PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    def _extract_grid_signature(self, packet: VideoFramePacket) -> str:
        """
        Extracts a grid-based signature from a frame.

        Args:
            packet (VideoFramePacket): The packet containing the frame to analyze

        Returns:
            str: The concatenated string of cell hashes.
        """
        pil_image = self._get_pil_image(packet)
        img_width, img_height = pil_image.size

        if self.grid_cols <= 0 or self.grid_rows <= 0:
            return str(imagehash.average_hash(pil_image, hash_size=self.cell_hash_size))

        base_cell_width = img_width // self.grid_cols
        base_cell_height = img_height // self.grid_rows

        cell_hashes = []
        placeholder_hex_len = (self.cell_hash_size * self.cell_hash_size) // 4
        placeholder_hash = "0" * placeholder_hex_len

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                left = c * base_cell_width
                upper = r * base_cell_height
                right = min((c + 1) * base_cell_width, img_width)
                lower = min((r + 1) * base_cell_height, img_height)

                if right > left and lower > upper:
                    cell_image = pil_image.crop((left, upper, right, lower))
                    try:
                        cell_hash_obj = imagehash.average_hash(
                            cell_image, hash_size=self.cell_hash_size
                        )
                        cell_hashes.append(str(cell_hash_obj))
                    except Exception:
                        cell_hashes.append(placeholder_hash)
                else:
                    cell_hashes.append(placeholder_hash)

        return "".join(cell_hashes)

    def process_frame(self, packet: VideoFramePacket) -> GateResult:
        """
        Process a single video frame packet using grid-based signatures to determine if
        it is similar to previously seen frames.

        Args:
            packet (VideoFramePacket): The packet containing the frame to analyze

        Returns:
            GateResult: Result containing gate decision and grid signature information
        """
        # Extract grid signature for the current frame
        signature = self._extract_grid_signature(packet)

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "signature_preview": signature[:20] + "..." if len(signature) > 20 else signature,
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "compute_grid_signature",
            "outcome": "computed",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed with grid signature.",
            extra=log_dict,
        )

        # Return GateResult with signature only; policy is external
        return GateResult(
            gate_name="grid_gate",
            metadata={
                "signature": signature,
                "signature_length": len(signature),
                "grid_config": {
                    "rows": self.grid_rows,
                    "cols": self.grid_cols,
                    "cell_hash_size": self.cell_hash_size,
                },
                "processing_metadata": log_metadata,
            },
        )

    def clear_signatures(self) -> None:
        """No-op retained for API compatibility; no internal state kept anymore."""
        return None
