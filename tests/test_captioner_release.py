"""Tests for captioner resource release and GPU memory cleanup"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cortexia.features.caption.captioner import CaptionFeature
from cortexia.data.models.video import VideoFramePacket


class DummyCaptioner:
    """Simple captioner allocating GPU memory for testing."""

    def __init__(self, config: dict[str, Any]):
        self.buffer = torch.empty((1024, 1024), device=torch.device("cuda"))

    def caption_image(self, image: Any) -> str:  # pragma: no cover - trivial
        return "dummy caption"

    def release(self) -> None:  # pragma: no cover - trivial
        del self.buffer


class TestCaptionerRelease:
    """Regression tests for GPU memory cleanup in CaptionFeature."""

    def test_release_clears_gpu_memory(self, monkeypatch: Any) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monkeypatch.setattr(
            "cortexia.features.caption.captioner.MoonDreamCaptioner",
            DummyCaptioner,
        )

        feature = CaptionFeature()
        frame = VideoFramePacket(
            frame_data=np.zeros((2, 2, 3), dtype=np.uint8),
            frame_number=0,
            timestamp=datetime.timedelta(0),
            source_video_id="vid",
            trajectory=[],
            current_traj_index=0,
        )
        feature.process_frame(frame)

        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        assert allocated_before > 0

        feature.release()

        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()
        assert allocated_after < allocated_before
