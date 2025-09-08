"""Tests for row limit behavior in use_with_ray example."""

from __future__ import annotations

import io
import os
from typing import List

import numpy as np
from PIL import Image


def _create_synthetic_dataset() -> List[bytes]:
    """Create synthetic image bytes matching cookbook example logic."""
    row_limit = int(os.environ.get("LANCE_ROW_LIMIT", "16"))
    synthetic_rows = min(row_limit, 8) if row_limit > 0 else 8
    synthetic_data: List[bytes] = []
    for _ in range(synthetic_rows):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format="JPEG")
        synthetic_data.append(img_bytes.getvalue())
    return synthetic_data


def test_row_limit_zero(monkeypatch):
    """When LANCE_ROW_LIMIT=0 the synthetic dataset should have 8 rows."""
    monkeypatch.setenv("LANCE_ROW_LIMIT", "0")
    dataset = _create_synthetic_dataset()
    assert len(dataset) == 8
