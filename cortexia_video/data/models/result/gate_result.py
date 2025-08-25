"""Gate result schema for gate outputs (metrics/features)."""

from typing import Any, Dict, Optional

import numpy as np

from .base_result import BaseResult


class GateResult(BaseResult):
    """
    Result schema for gate computations.

    Gates are calculators: they emit per-frame measurements or features and do
    not decide pass/fail. Any thresholding/comparison policy is handled by
    downstream components (e.g., buffers/policies).
    """
    gate_name: str  # Name of the gate/calculator
    score: Optional[float] = None  # Scalar metric when applicable
    vector: Optional[np.ndarray] = None  # Feature vector (e.g., CLIP embedding)
    threshold: Optional[float] = None  # Optional: policy threshold recorded downstream
    metadata: Optional[Dict[str, Any]] = None  # Additional gate-specific data
    processing_time_ms: Optional[float] = None

    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        fields.append(f"gate={self.gate_name}")
        if self.score is not None:
            fields.append(f"score={self.score:.3f}")
        if self.vector is not None and hasattr(self.vector, "shape"):
            try:
                fields.append(f"vector.shape={tuple(self.vector.shape)}")
            except Exception:
                pass
        if self.threshold is not None:
            fields.append(f"threshold={self.threshold:.3f}")
        return ", ".join(fields)

    @classmethod
    def from_dict(cls, data: dict) -> "GateResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)
