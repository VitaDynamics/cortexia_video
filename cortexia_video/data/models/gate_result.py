"""Gate result schema for gate decision outputs."""

from typing import Any, Dict, Optional

from .base_result import BaseResult


class GateResult(BaseResult):
    """Result schema for gate decision operations."""
    
    passes: bool  # Whether the frame passes the gate
    gate_name: str  # Name of the gate that made the decision
    score: Optional[float] = None  # Confidence or numeric score
    threshold: Optional[float] = None  # Threshold used for decision
    metadata: Optional[Dict[str, Any]] = None  # Additional gate-specific data
    processing_time_ms: Optional[float] = None
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        fields.append(f"gate={self.gate_name}")
        fields.append(f"passes={self.passes}")
        if self.score is not None:
            fields.append(f"score={self.score:.3f}")
        if self.threshold is not None:
            fields.append(f"threshold={self.threshold:.3f}")
        return ", ".join(fields)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GateResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)