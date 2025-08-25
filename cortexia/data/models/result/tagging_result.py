"""Tagging/listing result schema for object listing features."""

from typing import List, Optional

from .base_result import BaseResult
from ..registry import schema_registry

@schema_registry.register("result.tagging")
class TaggingResult(BaseResult):
    """Result schema for object tagging/listing operations."""
    
    tags: List[str]  # List of detected object tags
    raw_response: Optional[str] = None  # Raw response from the model
    confidence_scores: Optional[List[float]] = None  # Per-tag confidence if available  
    model_name: Optional[str] = None
    processing_time_ms: Optional[float] = None
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        if self.tags:
            tags_preview = ', '.join(self.tags[:3])
            if len(self.tags) > 3:
                tags_preview += f"... (+{len(self.tags)-3} more)"
            fields.append(f"tags=[{tags_preview}]")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TaggingResult":
        """Reconstruct from dictionary.""" 
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)