"""Caption result schema for image captioning features."""

from typing import Optional

from .base_result import BaseResult


class CaptionResult(BaseResult):
    """Result schema for image captioning operations."""
    
    caption: str
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    caption_length: Optional[str] = None  # "short", "long", etc.
    processing_time_ms: Optional[float] = None
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        if self.caption:
            # Show first 50 chars of caption
            caption_preview = self.caption[:50] + "..." if len(self.caption) > 50 else self.caption
            fields.append(f'caption="{caption_preview}"')
        if self.confidence is not None:
            fields.append(f"confidence={self.confidence:.3f}")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CaptionResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)