"""Caption result schema for image captioning features."""

from typing import Optional, Dict, Any

from .base_result import BaseResult

class CaptionResult(BaseResult):
    """Result schema for image captioning operations."""
    
    def __init__(self, caption: str, confidence: Optional[float] = None, model_name: Optional[str] = None, 
                 caption_length: Optional[str] = None, processing_time_ms: Optional[float] = None):
        super().__init__(caption=caption, confidence=confidence, model_name=model_name, 
                        caption_length=caption_length, processing_time_ms=processing_time_ms)
    
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
    def from_dict(cls, data: Dict[str, Any]) -> "CaptionResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)