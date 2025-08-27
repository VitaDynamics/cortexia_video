"""Tagging/listing result schema for object listing features."""

from typing import List, Optional, Dict, Any

from .base_result import BaseResult

class TaggingResult(BaseResult):
    """Result schema for object tagging/listing operations."""
    
    def __init__(self, tags: List[str], raw_response: Optional[str] = None, confidence_scores: Optional[List[float]] = None, 
                 model_name: Optional[str] = None, processing_time_ms: Optional[float] = None):
        super().__init__(tags=tags, raw_response=raw_response, confidence_scores=confidence_scores, 
                        model_name=model_name, processing_time_ms=processing_time_ms)
        
        # Explicitly set attributes for type checking
        self.tags = tags
        self.raw_response = raw_response
        self.confidence_scores = confidence_scores
        self.model_name = model_name
        self.processing_time_ms = processing_time_ms
    
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
    def from_dict(cls, data: Dict[str, Any]) -> "TaggingResult":
        """Reconstruct from dictionary.""" 
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)