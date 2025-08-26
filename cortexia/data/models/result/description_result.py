"""Description result schema for object description features."""

from typing import List, Optional, Dict, Any

from .base_result import BaseResult
from ..registry import schema_registry

@schema_registry.register("result.description")
class DescriptionResult(BaseResult):
    """Result schema for object description operations."""
    
    def __init__(self, descriptions: List[str], object_ids: Optional[List[str]] = None, 
                 model_name: Optional[str] = None, processing_time_ms: Optional[float] = None):
        super().__init__(descriptions=descriptions, object_ids=object_ids, model_name=model_name, 
                        processing_time_ms=processing_time_ms)
    
    def _get_repr_fields(self) -> str:
        """Show key fields for repr."""
        fields = []
        if self.descriptions:
            desc_count = len(self.descriptions)
            desc_preview = self.descriptions[0][:30] + "..." if self.descriptions[0] else "empty"
            fields.append(f"descriptions={desc_count} ('{desc_preview}')")
        if self.model_name:
            fields.append(f"model={self.model_name}")
        return ", ".join(fields)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DescriptionResult":
        """Reconstruct from dictionary."""
        deserialized_data = cls._deserialize_special_types(data)
        return cls(**deserialized_data)