"""Video data models"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import datetime

import numpy as np
from pydantic import BaseModel, Field

from .result.detection_result import DetectionResult
from .result.segmentation_result import SegmentationResult


@dataclass
class VideoFramePacket:
    """
    Standardized data packet for a single video frame and its metadata.
    This structure will be used for passing frame data between samplers,
    gates, buffers, and other processing modules.
    
    Extended to support unified data flow between features and gates with
    optional fields for computed embeddings, annotations, gate results,
    and derived metadata.
    """

    frame_data: np.ndarray  # The raw frame image data as a NumPy array (from decord)
    frame_number: (
        int  # Sequential frame number within the source video (0-indexed or 1-indexed)
    )
    timestamp: datetime.timedelta  # Timestamp of the frame relative to the video start
    source_video_id: str  # A unique identifier for the source video file or stream

    # Core metadata (backwards compatible)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Extended fields for unified processing
    embeddings: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)  # Named embeddings (e.g., "clip", "dino")
    annotations: Optional['AnnotationResults'] = None  # Results from feature processing
    gate_results: Optional['GateResults'] = None  # Results from gate processing  
    derived_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # Computed features/stats

    def __post_init__(self):
        if not isinstance(self.frame_data, np.ndarray):
            raise TypeError(
                f"frame_data must be a NumPy array, got {type(self.frame_data)}"
            )
        if not (
            self.frame_data.ndim == 3 and self.frame_data.shape[2] in [1, 3, 4]
        ) and not (self.frame_data.ndim == 2):  # (H, W, C) or (H, W)
            raise ValueError(
                "frame_data must be a 2D (grayscale) or 3D (e.g., RGB, RGBA) NumPy array. "
                f"Received shape: {self.frame_data.shape}"
            )
        if not isinstance(self.frame_number, int) or self.frame_number < 0:
            raise ValueError(
                f"frame_number must be a non-negative integer, got {self.frame_number}"
            )
        if not isinstance(self.timestamp, datetime.timedelta):
            raise TypeError(
                f"timestamp must be a datetime.timedelta, got {type(self.timestamp)}"
            )
        if (
            not isinstance(self.source_video_id, str)
            or not self.source_video_id.strip()
        ):
            raise ValueError(
                f"source_video_id must be a non-empty string, got '{self.source_video_id}'"
            )
        if not isinstance(self.additional_metadata, dict):
            raise TypeError(
                f"additional_metadata must be a dict, got {type(self.additional_metadata)}"
            )
        
        # Validate new extended fields
        if self.embeddings is None:
            self.embeddings = {}
        if not isinstance(self.embeddings, dict):
            raise TypeError(
                f"embeddings must be a dict, got {type(self.embeddings)}"
            )
        
        # Validate embedding values are numpy arrays
        for key, embedding in self.embeddings.items():
            if not isinstance(embedding, np.ndarray):
                raise TypeError(
                    f"embeddings['{key}'] must be a NumPy array, got {type(embedding)}"
                )
        
        if self.derived_metadata is None:
            self.derived_metadata = {}
        if not isinstance(self.derived_metadata, dict):
            raise TypeError(
                f"derived_metadata must be a dict, got {type(self.derived_metadata)}"
            )

    def __eq__(self, other):
        """
        Compare two VideoFramePacket objects for equality.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if objects are equal, False otherwise
        """
        if not isinstance(other, VideoFramePacket):
            return False

        # Compare all fields except frame_data and embeddings which contain numpy arrays
        fields_equal = (
            self.frame_number == other.frame_number
            and self.timestamp == other.timestamp
            and self.source_video_id == other.source_video_id
            and self.additional_metadata == other.additional_metadata
            and self.annotations == other.annotations
            and self.gate_results == other.gate_results
            and self.derived_metadata == other.derived_metadata
        )

        # Compare the numpy arrays
        # TODO: I thknk for emebedding, we need to more light way for it.
        arrays_equal = np.array_equal(self.frame_data, other.frame_data)
        
        # Compare embeddings dictionaries
        embeddings_equal = True
        
        if self.embeddings:
            # Only compare when embeddings are exsiting.
            if len(self.embeddings) != len(other.embeddings):
                embeddings_equal = False
            else:
                for key in self.embeddings:
                    if key not in other.embeddings:
                        embeddings_equal = False
                        break
                    if not np.array_equal(self.embeddings[key], other.embeddings[key]):
                        embeddings_equal = False
                        break

        return fields_equal and arrays_equal and embeddings_equal
    
    # Convenience methods for unified data flow
    
    def add_annotation_result(self, result_type: str, result_data: Any) -> None:
        """
        Add annotation result to the frame.
        
        Args:
            result_type: Type of annotation result (e.g., 'caption', 'detections')
            result_data: The annotation data to add
        """
        if self.annotations is None:
            self.annotations = AnnotationResults()
        
        if result_type == 'caption':
            self.annotations.caption = result_data
        elif result_type == 'detections':
            if isinstance(result_data, list):
                self.annotations.detections.extend(result_data)
            else:
                self.annotations.detections.append(result_data)
        elif result_type == 'segments':
            if isinstance(result_data, list):
                self.annotations.segments.extend(result_data)
            else:
                self.annotations.segments.append(result_data)
        elif result_type == 'depth_map':
            self.annotations.depth_map = result_data
        elif result_type == 'depth_statistics':
            self.annotations.depth_statistics = result_data
        elif result_type == 'scene_clip_features':
            self.annotations.scene_clip_features = result_data
        elif result_type == 'lister_results':
            self.annotations.lister_results = result_data
        elif result_type == 'dino_prompt':
            self.annotations.dino_prompt = result_data
        else:
            # Store in general features dict
            self.annotations.features[result_type] = result_data
    
    def add_gate_result(self, gate_name: str, passes: bool, score: Optional[float] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add gate result to the frame.
        
        Args:
            gate_name: Name of the gate
            passes: Whether the frame passed the gate
            score: Optional confidence score
            metadata: Optional gate-specific metadata
        """
        if self.gate_results is None:
            self.gate_results = GateResults()
        
        self.gate_results.add_gate_result(gate_name, passes, score, metadata)
    
    def add_embedding(self, embedding_name: str, embedding: np.ndarray) -> None:
        """
        Add named embedding to the frame.
        
        Args:
            embedding_name: Name of the embedding (e.g., 'clip', 'dino')
            embedding: The embedding vector as numpy array
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"Embedding must be a NumPy array, got {type(embedding)}")
        
        if self.embeddings is None:
            self.embeddings = {}
        
        self.embeddings[embedding_name] = embedding
    
    def get_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        """
        Get named embedding from the frame.
        
        Args:
            embedding_name: Name of the embedding
            
        Returns:
            The embedding array, or None if not found
        """
        if self.embeddings is None:
            return None
        return self.embeddings.get(embedding_name)
    
    def has_annotations(self) -> bool:
        """Check if frame has any annotation results."""
        return self.annotations is not None
    
    def has_gate_results(self) -> bool:
        """Check if frame has any gate results."""
        return self.gate_results is not None
    
    def passes_all_gates(self) -> bool:
        """Check if frame passes all gate criteria."""
        if self.gate_results is None:
            return True  # No gates means all pass
        return self.gate_results.passes
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert frame packet to dictionary for serialization.
        
        Returns:
            Dictionary representation of the frame packet
        """
        result = {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.total_seconds(),
            'source_video_id': self.source_video_id,
            'additional_metadata': self.additional_metadata.copy(),
            'derived_metadata': self.derived_metadata.copy() if self.derived_metadata else {},
            'frame_shape': list(self.frame_data.shape),
            'frame_dtype': str(self.frame_data.dtype),
        }
        
        if self.embeddings:
            result['embeddings'] = {name: emb.shape for name, emb in self.embeddings.items()}
        
        if self.annotations:
            result['has_annotations'] = True
            result['annotations_summary'] = {
                'detections_count': len(self.annotations.detections),
                'segments_count': len(self.annotations.segments),
                'has_caption': self.annotations.caption is not None,
                'has_depth_map': self.annotations.depth_map is not None,
            }
        
        if self.gate_results:
            result['gate_results'] = {
                'passes': self.gate_results.passes,
                'gate_count': len(self.gate_results.gate_decisions),
                'gate_decisions': self.gate_results.gate_decisions.copy(),
                'gate_scores': self.gate_results.gate_scores.copy(),
            }
        
        return result


class AnnotationResults(BaseModel):
    """Container for all annotation results from features"""
    
    detections: list[DetectionResult] = Field(default_factory=list)
    segments: list[SegmentationResult] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    
    # Specific feature results
    lister_results: Optional[list[str]] = None
    dino_prompt: Optional[str] = None
    scene_clip_features: Optional[np.ndarray] = None
    caption: Optional[str] = None
    depth_map: Optional[np.ndarray] = None
    depth_statistics: Optional[Dict[str, float]] = None
    
    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Override dict method to handle numpy arrays"""
        data = super().dict(**kwargs)
        # Skip large data in serialization
        data["scene_clip_features"] = None
        data["depth_map"] = None
        return data


class GateResults(BaseModel):
    """Container for gate decision results and metadata"""
    
    passes: bool = True  # Whether frame passes all gates
    gate_decisions: Dict[str, bool] = Field(default_factory=dict)  # Individual gate results
    gate_scores: Dict[str, float] = Field(default_factory=dict)    # Gate confidence scores
    gate_metadata: Dict[str, Any] = Field(default_factory=dict)    # Gate-specific data
    
    def add_gate_result(self, gate_name: str, passes: bool, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add result from a specific gate"""
        self.gate_decisions[gate_name] = passes
        if score is not None:
            self.gate_scores[gate_name] = score
        if metadata:
            self.gate_metadata[gate_name] = metadata
        # Update overall pass/fail
        self.passes = self.passes and passes