from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional, Union, Callable
import heapq
import time
from ..models.video import VideoFramePacket


# TODO: this should merged to @video schemas

@dataclass
class TaggedFrame:
    """Frame with gate-generated tag/score"""

    packet: VideoFramePacket
    tag: Any  # Gate output (score, classification, etc.)
    timestamp: float
    priority: float = 0.0  # Computed priority for eviction


class EvictionStrategy(ABC):
    """Abstract base for eviction policies"""

    @abstractmethod
    def select_frames_to_evict(
        self, current_frames: List[TaggedFrame], new_frame: TaggedFrame, max_size: int
    ) -> List[int]:
        """Return indices of frames to evict when adding new_frame"""
        pass


class FIFOEvictionStrategy(EvictionStrategy):
    """Traditional FIFO eviction"""

    def select_frames_to_evict(self, current_frames, new_frame, max_size):
        if len(current_frames) >= max_size:
            return [0]  # Evict oldest
        return []


class PriorityEvictionStrategy(EvictionStrategy):
    """Keep highest priority frames"""

    def __init__(self, score_func: Callable[[Any], float]):
        self.score_func = score_func

    def select_frames_to_evict(self, current_frames, new_frame, max_size):
        if len(current_frames) < max_size:
            return []

        # Calculate priorities
        all_frames = current_frames + [new_frame]
        for frame in all_frames:
            frame.priority = self.score_func(frame.tag)

        # Sort by priority (descending) and timestamp (ascending for tie-breaking)
        sorted_frames = sorted(all_frames, key=lambda f: (-f.priority, f.timestamp))

        # Keep top max_size frames
        frames_to_keep = sorted_frames[:max_size]
        keep_indices = {id(f.packet) for f in frames_to_keep if f != new_frame}

        # Return indices of frames to evict
        return [
            i
            for i, frame in enumerate(current_frames)
            if id(frame.packet) not in keep_indices
        ]


class SmartBuffer:
    """Enhanced buffer with gate integration and configurable eviction"""

    def __init__(
        self,
        max_size: int,
        gate: Optional["BaseGate"] = None,
        eviction_strategy: Optional[EvictionStrategy] = None,
        enabled: bool = True,
        session_id: str = "default_session",
    ):
        self.max_size = max_size
        self.gate = gate
        self.eviction_strategy = eviction_strategy or FIFOEvictionStrategy()
        self.enabled = enabled
        self.session_id = session_id
        self.frames: List[TaggedFrame] = []
        self.logger = logging.getLogger(f"Cortexia.{self.__class__.__name__}")

    def add(self, packet: VideoFramePacket, **kwargs) -> bool:
        """Add frame with gate tagging and smart eviction"""
        if not self.enabled:
            return False

        # Tag frame with gate if available
        tag = self.gate(packet) if self.gate else True

        new_frame = TaggedFrame(packet=packet, tag=tag, timestamp=time.time())

        # Determine what to evict
        indices_to_evict = self.eviction_strategy.select_frames_to_evict(
            self.frames, new_frame, self.max_size
        )

        # Remove evicted frames (in reverse order to preserve indices)
        for idx in sorted(indices_to_evict, reverse=True):
            evicted = self.frames.pop(idx)
            self.logger.debug(f"Evicted frame with tag: {evicted.tag}")

        # Add new frame
        self.frames.append(new_frame)

        return True

    def get_window(self) -> List[VideoFramePacket]:
        """Get current frames (maintaining API compatibility)"""
        return [f.packet for f in self.frames]

    def get_tagged_window(self) -> List[TaggedFrame]:
        """Get current tagged frames"""
        return self.frames.copy()
