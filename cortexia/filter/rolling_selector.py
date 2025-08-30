"""RollingSelector: composable, policy-driven sliding-window filtering.

This module provides:
- FilterSpec (Specification pattern) with AND/OR/NOT composition
- ComparatorRegistry for pluggable metric comparisons
- EvictionPolicy (Strategy) with FIFO implementation
- Sliding window iterator utilities
- RollingSelector orchestrator with pre/post admission evaluation

Design goals:
- Gates are calculators; all decision policy lives here
- Support AND/OR logic, pre/post evaluation, capacity limits, step-wise sliding
- Keep implementation minimal and focused
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple, TypeVar
from ..data.models.filter import FilterSpec


T = TypeVar("T")  # Item type (e.g., GateResult or any record)

# -----------------------------
# Eviction policies
# -----------------------------

class EvictionPolicy(ABC):
    @abstractmethod
    def select_indices_to_evict(self, current: Sequence[T], incoming_count: int, capacity: int) -> List[int]:
        """Return indices in current to evict to make room for incoming_count items."""
        pass


class FifoEviction(EvictionPolicy):
    """Evict oldest by index (smallest first)."""

    def select_indices_to_evict(self, current: Sequence[T], incoming_count: int, capacity: int) -> List[int]:
        free = capacity - len(current)
        need = max(0, incoming_count - max(0, free))
        if need <= 0:
            return []
        # Evict earliest indices [0..need-1]
        return list(range(min(need, len(current))))


# -----------------------------
# Sliding window iterator
# -----------------------------

def iter_windows(items: Sequence[T], window_size: int, step: int) -> Iterator[Tuple[int, Sequence[T]]]:
    """Yield (start_index, window_items) over items with given window size and step."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    n = len(items)
    i = 0
    while i < n:
        yield i, items[i : min(n, i + window_size)]
        i += step


# -----------------------------
# Rolling selector orchestrator
# -----------------------------

class AdmissionMode:
    PRE_ADD = "pre" # Just Compare with older existing elements
    POST_ADD = "post"  # Compare with New added elements


class RollingSelector:
    """Policy-driven sliding window selector over a sequence of results.

    - Applies a FilterSpec under current buffer window context (pre/post)
    - Admits up to max_admit_per_step from each candidate window
    - Maintains capacity with an EvictionPolicy (FIFO by default)
    - Advances through input by step until exhausted
    """

    def __init__(
        self,
        capacity: int,
        window_size: int,
        step: int = 1,
        admission_mode: str = AdmissionMode.POST_ADD,
        eviction: Optional[EvictionPolicy] = None,
        max_admit_per_step: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.window_size = window_size
        self.step = step
        self.admission_mode = admission_mode
        self.eviction = eviction or FifoEviction()
        self.max_admit_per_step = max_admit_per_step
        self._buffer: List[T] = []

    @property
    def buffer(self) -> List[T]:
        return list(self._buffer)

    def _admit(self, candidates: Sequence[T], spec: FilterSpec) -> List[T]:
        """Decide which candidates to keep based on spec and admission mode."""
        admitted: List[T] = []
        if self.admission_mode == AdmissionMode.PRE_ADD:
            for c in candidates:
                if spec.is_satisfied(c, self._buffer):
                    admitted.append(c)
                    if self.max_admit_per_step is not None and len(admitted) >= self.max_admit_per_step:
                        break
        elif self.admission_mode == AdmissionMode.POST_ADD:
            # Tentatively add all, then filter and possibly evict newcomers
            tentative = list(self._buffer) + list(candidates)
            keep_mask = [True] * len(tentative)
            for idx, c in enumerate(tentative):
                if not spec.is_satisfied(c, tentative[:idx] + tentative[idx + 1 :]):
                    keep_mask[idx] = False
            # Newcomers occupy the tail; select those that are marked True
            newcomers = [c for k, c in zip(keep_mask[len(self._buffer):], candidates) if k]
            admitted = newcomers
            if self.max_admit_per_step is not None:
                admitted = admitted[: self.max_admit_per_step]
        else:
            raise ValueError(f"Unknown admission_mode: {self.admission_mode}")
        return admitted

    def _evict_for_incoming(self, incoming_count: int) -> None:
        indices = self.eviction.select_indices_to_evict(self._buffer, incoming_count, self.capacity)
        # remove in reverse index order
        for i in sorted(indices, reverse=True):
            if 0 <= i < len(self._buffer):
                del self._buffer[i]

    def run(self, items: Sequence[T], spec: FilterSpec) -> List[T]:
        """Run selection over items; return the final buffer contents.

        The selector traverses `items` in sliding windows and applies the spec
        to admit candidates while enforcing capacity.
        """
        for _, window_items in iter_windows(items, self.window_size, self.step):
            candidates = list(window_items)
            admitted = self._admit(candidates, spec)
            if not admitted:
                continue
            self._evict_for_incoming(len(admitted))
            # Append admitted (FIFO ordering by default)
            self._buffer.extend(admitted)
            # If overflow still occurs (e.g., policy returned too few evictions), trim FIFO
            if len(self._buffer) > self.capacity:
                overflow = len(self._buffer) - self.capacity
                del self._buffer[:overflow]
        return list(self._buffer)
