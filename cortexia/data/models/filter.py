"""Filter model primitives (specifications) used by selection policies.

This module centralizes the Specification pattern used by filtering logic,
decoupling it from execution/orchestration layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


class FilterSpec(ABC):
    """Composable predicate over a candidate and current window state."""

    @abstractmethod
    def is_satisfied(self, candidate: T, window: Sequence[T]) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def __and__(self, other: "FilterSpec") -> "FilterSpec":
        return AndSpec(self, other)

    def __or__(self, other: "FilterSpec") -> "FilterSpec":
        return OrSpec(self, other)

    def __invert__(self) -> "FilterSpec":
        return NotSpec(self)


@dataclass(frozen=True)
class AndSpec(FilterSpec):
    left: FilterSpec
    right: FilterSpec

    def is_satisfied(self, candidate: T, window: Sequence[T]) -> bool:
        return self.left.is_satisfied(candidate, window) and self.right.is_satisfied(candidate, window)


@dataclass(frozen=True)
class OrSpec(FilterSpec):
    left: FilterSpec
    right: FilterSpec

    def is_satisfied(self, candidate: T, window: Sequence[T]) -> bool:
        return self.left.is_satisfied(candidate, window) or self.right.is_satisfied(candidate, window)


@dataclass(frozen=True)
class NotSpec(FilterSpec):
    inner: FilterSpec

    def is_satisfied(self, candidate: T, window: Sequence[T]) -> bool:
        return not self.inner.is_satisfied(candidate, window)


__all__ = [
    "FilterSpec",
    "AndSpec",
    "OrSpec",
    "NotSpec",
]

