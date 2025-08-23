"""Universal lightweight registry with decorator support.

Provides a generic `Registry` that can be used across features, gates,
and inner model selections (e.g., captioners, listers). Designed to be
simple, explicit, and avoid hidden magic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, MutableMapping, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A small, explicit registry for classes or callables.

    - Case-insensitive keys
    - Supports `@registry.register(name, aliases=[...])` decorator
    - Provides dict-like `get`, `keys`, `items`, `__contains__`
    - Optional `create(name, *args, **kwargs)` helper for instantiation
    """

    def __init__(self, kind: str = "registry"):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str, obj: Any, *, aliases: Optional[list[str]] = None) -> None:
        key = name.lower()
        self._items[key] = obj
        if aliases:
            for alias in aliases:
                self._items[alias.lower()] = obj

    def decorator(self, name: Optional[str] = None, *, aliases: Optional[list[str]] = None) -> Callable[[T], T]:
        """Return a class decorator that registers the class into this registry."""

        def _decorator(cls: T) -> T:
            reg_name = name or getattr(cls, "__name__", None)
            if not reg_name:
                raise ValueError(f"Cannot infer name for registration in {self.kind}")
            self.register(reg_name, cls, aliases=aliases)
            return cls

        return _decorator

    def get(self, name: str, default: Any = None) -> Any:
        return self._items.get(name.lower(), default)

    def create(self, name: str, *args, **kwargs) -> Any:
        cls = self.get(name)
        if cls is None:
            raise KeyError(f"{self.kind} '{name}' not found. Available: {list(self._items.keys())}")
        return cls(*args, **kwargs)

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


def create_registry(kind: str) -> Registry:
    """Convenience to create a named registry."""
    return Registry(kind)

