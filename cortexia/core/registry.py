"""Simple decorator-friendly Registry with dotted namespaces.

Features
- Register classes/functions via a decorator.
- Retrieve items by dotted name (e.g. "features.caption").
- Chain attribute access to traverse namespaces: ``registry.features.caption``.

Notes
- Names are stored as dotted full paths. Attribute access returns namespace
  proxies until a terminal item is reached.
- If both a terminal item and deeper names exist for the same prefix, the
  namespace takes precedence for chaining; to get the terminal item, call
  ``registry.get('prefix')`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Type, TypeVar


T = TypeVar("T")


class Registry:
    """A generic registry that supports dotted namespaces and decorators.

    Example
        registry = Registry("cortexia")

        @registry.register("features.caption")
        class CaptionFeature:
            ...

        # Access in multiple ways
        assert registry.get("features.caption") is CaptionFeature
        assert registry["features.caption"] is CaptionFeature
        assert registry.features.caption is CaptionFeature
    """

    def __init__(self, name: str = "registry") -> None:
        self._name = name
        self._items: Dict[str, Any] = {}

    # ---- Registration ----
    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Decorator to register an item under a given (dotted) name.

        If ``name`` is None, the class/function name (lowercased) is used.
        """

        def _decorator(obj: T) -> T:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError("Cannot infer name for registration; provide a name explicitly.")
            key = str(key)
            if name is None:
                key = key.lower()
            if key in self._items:
                raise KeyError(f"{self._name}: '{key}' is already registered")
            self._items[key] = obj
            return obj

        return _decorator

    def add(self, name: str, obj: Any, overwrite: bool = False) -> None:
        """Programmatically register an item.

        - ``name``: dotted path like "features.caption".
        - ``overwrite``: if True, replaces existing entry.
        """
        if not overwrite and name in self._items:
            raise KeyError(f"{self._name}: '{name}' is already registered")
        self._items[name] = obj

    # ---- Lookup ----
    def get(self, name: str, default: Any = None) -> Any:
        """Get an item by its dotted name. Returns ``default`` if not found."""
        return self._items.get(name, default)

    def require(self, name: str) -> Any:
        """Get an item or raise a KeyError if missing."""
        if name not in self._items:
            raise KeyError(f"{self._name}: '{name}' is not registered")
        return self._items[name]

    def __getitem__(self, name: str) -> Any:
        return self.require(name)

    def __contains__(self, name: object) -> bool:
        return bool(isinstance(name, str) and name in self._items)

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._items.items()

    # ---- Namespaces via attribute chaining ----
    def __getattr__(self, attr: str) -> "_Namespace":  # type: ignore[override]
        # Provide a namespace proxy starting at the first segment
        return _Namespace(self, attr)

    def namespace(self, prefix: str) -> "_Namespace":
        return _Namespace(self, prefix)

    # ---- Introspection ----
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={len(self._items)})"


@dataclass
class _Namespace:
    """Lightweight namespace proxy for dotted access (e.g., a.b.c).

    Resolution rules for attribute access on a prefix 'p':
    - If any registered key starts with 'p.<next>', return another namespace
      proxy to allow chaining.
    - Else if an exact item 'p' exists, return that item.
    - Else raise AttributeError.
    """

    _registry: Registry
    _prefix: str

    def __getattr__(self, attr: str) -> Any:  # type: ignore[override]
        full = f"{self._prefix}.{attr}" if self._prefix else attr
        # If there are deeper names, prefer returning a namespace for chaining
        has_deeper = any(k.startswith(full + ".") for k in self._registry._items.keys())
        if has_deeper:
            return _Namespace(self._registry, full)
        # Otherwise, try to return the exact item
        if full in self._registry._items:
            return self._registry._items[full]
        raise AttributeError(f"No such namespace or item: {full}")

    def get(self, name: str, default: Any = None) -> Any:
        full = f"{self._prefix}.{name}" if self._prefix else name
        return self._registry.get(full, default)

    def require(self, name: str) -> Any:
        full = f"{self._prefix}.{name}" if self._prefix else name
        return self._registry.require(full)

    def __repr__(self) -> str:
        return f"Namespace(prefix={self._prefix!r})"
