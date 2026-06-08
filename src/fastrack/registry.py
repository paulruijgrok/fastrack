"""A tiny name->factory registry for pluggable strategies.

Each swappable category (filament detectors, frame-to-frame linkers, movie
writers, filament stores, ...) owns a ``Registry`` instance.  Implementation
modules register their class/factory under a string name; the ``Settings``
object selects one by name at run time.  This is the single mechanism behind
every pluggability seam in the package, so new algorithms are added by writing
a module and registering it -- no edits to the call sites.
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps case-insensitive names to factories producing objects of type T."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Optional[Callable[..., T]] = None):
        """Register ``factory`` under ``name``.

        Usable directly (``reg.register("greedy", GreedyLinker)``) or as a
        decorator (``@reg.register("greedy")``).
        """
        def _add(f: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._items:
                raise ValueError("%s '%s' is already registered" % (self.kind, name))
            self._items[key] = f
            return f

        return _add(factory) if factory is not None else _add

    def create(self, name: str, *args, **kwargs) -> T:
        """Instantiate the implementation registered under ``name``."""
        key = str(name).lower()
        if key not in self._items:
            raise KeyError(
                "Unknown %s '%s'. Available: %s"
                % (self.kind, name, ", ".join(self.available()))
            )
        return self._items[key](*args, **kwargs)

    def get(self, name: str) -> Callable[..., T]:
        """Return the registered factory (not instantiated)."""
        return self._items[str(name).lower()]

    def available(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._items
