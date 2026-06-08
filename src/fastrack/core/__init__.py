"""Core image-processing and tracking objects.

Kept import-light: pulling in :mod:`fastrack.core.frame` (and the scipy /
scikit-image / opencv stack it needs) is deferred to the point of use.  Import
the specific submodule you need, e.g. ``from fastrack.core.frame import Frame``.
"""
__all__ = [
    "frame",
    "island",
    "filament",
    "link",
    "motility",
    "detection",
    "tracking",
]
