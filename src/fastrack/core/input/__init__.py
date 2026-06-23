"""Pluggable movie input sources (frame folders, TIFF stacks, ...)."""
from . import mm_dir  # noqa: F401  (registers MicroManagerDirSource)
from . import tiff_stack  # noqa: F401  (registers TiffStackSource)
from .base import (  # noqa: F401
    FrameSource, open_movie, open_source, register_source, to_pipeline_image,
)

__all__ = [
    "FrameSource", "open_movie", "open_source", "register_source",
    "to_pipeline_image",
]
