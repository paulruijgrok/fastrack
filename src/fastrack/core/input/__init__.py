"""Pluggable movie input sources (frame folders, TIFF stacks, ...)."""
from . import mm_dir  # noqa: F401  (registers MicroManagerDirSource)
from . import tiff_stack  # noqa: F401  (registers TiffStackSource)
from . import preprocessed  # noqa: F401  (registers PreprocessedSource)
from .base import (  # noqa: F401
    FrameSource, open_movie, open_source, register_source, to_pipeline_image,
)
from .preprocessed import PreprocessedSource  # noqa: F401

__all__ = [
    "FrameSource", "PreprocessedSource", "open_movie", "open_source",
    "register_source", "to_pipeline_image",
]
