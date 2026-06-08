"""Analysis pipelines (orchestration).

Importing this package registers the built-in pipelines under ``PIPELINES``.
``run`` is re-exported as the unloaded gliding-assay driver for convenience and
backward compatibility.
"""
from . import gliding  # noqa: F401  (registers GlidingPipeline)
from .base import PIPELINES, Pipeline
from .gliding import run

__all__ = ["PIPELINES", "Pipeline", "run", "gliding", "loaded"]
