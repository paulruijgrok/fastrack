"""Analysis-pipeline interface.

A pipeline is *orchestration*: it wires together the reusable machinery in
``core`` / ``analysis`` / ``io`` / ``viz`` for a particular assay.  New
pipelines (a different assay, a different workflow) are added by registering a
class under ``PIPELINES`` and selecting it by name -- the shared building blocks
do not change.
"""
from abc import ABC, abstractmethod

from ..registry import Registry

#: Registry of available pipelines (populated on import).
PIPELINES = Registry("pipeline")


class Pipeline(ABC):
    """Runs an analysis over a dataset directory given a ``Settings`` object."""

    @abstractmethod
    def run(self, main_dir, settings):
        """Analyze the dataset at ``main_dir`` using ``settings``."""
        raise NotImplementedError
