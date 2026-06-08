"""Frame-to-frame linking (tracking) strategy interface.

A ``Linker`` decides which filament in frame N+1 is the same physical filament
as one in frame N.  Swapping linkers (greedy score-based today; a global/
Hungarian or overlap-based tracker tomorrow) requires only registering a new
class under ``LINKERS`` and selecting it by name in the settings -- the pipeline
and path-construction code are unchanged.
"""
from abc import ABC, abstractmethod

from ...registry import Registry

#: Registry of available linker implementations (populated on import).
LINKERS = Registry("linker")


class Linker(ABC):
    """Builds links between the filaments of two adjacent frames."""

    @abstractmethod
    def link(self, frame1, frame2, dt, elapsed_times):
        """Link ``frame1`` -> ``frame2``.

        Implementations may mutate the frames' filaments (setting their
        ``forward_link`` / ``reverse_link``) exactly as the original algorithm
        did.  Returns ``(new_frame_links, dt_used)`` where ``new_frame_links`` is
        the list of accepted links and ``dt_used`` is the (possibly metadata-
        derived) time step for this pair.
        """
        raise NotImplementedError
