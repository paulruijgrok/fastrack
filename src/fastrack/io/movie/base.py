"""Tracking-movie writer interface.

A ``MovieWriter`` turns the per-frame skeleton PNGs produced during plotting
into a movie.  The default encodes H.264/MP4 via ffmpeg; alternative codecs,
containers, or overlay styles are added by registering a new writer under
``MOVIE_WRITERS`` and selecting it by name.
"""
from abc import ABC, abstractmethod

from ...registry import Registry

#: Registry of available movie writers (populated on import).
MOVIE_WRITERS = Registry("movie writer")


class MovieWriter(ABC):
    """Assembles per-frame images in ``directory`` into a movie."""

    @abstractmethod
    def write(self, directory, extra_fname=None,
              input_pattern="skeletons_%03d.png", output_name="filament_tracks.mp4",
              fps=1):
        """Encode the ``input_pattern`` image sequence in ``directory`` to a movie.

        ``extra_fname`` is an optional prefix for an additional copy placed
        alongside the length-velocity outputs (matching the original API).
        ``input_pattern`` / ``output_name`` default to the per-frame skeleton
        sequence and the standard tracking movie, so existing callers are
        unchanged; the overlay movie passes its own pattern/name.
        """
        raise NotImplementedError
