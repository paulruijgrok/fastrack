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
    def write(self, directory, extra_fname=None):
        """Write the movie for the frames in ``directory``.

        ``extra_fname`` is an optional prefix for an additional copy placed
        alongside the length-velocity outputs (matching the original API).
        """
        raise NotImplementedError
