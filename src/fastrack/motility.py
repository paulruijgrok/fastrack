"""Backward-compatibility shim for the pre-refactor ``fastrack.motility`` module.

The original monolithic ``motility.py`` was split into sub-packages (``core``,
``analysis``, ``io``, ``viz``).  This module re-exports the names it used to
expose so existing imports keep working, e.g.::

    from fastrack.motility import Motility, Frame, length_velocity

New code should import from the specific modules instead.
"""
from .analysis import (
    bin_length_velocity,
    contour2contour,
    coupling_velocity,
    fit_coupling_velocity,
    fit_gaussian,
    fit_length_velocity,
    gaussian,
    length_velocity,
    vec_length,
)
from .core.filament import Filament
from .core.frame import Frame
from .core.island import Island
from .core.link import Link, Path
from .core.motility import Motility
from .io.images import alpha_composite as _alpha_composite
from .io.images import stack_to_tiffs
from .viz.plots import make_N_colors

__all__ = [
    "Motility",
    "Frame",
    "Island",
    "Filament",
    "Link",
    "Path",
    "gaussian",
    "fit_gaussian",
    "fit_length_velocity",
    "length_velocity",
    "coupling_velocity",
    "fit_coupling_velocity",
    "bin_length_velocity",
    "contour2contour",
    "vec_length",
    "stack_to_tiffs",
    "make_N_colors",
    "_alpha_composite",
]
