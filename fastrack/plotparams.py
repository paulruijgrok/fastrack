"""Plotting parameters for FASTrack (modernized Python 3 port).

Original author: Tural Aksel.  Modernized for Matplotlib >= 3.6 and a headless
(Agg) backend so plots render without a display server.
"""
import math

import matplotlib

# Use a non-interactive backend by default; callers running interactively can
# override with ``matplotlib.use(...)`` before importing this module.
matplotlib.use("Agg")

import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as py  # noqa: E402
import numpy as np  # noqa: E402

params = {
    "axes.labelsize": 40,
    "axes.titlesize": 40,
    "font.size": 40,
    "legend.fontsize": 40,
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.borderaxespad": 1.0,
    "legend.numpoints": 1,
    "legend.handlelength": 0.5,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.borderpad": 0.2,
    "lines.markersize": 25,
    "lines.markeredgewidth": 3,
    "axes.linewidth": 1.0,
    "lines.linewidth": 10,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.formatter.limits": (-4, 4),
    "figure.subplot.top": 0.95,
    "figure.subplot.bottom": 0.135,
}

py.rcParams.update(params)

font = {"family": "sans-serif", "sans-serif": ["Arial", "DejaVu Sans"], "weight": "normal"}
py.rc("font", **font)

yTick = {"major.pad": 10, "major.size": 20, "minor.size": 10, "labelsize": 40}
xTick = {"major.pad": 10, "major.size": 20, "minor.size": 10, "labelsize": 40}
py.rc("xtick", **xTick)
py.rc("ytick", **yTick)


def get_figsize(fig_width_pt):
    """Convert a width in points to a (width, height) figure size in inches.

    Height is chosen via the golden ratio for a pleasing aspect.
    """
    inches_per_pt = 1.0 / 72.0
    golden_mean = (math.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    return [fig_width, fig_height]


def make_N_colors(cmap_name, N):
    """Return ``N`` evenly spaced colors from a named colormap.

    Works across Matplotlib versions (the ``cm.get_cmap`` API was deprecated in
    3.7 and removed later).
    """
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(N)
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(cmap_name, N)
    return cmap(np.arange(N))
