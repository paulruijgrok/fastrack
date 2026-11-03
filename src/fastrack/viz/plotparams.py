"""Plot helpers kept from the original FASTrack ``plotparams`` module.

The original module set FAST's rcParams (40 pt fonts, 10 pt lines, ...)
*globally* at import, which restyled every figure in the process -- including
other people's.  Those values now live in the plot theme
(``viz/themes/default.toml``, ``[rc]``) and are applied only while a FASTrack
figure is drawn (:mod:`fastrack.viz.theme`).  ``params`` is kept for reference.
"""
import math

import matplotlib

# Use a non-interactive backend by default; callers running interactively can
# override with ``matplotlib.use(...)`` before importing this module.
matplotlib.use("Agg")

import matplotlib.cm as cm  # noqa: E402
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
