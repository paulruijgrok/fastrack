"""Catalogue of every figure FASTrack writes.

Each figure has a name (the key used in themes, configs and ``--plot``), the
pipeline that makes it, whether it is a *summary* figure (vector formats by
default: PNG + PDF + SVG) or a *diagnostic* one (PNG), its option defaults, and
the backends that can draw it.

Figures are drawn in two steps, so other outputs can reuse the data:

1. a **data** step that computes plain tables (numpy arrays / dicts) -- the
   slow part, done once;
2. a **renderer** per backend: ``"matplotlib"`` (static files) today;
   ``"bokeh"`` (standalone interactive HTML) can be added per figure with
   :func:`register_renderer` without touching the data step.

``docs/plotting.md`` is the human-readable version of this table
(:func:`catalogue_markdown` generates it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class FigureSpec:
    name: str
    title: str
    pipeline: str                       # fast | fastplus | lima | fast-preview
    kind: str                           # "summary" | "diagnostic"
    files: str                          # output file pattern(s)
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    renderers: Dict[str, Callable] = field(default_factory=dict)

    @property
    def backends(self) -> Tuple[str, ...]:
        return ("matplotlib",) + tuple(b for b in self.renderers if b != "matplotlib")


FIGURES: Dict[str, FigureSpec] = {}


def register_figure(spec: FigureSpec) -> FigureSpec:
    FIGURES[spec.name] = spec
    return spec


def register_renderer(figure: str, backend: str, fn: Callable) -> None:
    """Attach a renderer (e.g. ``"bokeh"``) to a figure.

    The renderer is called as ``fn(data, fx, path)`` with the figure's data
    tables, its :class:`~fastrack.viz.theme.FigureContext` and the output stem.
    """
    FIGURES[figure].renderers[backend] = fn


def renderer(figure: str, backend: str) -> Optional[Callable]:
    spec = FIGURES.get(figure)
    return spec.renderers.get(backend) if spec else None


# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #
register_figure(FigureSpec(
    "length_velocity", "Length vs velocity", "fast", "summary",
    "<movie>_length_velocity.*, combined/<dataset>_length_velocity.*",
    "Scatter of filament length vs velocity with length / velocity histograms, "
    "the tolerance curve, summary labels (MVEL, TOP5%, PLATEAU, %STUCK, FIL-LENGTH) "
    "and the optional plateau fit.",
    defaults=dict(dpi=200, ymax=0, label_position="bottom", label_units=False,
                  text_font_size=30, pad_inches=0.3),
))
register_figure(FigureSpec(
    "paths_2D", "Filament tracks", "fast", "summary",
    "<movie>_2D.*, <movie>/paths_2D.*",
    "Every track, coloured by its median velocity, with one arrowhead along the "
    "motion, dots at each frame's position, a scale bar; or the original FAST "
    "figure (style = \"legacy\").",
    defaults=dict(dpi=300, style="velocity", colormap="blue", points=True, labels=False,
                  scale_bar_um=10.0, width_in=8.0, line_width=1.2, legacy_dpi=400,
                  pad_inches=0.05),
))
register_figure(FigureSpec(
    "correlation_length", "Contour correlation length", "fast", "summary",
    "correlation_length.*",
    "Tangent correlation along filaments (Motility.plot_correlation_profile).",
    defaults=dict(dpi=200),
))
register_figure(FigureSpec(
    "filaments_frame", "Detected filaments (one frame)", "fast", "diagnostic",
    "filaments_<frame>.*",
    "Debug image of the filaments detected in a frame (Frame.save_filament_img).",
    defaults=dict(dpi=200),
))
register_figure(FigureSpec(
    "preview_dashboard", "Preview dashboard", "fast-preview", "diagnostic",
    "<movie>_preview.*",
    "Quick-look dashboard: overlay of moving / stuck / floppy / pinned "
    "filaments, length and velocity histograms, ranked quality bars, examples.",
    defaults=dict(dpi=80),
))
register_figure(FigureSpec(
    "polarity_montage", "Polarity classification montage", "fastplus", "diagnostic",
    "qc_overlay.*",
    "Sampled head-channel frames with every head coloured by its polarity class.",
    defaults=dict(dpi=110),
))
register_figure(FigureSpec(
    "frame_average", "Per-frame mean directional velocity", "fastplus", "summary",
    "frame_average.*",
    "Mean signed velocity per frame across movies with SEM band, perturbation "
    "markers and kinetic fits.",
    defaults=dict(dpi=150),
))
register_figure(FigureSpec(
    "lima_load", "LIMA: mobility and MVIS vs load", "lima", "summary",
    "<A>_vs_<B>_stop_model*.*",
    "% time mobile and MVIS against utrophin (load) concentration, per protein, "
    "with the fitted load model.",
    defaults=dict(dpi=100),
))
register_figure(FigureSpec(
    "lima_length", "LIMA: filament length vs load", "lima", "summary",
    "<A>_vs_<B>_length*.*",
    "Mean filament length against load concentration, per protein.",
    defaults=dict(dpi=200),
))


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"%s"' % v
    return repr(v)


def catalogue_markdown() -> str:
    """Markdown table of the catalogue (for docs/plotting.md)."""
    rows = ["| figure | pipeline | kind | files | options (defaults) |",
            "|---|---|---|---|---|"]
    for s in FIGURES.values():
        opts = ", ".join("`%s = %s`" % (k, _toml_value(v)) for k, v in s.defaults.items()
                         if k != "dpi")
        rows.append("| `%s` | %s | %s | `%s` | dpi %s%s |" % (
            s.name, s.pipeline, s.kind, s.files, s.defaults.get("dpi", "-"),
            ("; " + opts) if opts else ""))
    return "\n".join(rows)
