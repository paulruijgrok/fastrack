"""Plot themes: one master stylesheet that every FASTrack figure derives from.

A *theme* is a TOML file (built-ins live in ``viz/themes/``) with:

* ``[tokens]``    backend-neutral design values (font, scale, ink, background);
* ``[colors]``    named colours the figures use;
* ``[colormaps]`` named colour scales as colour stops (matplotlib colormap or a
  Bokeh palette);
* ``[rc]``        matplotlib rcParams, applied only while a figure is drawn;
* ``[bokeh]``     Bokeh-only extras (for the interactive outputs to come);
* ``[output]``    default formats for summary / diagnostic figures;
* ``[figures.<name>]`` per-figure options (see :mod:`fastrack.viz.figures`).

A theme may start with ``extends = "<theme>"`` to inherit another one.  On top
of the theme come run-config and CLI overrides (:func:`load_theme`), merged
key by key, so

    built-in theme < theme file < [plotting.*] in the run config < --plot key=value

Figures use it through :meth:`Theme.figure`::

    with theme.figure("paths_2D", style="velocity") as fx:   # rc applied, nothing leaks
        fig = plt.figure(figsize=...)
        ... fx.opt["colormap"], fx.color("stuck"), fx.fs(12) ...
        fx.save(fig, out_dir + "/paths_2D")                   # -> .png/.pdf/.svg + manifest

Nothing here changes matplotlib's global state, so importing FASTrack no
longer restyles other people's plots.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

THEME_DIR = os.path.join(os.path.dirname(__file__), "themes")
BUILTIN_THEMES = ("default", "legacy", "paper")

#: File formats matplotlib can write directly.
STATIC_FORMATS = ("png", "pdf", "svg", "eps", "jpg", "jpeg", "tif", "tiff")
#: Formats produced by an interactive backend (Bokeh) -- prepared, not built yet.
INTERACTIVE_FORMATS = ("html",)

MANIFEST_NAME = "figures.json"


# --------------------------------------------------------------------------- #
# loading / merging
# --------------------------------------------------------------------------- #
def _toml_load(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            raise RuntimeError("Reading plot themes needs Python 3.11+ or the 'tomli' package")
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_merge(base: Mapping[str, Any], *overs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursively merge mappings; later ones win, nested tables merge key by key."""
    out = copy.deepcopy(dict(base))
    for over in overs:
        if not over:
            continue
        for k, v in over.items():
            if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
    return out


def theme_path(name_or_path: str) -> str:
    """Resolve a built-in theme name or a path to a TOML file."""
    if os.path.isfile(name_or_path):
        return name_or_path
    cand = os.path.join(THEME_DIR, name_or_path + ".toml")
    if os.path.isfile(cand):
        return cand
    raise ValueError("Unknown plot theme %r (built-in: %s, or a path to a .toml file)"
                     % (name_or_path, ", ".join(BUILTIN_THEMES)))


def _load_raw(name_or_path: str, _seen: Sequence[str] = ()) -> Dict[str, Any]:
    path = os.path.abspath(theme_path(name_or_path))
    if path in _seen:
        raise ValueError("Plot theme inheritance loop: %s" % " -> ".join(list(_seen) + [path]))
    data = _toml_load(path)
    parent = data.pop("extends", None)
    if parent:
        base_dir = os.path.dirname(path)
        rel = os.path.join(base_dir, parent)
        parent_ref = rel if os.path.isfile(rel) else parent
        data = deep_merge(_load_raw(parent_ref, tuple(_seen) + (path,)), data)
    data.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return data


def parse_override(item: str) -> Dict[str, Any]:
    """``"paths_2D.colormap=blue-red"`` -> ``{"figures": {"paths_2D": {"colormap": "blue-red"}}}``.

    The first dotted part names a theme section (``rc``, ``tokens``, ``colors``,
    ``colormaps``, ``output``, ``bokeh``) or else a figure.  ``rc`` keys keep
    their own dots (``rc.font.size=12``).  Values are parsed as TOML scalars /
    arrays when possible (``12``, ``true``, ``["png","pdf"]``), else as strings.
    """
    if "=" not in item:
        raise ValueError("Plot override must look like key=value, got %r" % item)
    key, raw = item.split("=", 1)
    key, raw = key.strip(), raw.strip()
    value = _parse_value(raw)
    parts = key.split(".")
    if parts[0] == "rc":
        if len(parts) < 2:
            raise ValueError("rc override needs a key: rc.<rcParam>=value")
        return {"rc": {".".join(parts[1:]): value}}
    if parts[0] in ("tokens", "colors", "output", "bokeh", "colormaps"):
        tree: Dict[str, Any] = {}
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
        return tree
    if len(parts) < 2:
        raise ValueError("Plot override %r: use <figure>.<option>=value or rc.<param>=value" % item)
    fig, opt = parts[0], ".".join(parts[1:])
    return {"figures": {fig: {opt: value}}}


def _parse_value(raw: str) -> Any:
    try:
        import tomllib
        return tomllib.loads("v = " + raw)["v"]
    except Exception:
        return raw


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
class Theme:
    """A resolved theme: the merged TOML tree plus helpers for each backend."""

    def __init__(self, data: Mapping[str, Any], formats: Optional[Sequence[str]] = None,
                 skip: Iterable[str] = (), manifest: bool = True):
        self.data = dict(data)
        self.name = self.data.get("name", "custom")
        self.tokens = dict(self.data.get("tokens", {}))
        self.colors = dict(self.data.get("colors", {}))
        self.colormaps = dict(self.data.get("colormaps", {}))
        self.rc = dict(self.data.get("rc", {}))
        self.bokeh = dict(self.data.get("bokeh", {}))
        self.output = dict(self.data.get("output", {}))
        self.figures = {k: dict(v) for k, v in self.data.get("figures", {}).items()}
        self.formats_override = list(formats) if formats else None
        self.skip = set(skip or ())
        self.manifest = manifest

    # -- options ----------------------------------------------------------- #
    def options(self, figure: str, **overrides: Any) -> Dict[str, Any]:
        """Figure options: registry defaults < theme [figures.X] < overrides (None ignored)."""
        from .figures import FIGURES
        spec = FIGURES.get(figure)
        base = dict(spec.defaults) if spec else {}
        opt = deep_merge(base, self.figures.get(figure, {}))
        opt.update({k: v for k, v in overrides.items() if v is not None})
        return opt

    def formats(self, figure: str) -> List[str]:
        from .figures import FIGURES
        opt = self.figures.get(figure, {})
        if "formats" in opt:
            return list(opt["formats"])
        spec = FIGURES.get(figure)
        kind = spec.kind if spec else "diagnostic"
        if kind == "summary" and self.formats_override:
            return list(self.formats_override)
        return list(self.output.get(kind, ["png"]))

    def enabled(self, figure: str) -> bool:
        return figure not in self.skip and self.figures.get(figure, {}).get("enabled", True)

    # -- colours ------------------------------------------------------------ #
    def color(self, name: str) -> Any:
        return self.colors.get(name, name)

    def scheme(self, name: str) -> Dict[str, Any]:
        """A colour scheme by name: dict(cmap, ink, stuck, background, centred).

        ``[colormaps]`` names come first; any other name is a matplotlib colormap.
        """
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
        tok_ink = self.tokens.get("ink", "k")
        if name in self.colormaps:
            sc = dict(self.colormaps[name])
            stops = sc.pop("stops")
            sc["cmap"] = LinearSegmentedColormap.from_list("fastrack_" + name, stops)
            sc.setdefault("ink", tok_ink)
            sc.setdefault("stuck", self.color("stuck"))
            sc.setdefault("background", self.tokens.get("background", "white"))
            sc.setdefault("centred", False)
            return sc
        try:
            cmap = matplotlib.colormaps[name]
        except KeyError:
            raise ValueError("unknown paths colormap %r" % name) from None
        return dict(cmap=cmap, ink="k", stuck=self.color("stuck"),
                    background=self.tokens.get("background", "white"), centred=False)

    def palette(self, name: str, n: int = 256) -> List[str]:
        """A colour scale as ``n`` hex colours (what Bokeh wants for palettes)."""
        from matplotlib.colors import to_hex
        cmap = self.scheme(name)["cmap"]
        return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

    # -- matplotlib --------------------------------------------------------- #
    def mpl_rc(self, figure: Optional[str] = None) -> Dict[str, Any]:
        """rcParams for a figure: theme [rc] (or matplotlib's defaults when the
        figure sets ``rc_base = "matplotlib"``) + the figure's own ``rc`` table."""
        fig_opt = self.figures.get(figure, {}) if figure else {}
        if fig_opt.get("rc_base") == "matplotlib":
            rc: Dict[str, Any] = {}
        else:
            rc = {}
            fam = self.tokens.get("font_family")
            if fam:
                rc["font.sans-serif"] = list(fam) if not isinstance(fam, str) else [fam]
            rc.update(self.rc)
        rc.update(fig_opt.get("rc", {}))
        return rc

    def figure(self, name: str, **overrides: Any) -> "FigureContext":
        return FigureContext(self, name, overrides)

    # -- bokeh (prepared; Bokeh itself is imported only when used) ---------- #
    def bokeh_theme_json(self) -> Dict[str, Any]:
        """The theme as a Bokeh theme dict (``bokeh.themes.Theme(json=...)``).

        Built from the shared tokens so an HTML figure matches the static one.
        Pure data -- doesn't import Bokeh.
        """
        fam = self.tokens.get("font_family", "sans-serif")
        fam = ", ".join(fam) if not isinstance(fam, str) else fam
        size = "%gpt" % (10 * float(self.tokens.get("font_scale", 1.0)))
        attrs = {
            "Plot": {"background_fill_color": self.tokens.get("background", "white"),
                     "toolbar_location": self.bokeh.get("toolbar_location", "right")},
            "Axis": {"axis_label_text_font": fam, "major_label_text_font": fam,
                     "axis_label_text_font_size": size, "major_label_text_font_size": size},
            "Title": {"text_font": fam},
            "Legend": {"label_text_font": fam, "label_text_font_size": size},
        }
        return {"attrs": deep_merge(attrs, self.bokeh.get("attrs", {}))}

    def bokeh_theme(self):
        try:
            from bokeh.themes import Theme as BokehTheme
        except ModuleNotFoundError:
            raise RuntimeError("Interactive (html) figures need Bokeh: "
                               "pip install 'fastrack[interactive]'") from None
        return BokehTheme(json=self.bokeh_theme_json())


class FigureContext:
    """Per-figure drawing context: rc applied on enter, restored on exit."""

    def __init__(self, theme: Theme, name: str, overrides: Mapping[str, Any]):
        self.theme = theme
        self.name = name
        self.opt = theme.options(name, **overrides)
        self.saved: List[str] = []
        self._stack: Optional[contextlib.ExitStack] = None

    def __enter__(self) -> "FigureContext":
        import matplotlib
        self._stack = contextlib.ExitStack()
        if self.theme.figures.get(self.name, {}).get("rc_base") == "matplotlib":
            self._stack.enter_context(matplotlib.rc_context(matplotlib.rcParamsDefault))
        self._stack.enter_context(matplotlib.rc_context(self.theme.mpl_rc(self.name)))
        return self

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.close()

    # helpers used by the figure code
    def fs(self, size: float) -> float:
        """A font size scaled by the theme's ``font_scale``."""
        return float(size) * float(self.theme.tokens.get("font_scale", 1.0))

    def color(self, name: str) -> Any:
        return self.theme.color(name)

    def scheme(self, name: Optional[str] = None) -> Dict[str, Any]:
        return self.theme.scheme(name or self.opt.get("colormap", "viridis"))

    @property
    def enabled(self) -> bool:
        return self.theme.enabled(self.name)

    def save(self, fig, path: str, formats: Optional[Sequence[str]] = None,
             title: Optional[str] = None, **savefig_kw: Any) -> List[str]:
        """Save ``fig`` once per format next to ``path`` (its extension is ignored).

        Uses the figure's ``dpi`` / ``pad_inches`` options unless given.  ``html``
        is skipped with a note unless the figure has an interactive renderer.
        Each written file is recorded in ``figures.json`` in that directory (the
        index a PDF report or dashboard can be built from).
        """
        stem = _strip_ext(path)
        fmts = list(formats) if formats else self.theme.formats(self.name)
        kw = {"dpi": self.opt.get("dpi", 200)}
        if "pad_inches" in self.opt:
            kw["pad_inches"] = self.opt["pad_inches"]
        kw.update(savefig_kw)
        written = []
        for fmt in fmts:
            fmt = fmt.lower().lstrip(".")
            if fmt in INTERACTIVE_FORMATS:
                warnings.warn("%s: no interactive renderer yet, skipping %s" % (self.name, fmt))
                continue
            if fmt not in STATIC_FORMATS:
                raise ValueError("%s: unknown figure format %r" % (self.name, fmt))
            out = stem + "." + fmt
            fig.savefig(out, format=fmt, **kw)
            written.append(out)
        self.saved.extend(written)
        if self.theme.manifest and written:
            record_figure(os.path.dirname(stem) or ".", self.name, written, title=title,
                          theme=self.theme.name)
        return written


def _strip_ext(path: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower().lstrip(".") in STATIC_FORMATS + INTERACTIVE_FORMATS:
        return root
    return path


def record_figure(directory: str, figure: str, files: Sequence[str],
                  title: Optional[str] = None, theme: Optional[str] = None) -> None:
    """Append (or update) an entry in ``<directory>/figures.json``.

    One entry per output stem: ``{"figure", "title", "theme", "files": [...]}``,
    with file names relative to ``directory``.  This is the index a later PDF
    report or interactive viewer collects the figures from.
    """
    from .figures import FIGURES
    path = os.path.join(directory, MANIFEST_NAME)
    try:
        with open(path) as f:
            entries = json.load(f)
    except (OSError, ValueError):
        entries = []
    rel = [os.path.relpath(p, directory) for p in files]
    stem = os.path.splitext(rel[0])[0]
    spec = FIGURES.get(figure)
    entry = {"figure": figure, "stem": stem,
             "title": title or (spec.title if spec else figure),
             "pipeline": spec.pipeline if spec else None,
             "kind": spec.kind if spec else None,
             "theme": theme, "files": rel}
    entries = [e for e in entries if e.get("stem") != stem] + [entry]
    try:
        with open(path, "w") as f:
            json.dump(entries, f, indent=1)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
PlotSpec = Union[None, str, Mapping[str, Any], Theme]


def load_theme(spec: PlotSpec = None, *overrides: Optional[Mapping[str, Any]]) -> Theme:
    """Resolve a plotting spec into a :class:`Theme`.

    ``spec`` is ``None`` (default theme), a theme name / path, a :class:`Theme`
    (returned as is when there are no overrides) or the dict built by
    :meth:`fastrack.config.Settings.plot_spec`::

        {"theme": "default", "formats": [...] | None, "skip": [...],
         "overrides": {"rc": {...}, "figures": {"paths_2D": {...}}, ...},
         "cli": ["paths_2D.colormap=blue-red", ...]}

    Extra ``overrides`` mappings (theme-shaped) are merged last.
    """
    if isinstance(spec, Theme):
        if not any(overrides):
            return spec
        return Theme(deep_merge(spec.data, *overrides), spec.formats_override, spec.skip,
                     spec.manifest)
    if spec is None or isinstance(spec, str):
        spec = {"theme": spec or "default"}
    spec = dict(spec)
    data = _load_raw(spec.get("theme") or "default")
    layers = [spec.get("overrides") or {}]
    layers += [parse_override(s) for s in spec.get("cli") or ()]
    layers += list(overrides)
    data = deep_merge(data, *layers)
    return Theme(data, formats=spec.get("formats"), skip=spec.get("skip") or (),
                 manifest=spec.get("manifest", True))


_DEFAULT: Optional[Theme] = None


def default_theme() -> Theme:
    """The built-in default theme (cached)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_theme("default")
    return _DEFAULT


def as_theme(theme: PlotSpec) -> Theme:
    return default_theme() if theme is None else load_theme(theme)
