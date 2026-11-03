"""Plot themes: loading, layering, per-figure rc without leaks, formats, manifest."""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

import matplotlib  # noqa: E402

from fastrack.config import Settings  # noqa: E402
from fastrack.viz import figures as F  # noqa: E402
from fastrack.viz.theme import (BUILTIN_THEMES, MANIFEST_NAME, THEME_DIR,  # noqa: E402
                                load_theme, parse_override)

from test_plot_layout import _motility_with_data, _paths_motility  # noqa: E402


# --------------------------------------------------------------------------- #
# loading and layering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", BUILTIN_THEMES)
def test_builtin_themes_load_and_cover_every_figure(name):
    th = load_theme(name)
    assert th.name == name
    for fig in F.FIGURES:
        opt = th.options(fig)
        assert "dpi" in opt
        assert th.formats(fig)


def test_theme_formats_summary_vs_diagnostic():
    d, legacy, paper = (load_theme(n) for n in ("default", "legacy", "paper"))
    assert d.formats("length_velocity") == ["png", "pdf", "svg"]
    assert d.formats("paths_2D") == ["png", "pdf", "svg"]
    assert d.formats("filaments_frame") == ["png"]
    assert d.formats("preview_dashboard") == ["png"]
    assert legacy.formats("paths_2D") == ["png"]
    assert legacy.options("paths_2D")["style"] == "legacy"
    assert legacy.options("length_velocity")["ymax"] == 1500
    assert paper.formats("length_velocity") == ["pdf", "svg"]


def test_registry_defaults_match_the_default_theme_file():
    import tomllib
    with open(os.path.join(THEME_DIR, "default.toml"), "rb") as f:
        tree = tomllib.load(f)["figures"]
    for name, spec in F.FIGURES.items():
        assert name in tree, name
        for k, v in spec.defaults.items():
            assert tree[name].get(k) == v, (name, k)


def test_user_theme_file_extends_a_builtin(tmp_path):
    p = tmp_path / "mine.toml"
    p.write_text('extends = "legacy"\n[figures.paths_2D]\ncolormap = "viridis"\n'
                 '[rc]\n"font.size" = 9\n')
    th = load_theme(str(p))
    assert th.options("paths_2D")["style"] == "legacy"          # from legacy
    assert th.options("paths_2D")["colormap"] == "viridis"      # own
    assert th.rc["font.size"] == 9 and th.rc["axes.linewidth"] == 1.0   # default base


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        load_theme("no-such-theme")


def test_parse_override_forms():
    assert parse_override("paths_2D.colormap=blue-red") == \
        {"figures": {"paths_2D": {"colormap": "blue-red"}}}
    assert parse_override("rc.font.size=12") == {"rc": {"font.size": 12}}
    assert parse_override("colors.stuck=0.5") == {"colors": {"stuck": 0.5}}
    assert parse_override('output.summary=["png"]') == {"output": {"summary": ["png"]}}
    assert parse_override("paths_2D.points=false") == {"figures": {"paths_2D": {"points": False}}}
    for bad in ("paths_2D", "rc=3", "colormap=blue"):
        with pytest.raises(ValueError):
            parse_override(bad)


def test_precedence_theme_config_cli():
    s = Settings.from_sources({"plotting": {"theme": "legacy",
                                            "paths_2D": {"colormap": "blue-red", "dpi": 150},
                                            "rc": {"font.size": 12}}})
    th = load_theme(s.plot_spec())
    o = th.options("paths_2D")
    assert (o["style"], o["colormap"], o["dpi"]) == ("legacy", "blue-red", 150)
    assert th.rc["font.size"] == 12
    s = s.with_plot_overrides(["paths_2D.colormap=viridis", "rc.font.size=8"],
                              formats=["svg"], skip=["filaments_frame"])
    th = load_theme(s.plot_spec())
    assert th.options("paths_2D")["colormap"] == "viridis"
    assert th.rc["font.size"] == 8
    assert th.formats("paths_2D") == ["svg"]                # summary override
    assert th.formats("polarity_montage") == ["png"]        # diagnostic untouched
    assert not th.enabled("filaments_frame")


def test_old_flat_keys_still_work():
    with pytest.warns(DeprecationWarning):
        s = Settings.from_sources({"plotting": {"ymax": 1500, "paths_colormap": "blue-red"}})
    th = load_theme(s.plot_spec())
    assert th.options("length_velocity")["ymax"] == 1500
    assert th.options("paths_2D")["colormap"] == "blue-red"
    # the CLI shortcuts (-ymax, -cl) map the same way, without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = Settings().with_overrides(ymax=2000, maxvel_color="g")
    th = load_theme(s.plot_spec())
    assert th.options("length_velocity")["ymax"] == 2000 and th.color("maxvel") == "g"


def test_unknown_plotting_table_is_rejected():
    with pytest.raises(KeyError):
        Settings.from_sources({"plotting": {"no_such_figure": {"dpi": 1}}})


def test_settings_round_trip_keeps_plot_overrides():
    s = Settings.from_sources({"plotting": {"paths_2D": {"colormap": "blue-red"}}})
    s2 = Settings.from_sources(s.as_dict())
    assert s2.plot_spec() == s.plot_spec()


def test_toml_config_with_plot_tables(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[plotting]\ntheme = "paper"\n[plotting.length_velocity]\nymax = 2500\n'
                   '[plotting.colors]\nstuck = "0.3"\n')
    th = load_theme(Settings.from_toml(str(cfg)).plot_spec())
    assert th.name == "paper" and th.options("length_velocity")["ymax"] == 2500
    assert th.color("stuck") == "0.3"


# --------------------------------------------------------------------------- #
# drawing: no global state, formats, manifest
# --------------------------------------------------------------------------- #
def test_importing_fastrack_plots_leaves_rcparams_alone():
    code = ("import matplotlib; before = dict(matplotlib.rcParams); "
            "import fastrack.viz.plots, fastrack.pipelines.loaded, fastrack.core.motility; "
            "after = dict(matplotlib.rcParams); "
            "changed = [k for k in before if before[k] != after[k] and not k.startswith('backend')]; "
            "print(changed); assert not changed")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_drawing_does_not_leak_rcparams(tmp_path):
    before = dict(matplotlib.rcParams)
    m = _paths_motility(tmp_path)
    m.plot_2D_path_data(5)
    m2 = _motility_with_data()
    m2.directory = str(tmp_path)
    m2.plot_length_velocity(min_points=10, percent_tolerance=500, fit_f="none")
    after = dict(matplotlib.rcParams)
    assert {k for k in before if before[k] != after[k]} <= {"backend", "backend_fallback"}


def test_summary_figure_written_in_all_formats_with_manifest(tmp_path):
    m = _paths_motility(tmp_path)
    m.plot_2D_path_data(5, extra_fname=str(tmp_path / "movie"))
    for stem in ("paths_2D", "movie_2D"):
        for ext in ("png", "pdf", "svg"):
            assert (tmp_path / ("%s.%s" % (stem, ext))).stat().st_size > 0
    entries = json.loads((tmp_path / MANIFEST_NAME).read_text())
    by_stem = {e["stem"]: e for e in entries}
    assert set(by_stem) == {"paths_2D", "movie_2D"}
    e = by_stem["paths_2D"]
    assert e["figure"] == "paths_2D" and e["pipeline"] == "fast" and e["kind"] == "summary"
    assert sorted(e["files"]) == ["paths_2D.pdf", "paths_2D.png", "paths_2D.svg"]
    # re-drawing replaces the entries, it doesn't duplicate them
    m.plot_2D_path_data(5, extra_fname=str(tmp_path / "movie"))
    assert len(json.loads((tmp_path / MANIFEST_NAME).read_text())) == 2


def test_skip_and_format_override(tmp_path):
    m = _paths_motility(tmp_path)
    th = load_theme({"theme": "default", "formats": ["svg"], "skip": []})
    m.plot_2D_path_data(5, theme=th)
    assert (tmp_path / "paths_2D.svg").exists() and not (tmp_path / "paths_2D.png").exists()
    skip = tmp_path / "skip"
    skip.mkdir()
    m.directory = str(skip)
    m.plot_2D_path_data(5, theme={"theme": "default", "skip": ["paths_2D"]})
    assert not os.listdir(skip)
    assert m.path_stats.shape == (12, 6)          # the data step still ran


def test_html_format_is_skipped_until_a_renderer_exists(tmp_path):
    m = _paths_motility(tmp_path)
    th = load_theme({"theme": "default", "formats": ["png", "html"]})
    with pytest.warns(UserWarning, match="html"):
        m.plot_2D_path_data(5, theme=th)
    assert (tmp_path / "paths_2D.png").exists() and not (tmp_path / "paths_2D.html").exists()


@pytest.mark.parametrize("name", BUILTIN_THEMES)
def test_every_motility_figure_renders_in_every_theme(tmp_path, name):
    m = _paths_motility(tmp_path)
    for style in ("velocity", "legacy"):
        m.plot_2D_path_data(5, extra_fname=str(tmp_path / ("m_" + style)), style=style,
                            theme=name)
    m2 = _motility_with_data()
    m2.directory = str(tmp_path)
    m2.plot_length_velocity(min_points=10, percent_tolerance=500, fit_f="none", theme=name)
    fmts = load_theme(name).formats("length_velocity")
    for ext in fmts:
        assert (tmp_path / ("length_velocity." + ext)).exists()


@pytest.mark.parametrize("name", BUILTIN_THEMES)
def test_polarity_figures_render_in_every_theme(tmp_path, name):
    from fastrack.analysis.kinetics import KineticModelFitter, exp_rise
    from fastrack.polarity.overlay import save_classification_montage, save_frame_average_plot
    out = save_classification_montage(np.zeros((2, 32, 40), np.uint8), {0: [], 1: []},
                                      str(tmp_path / "qc_overlay.png"), max_frames=2, theme=name)
    assert out and os.path.exists(out)
    t = np.linspace(0, 30, 31)
    mean = exp_rise(t, v0=0.0, amp=300.0, tau=5.0, t0=5.0)
    fit = KineticModelFitter([5.0]).fit(t, mean, model="exp_rise")
    out = save_frame_average_plot({"time_s": t, "mean": mean, "sem": np.full_like(t, 10.0)},
                                  str(tmp_path / "frame_average.png"), kinetics=[fit], theme=name)
    assert out and os.path.exists(out)


def test_movie_background_has_the_movie_frame_geometry(tmp_path):
    """Velocity-style paths_2D is cropped and has a colourbar; the overlay
    movie needs a background the size of its frames (default figure, 400 dpi)."""
    from PIL import Image
    m = _paths_motility(tmp_path)
    m.plot_2D_path_data(5)
    bg = m._render_movie_background(m._paths2d_data, load_theme("default"))
    w, h = Image.open(bg).size
    fw, fh = matplotlib.rcParamsDefault["figure.figsize"]
    assert (w, h) == (int(fw * 400), int(fh * 400))


# --------------------------------------------------------------------------- #
# prepared for interactive output
# --------------------------------------------------------------------------- #
def test_palette_and_bokeh_theme_are_plain_data():
    th = load_theme("default")
    pal = th.palette("blue", 5)
    assert len(pal) == 5 and all(p.startswith("#") and len(p) == 7 for p in pal)
    assert pal[0] == "#bdd7ee" and pal[-1] == "#08306b"
    js = th.bokeh_theme_json()
    assert js["attrs"]["Axis"]["axis_label_text_font_size"].endswith("pt")


def test_figure_catalogue_and_renderer_hook():
    md = F.catalogue_markdown()
    for name in F.FIGURES:
        assert "`%s`" % name in md
    assert F.FIGURES["paths_2D"].backends == ("matplotlib",)
    F.register_renderer("paths_2D", "bokeh", lambda data, fx, path: None)
    try:
        assert F.FIGURES["paths_2D"].backends == ("matplotlib", "bokeh")
        assert F.renderer("paths_2D", "bokeh") is not None
    finally:
        F.FIGURES["paths_2D"].renderers.pop("bokeh")


def test_docs_catalogue_is_up_to_date():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "docs", "plotting.md")) as f:
        assert F.catalogue_markdown() in f.read(), \
            "regenerate the table in docs/plotting.md from viz.figures.catalogue_markdown()"
