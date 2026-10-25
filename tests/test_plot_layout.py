"""Length-velocity figure: data-based velocity limit and label placement."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from fastrack.config import Settings  # noqa: E402
from fastrack.viz.plots import auto_velocity_limit  # noqa: E402


def test_auto_limit_covers_the_data_and_the_top5_line():
    v = np.r_[np.zeros(50), np.linspace(100, 2000, 1000)]        # zeros = stuck
    lim = auto_velocity_limit(v, top_5_velocity=1950)
    assert lim % 250 == 0
    assert lim >= 1.1 * 1950 and lim < 1.1 * 2000 + 250
    # a TOP5% above the 99.5th percentile still fits on the plot
    assert auto_velocity_limit(v, top_5_velocity=3000) >= 3000


def test_auto_limit_ignores_stuck_and_nonfinite_and_has_a_fallback():
    assert auto_velocity_limit(np.r_[np.zeros(10), np.nan, 800.0]) == 1000.0
    assert auto_velocity_limit(np.zeros(5)) == 1500.0
    assert auto_velocity_limit([]) == 1500.0


def test_plotting_defaults_auto_ymax_and_bottom_labels():
    s = Settings()
    assert s.plotting.ymax == 0
    assert s.plotting.label_position == "bottom"
    kw = s.to_run_kwargs()                 # what `fast` passes to gliding.run
    assert kw["plot_ymax"] == 0 and kw["label_position"] == "bottom"
    from fastrack.pipelines import gliding
    import inspect
    assert "label_position" in inspect.signature(gliding.run).parameters


def _motility_with_data(seed=0):
    from fastrack.core.motility import Motility
    rng = np.random.default_rng(seed)
    n = 400
    length = rng.uniform(800, 3000, n)
    vel = np.clip(rng.normal(1800, 300, n), 50, None)
    full = np.column_stack([length, vel, 0.05 * vel, np.full(n, 12.0)])
    m = Motility()
    m.full_len_vel = full
    m.max_len_vel = full[::4]
    return m


@pytest.mark.parametrize("pos", ["bottom", "top"])
def test_figure_renders_with_auto_limit_and_both_label_positions(tmp_path, pos):
    m = _motility_with_data()
    m.directory = str(tmp_path)
    stats = m.plot_length_velocity(header=pos + "_", min_points=10, percent_tolerance=500,
                                   fit_f="none", print_plot=True, label_position=pos)
    assert stats[0] > 0
    assert (tmp_path / (pos + "_length_velocity.png")).exists()


def test_bad_label_position_is_rejected(tmp_path):
    m = _motility_with_data()
    m.directory = str(tmp_path)
    with pytest.raises(ValueError):
        m.plot_length_velocity(print_plot=True, fit_f="none", label_position="middle")


def test_velocity_limit_does_not_change_the_statistics(tmp_path):
    a, b = _motility_with_data(), _motility_with_data()
    a.directory = b.directory = str(tmp_path)
    sa = a.plot_length_velocity(max_vel=0, min_points=10, percent_tolerance=500,
                                fit_f="none", print_plot=False)
    sb = b.plot_length_velocity(max_vel=1500, min_points=10, percent_tolerance=500,
                                fit_f="none", print_plot=False)
    assert sa == sb


# --------------------------------------------------------------------------- #
# plateau fit
# --------------------------------------------------------------------------- #
def _lv_data(n=600, U=1900.0, A=2500.0, tau=650.0, seed=1):
    rng = np.random.default_rng(seed)
    L = rng.uniform(700, 3500, n)
    V = U - A * np.exp(-L / tau) + rng.normal(0, 120, n)
    return L, V


def test_fit_plateau_recovers_parameters_with_sensible_errors():
    from fastrack.analysis import fit_plateau
    L, V = _lv_data()
    r = fit_plateau(L, V, "exp")
    assert r["ok"] and r["n"] == 600
    assert abs(r["plateau"] - 1900) < 3 * r["plateau_se"] + 20
    assert 5 < r["plateau_se"] < 200
    u = fit_plateau(L, V, "uyeda")
    assert u["ok"] and 1500 < u["plateau"] < 2600 and 0 < u["params"][1] < 1


def test_fit_plateau_fails_cleanly():
    from fastrack.analysis import fit_plateau
    r = fit_plateau([1000, 2000], [1500, 1700], "exp")
    assert not r["ok"] and r["plateau"] == -1
    with pytest.raises(ValueError):
        fit_plateau([1, 2, 3], [1, 2, 3], "linear")


def test_plot_fit_ignores_impossibly_long_filaments(tmp_path):
    """One merged clump 224 um 'long' used to drag the plateau to ~2e5 nm/s."""
    from fastrack.core.motility import Motility
    L, V = _lv_data(n=400)
    rows = np.column_stack([L, V, 0.05 * V, np.full(len(L), 12.0)])
    clump = np.array([[224000.0, 1900.0, 50.0, 12.0]])
    m = Motility()
    m.full_len_vel = np.vstack([rows, clump])
    m.max_len_vel = np.vstack([rows, clump])
    m.directory = str(tmp_path)
    stats = m.plot_length_velocity(min_points=10, percent_tolerance=500, fit_f="exp",
                                   print_plot=False)
    assert 1700 < stats[4] < 2100                 # plateau
    assert m.fit_result["ok"] and m.fit_result["n"] == 400


@pytest.mark.parametrize("units", [False, True])
def test_figure_with_fit_and_units_renders(tmp_path, units):
    from fastrack.core.motility import Motility
    L, V = _lv_data(n=300)
    rows = np.column_stack([L, V, 0.05 * V, np.full(len(L), 12.0)])
    m = Motility()
    m.full_len_vel, m.max_len_vel = rows, rows
    m.directory = str(tmp_path)
    m.plot_length_velocity(header="u%d_" % units, min_points=10, percent_tolerance=500,
                           fit_f="exp", print_plot=True, label_units=units)
    assert (tmp_path / ("u%d_length_velocity.png" % units)).exists()


# --------------------------------------------------------------------------- #
# paths_2D
# --------------------------------------------------------------------------- #
def _paths_motility(tmp_path, n=12, steps=8):
    """n straight tracks moving +x (columns) by 3 px/frame; the last one stuck."""
    from fastrack.core.link import Link, Path
    from fastrack.core.motility import Motility
    paths = []
    for k in range(n):
        links = []
        for t in range(steps):
            lk = Link()
            lk.frame1_no, lk.frame2_no = t, t + 1
            lk.filament1_midpoint = np.array([20 + 15 * k, 10 + 3 * t])     # [row, col]
            lk.filament2_midpoint = np.array([20 + 15 * k, 10 + 3 * (t + 1)])
            lk.instant_velocity = 300.0 + 100 * k
            lk.average_length = 1500.0
            links.append(lk)
        p = Path()
        p.links = links[::-1]                    # latest first, as create_paths stores them
        p.stuck = (k == n - 1)
        paths.append(p)
    m = Motility()
    m.paths, m.width, m.height, m.dx = paths, 256, 256, 80.65
    m.directory = str(tmp_path)
    return m


def test_track_midpoints_run_forward_in_time(tmp_path):
    from fastrack.viz.plots import track_midpoints
    m = _paths_motility(tmp_path)
    xy = track_midpoints(m.paths[0])
    assert len(xy) == 9
    assert np.all(np.diff(xy[:, 0]) == 3)        # columns increase: earliest -> latest
    assert np.all(xy[:, 1] == 20)


@pytest.mark.parametrize("style,labels", [("velocity", False), ("velocity", True), ("legacy", False)])
def test_paths_2D_styles_render(tmp_path, style, labels):
    m = _paths_motility(tmp_path)
    extra = str(tmp_path / "movie")
    m.plot_2D_path_data(5, extra_fname=extra, style=style, show_labels=labels, scale_bar_um=10)
    assert (tmp_path / "paths_2D.png").exists() and (tmp_path / "movie_2D.png").exists()
    assert m.path_stats.shape == (12, 6)


def test_paths_2D_rejects_unknown_style(tmp_path):
    m = _paths_motility(tmp_path)
    with pytest.raises(ValueError):
        m.plot_2D_path_data(5, style="rainbow")


def test_paths_settings_reach_the_pipeline():
    s = Settings()
    kw = s.to_run_kwargs()
    assert (kw["paths_style"], kw["paths_labels"], kw["scale_bar_um"]) == ("velocity", False, 10.0)
    import inspect
    from fastrack.pipelines import gliding
    params = inspect.signature(gliding.run).parameters
    assert {"paths_style", "paths_labels", "scale_bar_um"} <= set(params)


def test_paths_colormap_blue_default_and_matplotlib_names(tmp_path):
    from fastrack.viz.plots import paths_colormap
    assert Settings().to_run_kwargs()["paths_colormap"] == "blue"
    r, g, b, _ = paths_colormap("blue")["cmap"](1.0)
    assert b > r and b > g                              # fast end is blue
    assert paths_colormap("viridis")["cmap"].name == "viridis"
    assert paths_colormap("blue-red")["centred"]
    with pytest.raises(ValueError):
        paths_colormap("no-such-map")
    m = _paths_motility(tmp_path)
    m.plot_2D_path_data(5, colormap="viridis")
    m.plot_2D_path_data(5, colormap="blue-red")
    assert (tmp_path / "paths_2D.png").exists()
