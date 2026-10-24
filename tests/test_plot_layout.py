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
