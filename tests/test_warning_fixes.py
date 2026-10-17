"""Regression tests for the numerical warnings cleaned up on 2026-10-16.

Each case used to emit a RuntimeWarning (0/0 or x/0) or a noisy UserWarning;
the behaviour is unchanged, the warning is gone.  pyproject's pytest config
turns RuntimeWarnings from fastrack into errors, so these would fail loudly.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("cv2")


def test_quality_gate_blank_frame_is_bad_without_warning():
    from fastrack.core.frame import Frame
    fr = Frame()
    fr.img = np.zeros((64, 64), dtype=np.uint16)
    fr.width, fr.height = fr.img.shape
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fr.check_picture_quality() == "bad"


def test_exact_rank_percentile_is_quiet_and_unchanged():
    from skimage.filters import rank
    from skimage.morphology import disk
    from fastrack.core.frame import _rank_percentile
    rng = np.random.default_rng(0)
    img = rng.integers(0, 10000, (48, 48)).astype(np.uint16)   # ~13-bit range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = rank.percentile(img, disk(5), p0=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _rank_percentile(img, disk(5), 0.95)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("which", ["coupling", "uyeda"])
def test_fit_residuals_finite_with_zero_weights(which):
    from fastrack.analysis import fitting
    rng = np.random.default_rng(1)
    L = rng.uniform(500, 5000, 200)
    v = 1500 - 600 * np.exp(-L / 1200) + rng.normal(0, 30, 200)
    w = np.ones_like(L)
    w[:20] = 0.0                                   # some zero weights
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if which == "coupling":
            u, amp, tau, res, ok = fitting.fit_coupling_velocity(L, v, w, weighted=True)
            model = fitting.coupling_velocity(L, u, amp, tau)
        else:
            u, r, res, ok = fitting.fit_length_velocity(L, v, w, weighted=True)
            model = fitting.length_velocity(L, u, r)
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res, model - v)


def test_plot_length_velocity_stats_without_warning(tmp_path):
    """The stats path (incl. the former dead weighted fit) runs warning-free
    and returns the same MVEL numbers as computed by hand."""
    from fastrack.core.motility import Motility
    rng = np.random.default_rng(2)
    n = 300
    L = rng.uniform(800, 6000, n)
    V = 1500 - 500 * np.exp(-L / 1500) + rng.normal(0, 40, n)
    full = np.column_stack([L, V, 0.1 * V, np.full(n, 8.0)])
    m = Motility()
    m.full_len_vel = full.copy()
    m.max_len_vel = full.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stats = m.plot_length_velocity(print_plot=False, fit_f="none", min_points=10)
    assert stats[3] == pytest.approx(V.mean())        # MVEL_filtered
    assert stats[4] == -1                              # no fit requested
