"""Smoke tests for FASTrack's pure-numeric helpers.

These exercise the analytic / geometric functions in ``fastrack.motility`` that
do not touch the image pipeline.  They require the package's runtime
dependencies (numpy/scipy/scikit-image/opencv/matplotlib) to be installed, since
``fastrack.motility`` imports them at module load.  Run with::

    pip install -e .[test]
    pytest -q
"""
import numpy as np
import pytest

from fastrack import motility as m


def test_gaussian_peak():
    # At the mean the Gaussian equals its amplitude.
    assert m.gaussian(5.0, 3.0, 5.0, 2.0) == pytest.approx(3.0)
    # Symmetric about the mean.
    assert m.gaussian(4.0, 3.0, 5.0, 2.0) == pytest.approx(m.gaussian(6.0, 3.0, 5.0, 2.0))


def test_fit_gaussian_recovers_params():
    x = np.linspace(200, 1200, 200)
    amp, mu, sd = 40.0, 700.0, 120.0
    y = m.gaussian(x, amp, mu, sd)
    famp, fmu, fsd = m.fit_gaussian(x, y)
    assert famp == pytest.approx(amp, rel=1e-2)
    assert fmu == pytest.approx(mu, rel=1e-2)
    assert abs(fsd) == pytest.approx(sd, rel=1e-2)


def test_length_velocity_monotonic_and_saturating():
    lengths = np.array([36.0, 360.0, 3600.0])
    v = m.length_velocity(lengths, max_vel=800.0, f=0.05)
    # velocity increases with length and stays below the plateau
    assert v[0] < v[1] < v[2] <= 800.0
    # very long filaments approach the plateau
    assert m.length_velocity(1e6, 800.0, 0.05) == pytest.approx(800.0, rel=1e-6)


def test_fit_length_velocity_recovers_plateau():
    length = np.linspace(36, 5000, 300)
    true_max, true_f = 750.0, 0.01
    velocity = m.length_velocity(length, true_max, true_f)
    weights = np.ones_like(length)
    vmax, f, residuals, success = m.fit_length_velocity(length, velocity, weights)
    assert vmax == pytest.approx(true_max, rel=1e-2)
    assert np.allclose(residuals, 0.0, atol=1e-3)


def test_coupling_velocity_and_fit():
    length = np.linspace(50, 4000, 300)
    true = (900.0, 400.0, 600.0)
    velocity = m.coupling_velocity(length, *true)
    weights = np.ones_like(length)
    vmax, amp, tau, residuals, success = m.fit_coupling_velocity(length, velocity, weights)
    assert vmax == pytest.approx(true[0], rel=1e-2)
    assert amp == pytest.approx(true[1], rel=1e-2)
    assert tau == pytest.approx(true[2], rel=1e-2)


def test_vec_length():
    vecs = np.array([[3.0, 4.0], [5.0, 12.0], [0.0, 0.0]])
    assert np.allclose(m.vec_length(vecs), [5.0, 13.0, 0.0])


def test_bin_length_velocity():
    # bin_length_velocity only emits bins up to int(max_len/dx); the top partial
    # bin is dropped (verbatim original behavior).  Include a point in a third
    # bin so the (0,100] and (100,200] bins are both fully formed.
    length = np.array([10.0, 20.0, 110.0, 120.0, 210.0])
    velocity = np.array([1.0, 3.0, 10.0, 20.0, 99.0])
    binned = m.bin_length_velocity(length, velocity, dx=100)
    # two bins: (0,100] and (100,200]; the (200,300] point is dropped
    assert binned.shape == (2, 2)
    assert binned[0, 1] == pytest.approx(2.0)   # mean of 1,3
    assert binned[1, 1] == pytest.approx(15.0)  # mean of 10,20


def test_contour2contour_identical_is_zero():
    contour = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
    assert m.contour2contour(contour, contour.copy(), 1) == pytest.approx(0.0)


def test_contour2contour_shifted():
    c1 = np.array([[0, 0], [0, 1], [0, 2]], dtype=float)
    c2 = c1 + np.array([3.0, 0.0])  # rigid shift by 3 in x
    assert m.contour2contour(c1, c2, 1) == pytest.approx(3.0)


def test_legacy_linking_flag_defaults_off():
    # The corrected (self-consistent) linking is the default.
    mot = m.Motility()
    assert mot.legacy_linking is False
    # The flag is a plain attribute that the driver/CLI can toggle.
    mot.legacy_linking = True
    assert mot.legacy_linking is True


def test_make_frame_links_partner_selection():
    """Corrected vs. legacy partner recovery in make_frame_links.

    Build a frame1 with one filament and a frame2 with two accepted candidates
    whose *sorted-best* (lowest overlap score) is NOT the last one iterated, so
    the corrected path and the legacy (leftover-loop-variable) path pick
    different partners.  We stub ``sim_score`` so no real image data is needed.
    """
    class StubFil:
        def __init__(self, label, cm, length, overlap):
            self.label = label
            self.cm = np.array(cm, dtype=float)
            self.fil_length = length
            self.contour = np.array([cm], dtype=float)
            self.midpoint = np.array(cm, dtype=float)
            self.frame_no = 1
            self.reverse_link = None
            self.forward_link = None
            self._overlap = overlap

        def sim_score(self, other):
            # (overlap, area, distance, fil_direction, mov_direction)
            # area scores well separated so the >1-candidate criterion passes.
            return (other._overlap, other._area, 5.0, 1, 1)

    def build(legacy):
        mot = m.Motility()
        mot.legacy_linking = legacy
        mot.dt = 1.0
        mot.elapsed_times = np.array([])
        mot.max_velocity = 1e9
        mot.max_length_dif = 1e9
        mot.min_fil_length = -1
        mot.overlap_score_cutoff = 0.1
        mot.log_area_score_cutoff = 10.0
        mot.dif_log_area_score_cutoff = 0.5
        mot.frame_links = []
        frame1 = m.Frame(); frame1.frame_no = 0
        frame2 = m.Frame(); frame2.frame_no = 1
        frame1.filaments = [StubFil(0, [0.0, 0.0], 100.0, 0.0)]
        frame1.filaments[0].frame_no = 0
        frame2.filaments = [
            StubFil(0, [1.0, 0.0], 100.0, 0.5),
            StubFil(1, [9.0, 0.0], 100.0, 0.9),
        ]
        frame2.filaments[0]._area = 1.0
        frame2.filaments[1]._area = 100.0
        mot.frame1, mot.frame2 = frame1, frame2
        mot.make_frame_links()
        return mot.frame_links[0][0]

    corrected = build(legacy=False)
    legacy = build(legacy=True)
    # corrected links to the sorted-best partner (label 0, cm x=1)
    assert corrected.filament2_label == 0
    assert corrected.filament2_cm[0] == pytest.approx(1.0)
    # legacy links to the leftover last-iterated partner (label 1, cm x=9)
    assert legacy.filament2_label == 1
    assert legacy.filament2_cm[0] == pytest.approx(9.0)
