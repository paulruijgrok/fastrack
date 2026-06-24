"""Tests for the FASTplus directional add-on.

These exercise the numpy-only core (head detection, Kalman-LAP tracking, head<->
filament association, disambiguation, signed scoring, per-frame averaging, and
kinetic fitting) plus the config/registry wiring -- none of which needs the
heavy scikit-image filament path, so they run anywhere numpy is available.
"""
import numpy as np
import pytest

from fastrack.config import Settings, DirectionalSettings
from fastrack.core.detection.heads import detect_spots, log_response
from fastrack.core.tracking import HEAD_TRACKERS
from fastrack.polarity import (DirectionalScorer, HeadFilamentAssociator,
                               PolarityClassifier, SpotRecord, SpotTable)
from fastrack.polarity.datamodel import (BOTH_ENDS, MIDDLE, NONE, PLUS_END,
                                         PolarFilament)
from fastrack.analysis.frame_average import FrameVelocityAggregator
from fastrack.analysis.kinetics import KineticModelFitter, exp_rise


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Fil:
    """Minimal filament stand-in: a straight contour stored as (row, col)."""
    def __init__(self, p0_xy, p1_xy, n=21):
        xs = np.linspace(p0_xy[0], p1_xy[0], n)
        ys = np.linspace(p0_xy[1], p1_xy[1], n)
        self.contour = np.column_stack([ys, xs])      # (row, col)
        self.cm = np.array([ys.mean(), xs.mean()])    # (row, col)
        self.fil_length = float(np.hypot(p1_xy[0] - p0_xy[0], p1_xy[1] - p0_xy[1]))


def _gaussian_image(h, w, spots, amp=120, sigma=2.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), float)
    for (x, y) in spots:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img


# --------------------------------------------------------------------------- #
# config / registry wiring
# --------------------------------------------------------------------------- #
def test_directional_settings_in_sections_and_overrides():
    s = Settings()
    assert isinstance(s.directional, DirectionalSettings)
    s2 = s.with_overrides(mode="filament-centric", head_radius=7.0, max_frame_gap=6)
    assert s2.directional.mode == "filament-centric"
    assert s2.directional.head_radius == 7.0
    assert s2.directional.max_frame_gap == 6
    # base analysis section still reachable via the same flat override path
    s3 = s.with_overrides(stuck_velocity_nm_s=120.0)
    assert s3.analysis.stuck_velocity_nm_s == 120.0


def test_to_directional_kwargs_roundtrip():
    s = Settings().with_overrides(head_quality=9.0, pixel_size_nm=100.0,
                                  perturbation_times_s=(10.0, 20.0))
    kw = s.to_directional_kwargs()
    assert kw["head_quality"] == 9.0
    assert kw["pixel_size_nm"] == 100.0
    assert kw["perturbation_times_s"] == (10.0, 20.0)
    assert kw["mode"] == "head-centric"


def test_registries_have_fastplus_entries():
    from fastrack.core.detection import DETECTORS
    assert "heads-log" in DETECTORS
    assert "kalman-lap" in HEAD_TRACKERS
    from fastrack.pipelines import PIPELINES
    assert "polarity-head-centric" in PIPELINES
    assert "gliding-directional" in PIPELINES


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def test_detect_spots_finds_known_points():
    pts = [(30.0, 40.0), (70.0, 55.0), (50.0, 80.0)]
    img = _gaussian_image(128, 128, pts)
    spots = detect_spots(img, frame=0, gaussian_sigma=1.0, radius=2.0,
                         quality_threshold=5.0)
    assert len(spots) >= len(pts)
    for (px, py) in pts:
        assert any(abs(s.x - px) <= 1.5 and abs(s.y - py) <= 1.5 for s in spots)


def test_log_response_positive_on_bright_blob():
    img = _gaussian_image(64, 64, [(32, 32)])
    resp = log_response(img, 2.0 / np.sqrt(2))
    assert resp[32, 32] > 0


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def test_kalman_lap_tracks_linear_motion_with_gap():
    # one head moving at constant velocity; drop frame 3 to exercise gap closing
    spots = []
    for f in range(8):
        if f == 3:
            continue
        spots.append(SpotRecord(frame=f, x=10.0 + 3 * f, y=20.0 + 1.0 * f, quality=50))
    tracker = HEAD_TRACKERS.create("kalman-lap", initial_search_radius=20,
                                   kalman_search_radius=15, max_frame_gap=4)
    out = tracker.track(spots)
    assert len({s.track_id for s in out}) == 1          # single coherent track


def test_kalman_lap_two_crossing_tracks_stay_separate():
    spots = []
    for f in range(10):
        spots.append(SpotRecord(frame=f, x=5.0 + 4 * f, y=50.0, quality=50))   # ->
        spots.append(SpotRecord(frame=f, x=50.0, y=5.0 + 4 * f, quality=50))   # up
    tracker = HEAD_TRACKERS.create("kalman-lap")
    out = tracker.track(spots)
    assert len({s.track_id for s in out}) == 2


# --------------------------------------------------------------------------- #
# association + disambiguation
# --------------------------------------------------------------------------- #
def _assoc():
    return HeadFilamentAssociator(max_end_distance_px=6.0, end_fraction=0.2)


def test_one_head_on_one_end_is_unambiguous():
    fil = _Fil((10, 10), (10, 50))           # vertical filament, tips at y=10 and y=50
    head = SpotRecord(frame=0, x=10, y=10, quality=50, radius=2)
    head.track_id = 7
    pf = _assoc().associate_frame([fil], [head], 0)[0]
    PolarityClassifier().classify(pf)
    assert pf.classification == PLUS_END
    assert pf.is_unambiguous
    assert pf.head_ids == [7]
    # plus end is the marked tip (10,10); polarity vector points from minus->plus
    assert np.allclose(pf.plus_end_xy, [10, 10], atol=1e-6)
    assert pf.polarity_vector[1] < 0          # points toward smaller y (the head)


def test_heads_on_both_ends_excluded():
    fil = _Fil((10, 10), (10, 50))
    h0 = SpotRecord(frame=0, x=10, y=10, quality=50); h0.track_id = 1
    h1 = SpotRecord(frame=0, x=10, y=50, quality=50); h1.track_id = 2
    pf = _assoc().associate_frame([fil], [h0, h1], 0)[0]
    PolarityClassifier().classify(pf)
    assert pf.classification == BOTH_ENDS
    assert not pf.is_unambiguous


def test_head_in_middle_excluded():
    fil = _Fil((10, 10), (10, 50))
    hm = SpotRecord(frame=0, x=10, y=30, quality=50); hm.track_id = 3   # mid filament
    pf = _assoc().associate_frame([fil], [hm], 0)[0]
    PolarityClassifier().classify(pf)
    assert pf.classification == MIDDLE
    assert not pf.is_unambiguous


def test_no_head_excluded():
    fil = _Fil((10, 10), (10, 50))
    pf = _assoc().associate_frame([fil], [], 0)[0]
    PolarityClassifier().classify(pf)
    assert pf.classification == NONE


# --------------------------------------------------------------------------- #
# signed scoring
# --------------------------------------------------------------------------- #
def test_signed_velocity_plus_and_minus():
    scorer = DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0)
    # axis points +x (minus->plus along +x)
    axis = {f: np.array([1.0, 0.0]) for f in range(5)}
    plus = [SpotRecord(frame=f, x=10.0 + 2 * f, y=5.0) for f in range(5)]   # moving +x
    minus = [SpotRecord(frame=f, x=10.0 - 2 * f, y=5.0) for f in range(5)]  # moving -x
    dp_plus = scorer.score_head_track(0, plus, axis)
    dp_minus = scorer.score_head_track(1, minus, axis)
    assert dp_plus.mean_signed_velocity() > 0
    assert dp_minus.mean_signed_velocity() < 0
    # 2 px/frame * 100 nm/px / 1 s = 200 nm/s
    assert dp_plus.mean_signed_velocity() == pytest.approx(200.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# per-frame averaging
# --------------------------------------------------------------------------- #
def test_frame_average_across_movies():
    from fastrack.polarity.datamodel import DirectionalPath
    agg = FrameVelocityAggregator(dt_s=1.0)
    for _movie in range(3):
        dp = DirectionalPath(path_id=0, frames=[0, 1, 2],
                             signed_velocity_nm_s=[100.0, 200.0, 300.0])
        agg.add_movie([dp])
    st = agg.frame_means()
    assert list(st["frame"]) == [0, 1, 2]
    assert st["mean"][0] == pytest.approx(100.0)
    assert st["n"][0] == 3
    assert agg.n_movies == 3


# --------------------------------------------------------------------------- #
# kinetics
# --------------------------------------------------------------------------- #
def test_kinetic_exp_rise_recovers_tau():
    t = np.linspace(0, 60, 61)
    truth = exp_rise(t, v0=0.0, amp=500.0, tau=10.0, t0=10.0)
    fitter = KineticModelFitter(perturbation_times_s=[10.0])
    res = fitter.fit(t, truth, model="exp_rise")
    assert res["tau"] == pytest.approx(10.0, rel=0.15)
    assert res["amp"] == pytest.approx(500.0, rel=0.15)
    assert res["r2"] > 0.98


# --------------------------------------------------------------------------- #
# end-to-end head-centric core (no scikit-image needed)
# --------------------------------------------------------------------------- #
def test_head_centric_core_end_to_end():
    from fastrack.pipelines.directional import analyze_head_centric

    n_frames = 8
    # a vertical filament drifting in +x; head on its lower-y (plus) tip
    head_frames, filament_frames = [], []
    for f in range(n_frames):
        x = 20.0 + 3.0 * f
        fil = _Fil((x, 20), (x, 60))                  # tips at y=20 and y=60
        filament_frames.append([fil])
        head = SpotRecord(frame=f, x=x, y=20.0, quality=50, radius=2)  # on y=20 tip
        head_frames.append([head])

    tracker = HEAD_TRACKERS.create("kalman-lap")
    paths, qc = analyze_head_centric(
        head_frames, filament_frames,
        scorer=DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0),
        associator=HeadFilamentAssociator(max_end_distance_px=6.0, end_fraction=0.2),
        classifier=PolarityClassifier(), head_tracker=tracker)

    assert qc[PLUS_END] == n_frames        # every frame: one head on one tip
    assert len(paths) == 1
    # head moves +x; polarity axis is along y (toward the head), so motion is
    # perpendicular to polarity -> signed velocity ~ 0
    assert abs(paths[0].mean_signed_velocity()) < 1e-6


def test_find_rgb_movies_recursive_case_insensitive(tmp_path=None):
    import tempfile, os
    from fastrack.pipelines.directional import find_rgb_movies
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "a", "b"); os.makedirs(sub)
    for name in ["m1 RGB.tif", "m2 rgb.tif", "skip_fil.tif", "notes.txt"]:
        open(os.path.join(sub, name), "w").close()
    hits = find_rgb_movies(d)
    assert len(hits) == 2
    assert all(h.lower().endswith("rgb.tif") for h in hits)


def test_run_raises_on_missing_directory():
    from fastrack.pipelines import directional
    with pytest.raises(NotADirectoryError):
        directional.run("/no/such/path/here", mode="head-centric")


def test_head_centric_signed_motion_along_axis():
    from fastrack.pipelines.directional import analyze_head_centric
    # filament oriented along x; head on the +x tip; whole thing translates +x
    head_frames, filament_frames = [], []
    for f in range(6):
        x0 = 20.0 + 4.0 * f
        fil = _Fil((x0, 30), (x0 + 40, 30))           # tips at x0 and x0+40
        filament_frames.append([fil])
        head = SpotRecord(frame=f, x=x0 + 40, y=30.0, quality=50)  # +x tip
        head_frames.append([head])
    tracker = HEAD_TRACKERS.create("kalman-lap")
    paths, qc = analyze_head_centric(
        head_frames, filament_frames,
        scorer=DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0),
        associator=HeadFilamentAssociator(max_end_distance_px=6.0, end_fraction=0.25),
        classifier=PolarityClassifier(), head_tracker=tracker)
    assert len(paths) == 1
    # moving toward the plus end at 4 px/frame * 100 nm = 400 nm/s
    assert paths[0].mean_signed_velocity() == pytest.approx(400.0, rel=0.05)
