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
def test_sign_convention_plus_and_minus_end_labels():
    # polar axis points +x toward the labelled tip (head); head moving +x = LEADING.
    axis = {f: np.array([1.0, 0.0]) for f in range(5)}
    leading = [SpotRecord(frame=f, x=10.0 + 2 * f, y=5.0) for f in range(5)]  # head leads

    # (+)-end label: leading head -> NEGATIVE (motors stroke toward (-) end)
    plus_scorer = DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0, head_marks_end="plus")
    dp = plus_scorer.score_head_track(0, leading, axis)
    assert dp.mean_signed_velocity() == pytest.approx(-200.0, abs=1e-6)

    # (-)-end label: same leading head -> POSITIVE (opposite sign)
    minus_scorer = DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0, head_marks_end="minus")
    dp2 = minus_scorer.score_head_track(0, leading, axis)
    assert dp2.mean_signed_velocity() == pytest.approx(+200.0, abs=1e-6)

    # lagging head flips both
    lagging = [SpotRecord(frame=f, x=10.0 - 2 * f, y=5.0) for f in range(5)]
    assert plus_scorer.score_head_track(0, lagging, axis).mean_signed_velocity() > 0
    assert minus_scorer.score_head_track(0, lagging, axis).mean_signed_velocity() < 0


# --------------------------------------------------------------------------- #
# per-frame averaging
# --------------------------------------------------------------------------- #
def test_frame_percentile_bands():
    from fastrack.polarity.datamodel import DirectionalPath
    agg = FrameVelocityAggregator(dt_s=1.0)
    # frame 0 gets velocities 0..100 across many "movies"
    for v in range(101):
        agg.add_path(DirectionalPath(path_id=0, frames=[0], signed_velocity_nm_s=[float(v)]))
    bands = agg.frame_percentile_bands([(14, 86), (2, 98)])
    (lo1, hi1), (lo2, hi2) = bands
    assert lo1[0] == pytest.approx(14.0, abs=1.0)
    assert hi1[0] == pytest.approx(86.0, abs=1.0)
    assert lo2[0] == pytest.approx(2.0, abs=1.0)
    assert hi2[0] == pytest.approx(98.0, abs=1.0)
    # outer band brackets inner band
    assert lo2[0] <= lo1[0] and hi2[0] >= hi1[0]


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
# --------------------------------------------------------------------------- #
# perturbation / switch frames
# --------------------------------------------------------------------------- #
def test_perturbation_from_led_csv(tmp_path=None):
    import os, tempfile, csv
    from fastrack.analysis.perturbation import from_led_csv
    d = tempfile.mkdtemp()
    # 10 frames, ms timestamps; switches at frames 3 and 7 (after dropping first)
    times = [1000.0 + 100.0 * i for i in range(11)]   # 11 rows (first dropped)
    tcsv = os.path.join(d, "m.csv")
    with open(tcsv, "w", newline="") as f:
        for t in times:
            csv.writer(f).writerow([t])
    # times[1:] -> indices 0..9 map to times[1..10]; pick switch times at idx 3 & 7
    st = [times[1 + 3], times[1 + 7]]
    led = os.path.join(d, "m led.csv")
    # row 1 = LED state BEFORE each switch (0 then on): off->on at f3, on->off at f7
    with open(led, "w", newline="") as f:
        w = csv.writer(f); w.writerow(st); w.writerow([0.0, 1.1])
    pert = from_led_csv(tcsv, led)
    assert pert is not None
    assert list(pert.switch_frames) == [3, 7]
    # state AFTER each switch = toggle of state-before -> on at f3, off at f7
    assert list(pert.states) == [1.0, 0.0]
    assert pert.initial_state == 0.0
    segs = pert.segments(n_frames=10, frame_interval_s=1.0)
    assert [s["model"] for s in segs] == ["exp_rise", "exp_decay"]
    assert segs[0]["t0_s"] == 3.0 and segs[1]["t0_s"] == 7.0 and segs[1]["end_s"] == 10.0


def _has_toml():
    try:
        import tomllib  # noqa: F401
        return True
    except ModuleNotFoundError:
        try:
            import tomli  # noqa: F401
            return True
        except ModuleNotFoundError:
            return False


def test_perturbation_sidecar_toml(tmp_path=None):
    import os, tempfile
    if not _has_toml():
        return  # no TOML parser in this interpreter (py<3.11 without tomli)
    from fastrack.analysis.perturbation import from_sidecar
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.perturb.toml")
    open(p, "w").write("[perturbation]\nswitch_frames = [100, 200, 300, 400]\n"
                       "states = [1, 0, 1, 0]\n")
    pert = from_sidecar(p)
    assert list(pert.switch_frames) == [100, 200, 300, 400]
    assert list(pert.states) == [1, 0, 1, 0]


def test_perturbation_segments_two_cycles():
    from fastrack.analysis.perturbation import from_frames
    # two full rise/decay cycles, dt = 1 s, trace 500 frames
    pert = from_frames([100, 200, 300, 400], states=[1, 0, 1, 0])
    segs = pert.segments(n_frames=500, frame_interval_s=1.0)
    models = [s["model"] for s in segs]
    assert models == ["exp_rise", "exp_decay", "exp_rise", "exp_decay"]
    assert segs[0]["t0_s"] == 100.0 and segs[0]["end_s"] == 200.0
    # on/off pairs
    assert pert.on_off_frames() == [(100, 200), (300, 400)]


def test_perturbation_resolve_prefers_sidecar(tmp_path=None):
    import os, tempfile
    if not _has_toml():
        return  # no TOML parser in this interpreter (py<3.11 without tomli)
    from fastrack.analysis import perturbation as P
    d = tempfile.mkdtemp()
    movie = os.path.join(d, "mymovie 01 RGB.tif")
    open(movie, "w").close()
    open(os.path.join(d, "mymovie 01 RGB.perturb.toml"), "w").write(
        "[perturbation]\nswitch_frames = [42]\nstates=[1]\n")
    pert = P.resolve(movie, source="auto")
    assert list(pert.switch_frames) == [42]
    assert pert.source.startswith("sidecar")


def test_fit_schedule_two_cycles_recovers_taus():
    import numpy as np
    from fastrack.analysis.kinetics import KineticModelFitter, exp_rise, exp_decay
    from fastrack.analysis.perturbation import from_frames
    t = np.linspace(0, 40, 401)            # dt = 0.1 s
    y = np.zeros_like(t)
    y += np.where(t < 10, 0.0, 0.0)
    # cycle 1: rise at t=10 (tau 2), cycle 2: decay at t=25 (tau 3)
    rise = exp_rise(t, 0.0, 500.0, 2.0, 10.0)
    y = rise.copy()
    decay = exp_decay(t, 0.0, rise.max(), 3.0, 25.0)
    y[t >= 25] = decay[t >= 25]
    pert = from_frames([100, 250], states=[1, 0])   # frames at dt=0.1 -> 10s, 25s
    segs = pert.segments(n_frames=401, frame_interval_s=0.1)
    fits = KineticModelFitter.fit_schedule(t, y, segs)
    assert len(fits) == 2
    assert fits[0]["model"] == "exp_rise" and fits[0]["tau"] == pytest.approx(2.0, rel=0.2)
    assert fits[1]["model"] == "exp_decay" and fits[1]["tau"] == pytest.approx(3.0, rel=0.2)


def test_fit_continuous_is_piecewise_continuous():
    import numpy as np
    from fastrack.analysis.kinetics import fit_continuous, exp_rise, exp_decay
    from fastrack.analysis.perturbation import from_frames
    # build a continuous rise(0->500, tau 2) then decay back, dt=0.1
    t = np.linspace(0, 40, 401)
    rise = exp_rise(t, 0.0, 500.0, 2.0, 10.0)
    v = rise.copy()
    v_at_off = exp_rise(np.array([25.0]), 0.0, 500.0, 2.0, 10.0)[0]
    v[t >= 25] = 0.0 + (v_at_off - 0.0) * np.exp(-(t[t >= 25] - 25.0) / 3.0)
    pert = from_frames([100, 250], states=[1, 0])
    segs = pert.segments(401, 0.1)
    res = fit_continuous(t, v, segs)
    assert res is not None and res["r2"] > 0.98
    rise_c, decay_c = res["cycles"]
    assert rise_c["tau"] == pytest.approx(2.0, rel=0.15)
    assert decay_c["tau"] == pytest.approx(3.0, rel=0.15)
    # continuity: decay starts exactly where the rise ended
    assert decay_c["start_level"] == pytest.approx(rise_c["end_level"], abs=1e-9)
    # and the sampled curve has no jump at the switch (t=25 s)
    ct = np.asarray(res["curve_t"]); cv = np.asarray(res["curve_v"])
    j = int(np.argmin(np.abs(ct - 25.0)))
    assert abs(cv[j + 1] - cv[j]) < 50.0     # smooth, no step


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


# --------------------------------------------------------------------------- #
# detection cache
# --------------------------------------------------------------------------- #
def _fil_record(frame, label, n=11):
    from fastrack.datamodel import FilamentRecord
    contour = np.column_stack([np.arange(n), np.full(n, label)]).astype(int)
    return FilamentRecord(frame=frame, label=label, contour=contour,
                          length=float(10 * (label + 1)),
                          cm=np.array([float(n // 2), float(label)]))


def _fil_frames(n_frames=4, per=3):
    return [[_fil_record(f, l) for l in range(per)] for f in range(n_frames)]


def test_detection_cache_roundtrip(tmp_path=None):
    import tempfile
    from fastrack.io.detection_cache import DetectionCache
    for layout in ("per-movie", "per-frame"):
        d = tempfile.mkdtemp()
        params = {"detector": "entropy", "channel": "green"}
        cache = DetectionCache(d, "fil", params, layout=layout)
        frames = _fil_frames(4, 3)

        assert not cache.has_all(4)
        cache.save(frames)
        assert cache.has_all(4), layout
        assert cache.count() == 4

        out = cache.load()
        assert len(out) == 4 and all(len(fr) == 3 for fr in out)
        # records survive the np.save round-trip (contour + scalars intact)
        assert out[2][1].label == 1            # frame 2, second record (label index 1)
        assert out[2][1].length == frames[2][1].length
        assert np.array_equal(out[3][0].contour, frames[3][0].contour)


def test_detection_cache_invalidates_on_param_change(tmp_path=None):
    import tempfile
    from fastrack.io.detection_cache import DetectionCache
    d = tempfile.mkdtemp()
    a = DetectionCache(d, "fil", {"detector": "entropy", "quality": 8}, layout="per-movie")
    a.save(_fil_frames(3, 2))
    assert a.has_all(3)
    # a different parameter -> different tag -> cache miss (no stale reuse)
    b = DetectionCache(d, "fil", {"detector": "entropy", "quality": 5}, layout="per-movie")
    assert not b.has_all(3)
    assert a.tag != b.tag
    # identical params -> same tag -> hit
    c = DetectionCache(d, "fil", {"detector": "entropy", "quality": 8}, layout="per-movie")
    assert c.tag == a.tag and c.has_all(3)


def test_detection_cache_caches_head_spots(tmp_path=None):
    import tempfile
    from fastrack.io.detection_cache import DetectionCache
    d = tempfile.mkdtemp()
    heads = [[SpotRecord(frame=f, x=1.0 * f, y=2.0, quality=50.0, radius=5.0)]
             for f in range(3)]
    cache = DetectionCache(d, "head", {"channel": "red"}, layout="per-movie")
    cache.save(heads)
    out = cache.load()
    assert len(out) == 3 and out[1][0].x == 1.0 and out[1][0].frame == 1


def test_parallel_movies_setting_roundtrip():
    s = Settings().with_overrides(parallel_movies=4)
    assert s.directional.parallel_movies == 4
    assert s.to_directional_kwargs()["parallel_movies"] == 4


def test_across_movie_dispatch_handles_errors_serial_and_pool(tmp_path=None):
    # Two bogus *RGB.tif files: each movie load fails, but the run must not
    # crash — it warns per movie and returns. Exercises the serial loop and the
    # across-movie Pool dispatch + the picklable per-movie worker/results.
    import os, tempfile, warnings
    from fastrack.pipelines import directional
    d = tempfile.mkdtemp()
    for nm in ("m1 RGB.tif", "m2 RGB.tif"):
        open(os.path.join(d, nm), "wb").close()      # empty -> load raises

    for movie_workers in (1, 2):
        out = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = directional.run(main_dir=d, output_dir=out, register_channels=False,
                                  parallel_movies=movie_workers, verbose=False)
        assert res["movies"] == 2          # both discovered
        assert res["qc"] == {}             # both failed cleanly -> nothing aggregated


def test_resolve_workers():
    from fastrack.pipelines.directional import resolve_workers
    import multiprocessing
    assert resolve_workers(1) == 1
    assert resolve_workers(4) == 4
    assert resolve_workers(None) == max(1, multiprocessing.cpu_count())
    assert resolve_workers(0) == 1


def test_pmap_serial_and_parallel_match():
    from fastrack.pipelines.directional import _pmap, _selftest_double
    tasks = list(range(10))
    expected = [x * 2 for x in tasks]
    assert _pmap(_selftest_double, tasks, nprocs=1) == expected   # serial
    assert _pmap(_selftest_double, tasks, nprocs=2) == expected   # pool, ordered


def test_run_raises_on_missing_directory():
    from fastrack.pipelines import directional
    with pytest.raises(NotADirectoryError):
        directional.run("/no/such/path/here", mode="head-centric")


def test_head_centric_collects_polar_by_frame_and_overlay(tmp_path=None):
    import os, tempfile
    from fastrack.pipelines.directional import analyze_head_centric
    from fastrack.polarity.overlay import save_classification_montage
    from fastrack.polarity.datamodel import PLUS_END

    n_frames = 5
    head_frames, filament_frames = [], []
    for f in range(n_frames):
        x0 = 20.0 + 4.0 * f
        fil = _Fil((x0, 30), (x0 + 40, 30))
        filament_frames.append([fil])
        head_frames.append([SpotRecord(frame=f, x=x0 + 40, y=30.0, quality=50)])

    polar_by_frame = {}
    tracker = HEAD_TRACKERS.create("kalman-lap")
    paths, qc = analyze_head_centric(
        head_frames, filament_frames,
        scorer=DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0),
        associator=HeadFilamentAssociator(max_end_distance_px=6.0, end_fraction=0.25),
        classifier=PolarityClassifier(), head_tracker=tracker,
        polar_by_frame=polar_by_frame)

    assert set(polar_by_frame) == set(range(n_frames))
    assert all(pf.classification == PLUS_END
               for pfs in polar_by_frame.values() for pf in pfs)

    # the montage renderer should produce a PNG (matplotlib available)
    stack = np.zeros((n_frames, 64, 80), dtype=np.uint8)
    out = os.path.join(tempfile.mkdtemp(), "qc_overlay.png")
    res = save_classification_montage(stack, polar_by_frame, out, max_frames=n_frames)
    assert res == out and os.path.exists(out) and os.path.getsize(out) > 0


def test_montage_with_filament_contours_renders():
    import os, tempfile
    from fastrack.polarity.overlay import save_classification_montage
    polar_by_frame = {0: [], 1: []}
    fil_by_frame = {0: [_Fil((10, 10), (10, 50))], 1: [_Fil((12, 10), (12, 50))]}
    stack = np.zeros((2, 64, 80), dtype=np.uint8)
    out = os.path.join(tempfile.mkdtemp(), "qc_overlay.png")
    res = save_classification_montage(stack, polar_by_frame, out, max_frames=2,
                                      filament_by_frame=fil_by_frame)
    assert res == out and os.path.exists(out) and os.path.getsize(out) > 0


def test_frame_average_plot_renders_with_fit():
    import os, tempfile
    from fastrack.polarity.overlay import save_frame_average_plot
    from fastrack.analysis.kinetics import KineticModelFitter, exp_rise
    t = np.linspace(0, 60, 61)
    mean = exp_rise(t, v0=0.0, amp=400.0, tau=8.0, t0=10.0)
    stats = {"time_s": t, "mean": mean, "sem": np.full_like(t, 20.0)}
    fit = KineticModelFitter([10.0]).fit(t, mean, model="exp_rise")
    out = os.path.join(tempfile.mkdtemp(), "frame_average.png")
    res = save_frame_average_plot(stats, out, perturbation_times_s=[10.0],
                                  kinetics=[fit])
    assert res == out and os.path.exists(out) and os.path.getsize(out) > 0


def test_head_centric_signed_motion_along_axis():
    from fastrack.pipelines.directional import analyze_head_centric
    # filament along x; head on the labelled tip (+x); whole thing translates +x
    # so the head LEADS. With a (+)-end label, a leading head is NEGATIVE.
    head_frames, filament_frames = [], []
    for f in range(6):
        x0 = 20.0 + 4.0 * f
        fil = _Fil((x0, 30), (x0 + 40, 30))           # tips at x0 and x0+40
        filament_frames.append([fil])
        head = SpotRecord(frame=f, x=x0 + 40, y=30.0, quality=50)  # labelled (+x) tip
        head_frames.append([head])
    tracker = HEAD_TRACKERS.create("kalman-lap")
    paths, qc = analyze_head_centric(
        head_frames, filament_frames,
        scorer=DirectionalScorer(pixel_size_nm=100.0, dt_s=1.0, head_marks_end="plus"),
        associator=HeadFilamentAssociator(max_end_distance_px=6.0, end_fraction=0.25),
        classifier=PolarityClassifier(), head_tracker=tracker)
    assert len(paths) == 1
    # 4 px/frame * 100 nm = 400 nm/s; head leading + (+)-end label -> negative
    assert paths[0].mean_signed_velocity() == pytest.approx(-400.0, rel=0.05)
