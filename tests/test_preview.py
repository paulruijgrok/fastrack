"""``fast-preview``: the quick-look pipeline (pipelines/preview.py).

Unit tests for binning / parameter scaling / quality scoring need no optional
dependency.  The end-to-end tests run a synthetic movie with known answers
(6 filaments gliding at 4 px/frame, 3 stuck filaments, all 60 px long) through
the API and the CLI; they need a ridge detector (``ridge-detector`` or
``ridge-detector-fast``).
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
tifffile = pytest.importorskip("tifffile")

from fastrack.pipelines import preview as P  # noqa: E402

PX = 80.65


def _has_ridge():
    for mod in ("ridge_detector_fast", "ridge_detector"):
        try:
            __import__(mod)
            return True
        except ImportError:
            pass
    return False


needs_ridge = pytest.mark.skipif(not _has_ridge(), reason="no ridge detector installed")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_bin_means_blocks_and_crops_remainder():
    img = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    out = P._bin(img, 2)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(np.mean([0, 1, 7, 8]))
    assert P._bin(img, 1).dtype == np.float32


def test_ridge_params_scale_with_binning():
    p = P._ridge_params(P.DEFAULT_RIDGE, 2)
    assert p["line_widths"] == [2] and p["min_len"] == pytest.approx(4.0)
    assert p["low_contrast"] == P.DEFAULT_RIDGE["low_contrast"]       # contrast unchanged
    assert P._ridge_params(P.DEFAULT_RIDGE, 1) == P.DEFAULT_RIDGE


def test_score_is_linear_between_anchors_in_both_directions():
    assert P._score(12.0, 12.0, 4.0) == 1.0
    assert P._score(8.0, 12.0, 4.0) == pytest.approx(0.5)
    assert P._score(1.0, 12.0, 4.0) == 0.0
    assert P._score(25.0, 15.0, 35.0) == pytest.approx(0.5)            # lower is better
    assert P._score(None, 1, 0) is None and P._score(np.nan, 1, 0) is None


def test_quality_ranking_worst_first_and_verdict():
    good = dict(snr=20, focus_width_nm=270, flatness=0.95, saturation_pct=0.0,
                background_to_signal=3, density=3, stuck_pct=5, floppy_pct=5, pinned_pct=0,
                bleach_pct_per_frame=0.2, step_over_length=0.4)
    rows = P.score_quality(good)
    assert P.verdict(rows) == "GOOD"
    assert len(rows) == len(P.QUALITY_SPECS)
    bad = dict(good, snr=3.0, stuck_pct=25.0)
    rows = P.score_quality(bad)
    assert [r["status"] for r in rows[:2]] == ["poor", "marginal"]
    assert "Signal" in rows[0]["label"] or "signal" in rows[0]["label"]
    assert P.verdict(rows) == "POOR"
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores)
    assert P.verdict([]) == "unknown"


def test_missing_metric_is_reported_not_scored():
    rows = P.score_quality(dict(snr=20))
    assert P.verdict(rows) == "GOOD"
    assert any(r["score"] is None for r in rows) or len(rows) < len(P.QUALITY_SPECS)


# --------------------------------------------------------------------------- #
# end to end on a synthetic movie with known answers
# --------------------------------------------------------------------------- #
def _synthetic(root, T=20, H=384, W=384, step=4, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "movA", "Pos0")
    os.makedirs(d)
    for t in range(T):
        img = np.zeros((H, W))
        for k in range(6):                                   # gliding
            r, c = 30 + k * 55, 10 + step * t + k * 7
            img[r, c:c + 60] = 1
        for k in range(3):                                   # stuck
            c = 320 + k * 20
            img[40 + k * 100:100 + k * 100, c] = 1
        img = 300 + gaussian_filter(img, 1.5) * 3000 + rng.normal(0, 6, (H, W))
        tifffile.imwrite(os.path.join(d, "img_%09d_561 nm_000.tif" % t),
                         np.clip(img, 0, 65535).astype(np.uint16))
    return root


@needs_ridge
def test_preview_recovers_known_velocity_length_and_stuck(tmp_path):
    """4 px/frame x 20 frames is only 1.3 filament lengths: a temporal-median
    background would contain the gliding filaments and cut them short (half
    the length and velocity); the low-order background must not."""
    root = _synthetic(str(tmp_path / "data"))
    out = str(tmp_path / "out")
    res = P.run(root, P.PreviewSettings(nprocs=1), out_dir=out, plot=True, quiet=True)
    (name, r), = res.items()
    assert r["timing_source"] == "assumed 1 s/frame"
    assert r["velocity_nm_s"]["mean_moving"] == pytest.approx(4 * PX, rel=0.05)
    assert r["length_um"]["median"] == pytest.approx(60 * PX / 1000, rel=0.15)
    assert r["static_filaments"] == 3
    assert r["moving_per_frame"] == pytest.approx(6, abs=0.5)
    assert r["stuck_pct"] == pytest.approx(100 / 3, abs=5)
    assert r["metrics"]["floppy_pct"] == 0.0          # straight gliding stays on track
    assert r["metrics"]["pinned_pct"] == 0.0
    assert set(r["seconds"]) >= {"read", "preprocess", "detect", "link", "total"}
    safe = name.replace(os.sep, "__")
    for f in (safe + "_preview.json", safe + "_preview.png", "preview_summary.csv"):
        assert os.path.exists(os.path.join(out, f)), f
    js = json.load(open(os.path.join(out, safe + "_preview.json")))
    assert js["verdict"] == r["verdict"] and js["settings"]["bin"] == 2


@needs_ridge
def test_latest_skips_the_newest_file_and_cli_runs(tmp_path, capsys):
    from fastrack.cli import fast_preview_main
    root = _synthetic(str(tmp_path / "data"), T=24)
    r, = P.run(root, P.PreviewSettings(nprocs=1, latest=True), quiet=True).values()
    assert r["frames"][0] == 3 and r["frames"][-1] == 22        # 20 frames, newest skipped
    out = str(tmp_path / "cli")
    fast_preview_main([root, "-n", "10", "-j", "1", "-o", out, "--no-plot",
                       "--frame-rate", "2"])
    assert "velocity" in capsys.readouterr().out
    rows = open(os.path.join(out, "preview_summary.csv")).read().splitlines()
    assert len(rows) == 2
