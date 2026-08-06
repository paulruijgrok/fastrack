"""Tests for the standalone low-SNR preprocessing tool (tools/preprocess_lowsnr.py).

Covers the pure core (`preprocess_stack`) and the movie-discovery helpers. The
tool lives under tools/ (not in the package), so it is imported by path.
"""
import os
import sys

import numpy as np
import pytest

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, os.path.abspath(_TOOLS))

pp = pytest.importorskip("preprocess_lowsnr")


def _synthetic_stack(T=8, H=64, W=64):
    """Low-valued stack: flat background + top-bright gradient + one moving spot."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = np.linspace(1.4, 1.0, H)[:, None] * np.ones((1, W))   # brighter at top
    stack = np.empty((T, H, W), np.float32)
    for t in range(T):
        bg = 300.0 * grad + rng.normal(0, 2, (H, W))
        spot = 150.0 * np.exp(-(((yy - (10 + 4 * t)) ** 2 + (xx - 30) ** 2) / 8.0))
        stack[t] = bg + spot
    return stack


def test_preprocess_stack_stretches_to_full_8bit():
    stack = _synthetic_stack()
    out = pp.preprocess_stack(stack, background="median", flat_sigma=20,
                              denoise_sigma=0.8, clip_hi_pct=99.9, bit_depth=8)
    assert out.dtype == np.uint8
    assert out.shape == stack.shape
    assert out.min() == 0                      # background maps to 0
    assert out.max() == 255                    # stretch reaches the top of the range
    assert len(np.unique(out)) > 32            # real dynamic range, not crushed


def test_preprocess_stack_16bit_option():
    stack = _synthetic_stack()
    out = pp.preprocess_stack(stack, bit_depth=16)
    assert out.dtype == np.uint16
    assert out.max() == 65535


def test_background_none_keeps_signal_positive():
    stack = _synthetic_stack()
    out = pp.preprocess_stack(stack, background="none", flat_sigma=0,
                              denoise_sigma=0, clip_hi_pct=99.9, bit_depth=8)
    assert out.dtype == np.uint8 and out.max() == 255


def test_invalid_params_raise():
    stack = _synthetic_stack()
    with pytest.raises(ValueError):
        pp.preprocess_stack(stack, background="bogus")
    with pytest.raises(ValueError):
        pp.preprocess_stack(stack, bit_depth=12)


def test_frame_paths_sorted_by_index(tmp_path):
    import tifffile
    d = tmp_path / "Pos0"
    d.mkdir()
    # write out of order; expect numeric-index ordering
    for i in (2, 0, 10, 1):
        tifffile.imwrite(str(d / f"img_{i:09d}_561 nm_000.tif"),
                         np.zeros((4, 4), np.uint16))
    (d / "not_a_frame.tif").write_bytes(b"")
    got = [int(os.path.basename(p).split("_")[1]) for p in pp.frame_paths(str(d))]
    assert got == [0, 1, 2, 10]


def test_correct_and_stretch_compose_to_preprocess():
    stack = _synthetic_stack()
    a = pp.preprocess_stack(stack, flat_sigma=20)
    b = pp.stretch_to_uint(pp.correct_stack(stack, flat_sigma=20))
    assert np.array_equal(a, b)


def test_smallest_row_band_picks_tight_window():
    d = np.array([0, 0, 5, 5, 5, 0, 0], float)   # total 15
    assert pp._smallest_row_band(d, 12.0) == (2, 5)   # shortest window with sum>=12


def test_estimate_crop_rows_localizes_to_dense_band():
    # method="bright" for a deterministic test (the ridge metric needs real
    # filaments + ridge_detector; it is validated separately on real data).
    rng = np.random.default_rng(0)
    T, H, W = 6, 80, 48
    corr = rng.normal(0, 0.05, (T, H, W)).clip(0).astype(np.float32)
    corr[:, 30:50, :] += 10.0                    # bright band in rows 30-50
    r0, r1 = pp.estimate_crop_rows(corr, frac=0.8, n_sample=4, margin=2,
                                   method="bright")
    assert (r1 - r0) < H                          # actually crops
    assert 28 <= (r0 + r1) / 2 <= 52              # centered on the bright band


def test_crop_reduces_height_and_keeps_uint8():
    rng = np.random.default_rng(1)
    T, H, W = 6, 80, 48
    corr = rng.normal(0, 0.05, (T, H, W)).clip(0).astype(np.float32)
    corr[:, 10:30, :] += 10.0
    r0, r1 = pp.estimate_crop_rows(corr, frac=0.8, n_sample=4, margin=2,
                                   method="bright")
    out = pp.stretch_to_uint(corr[:, r0:r1, :])
    assert out.shape[1] == r1 - r0 < H and out.dtype == np.uint8


def test_estimate_crop_box_rows_only_keeps_full_width():
    rng = np.random.default_rng(3)
    corr = rng.normal(0, 0.05, (6, 80, 64)).clip(0).astype(np.float32)
    corr[:, 20:50, :] += 10.0
    r0, r1, c0, c1 = pp.estimate_crop_box(corr, frac=0.8, n_sample=4, margin=2,
                                          method="bright", crop_cols=False)
    assert (c0, c1) == (0, 64)                    # full width when crop_cols=False
    assert (r1 - r0) < 80


def test_estimate_crop_box_2d_localizes_both_axes():
    rng = np.random.default_rng(4)
    corr = rng.normal(0, 0.05, (6, 80, 80)).clip(0).astype(np.float32)
    corr[:, 20:50, 30:60] += 10.0                 # a 2D blob
    r0, r1, c0, c1 = pp.estimate_crop_box(corr, frac=0.8, n_sample=4, margin=2,
                                          method="bright", crop_cols=True)
    assert (r1 - r0) < 80 and (c1 - c0) < 80      # both axes cropped
    assert 20 <= (r0 + r1) / 2 <= 50              # centered on the blob (rows)
    assert 30 <= (c0 + c1) / 2 <= 60              # centered on the blob (cols)


def test_estimate_crop_rows_ridge_falls_back_without_detector(monkeypatch):
    """method='ridge' without ridge_detector warns and still returns a band."""
    import builtins
    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name == "ridge_detector":
            raise ImportError("simulated missing ridge_detector")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fail)
    rng = np.random.default_rng(2)
    corr = rng.normal(0, 0.05, (4, 60, 40)).clip(0).astype(np.float32)
    corr[:, 20:40, :] += 10.0
    with pytest.warns(RuntimeWarning, match="ridge_detector"):
        r0, r1 = pp.estimate_crop_rows(corr, frac=0.8, n_sample=3, method="ridge")
    assert 0 <= r0 < r1 <= 60


def test_find_movie_dirs(tmp_path):
    import tifffile
    mv = tmp_path / "movieA" / "Pos0"
    mv.mkdir(parents=True)
    tifffile.imwrite(str(mv / "img_000000000_561 nm_000.tif"), np.zeros((4, 4), np.uint16))
    (tmp_path / "empty").mkdir()
    found = pp.find_movie_dirs(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["Pos0"]
