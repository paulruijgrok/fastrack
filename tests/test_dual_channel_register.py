"""Tests for io.dual_channel's two input modes.

FASTplus accepts either an already-registered RGB movie (default) or a raw,
spatially-packed movie that it aligns + merges in-process via the optional
optomerge dependency (register=True). optomerge is optional: when it is absent,
register=True must warn and fall back to reading the path as pre-registered RGB.
"""
import numpy as np
import pytest
import tifffile

from fastrack.io.dual_channel import TwoChannelMovie, parse_channel_map


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rgb_stack(T=4, h=32, w=32):
    """A tiny (T, H, W, 3) stack standing in for a pre-registered RGB movie."""
    rng = np.random.default_rng(0)
    return (rng.random((T, h, w, 3)) * 255).astype(np.uint8)


def _raw_packed_movie(path, T=8, half=64, w=128, dy=4, dx=-3):
    """Write a raw spatially-packed movie: top half = filament lines, bottom
    half = head dots shifted by a known (dy, dx). Mirrors the optomerge layout
    ``top_green_fils_bottom_red_heads``."""
    rng = np.random.default_rng(0)
    H = 2 * half
    fil = np.zeros((half, w))
    for c in (20, 50, 90):
        fil[:, c:c + 2] = 1.0
    for r in (15, 40):
        fil[r:r + 2, :] = 0.8
    yy, xx = np.mgrid[0:half, 0:w]
    heads = np.zeros((half, w))
    for (r, c) in [(15, 22), (40, 52), (30, 92), (50, 30)]:
        heads += np.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / (2 * 3.0 ** 2)))
    heads = np.roll(np.roll(heads, dy, axis=0), dx, axis=1)
    raw = np.zeros((T, H, w), np.uint16)
    for t in range(T):
        frame = np.zeros((H, w))
        frame[:half] = fil * 40000 + rng.normal(0, 200, (half, w))
        frame[half:] = heads * 50000 + rng.normal(0, 200, (half, w))
        raw[t] = np.clip(frame, 0, 65535).astype(np.uint16)
    tifffile.imwrite(str(path), raw)


# --------------------------------------------------------------------------- #
# parsing + pre-registered RGB mode (no optomerge needed)
# --------------------------------------------------------------------------- #
def test_parse_channel_map():
    assert parse_channel_map("red=heads,green=filaments") == {
        "heads": "red", "filaments": "green"}


def test_register_off_is_default():
    movie = TwoChannelMovie("<in-memory>")
    assert movie.register_channels is False


def test_split_pre_registered_rgb(tmp_path):
    stack = _rgb_stack()
    path = tmp_path / "movie_RGB.tif"
    tifffile.imwrite(str(path), stack)
    movie = TwoChannelMovie(str(path), head_channel="red", filament_channel="green")
    heads, fils = movie.split()
    assert heads.shape == fils.shape == stack.shape[:3]
    np.testing.assert_array_equal(heads, stack[..., 0])   # red = heads
    np.testing.assert_array_equal(fils, stack[..., 1])    # green = fils
    assert not movie.registered


# --------------------------------------------------------------------------- #
# raw spatially-packed mode (needs optomerge)
# --------------------------------------------------------------------------- #
def test_register_raw_packed_via_optomerge(tmp_path):
    pytest.importorskip("optomerge")
    src = tmp_path / "raw.tif"
    _raw_packed_movie(src)
    movie = TwoChannelMovie(str(src), register=True,
                            channel_order="top_green_fils_bottom_red_heads",
                            max_shift=30)
    heads, fils = movie.split()
    assert movie.registered
    assert heads.ndim == fils.ndim == 3            # (T, H, W)
    assert heads.shape == fils.shape
    assert heads.dtype == np.uint8                 # 8-bit merged RGB by default
    # heads (red) and filaments (green) carry distinct signal after the merge.
    assert heads.max() > 0 and fils.max() > 0


def test_missing_optomerge_falls_back_to_rgb(tmp_path, monkeypatch):
    """register=True without optomerge: warn and read path as pre-registered RGB."""
    import builtins
    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name == "optomerge" or name.startswith("optomerge."):
            raise ImportError("simulated missing optomerge")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fail)

    stack = _rgb_stack()
    path = tmp_path / "already_rgb.tif"
    tifffile.imwrite(str(path), stack)
    movie = TwoChannelMovie(str(path), register=True)
    with pytest.warns(RuntimeWarning, match="optomerge"):
        heads, fils = movie.split()
    assert not movie.registered
    np.testing.assert_array_equal(heads, stack[..., 0])


# --------------------------------------------------------------------------- #
# fastplus --register end-to-end (directional driver)
# --------------------------------------------------------------------------- #
def test_register_broadens_movie_discovery(tmp_path):
    """With --register, discovery matches raw *.tif (not just *RGB.tif)."""
    import warnings
    from fastrack.pipelines import directional
    # An empty, non-RGB-named .tif: it will be discovered but fail to merge/load;
    # the run must still count it (fail-isolated) rather than find nothing.
    (tmp_path / "raw01.tif").write_bytes(b"")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = directional.run(main_dir=str(tmp_path), output_dir=str(tmp_path / "o"),
                              register_channels=True, nprocs=1, verbose=False)
    assert res["movies"] == 1        # found via the broadened '.tif' suffix


def test_fastplus_register_runs_end_to_end(tmp_path):
    """fastplus --register: discover a raw movie, merge via optomerge, run."""
    pytest.importorskip("optomerge")
    pytest.importorskip("skimage")
    import os
    from fastrack.pipelines import directional
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw_packed_movie(raw_dir / "sample01.tif", T=6)      # not named *RGB.tif
    out = tmp_path / "out"
    res = directional.run(
        main_dir=str(raw_dir), output_dir=str(out),
        register_channels=True,
        channel_order="top_green_fils_bottom_red_heads",
        register_max_shift=30, register_frames=None,
        mode="head-centric", head_quality=1.0,
        frame_rate_hz=1.0, nprocs=1, verbose=False,
    )
    # Discovered via the auto '.tif' suffix and run end-to-end; optomerge merged
    # the spatially-packed channels into RGB before detection.
    assert res["movies"] == 1
    assert os.path.isdir(str(out))
