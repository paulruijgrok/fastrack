"""Tests for the filament stores (dependency-light: numpy only).

The key guarantee is that the per-movie (.npz) backend round-trips the exact
same ``filXYs`` content as the original per-frame (.npy) backend -- so switching
``cache_layout`` cannot change the science, only the file count.
"""
import os

import numpy as np

from fastrack.io.stores import STORES


def _sample(n):
    """A filXYs entry like the detectors produce: [contour, width, density, mid]."""
    contour = np.array([[i, i + 1] for i in range(n)], dtype=int)
    return [[contour, float(n), float(10 * n), contour[len(contour) // 2 - 1]]]


def _frames():
    return {0: _sample(3), 1: _sample(5), 2: _sample(2)}


def test_both_layouts_registered():
    assert "npy" in STORES
    assert "per-movie" in STORES


def _write_all(store, d, tag, frames, force=True):
    store.open_write(d, tag, force=force)
    try:
        for no, fx in frames.items():
            store.write(d, tag, no, fx)
    finally:
        store.close()


def test_per_frame_makes_one_file_per_frame(tmp_path):
    d = str(tmp_path)
    _write_all(STORES.create("npy"), d, "", _frames())
    npys = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    assert npys == ["filXYs000.npy", "filXYs001.npy", "filXYs002.npy"]


def test_per_movie_makes_one_file_per_movie(tmp_path):
    d = str(tmp_path)
    _write_all(STORES.create("per-movie"), d, "", _frames())
    files = os.listdir(d)
    assert files == ["filXYs.npz"]                  # exactly one file


def test_layouts_roundtrip_identically(tmp_path):
    frames = _frames()
    contents = {}
    for layout in ("npy", "per-movie"):
        sub = tmp_path / layout
        sub.mkdir()
        d, tag = str(sub), "_ridge"
        store = STORES.create(layout)
        _write_all(store, d, tag, frames)
        assert store.frames(d, tag) == [0, 1, 2]
        out = {}
        for no in frames:
            assert store.has(d, tag, no)
            entry = store.read(d, tag, no)[0]
            out[no] = (entry[0].tolist(), entry[1], entry[2], entry[3].tolist())
        contents[layout] = out
    assert contents["npy"] == contents["per-movie"]   # identical content


def test_per_movie_resume_appends_missing(tmp_path):
    d, tag = str(tmp_path), ""
    store = STORES.create("per-movie")
    # initial cache: frames 0,1
    _write_all(store, d, tag, {0: _sample(3), 1: _sample(4)}, force=True)
    # resume (force=False) appends only the missing frame 2, keeps 0,1
    store.open_write(d, tag, force=False)
    try:
        store.write(d, tag, 2, _sample(5))
    finally:
        store.close()
    assert store.frames(d, tag) == [0, 1, 2]
    assert len(store.read(d, tag, 0)[0][0]) == 3
    assert len(store.read(d, tag, 2)[0][0]) == 5
