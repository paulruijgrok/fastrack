"""Tests for the frame-source seam (movie input).

The pixel-quantization, prober dispatch and descriptor round-trip are
dependency-light. The stack-vs-frames equivalence runs against the example data
(``examples/.../stacks/..._2.tif`` mirrors ``.../micromanager_tifs/..._2/``) and
skips if that data isn't present.
"""
import glob
import os

import numpy as np
import pytest

from fastrack.core.input import open_movie, open_source, to_pipeline_image
from fastrack.core.input.mm_dir import MicroManagerDirSource
from fastrack.core.input.tiff_stack import TiffStackSource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_to_pipeline_image_matches_legacy_read():
    # 16-bit -> (>>8) -> (*257), reproducing cv2 IMREAD_GRAYSCALE + img_as_uint
    page = np.array([[0, 256, 512, 65535]], dtype=np.uint16)
    out = to_pipeline_image(page)
    assert out.dtype == np.uint16
    assert out.tolist() == [[0, 1 * 257, 2 * 257, 255 * 257]]
    # 8-bit input skips the downshift
    u8 = np.array([[0, 1, 255]], dtype=np.uint8)
    assert to_pipeline_image(u8).tolist() == [[0, 257, 65535]]


def test_prober_and_descriptor_roundtrip(tmp_path):
    # a folder of mm frames -> MicroManagerDirSource
    movie = tmp_path / "_1"
    movie.mkdir()
    for i in range(3):
        open(movie / ("img_000000%03d__000.tif" % i), "wb").close()
    src = open_movie(str(movie))
    assert isinstance(src, MicroManagerDirSource)
    assert src.count == 3 and src.frame_numbers() == [0, 1, 2]
    # descriptor round-trips to an equivalent source
    src2 = open_source(src.descriptor())
    assert isinstance(src2, MicroManagerDirSource)
    assert src2.directory == src.directory and src2.tail == src.tail

    # a .tif file -> TiffStackSource; forcing works too
    stub = tmp_path / "movie.tif"
    stub.write_bytes(b"II*\x00")
    assert isinstance(open_movie(str(stub)), TiffStackSource)
    assert isinstance(open_movie(str(movie), input_format="stack"), TiffStackSource)
    d = open_movie(str(stub)).descriptor()
    assert open_source(d).path == os.path.abspath(str(stub))


def _example_pair():
    stacks = glob.glob(os.path.join(
        ROOT, "examples", "**", "stacks", "**", "alpha_0.04mg_ml", "_2.tif"), recursive=True)
    frames = glob.glob(os.path.join(
        ROOT, "examples", "**", "micromanager_tifs", "**", "alpha_0.04mg_ml", "_2"), recursive=True)
    frames = [f for f in frames if os.path.isdir(f)]
    if not stacks or not frames:
        return None
    return stacks[0], frames[0]


def test_stack_reads_equal_frame_folder_reads():
    pair = _example_pair()
    if pair is None:
        pytest.skip("example stack/frame data not present")
    stack_path, frame_dir = pair
    s = TiffStackSource(stack_path)
    d = MicroManagerDirSource(frame_dir)
    assert s.count == d.count, "frame counts differ"
    # the stack and its pre-split frames must read identically (same pixels,
    # same legacy quantization) -- this is the core correctness guarantee.
    for n in (0, s.count // 2, s.count - 1):
        a = s.read(n)
        b = d.read(n)
        assert a.shape == b.shape and a.dtype == np.uint16
        assert np.array_equal(a, b), "stack vs frame-folder pixels differ at frame %d" % n
