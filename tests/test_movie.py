"""Tests for the (generalized) movie writer.

These don't require ffmpeg: the writer returns early when no input frames match
the pattern, which is what we assert.  This guards the pattern/output-name
parameterization added for the overlay movie.
"""
import os

from fastrack.io.movie import MOVIE_WRITERS


def test_writer_registered():
    assert "ffmpeg_h264" in MOVIE_WRITERS


def test_writer_no_frames_is_noop(tmp_path):
    writer = MOVIE_WRITERS.create("ffmpeg_h264")
    d = str(tmp_path)
    # No skeletons_*.png and no overlay_*.png -> both calls are no-ops (no ffmpeg,
    # no output) and must not raise.
    writer.write(d)  # default skeleton pattern
    writer.write(d, input_pattern="overlay_%03d.png", output_name="overlay_tracks.mp4")
    assert not os.path.isfile(os.path.join(d, "filament_tracks.mp4"))
    assert not os.path.isfile(os.path.join(d, "overlay_tracks.mp4"))


def test_writer_pattern_prefix_guard(tmp_path):
    # A frame matching the overlay prefix is present but ffmpeg may be absent;
    # the call must still not raise (it either encodes or prints + returns).
    d = str(tmp_path)
    open(os.path.join(d, "overlay_000.png"), "wb").close()
    writer = MOVIE_WRITERS.create("ffmpeg_h264")
    writer.write(d, input_pattern="overlay_%03d.png", output_name="overlay_tracks.mp4")


def test_writer_moves_movie_to_extra_fname(tmp_path):
    # With extra_fname set, the movie must land at the destination and NOT be
    # left behind in the input/frame directory.
    import shutil as _sh
    if _sh.which("ffmpeg") is None:
        import pytest
        pytest.skip("ffmpeg not available")
    import numpy as np
    import cv2

    frames = tmp_path / "frames"
    out = tmp_path / "out"
    frames.mkdir()
    out.mkdir()
    for i in range(4):
        cv2.imwrite(str(frames / ("overlay_%03d.png" % i)),
                    np.full((40, 60, 3), i * 50, np.uint8))

    dest_prefix = str(out / "movie_")  # extra_fname is a path prefix
    MOVIE_WRITERS.create("ffmpeg_h264").write(
        str(frames), dest_prefix,
        input_pattern="overlay_%03d.png", output_name="overlay_tracks.mp4", fps=5)

    # Lands in outputs, removed from the frame directory, PNGs cleaned up.
    assert os.path.isfile(str(out / "movie_overlay_tracks.mp4"))
    assert not os.path.isfile(str(frames / "overlay_tracks.mp4"))
    assert not list(frames.glob("overlay_*.png"))
