"""Tests for the trajectory CSV export.

``write_rows_csv`` and the column schemas are dependency-light.  The row
builders live on :class:`Motility`, so the end-to-end builder test importorskips
the image stack and drives synthetic ``Path``/``Link`` objects (no real movie).
"""
import csv
import math

import numpy as np
import pytest

from fastrack.io.export import (
    TRAJECTORY_COLUMNS, CONTOUR_COLUMNS, write_rows_csv, append_rows_csv,
)


def test_write_rows_csv_roundtrips(tmp_path):
    rows = [
        {"movie": "m", "path_id": 0, "frame": 0, "x_nm": 1.5, "extra": "ignored"},
        {"movie": "m", "path_id": 0, "frame": 1, "x_nm": 2.5},
    ]
    cols = ["movie", "path_id", "frame", "x_nm"]
    p = tmp_path / "t.csv"
    n = write_rows_csv(rows, str(p), cols)
    assert n == 2
    with open(p) as f:
        got = list(csv.DictReader(f))
    assert list(got[0].keys()) == cols          # extra key dropped, order stable
    assert got[1]["x_nm"] == "2.5"


def test_append_builds_one_combined_file(tmp_path):
    """Appending two movies yields a single file with one header and all rows."""
    cols = ["movie", "path_id", "frame"]
    p = tmp_path / "all_trajectories.csv"
    movie_a = [{"movie": "a", "path_id": 0, "frame": 0},
               {"movie": "a", "path_id": 0, "frame": 1}]
    movie_b = [{"movie": "b", "path_id": 0, "frame": 0}]
    append_rows_csv(movie_a, str(p), cols, write_header=True)    # first movie
    append_rows_csv(movie_b, str(p), cols, write_header=False)   # second movie
    with open(p) as f:
        lines = f.read().splitlines()
    assert lines[0] == "movie,path_id,frame"          # exactly one header
    assert len(lines) == 1 + 3                          # header + 3 data rows
    assert {l.split(",")[0] for l in lines[1:]} == {"a", "b"}


def _make_motility():
    """Build a Motility with one synthetic 2-frame trajectory (no movie needed)."""
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    pytest.importorskip("cv2")
    from fastrack.core.motility import Motility
    from fastrack.core.link import Link, Path

    lk = Link()
    lk.frame1_no, lk.frame2_no = 0, 1
    lk.filament1_time, lk.filament2_time = 0.0, 0.5
    lk.filament1_length, lk.filament2_length = 10.0, 12.0
    lk.filament1_midpoint = np.array([2, 4])     # [row, col]
    lk.filament2_midpoint = np.array([2, 8])
    lk.filament1_cm = np.array([2, 4])
    lk.filament2_cm = np.array([2, 8])
    lk.filament1_contour = np.array([[0, 0], [1, 1]])
    lk.filament2_contour = np.array([[0, 0], [1, 1], [2, 2]])
    lk.instant_velocity = 3.0                    # pretend already nm/s

    path = Path()
    path.links = [lk]
    path.stuck = False

    m = Motility()
    m.dx = 10.0                                  # nm / pixel
    m.paths = [path]
    return m


def test_trajectory_rows_units_and_shape():
    m = _make_motility()
    rows = m.trajectory_rows(movie="mov")
    assert [r["frame"] for r in rows] == [0, 1]   # source frame + closing frame
    assert all(r["path_id"] == 0 and r["movie"] == "mov" for r in rows)
    r0, r1 = rows
    assert r0["length_nm"] == 100.0               # 10 px * dx 10
    assert (r0["x_nm"], r0["y_nm"]) == (40.0, 20.0)  # x=col*dx, y=row*dx
    assert r0["velocity_nm_s"] == 3.0
    assert r0["stuck"] == 0 and r0["n_points"] == 2
    assert r1["frame"] == 1 and r1["x_nm"] == 80.0 and r1["n_points"] == 3
    assert math.isnan(r1["velocity_nm_s"])        # last frame has no outgoing link
    assert set(TRAJECTORY_COLUMNS).issubset(rows[0].keys())


def test_contour_rows_long_format():
    m = _make_motility()
    pts = list(m.contour_rows(movie="mov"))
    # 2 points (frame 0) + 3 points (frame 1)
    assert len(pts) == 5
    assert [p["frame"] for p in pts] == [0, 0, 1, 1, 1]
    assert [p["point"] for p in pts[:2]] == [0, 1]
    # x = col*dx, y = row*dx ; contour point [2,2] -> (20, 20)
    assert (pts[-1]["x_nm"], pts[-1]["y_nm"]) == (20.0, 20.0)
    assert set(CONTOUR_COLUMNS) == set(pts[0].keys())
