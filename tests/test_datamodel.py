"""Tests for the filament data model and CSV export (no heavy deps required)."""
import csv
import os
import tempfile

import numpy as np

from fastrack.datamodel import FilamentRecord, FilamentTable
from fastrack.io.export import to_csv


class _FakeFilament:
    def __init__(self, frame_no, label, contour, length):
        self.frame_no = frame_no
        self.label = label
        self.contour = contour
        self.fil_length = length
        self.fil_width = 4.0
        self.fil_density = 900.0
        self.fil_area = 12.0
        self.end2end = 2.5
        self.midpoint = np.array([contour[0][0], contour[0][1]])


def test_record_from_filament():
    f = _FakeFilament(2, 3, np.array([[1, 1], [2, 2], [3, 3]]), 120.5)
    r = FilamentRecord.from_filament(f)
    assert r.frame == 2 and r.label == 3
    assert r.length == 120.5
    assert len(r.contour) == 3
    row = r.to_row()
    assert row["n_points"] == 3 and row["length"] == 120.5


def test_table_collects_and_exports():
    frames = [
        type("F", (), {"filaments": [_FakeFilament(0, 0, np.zeros((2, 2), int), 10.0)]})(),
        type("F", (), {"filaments": [
            _FakeFilament(1, 0, np.zeros((3, 2), int), 20.0),
            _FakeFilament(1, 1, np.zeros((4, 2), int), 30.0),
        ]})(),
    ]
    table = FilamentTable.from_frames(frames)
    assert len(table) == 3
    assert table.frames() == [0, 1]
    assert len(table.by_frame(1)) == 2

    path = os.path.join(tempfile.mkdtemp(), "filaments.csv")
    to_csv(table, path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["frame"] for r in rows} == {"0", "1"}
    assert rows[-1]["length"] == "30.0"
