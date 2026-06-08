"""Tests for the greedy linker's partner recovery (corrected vs legacy).

These exercise ``GreedyLinker`` directly with stub filaments, so no image data
or scipy/scikit-image is needed.
"""
import numpy as np

from fastrack.core.tracking.greedy import GreedyLinker


class StubFil:
    def __init__(self, label, cm, length, overlap, area):
        self.label = label
        self.cm = np.array(cm, dtype=float)
        self.fil_length = length
        self.contour = np.array([cm], dtype=float)
        self.midpoint = np.array(cm, dtype=float)
        self.frame_no = 0
        self.reverse_link = None
        self.forward_link = None
        self._overlap = overlap
        self._area = area

    def sim_score(self, other):
        # (overlap, area, distance, fil_direction, mov_direction)
        return (other._overlap, other._area, 5.0, 1, 1)


class StubFrame:
    def __init__(self, frame_no, filaments):
        self.frame_no = frame_no
        self.filaments = filaments

    def reset_filament_labels(self):
        for i, f in enumerate(self.filaments):
            f.label = i


def _run(legacy):
    linker = GreedyLinker(
        min_fil_length=-1,
        max_velocity=1e9,
        max_length_dif=1e9,
        overlap_score_cutoff=0.1,
        log_area_score_cutoff=10.0,
        dif_log_area_score_cutoff=0.5,
        legacy=legacy,
    )
    f1 = StubFrame(0, [StubFil(0, [0.0, 0.0], 100.0, 0.0, 1.0)])
    f1.filaments[0].frame_no = 0
    f2 = StubFrame(1, [
        StubFil(0, [1.0, 0.0], 100.0, 0.5, 1.0),    # sorted-best (lowest overlap)
        StubFil(1, [9.0, 0.0], 100.0, 0.9, 100.0),  # last iterated
    ])
    for f in f2.filaments:
        f.frame_no = 1
    links, _ = linker.link(f1, f2, dt=1.0, elapsed_times=np.array([]))
    return links[0]


def test_corrected_recovers_partner_by_label():
    link = _run(legacy=False)
    assert link.filament2_label == 0
    assert link.filament2_cm[0] == 1.0


def test_legacy_uses_leftover_loop_variable():
    link = _run(legacy=True)
    assert link.filament2_label == 1
    assert link.filament2_cm[0] == 9.0
