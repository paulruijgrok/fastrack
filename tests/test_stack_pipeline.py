"""End-to-end equivalence: TIFF stacks == split frames (timing held constant).

The ``examples/.../stacks`` tree is the same movies as ``.../micromanager_tifs``
(one multi-page ``.tif`` per movie vs a folder of frames).  Stacks carry no
acquisition clock, so we run BOTH trees with the same uniform ``--frame-rate``
(which overrides metadata.txt for the frame tree too) and require identical
combined statistics -- isolating the pixel/pipeline equivalence from timing.

Skips unless the heavy deps and both example trees are present.
"""
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("scipy", reason="image stack not installed")
pytest.importorskip("skimage", reason="image stack not installed")
pytest.importorskip("cv2", reason="opencv not installed")


def _find(kind):
    hits = glob.glob(os.path.join(ROOT, "examples", "**", kind), recursive=True)
    return next((h for h in hits if os.path.isdir(h)), None)


STACKS = _find("stacks")
FRAMES = _find("micromanager_tifs")
if STACKS is None or FRAMES is None:
    pytest.skip("example stacks/ and micromanager_tifs/ trees not both present",
                allow_module_level=True)


def _float_rows(path):
    rows = []
    with open(path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().lstrip("#").split("\t")
            floats = []
            for p in parts:
                try:
                    floats.append(round(float(p), 4))
                except ValueError:
                    pass
            if floats:
                rows.append(tuple(floats))
    return sorted(rows)


def _run_combined_means(dataset, tmp_path, label):
    from fastrack.pipelines import gliding
    work = tmp_path / label
    work.mkdir()
    cwd = os.getcwd()
    os.chdir(work)
    try:
        gliding.run(
            main_dir=dataset,
            force_analysis=True,
            fast_rank=False,
            morph_contrast=False,
            frame_rate=2.0,          # SAME uniform clock for both -> comparable
        )
        means = sorted(glob.glob("outputs/**/combined/MEAN_values.txt", recursive=True))
    finally:
        os.chdir(cwd)
    assert means, "%s run produced no combined MEAN_values.txt" % label
    return _float_rows(os.path.join(work, means[0]))


def test_stack_equals_frames_with_matched_timing(tmp_path):
    frames = _run_combined_means(FRAMES, tmp_path, "frames")
    stacks = _run_combined_means(STACKS, tmp_path, "stacks")
    assert stacks == frames, "stack vs frame-folder combined statistics differ"
