"""End-to-end equivalence: running on TIFF stacks == running on split frames.

The ``examples/.../stacks`` tree is the same movies as ``.../micromanager_tifs``
(one multi-page ``.tif`` per movie instead of a folder of frames), so the full
pipeline must produce the same combined velocity statistics -- i.e. it must
reproduce the committed golden baseline.  Skips unless the heavy deps, the
stacks dataset, and the baseline are all present.
"""
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, "tests", "baseline")

pytest.importorskip("scipy", reason="image stack not installed")
pytest.importorskip("skimage", reason="image stack not installed")
pytest.importorskip("cv2", reason="opencv not installed")


def _find_stacks_dataset():
    """The dataset dir to pass as -d: a tree containing multi-page .tif movies."""
    hits = glob.glob(os.path.join(ROOT, "examples", "**", "stacks"), recursive=True)
    return next((h for h in hits if os.path.isdir(h)), None)


def _float_rows(path):
    """Sorted set of float-only rows from a *_values.txt (order-independent)."""
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


DATASET = _find_stacks_dataset()
if DATASET is None:
    pytest.skip("stacks example dataset not present", allow_module_level=True)
if not os.path.isfile(os.path.join(BASELINE_DIR, "MANIFEST.sha256")):
    pytest.skip("golden baseline not captured", allow_module_level=True)


def test_stack_run_matches_baseline(tmp_path):
    from fastrack.pipelines import gliding

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        gliding.run(
            main_dir=DATASET,
            force_analysis=True,
            fast_rank=False,        # exact reference path, as the baseline was captured
            morph_contrast=False,
        )
        produced = sorted(
            glob.glob("outputs/**/combined/MEAN_values.txt", recursive=True))
    finally:
        os.chdir(cwd)

    assert produced, "stack run produced no combined MEAN_values.txt"

    base = [
        p for p in glob.glob(
            os.path.join(BASELINE_DIR, "**", "combined", "MEAN_values.txt"),
            recursive=True)
        if os.path.relpath(p, BASELINE_DIR).count(os.sep) == 2     # canonical tree
    ]
    assert base, "no canonical baseline MEAN file"

    # Compare numeric content only (the filename column differs: stacks_* vs
    # micromanager_tifs_*), order-independent: same movies -> same float rows.
    got = _float_rows(os.path.join(tmp_path, produced[0]))
    expected = _float_rows(base[0])
    assert got == expected, "stack-run velocities differ from the frame baseline"
