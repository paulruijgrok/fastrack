"""Golden-master regression test.

Re-runs the analysis on the example dataset with the exact (deterministic)
settings and asserts the numeric outputs match the committed baseline in
``tests/baseline`` (captured before the refactor by ``tools/capture_baseline.py``).

This is the primary proof that the refactor preserved functionality.  It is
skipped automatically when the heavy dependencies, the example dataset, or the
baseline are unavailable (e.g. in CI without the >1 GB dataset), so the rest of
the suite still runs.  To exercise it, populate ``examples/`` and run::

    pytest tests/test_golden.py -q
"""
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(ROOT, "tests", "baseline")
DATASET = os.path.join(ROOT, "examples", "unloaded_motility", "micromanager_tifs")

pytest.importorskip("scipy", reason="image stack not installed")
pytest.importorskip("skimage", reason="image stack not installed")
pytest.importorskip("cv2", reason="opencv not installed")

if not os.path.isdir(DATASET):
    pytest.skip("example dataset not present", allow_module_level=True)
if not os.path.isfile(os.path.join(BASELINE_DIR, "MANIFEST.sha256")):
    pytest.skip("baseline not captured", allow_module_level=True)


def _read_values(path):
    """Parse a *_values.txt table into a list of float lists (order-stable)."""
    rows = []
    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.lstrip("#").split("\t")
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    pass
            rows.append(floats)
    return rows


def test_outputs_match_baseline(tmp_path):
    from fastrack.pipelines import gliding

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        gliding.run(
            main_dir=DATASET,
            force_analysis=True,
            fast_rank=False,        # exact 16-bit reference path
            morph_contrast=False,
        )
        produced = sorted(
            glob.glob("outputs/**/combined/MEAN_values.txt", recursive=True)
            + glob.glob("outputs/**/combined/SEM_values.txt", recursive=True)
        )
    finally:
        os.chdir(cwd)

    assert produced, "pipeline produced no combined value files"

    # Pair each baseline file with the produced file at the SAME path relative to
    # the outputs root.  (Basenames are not unique: the dataset produces both a
    # clean output tree and a duplicate flattened-path tree, so pairing by
    # basename would compare mismatched files.)  Baseline files were stored
    # relative to "outputs", so the produced counterpart is tmp_path/outputs/<rel>.
    baseline_files = glob.glob(
        os.path.join(BASELINE_DIR, "**", "combined", "*_values.txt"), recursive=True
    )
    # Compare only the canonical combined tree (``<outdir>/combined/<file>``).
    # The dataset also emits a duplicate tree under a flattened full-path name;
    # that is a pre-existing output quirk and isn't the scientific deliverable,
    # so it doesn't gate the regression test.
    baseline_files = [
        p for p in baseline_files
        if os.path.relpath(p, BASELINE_DIR).count(os.sep) == 2
    ]
    assert baseline_files, "no canonical baseline value files found"

    checked = 0
    for baseline_path in baseline_files:
        rel = os.path.relpath(baseline_path, BASELINE_DIR)
        produced_path = os.path.join(tmp_path, "outputs", rel)
        assert os.path.isfile(produced_path), "refactor did not produce %s" % rel
        new_rows = _read_values(produced_path)
        old_rows = _read_values(baseline_path)
        assert len(new_rows) == len(old_rows), "row count differs in %s" % rel
        for new, old in zip(new_rows, old_rows):
            assert new == pytest.approx(old, rel=1e-6, abs=1e-6), (
                "values differ in %s" % rel
            )
        checked += 1
    assert checked > 0
