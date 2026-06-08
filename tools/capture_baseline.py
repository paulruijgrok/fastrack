#!/usr/bin/env python3
"""Capture a golden-master baseline of FAST outputs *before* refactoring.

Runs the analysis on a dataset with the exact, deterministic settings
(``--exact-rank`` path, no morphological approximation) and snapshots the
numeric result files plus a sha256 manifest into a baseline directory.  After
the refactor, a pytest golden test re-runs the same config and diffs the
outputs against this snapshot to prove functionality is preserved.

Only deterministic text files are captured (``*_values.txt`` and the per-movie
length-velocity tables).  PNG plots and the movie are intentionally skipped --
they are not byte-stable across runs/matplotlib versions.

Usage:
    python capture_baseline.py -d examples/unloaded_motility/micromanager_tifs
    python capture_baseline.py -d <dataset> -o tests/baseline -j 8
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", required=True, help="dataset directory to analyze")
    p.add_argument("-o", default="tests/baseline", help="baseline output dir (Default: tests/baseline)")
    p.add_argument("-j", default=None, type=int, help="parallel worker processes")
    args = p.parse_args(argv)

    dataset = os.path.abspath(args.d)
    if not os.path.isdir(dataset):
        sys.exit("Dataset directory does not exist: %s" % dataset)

    # Exact, deterministic path: native 16-bit percentile filters, no morph
    # approximation.  This is the strongest invariant to pin the refactor to.
    from fastrack.pipelines import gliding
    gliding.run(
        main_dir=dataset,
        force_analysis=True,
        fast_rank=False,
        morph_contrast=False,
        nprocs=args.j,
    )

    # Collect deterministic numeric outputs only.
    patterns = [
        "outputs/**/combined/MEAN_values.txt",
        "outputs/**/combined/SEM_values.txt",
        "outputs/**/*_length_velocity.txt",
        "outputs/**/*_max_length_velocity.txt",
    ]
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    files = sorted(set(files))
    if not files:
        sys.exit("No output value files found under outputs/. Did the run produce results?")

    if os.path.isdir(args.o):
        shutil.rmtree(args.o)
    os.makedirs(args.o, exist_ok=True)

    manifest = os.path.join(args.o, "MANIFEST.sha256")
    with open(manifest, "w") as mf:
        for fpath in files:
            rel = os.path.relpath(fpath, "outputs")
            dest = os.path.join(args.o, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(fpath, dest)
            mf.write("%s  %s\n" % (sha256(fpath), rel))

    print("Captured %d baseline file(s) into %s" % (len(files), args.o))
    print("Manifest: %s" % manifest)
    print("\nCommit this directory so the post-refactor golden test can diff against it:")
    print("    git add %s && git commit -m 'Add golden-master baseline'" % args.o)


if __name__ == "__main__":
    main()
