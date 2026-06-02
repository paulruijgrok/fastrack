#!/usr/bin/env python3
"""Run a dataset through FAST twice -- corrected vs. legacy frame-linking --
and print a side-by-side comparison of the resulting velocities.

This quantifies how much the ``make_frame_links`` partner-recovery fix actually
moves the numbers (see README, "Notes / things to review").

Each mode is run in its own working directory so their ``outputs/`` trees do not
collide, then the combined ``MEAN_values.txt`` files are parsed and compared.

Usage:
    python compare_linking.py -d examples/unloaded_motility/micromanager_tifs
    python compare_linking.py -d <dataset> -j 4 -o linking_comparison
"""
import argparse
import glob
import os
import sys


# Float-column layout of MEAN_values.txt (after slide/exp/filename/protein).
#   0 points-filtered  1 conc(mg/ml)  2 utrophin(nM)  3 top-vel-5  4 p-stuck
#   5 MVEL  6 MVEL-filtered  7 plateau  8 MVIS  9 mean-length-all
#  10 mean-length-filtered  11 mean-length-mobile
COL = {"points-filtered": 0, "top-vel-5": 3, "p-stuck": 4, "MVEL": 5,
       "MVEL-filtered": 6, "plateau": 7, "MVIS": 8}


def parse_mean_file(path):
    """Return {filename: {'protein':..., 'floats':[...]}} from a MEAN file."""
    rows = {}
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        entries = line[1:].strip().split("\t")
        if len(entries) < 5:
            continue
        fname = entries[2].strip()
        protein = entries[3].strip()
        try:
            floats = [float(x) for x in entries[4:]]
        except ValueError:
            continue
        rows[fname] = {"protein": protein, "floats": floats}
    return rows


def find_mean_file(run_dir):
    matches = glob.glob(os.path.join(run_dir, "outputs", "*", "combined", "MEAN_values.txt"))
    return matches[0] if matches else None


def run_mode(dataset_abs, run_dir, legacy, nprocs):
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        from fastrack import pipeline
        pipeline.run(
            main_dir=dataset_abs,
            force_analysis=True,
            legacy_linking=legacy,
            nprocs=nprocs,
        )
    finally:
        os.chdir(cwd)
    return find_mean_file(run_dir)


def pct(a, b):
    if a == 0:
        return float("nan")
    return 100.0 * (b - a) / a


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", required=True, help="dataset directory to analyze")
    parser.add_argument("-o", default="linking_comparison",
                        help="output directory for the two runs (Default: linking_comparison)")
    parser.add_argument("-j", default=None, type=int, help="parallel worker processes")
    parser.add_argument("--skip-runs", action="store_true",
                        help="don't re-run FAST; just re-parse existing run directories")
    args = parser.parse_args(argv)

    dataset_abs = os.path.abspath(args.d)
    if not os.path.isdir(dataset_abs):
        sys.exit("Dataset directory does not exist: %s" % dataset_abs)

    corrected_dir = os.path.join(args.o, "corrected")
    legacy_dir = os.path.join(args.o, "legacy")

    if args.skip_runs:
        corrected_file = find_mean_file(corrected_dir)
        legacy_file = find_mean_file(legacy_dir)
    else:
        print("=== Running CORRECTED linking (default) ===")
        corrected_file = run_mode(dataset_abs, corrected_dir, legacy=False, nprocs=args.j)
        print("=== Running LEGACY linking (original Python 2 behaviour) ===")
        legacy_file = run_mode(dataset_abs, legacy_dir, legacy=True, nprocs=args.j)

    corrected = parse_mean_file(corrected_file) if corrected_file else {}
    legacy = parse_mean_file(legacy_file) if legacy_file else {}

    if not corrected and not legacy:
        sys.exit("No MEAN_values.txt produced by either run. Check the dataset / install.")

    all_keys = sorted(set(corrected) | set(legacy))

    print("\n" + "=" * 96)
    print("FRAME-LINKING COMPARISON  (corrected = default, legacy = original)")
    print("=" * 96)
    metrics = ["top-vel-5", "MVEL", "MVEL-filtered", "MVIS"]
    for key in all_keys:
        c = corrected.get(key)
        l = legacy.get(key)
        protein = (c or l)["protein"]
        print("\n%-70s [%s]" % (key, protein))
        print("  %-16s %14s %14s %12s" % ("metric", "corrected", "legacy", "delta %"))
        for mt in metrics:
            idx = COL[mt]
            cv = c["floats"][idx] if c and idx < len(c["floats"]) else float("nan")
            lv = l["floats"][idx] if l and idx < len(l["floats"]) else float("nan")
            print("  %-16s %14.3f %14.3f %11.2f%%" % (mt, cv, lv, pct(cv, lv)))

    # Qualitative paper check: alpha-cardiac should glide faster than beta.
    print("\n" + "-" * 96)
    print("Qualitative paper check (alpha-cardiac should be faster than beta-cardiac):")
    for label, data in (("corrected", corrected), ("legacy", legacy)):
        a = [v["floats"][COL["top-vel-5"]] for k, v in data.items()
             if v["protein"].lower().startswith("alpha")]
        b = [v["floats"][COL["top-vel-5"]] for k, v in data.items()
             if v["protein"].lower().startswith("beta")]
        if a and b:
            am, bm = sum(a) / len(a), sum(b) / len(b)
            verdict = "PASS" if am > bm else "FAIL"
            print("  %-10s mean top-vel-5:  alpha=%.1f  beta=%.1f  ratio=%.2f  -> %s"
                  % (label, am, bm, am / bm if bm else float("nan"), verdict))
        else:
            print("  %-10s could not find both alpha and beta rows" % label)


if __name__ == "__main__":
    main()
