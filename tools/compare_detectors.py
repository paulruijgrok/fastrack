#!/usr/bin/env python3
"""Run a dataset through FAST twice -- entropy vs. ridge detection -- and report
wall-clock time and how much the velocities differ.

This establishes a baseline comparison (accuracy + speed) between the default
entropy/watershed detector and the optional ridge detector. The ridge detector
requires the optional dependency:  pip install 'fastrack[ridge]'.

Each detector runs in its own working directory so their ``outputs/`` trees do
not collide, then the combined ``MEAN_values.txt`` files are parsed and compared.

Usage:
    python tools/compare_detectors.py -d examples/unloaded_motility/micromanager_tifs
    python tools/compare_detectors.py -d <dataset> -j 8 --ridge-line-widths 3 5
"""
import argparse
import glob
import os
import sys
import time


# Float-column layout of MEAN_values.txt (after slide/exp/filename/protein).
#   0 points-filtered  1 conc(mg/ml)  2 utrophin(nM)  3 top-vel-5  4 p-stuck
#   5 MVEL  6 MVEL-filtered  7 plateau  8 MVIS ...
COL = {"top-vel-5": 3, "p-stuck": 4, "MVEL": 5, "MVEL-filtered": 6, "MVIS": 8}


def parse_mean_file(path):
    rows = {}
    if not path or not os.path.isfile(path):
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
    """Return the canonical combined MEAN file (shortest path; avoids the
    duplicate flattened-path output tree)."""
    matches = glob.glob(
        os.path.join(run_dir, "**", "combined", "MEAN_values.txt"), recursive=True
    )
    if not matches:
        return None
    return sorted(matches, key=lambda p: len(p.split(os.sep)))[0]


def run_mode(dataset_abs, run_dir, detection_algorithm, detection_params, nprocs, verbose):
    """Run FAST in `run_dir`; return (mean_file_path, elapsed_seconds)."""
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    t0 = time.perf_counter()
    try:
        from fastrack.pipelines import gliding
        gliding.run(
            main_dir=dataset_abs,
            force_analysis=True,
            detection_algorithm=detection_algorithm,
            detection_params=detection_params,
            nprocs=nprocs,
            verbose=verbose,
        )
    finally:
        elapsed = time.perf_counter() - t0
        os.chdir(cwd)
    return find_mean_file(run_dir), elapsed


def pct(a, b):
    if a == 0:
        return float("nan")
    return 100.0 * (b - a) / a


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", required=True, help="dataset directory to analyze")
    p.add_argument("-o", default="detector_comparison",
                   help="output directory for the two runs (Default: detector_comparison)")
    p.add_argument("-j", default=None, type=int, help="parallel worker processes")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose FAST output")
    # Ridge params (entropy uses its defaults).
    p.add_argument("--ridge-line-widths", nargs="*", type=int, default=[3])
    p.add_argument("--ridge-low-contrast", type=float, default=50)
    p.add_argument("--ridge-high-contrast", type=float, default=150)
    p.add_argument("--ridge-min-len", type=float, default=10)
    p.add_argument("--ridge-dark-line", action="store_true", default=False)
    args = p.parse_args(argv)

    dataset_abs = os.path.abspath(args.d)
    if not os.path.isdir(dataset_abs):
        sys.exit("Dataset directory does not exist: %s" % dataset_abs)

    ridge_params = dict(
        line_widths=args.ridge_line_widths,
        low_contrast=args.ridge_low_contrast,
        high_contrast=args.ridge_high_contrast,
        min_len=args.ridge_min_len,
        dark_line=args.ridge_dark_line,
        estimate_width=True,
    )

    print("=== Running ENTROPY detector (default) ===")
    ent_file, ent_t = run_mode(dataset_abs, os.path.join(args.o, "entropy"),
                               "entropy", {}, args.j, args.verbose)
    print("=== Running RIDGE detector ===")
    rdg_file, rdg_t = run_mode(dataset_abs, os.path.join(args.o, "ridge"),
                               "ridge", ridge_params, args.j, args.verbose)

    # ----- wall-clock --------------------------------------------------- #
    print("\n" + "=" * 96)
    print("WALL-CLOCK")
    print("=" * 96)
    print("  entropy:  %8.2f s" % ent_t)
    print("  ridge:    %8.2f s" % rdg_t)
    if rdg_t > 0 and ent_t > 0:
        faster, slower = sorted((ent_t, rdg_t))
        label = "ridge" if rdg_t < ent_t else "entropy"
        print("  %s is %.2fx faster" % (label, slower / faster))

    entropy = parse_mean_file(ent_file)
    ridge = parse_mean_file(rdg_file)
    if not entropy and not ridge:
        sys.exit("No MEAN_values.txt produced by either run. Check the dataset / install.")

    # ----- velocity comparison ------------------------------------------ #
    print("\n" + "=" * 96)
    print("VELOCITY COMPARISON  (delta = ridge relative to entropy)")
    print("=" * 96)
    metrics = ["top-vel-5", "MVEL", "MVEL-filtered", "MVIS", "p-stuck"]
    for key in sorted(set(entropy) | set(ridge)):
        e = entropy.get(key)
        r = ridge.get(key)
        protein = (e or r)["protein"]
        print("\n%-70s [%s]" % (key, protein))
        print("  %-16s %14s %14s %12s" % ("metric", "entropy", "ridge", "delta %"))
        for mt in metrics:
            idx = COL[mt]
            ev = e["floats"][idx] if e and idx < len(e["floats"]) else float("nan")
            rv = r["floats"][idx] if r and idx < len(r["floats"]) else float("nan")
            print("  %-16s %14.3f %14.3f %11.2f%%" % (mt, ev, rv, pct(ev, rv)))

    # ----- qualitative paper check both ways ---------------------------- #
    print("\n" + "-" * 96)
    print("Qualitative paper check (alpha-cardiac should be faster than beta-cardiac):")
    for label, data in (("entropy", entropy), ("ridge", ridge)):
        a = [v["floats"][COL["top-vel-5"]] for v in data.values()
             if v["protein"].lower().startswith("alpha")]
        b = [v["floats"][COL["top-vel-5"]] for v in data.values()
             if v["protein"].lower().startswith("beta")]
        if a and b:
            am, bm = sum(a) / len(a), sum(b) / len(b)
            verdict = "PASS" if am > bm else "FAIL"
            print("  %-10s mean top-vel-5:  alpha=%.1f  beta=%.1f  ratio=%.2f  -> %s"
                  % (label, am, bm, am / bm if bm else float("nan"), verdict))
        else:
            print("  %-10s could not find both alpha and beta rows" % label)

    print("\nInterpretation: compare runtime (ridge carries numba JIT warm-up on the")
    print("first frame) and whether the velocities/ordering agree. Large velocity")
    print("deltas mean the detectors disagree on what counts as a filament -- expected")
    print("to some degree, since ridge and entropy segment differently.")


if __name__ == "__main__":
    main()
