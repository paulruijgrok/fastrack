#!/usr/bin/env python3
"""Run a dataset through FAST twice -- default 16-bit rank filters vs. the
``--fast-rank`` 8-bit path -- and report both the wall-clock speedup and how
much the velocities move.

The percentile (rank) filters in ``entropy_clusters`` / ``check_picture_quality``
are the dominant per-frame cost; ``fast_rank`` rescales each frame to 8-bit
before them, shrinking scikit-image's local histogram from 65536 to 256 bins.
This script quantifies the resulting speed/accuracy trade-off on real data.

Each mode runs in its own working directory so their ``outputs/`` trees do not
collide, then the combined ``MEAN_values.txt`` files are parsed and compared.

Usage:
    python compare_fast_rank.py -d examples/unloaded_motility/micromanager_tifs
    python compare_fast_rank.py -d <dataset> -j 4 -o fast_rank_comparison
"""
import argparse
import glob
import os
import sys
import time


# Float-column layout of MEAN_values.txt (after slide/exp/filename/protein).
#   0 points-filtered  1 conc(mg/ml)  2 utrophin(nM)  3 top-vel-5  4 p-stuck
#   5 MVEL  6 MVEL-filtered  7 plateau  8 MVIS  9 mean-length-all ...
COL = {"top-vel-5": 3, "MVEL": 5, "MVEL-filtered": 6, "MVIS": 8}


def parse_mean_file(path):
    """Return {filename: {'protein':..., 'floats':[...]}} from a MEAN file."""
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
    matches = glob.glob(
        os.path.join(run_dir, "**", "combined", "MEAN_values.txt"), recursive=True
    )
    return matches[0] if matches else None


def run_mode(dataset_abs, run_dir, fast_rank, nprocs, verbose, morph_contrast=False):
    """Run FAST in `run_dir`; return (mean_file_path, elapsed_seconds)."""
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    t0 = time.perf_counter()
    try:
        from fastrack import pipeline
        pipeline.run(
            main_dir=dataset_abs,
            force_analysis=True,   # never reuse cached filaments across modes
            fast_rank=fast_rank,
            morph_contrast=morph_contrast,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", required=True, help="dataset directory to analyze")
    parser.add_argument("-o", default="fast_rank_comparison",
                        help="output directory for the two runs (Default: fast_rank_comparison)")
    parser.add_argument("-j", default=None, type=int, help="parallel worker processes")
    parser.add_argument("--morph", action="store_true",
                        help="also enable the morphological-gradient contrast in the fast run "
                             "(measures fast_rank + morph_contrast combined vs the 16-bit baseline)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose FAST output (per-frame progress and errors)")
    args = parser.parse_args(argv)

    dataset_abs = os.path.abspath(args.d)
    if not os.path.isdir(dataset_abs):
        sys.exit("Dataset directory does not exist: %s" % dataset_abs)

    baseline_dir = os.path.join(args.o, "baseline_16bit")
    fast_dir = os.path.join(args.o, "fast_rank_8bit")

    print("=== Running BASELINE (16-bit rank filters, default) ===")
    base_file, base_t = run_mode(dataset_abs, baseline_dir, fast_rank=False,
                                 nprocs=args.j, verbose=args.verbose)
    fast_label = "8-bit rank + morph gradient" if args.morph else "8-bit rank filters"
    print("=== Running FAST-RANK (%s) ===" % fast_label)
    fast_file, fast_t = run_mode(dataset_abs, fast_dir, fast_rank=True,
                                 nprocs=args.j, verbose=args.verbose,
                                 morph_contrast=args.morph)

    # ----- speedup ------------------------------------------------------- #
    print("\n" + "=" * 96)
    print("WALL-CLOCK")
    print("=" * 96)
    print("  baseline (16-bit):  %8.2f s" % base_t)
    print("  fast-rank (8-bit):  %8.2f s" % fast_t)
    if fast_t > 0:
        print("  speedup:            %8.2fx  (%.1f%% faster)"
              % (base_t / fast_t, 100.0 * (base_t - fast_t) / base_t))

    baseline = parse_mean_file(base_file)
    fast = parse_mean_file(fast_file)
    if not baseline and not fast:
        sys.exit("No MEAN_values.txt produced by either run. Check the dataset / install.")

    # ----- accuracy ------------------------------------------------------ #
    print("\n" + "=" * 96)
    print("VELOCITY DELTAS  (baseline = 16-bit, fast = 8-bit fast_rank)")
    print("=" * 96)
    metrics = ["top-vel-5", "MVEL", "MVEL-filtered", "MVIS"]
    for key in sorted(set(baseline) | set(fast)):
        b = baseline.get(key)
        fr = fast.get(key)
        protein = (b or fr)["protein"]
        print("\n%-70s [%s]" % (key, protein))
        print("  %-16s %14s %14s %12s" % ("metric", "baseline", "fast-rank", "delta %"))
        for mt in metrics:
            idx = COL[mt]
            bv = b["floats"][idx] if b and idx < len(b["floats"]) else float("nan")
            fv = fr["floats"][idx] if fr and idx < len(fr["floats"]) else float("nan")
            print("  %-16s %14.3f %14.3f %11.2f%%" % (mt, bv, fv, pct(bv, fv)))

    # ----- qualitative paper check both ways ----------------------------- #
    print("\n" + "-" * 96)
    print("Qualitative paper check (alpha-cardiac should be faster than beta-cardiac):")
    for label, data in (("baseline", baseline), ("fast-rank", fast)):
        a = [v["floats"][COL["top-vel-5"]] for v in data.values()
             if v["protein"].lower().startswith("alpha")]
        bt = [v["floats"][COL["top-vel-5"]] for v in data.values()
              if v["protein"].lower().startswith("beta")]
        if a and bt:
            am, bm = sum(a) / len(a), sum(bt) / len(bt)
            verdict = "PASS" if am > bm else "FAIL"
            print("  %-10s mean top-vel-5:  alpha=%.1f  beta=%.1f  ratio=%.2f  -> %s"
                  % (label, am, bm, am / bm if bm else float("nan"), verdict))
        else:
            print("  %-10s could not find both alpha and beta rows" % label)

    print("\nInterpretation: a large speedup with small (<~5%) velocity deltas and an")
    print("unchanged PASS verdict means fast_rank is a safe optimization for this data.")


if __name__ == "__main__":
    main()
