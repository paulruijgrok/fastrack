#!/usr/bin/env python3
"""Run a dataset through FAST with several detectors and report wall-clock time
and how much the velocities differ.

Compares any combination of the default entropy/watershed detector and the
optional ridge detectors:

* ``ridge``       -- reference (pip install 'fastrack[ridge]')
* ``ridge-fast``  -- ~4x faster, numerically-identical drop-in
                     (pip install 'fastrack[ridge-fast]')

The first detector listed is the baseline that the others' velocity deltas are
measured against.  Each detector runs in its own working directory so their
``outputs/`` trees do not collide, then the combined ``MEAN_values.txt`` files
are parsed and compared.

Usage:
    python tools/compare_detectors.py -d examples/unloaded_motility/micromanager_tifs
    python tools/compare_detectors.py -d <dataset> --detectors ridge ridge-fast   # speed test
    python tools/compare_detectors.py -d <dataset> --detectors entropy ridge ridge-fast -j 8
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


def _load_sample_frames(dataset_abs, max_frames):
    """Load up to ``max_frames`` frames (one movie folder) the same way the
    pipeline does (16-bit grayscale), so detection timing is apples-to-apples."""
    import cv2
    from skimage.util import img_as_uint

    for root, _dirs, files in os.walk(dataset_abs):
        tifs = sorted(f for f in files if f.lower().endswith(".tif"))
        if not tifs:
            continue
        imgs = []
        for f in tifs[: (max_frames or len(tifs))]:
            im = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(img_as_uint(im))
        if imgs:
            return root, imgs
    return None, []


def time_detection(dataset_abs, detectors, ridge_params, max_frames,
                   fast_rank, morph_contrast):
    """Time ONLY the detection stage, in-process, on a common set of frames.

    The full pipeline runs detection inside multiprocessing workers, so a parent
    timer can't see it; this isolates detection compute directly.  A one-frame
    warm-up per detector is excluded so numba JIT compilation doesn't skew the
    steady-state numbers.  Returns ``(source_dir, n_frames, {name: seconds})``.
    """
    from fastrack.core.frame import Frame
    from fastrack.core.detection import DETECTORS

    source, imgs = _load_sample_frames(dataset_abs, max_frames)
    if not imgs:
        return None, 0, {}

    def run_once(det, im):
        fr = Frame()
        fr.frame_no = 0
        fr.img = im.copy()
        fr.width, fr.height = im.shape
        det.detect(fr)

    results = {}
    for name in detectors:
        if name == "entropy":
            det = DETECTORS.create("entropy", fast_rank=fast_rank,
                                   morph_contrast=morph_contrast)
        else:
            det = DETECTORS.create(name, **ridge_params)
        try:
            run_once(det, imgs[0])          # warm-up (JIT compile), not timed
            t0 = time.perf_counter()
            for im in imgs:
                run_once(det, im)
            results[name] = time.perf_counter() - t0
        except Exception as e:              # keep going if one detector can't run here
            print("  (detection timing skipped for %s: %s)" % (name, e))
    return source, len(imgs), results


def report_detection_timing(dataset_abs, detectors, ridge_params, max_frames,
                            fast_rank, morph_contrast):
    """Print the detection-only timing section (first detector = baseline)."""
    print("\n" + "=" * 96)
    print("DETECTION-ONLY TIMING  (in-process, numba warm-up excluded)")
    print("=" * 96)
    source, n, det_times = time_detection(
        dataset_abs, detectors, ridge_params, max_frames, fast_rank, morph_contrast)
    if not det_times or not n:
        print("  no frames found to time.")
        return
    print("  sample: %d frames from %s" % (n, source))
    baseline = next((d for d in detectors if d in det_times), None)
    base_t = det_times.get(baseline)
    for name in detectors:
        if name not in det_times:
            continue
        t = det_times[name]
        per = 1000.0 * t / n
        note = ""
        if name != baseline and base_t and t > 0:
            note = ("  (%.2fx faster than %s)" % (base_t / t, baseline) if t < base_t
                    else "  (%.2fx slower than %s)" % (t / base_t, baseline))
        print("  %-12s %7.2f s total   %6.1f ms/frame%s" % (name, t, per, note))


def pct(a, b):
    if a == 0:
        return float("nan")
    return 100.0 * (b - a) / a


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", required=True, help="dataset directory to analyze")
    p.add_argument("-o", default="detector_comparison",
                   help="output directory for the runs (Default: detector_comparison)")
    p.add_argument("--detectors", nargs="+", default=["entropy", "ridge"],
                   choices=["entropy", "ridge", "ridge-fast"],
                   help="detectors to run; the first is the baseline for deltas "
                        "(Default: entropy ridge)")
    p.add_argument("-j", default=None, type=int, help="parallel worker processes")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose FAST output")
    p.add_argument("--time-detection", action="store_true",
                   help="also time the detection stage only (isolates the ~4x; in-process, "
                        "warm-up excluded). Directly comparable across detectors.")
    p.add_argument("--detection-only", action="store_true",
                   help="ONLY run the detection-stage timer; skip the full pipeline runs "
                        "and the velocity comparison")
    p.add_argument("--time-frames", type=int, default=40,
                   help="max frames (one movie folder) for the detection timer (Default: 40; "
                        "0 = all)")
    # Ridge params (shared by ridge and ridge-fast; entropy uses its defaults).
    p.add_argument("--ridge-line-widths", nargs="*", type=int, default=[3])
    p.add_argument("--ridge-low-contrast", type=float, default=50)
    p.add_argument("--ridge-high-contrast", type=float, default=150)
    p.add_argument("--ridge-min-len", type=float, default=10)
    p.add_argument("--ridge-dark-line", action="store_true", default=False)
    args = p.parse_args(argv)

    dataset_abs = os.path.abspath(args.d)
    if not os.path.isdir(dataset_abs):
        sys.exit("Dataset directory does not exist: %s" % dataset_abs)

    # de-duplicate while preserving order; first listed is the baseline
    detectors = list(dict.fromkeys(args.detectors))

    ridge_params = dict(
        line_widths=args.ridge_line_widths,
        low_contrast=args.ridge_low_contrast,
        high_contrast=args.ridge_high_contrast,
        min_len=args.ridge_min_len,
        dark_line=args.ridge_dark_line,
        estimate_width=True,
    )

    if args.detection_only:
        report_detection_timing(dataset_abs, detectors, ridge_params,
                                args.time_frames, True, False)
        return

    times = {}
    rows = {}
    for name in detectors:
        print("=== Running %s detector ===" % name.upper())
        params = {} if name == "entropy" else ridge_params
        mean_file, elapsed = run_mode(
            dataset_abs, os.path.join(args.o, name), name, params, args.j, args.verbose)
        times[name] = elapsed
        rows[name] = parse_mean_file(mean_file)

    if not any(rows.values()):
        sys.exit("No MEAN_values.txt produced by any run. Check the dataset / install.")

    baseline = detectors[0]

    # ----- wall-clock --------------------------------------------------- #
    print("\n" + "=" * 96)
    print("WALL-CLOCK  (speedup relative to baseline '%s')" % baseline)
    print("=" * 96)
    base_t = times[baseline]
    for name in detectors:
        t = times[name]
        note = ""
        if name != baseline and t > 0 and base_t > 0:
            ratio = base_t / t
            note = "  (%.2fx %s than %s)" % (
                ratio if ratio >= 1 else 1.0 / ratio,
                "faster" if ratio >= 1 else "slower", baseline)
        print("  %-12s %8.2f s%s" % (name, t, note))

    # ----- velocity comparison ------------------------------------------ #
    print("\n" + "=" * 96)
    print("VELOCITY COMPARISON  (delta = each detector relative to '%s')" % baseline)
    print("=" * 96)
    metrics = ["top-vel-5", "MVEL", "MVEL-filtered", "MVIS", "p-stuck"]
    all_keys = sorted(set().union(*[set(r) for r in rows.values()]))
    for key in all_keys:
        present = next((rows[n][key] for n in detectors if key in rows[n]), None)
        protein = present["protein"] if present else "?"
        print("\n%-70s [%s]" % (key, protein))
        for mt in metrics:
            idx = COL[mt]

            def value(name):
                r = rows[name].get(key)
                return r["floats"][idx] if r and idx < len(r["floats"]) else float("nan")

            base_v = value(baseline)
            parts = []
            for name in detectors:
                v = value(name)
                if name == baseline:
                    parts.append("%s=%.3f" % (name, v))
                else:
                    parts.append("%s=%.3f (%+.1f%%)" % (name, v, pct(base_v, v)))
            print("  %-14s %s" % (mt, "   ".join(parts)))

    # ----- qualitative paper check for each detector -------------------- #
    print("\n" + "-" * 96)
    print("Qualitative paper check (alpha-cardiac should be faster than beta-cardiac):")
    for name in detectors:
        data = rows[name]
        a = [v["floats"][COL["top-vel-5"]] for v in data.values()
             if v["protein"].lower().startswith("alpha")]
        b = [v["floats"][COL["top-vel-5"]] for v in data.values()
             if v["protein"].lower().startswith("beta")]
        if a and b:
            am, bm = sum(a) / len(a), sum(b) / len(b)
            verdict = "PASS" if am > bm else "FAIL"
            print("  %-12s mean top-vel-5:  alpha=%.1f  beta=%.1f  ratio=%.2f  -> %s"
                  % (name, am, bm, am / bm if bm else float("nan"), verdict))
        else:
            print("  %-12s could not find both alpha and beta rows" % name)

    print("\nInterpretation: ridge and ridge-fast agree to within float32-vs-float64")
    print("rounding (ridge-fast is a numerically-identical drop-in), so their velocity")
    print("deltas should be ~0. The ~4x speedup is for the DETECTION stage only; this")
    print("wall-clock is the whole pipeline, so the total speedup is diluted by the")
    print("unchanged linking/plotting/I/O work (Amdahl). If detection is a fraction f")
    print("of the run, total speedup ~= 1 / ((1-f) + f/4); it climbs toward 4x on")
    print("longer stacks where detection dominates. Larger entropy-vs-ridge deltas are")
    print("expected -- those detectors segment differently.")

    if args.time_detection:
        report_detection_timing(dataset_abs, detectors, ridge_params,
                                args.time_frames, True, False)


if __name__ == "__main__":
    main()
