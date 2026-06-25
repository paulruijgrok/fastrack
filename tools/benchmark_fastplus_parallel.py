#!/usr/bin/env python3
"""Wall-clock benchmark of FASTplus parallel filament detection.

Runs the directional pipeline (:func:`fastrack.pipelines.directional.run`) on a
dataset at several worker counts (``-j`` / ``nprocs``) and reports the wall-clock
time, speedup, and parallel efficiency for each.  Optionally verifies that every
parallel run reproduces the serial (``-j 1``) ``frame_average.csv`` byte-for-byte
(the determinism check), and writes a Markdown table + CSV suitable for pasting
into ``docs/fastplus_parallel_verification.md``.

USAGE
-----
    python tools/benchmark_fastplus_parallel.py \
        -d ".../PolarityLabeled/20170803/ch12" \
        --jobs 1 2 4 8 --repeats 2 --spf 0.1356 --head-quality 8 \
        --no-register --max-frames 300 --verify

Notes
-----
* Each (jobs, repeat) runs into its own temp output dir; the *minimum* time over
  repeats is reported (least affected by background load).
* The pipeline does no caching, so every run re-detects from scratch -- exactly
  the work being parallelized.  On macOS workers use the 'spawn' start method, so
  the first parallel run pays a one-off interpreter-import cost per worker.
* This times the whole directional run (load -> detect -> track -> fit -> plot);
  filament detection dominates, so it is a good proxy for detection scaling.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import statistics
import sys
import tempfile
import time


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", required=True, help="dataset directory of *RGB.tif movies")
    ap.add_argument("--jobs", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="worker counts to benchmark (Default: 1 2 4 8)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats per worker count; the minimum time is reported")
    # pipeline knobs (kept minimal; defaults match a typical head-centric run)
    ap.add_argument("--mode", default="head-centric",
                    choices=["head-centric", "filament-centric"])
    ap.add_argument("--head-channel", default="red")
    ap.add_argument("--filament-channel", default="green")
    ap.add_argument("--head-quality", type=float, default=8.0)
    ap.add_argument("--spf", type=float, default=None, help="seconds per frame")
    ap.add_argument("--no-register", action="store_true",
                    help="skip optomerge channel registration")
    ap.add_argument("--kinetic-model", default="exp_rise_decay")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per movie (speeds up the benchmark)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of movies")
    ap.add_argument("--verify", action="store_true",
                    help="check each run's frame_average.csv matches the -j 1 run")
    ap.add_argument("--report", default="benchmark_parallel_results.md",
                    help="Markdown report output path")
    ap.add_argument("--csv", default="benchmark_parallel_results.csv",
                    help="CSV report output path")
    ap.add_argument("--plot", default="benchmark_parallel_results.png",
                    help="speedup-vs-workers plot output path ('' to disable)")
    ap.add_argument("--keep-outputs", action="store_true",
                    help="keep the temporary per-run output folders")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.d):
        sys.exit("dataset directory not found: %s" % args.d)

    # import here so --help works without the heavy stack installed
    from fastrack.config import Settings
    from fastrack.pipelines import directional

    def build_kwargs(jobs):
        overrides = dict(
            mode=args.mode, head_channel=args.head_channel,
            filament_channel=args.filament_channel, head_quality=args.head_quality,
            register_channels=not args.no_register, kinetic_model=args.kinetic_model,
            nprocs=jobs,
        )
        if args.spf:
            overrides["frame_rate_hz"] = 1.0 / args.spf
        return Settings().with_overrides(**overrides).to_directional_kwargs()

    workdir = tempfile.mkdtemp(prefix="fastplus_bench_")
    results = []           # (jobs, best_time, all_times, ok, md5)
    serial_md5 = None
    try:
        for jobs in args.jobs:
            times = []
            last_out = None
            for r in range(args.repeats):
                odir = os.path.join(workdir, "j%d_r%d" % (jobs, r))
                kw = build_kwargs(jobs)
                t0 = time.perf_counter()
                directional.run(main_dir=args.d, output_dir=odir,
                                max_frames=args.max_frames, limit=args.limit, **kw)
                dt = time.perf_counter() - t0
                times.append(dt)
                last_out = odir
                print("  jobs=%-3d repeat=%d  %.1f s" % (jobs, r, dt), flush=True)
            best = min(times)
            fa = os.path.join(last_out, "frame_average.csv")
            md5 = _md5(fa) if os.path.isfile(fa) else None
            if jobs == args.jobs[0]:
                serial_md5 = md5
            ok = (md5 == serial_md5) if (args.verify and md5) else None
            results.append((jobs, best, times, ok, md5))
            print("jobs=%-3d  best=%.1f s%s" %
                  (jobs, best, "" if ok is None else ("  verify=%s" % ("OK" if ok else "MISMATCH"))),
                  flush=True)
    finally:
        if not args.keep_outputs:
            shutil.rmtree(workdir, ignore_errors=True)

    base = results[0][1]
    # console + report
    header = "| jobs | time (s) | speedup | efficiency |" + (" verify |" if args.verify else "")
    sep = "|------|----------|---------|------------|" + ("--------|" if args.verify else "")
    lines = [header, sep]
    for jobs, best, _all, ok, _hash in results:
        sp = base / best if best else float("nan")
        eff = sp / jobs if jobs else float("nan")
        row = "| %d | %.1f | %.2f× | %.0f%% |" % (jobs, best, sp, 100 * eff)
        if args.verify:
            row += " %s |" % ("—" if ok is None else ("OK" if ok else "MISMATCH"))
        lines.append(row)
    table = "\n".join(lines)
    print("\n" + table)

    import multiprocessing
    meta = ("Dataset: `%s`  \nWorker counts: %s, repeats: %d (min reported)  \n"
            "Host cores: %d  \nmax_frames=%s, mode=%s\n\n" %
            (args.d, args.jobs, args.repeats, multiprocessing.cpu_count(),
             args.max_frames, args.mode))
    plot_path = _save_plot(results, base, args.plot) if args.plot else None
    with open(args.report, "w") as f:
        f.write("## FASTplus parallel timing benchmark\n\n" + meta + table + "\n")
        if plot_path:
            f.write("\n![speedup vs workers](%s)\n" % os.path.basename(plot_path))
    with open(args.csv, "w") as f:
        f.write("jobs,best_time_s,speedup,efficiency,verify\n")
        for jobs, best, _all, ok, _hash in results:
            sp = base / best if best else float("nan")
            f.write("%d,%.3f,%.3f,%.3f,%s\n" %
                    (jobs, best, sp, sp / jobs if jobs else float("nan"),
                     "" if ok is None else ("ok" if ok else "mismatch")))
    print("\nwrote %s and %s%s" %
          (args.report, args.csv, (" and %s" % plot_path) if plot_path else ""))


def _save_plot(results, base, out_path):
    """Speedup-vs-workers plot with the ideal-linear reference line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    jobs = [r[0] for r in results]
    speed = [base / r[1] if r[1] else float("nan") for r in results]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    lo, hi = min(jobs), max(jobs)
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", label="ideal (linear)")
    ax.plot(jobs, speed, "o-", color="tab:red", lw=2, label="measured")
    ax.set_xlabel("worker processes (-j)", fontsize=11)
    ax.set_ylabel("speedup vs serial", fontsize=11)
    ax.set_title("FASTplus filament-detection scaling", fontsize=12)
    ax.set_xticks(jobs)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
