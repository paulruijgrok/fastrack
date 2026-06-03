#!/usr/bin/env python3
"""Per-stage profiler for FASTrack frame processing.

Times each step of ``Frame`` extraction (read -> low-pass -> entropy/percentile
+ watershed -> per-island skeletonization) on real frames so you can see where
the wall-clock time actually goes before deciding what to optimize / move to GPU.

Run on a machine with the full image stack installed (your Mac):

    python profile_frame.py -d examples/unloaded_motility/micromanager_tifs \
        --header img_000000 --tail "" -n 8

If you don't know the header/tail for a dataset, point ``-d`` at a single movie
folder (the one containing the ``*_000.tif`` frame files) and the script will try
to autodetect them from the first ``*_000.tif`` it finds.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

from fastrack.motility import Frame


def autodetect(directory):
    """Guess (header, tail, first_frame_no) from a *_000.tif in `directory`."""
    cands = sorted(glob.glob(os.path.join(directory, "*_000.tif")))
    if not cands:
        return None
    base = os.path.basename(cands[0])
    # Frame filenames are <header><NNN>_<tail>_000.tif  (NNN = 3-digit frame no).
    m = re.match(r"^(.*?)(\d{3})_(.*)_000\.tif$", base)
    if not m:
        return None
    header, frame_no, tail = m.group(1), int(m.group(2)), m.group(3)
    return header, tail, frame_no


class Timer:
    def __init__(self):
        self.t = {}

    def run(self, name, fn):
        t0 = time.perf_counter()
        out = fn()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0)
        return out


def profile_frame(directory, header, tail, frame_no, timer, fast_rank=False,
                  morph_contrast=False):
    f = Frame()
    f.directory = directory
    f.header = header
    f.tail = tail
    f.fast_rank = fast_rank
    f.morph_contrast = morph_contrast

    ok = timer.run("01_read_frame", lambda: f.read_frame(frame_no))
    if not ok:
        return False
    timer.run("02_low_pass_filter(gaussian)", lambda: f.low_pass_filter())
    timer.run("03_entropy_clusters(percentile+watershed)", lambda: f.entropy_clusters())
    timer.run("04_filter_islands", lambda: f.filter_islands())
    timer.run("05_skeletonize_islands", lambda: f.skeletonize_islands())
    timer.run("06_filaments2filXYs", lambda: f.filaments2filXYs())
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", required=True, help="movie folder containing *_000.tif frames")
    p.add_argument("--header", default=None, help="frame filename header (autodetected if omitted)")
    p.add_argument("--tail", default=None, help="frame filename tail (autodetected if omitted)")
    p.add_argument("-n", type=int, default=8, help="number of frames to profile (Default: 8)")
    p.add_argument("--start", type=int, default=None, help="first frame number")
    p.add_argument("--fast-rank", action="store_true", default=False,
                   help="profile the 8-bit fast_rank path instead of the 16-bit default")
    p.add_argument("--morph-contrast", action="store_true", default=False,
                   help="profile the one-pass morphological-gradient contrast path")
    args = p.parse_args(argv)

    directory = os.path.abspath(args.d)
    header, tail, start = args.header, args.tail, args.start
    if header is None or tail is None or start is None:
        det = autodetect(directory)
        if det is None:
            sys.exit("Could not autodetect frame naming in %s; pass --header/--tail/--start."
                     % directory)
        d_header, d_tail, d_start = det
        header = d_header if header is None else header
        tail = d_tail if tail is None else tail
        start = d_start if start is None else start

    print("directory: %s" % directory)
    print("header=%r tail=%r start=%d n=%d\n" % (header, tail, start, args.n))

    timer = Timer()
    wall0 = time.perf_counter()
    done = 0
    for i in range(args.n):
        if profile_frame(directory, header, tail, start + i, timer,
                         fast_rank=args.fast_rank, morph_contrast=args.morph_contrast):
            done += 1
        else:
            break
    wall = time.perf_counter() - wall0

    if done == 0:
        sys.exit("No frames processed -- check -d / --header / --tail / --start.")

    print("Profiled %d frame(s) in %.2f s wall (%.2f s/frame)\n" % (done, wall, wall / done))
    total = sum(timer.t.values())
    print("%-46s %10s %8s %12s" % ("stage", "total s", "%", "per-frame ms"))
    print("-" * 80)
    for name in sorted(timer.t):
        s = timer.t[name]
        print("%-46s %10.3f %7.1f%% %12.1f"
              % (name, s, 100.0 * s / total, 1000.0 * s / done))
    print("-" * 80)
    print("%-46s %10.3f %7.1f%% %12.1f"
          % ("TOTAL (summed stages)", total, 100.0, 1000.0 * total / done))


if __name__ == "__main__":
    main()
