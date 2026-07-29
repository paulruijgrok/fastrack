#!/usr/bin/env python3
"""Standalone low-SNR / uneven-illumination preprocessing for gliding-assay movies.

Some gliding-assay acquisitions are low-contrast (filaments only a few hundred
counts over background), sit in the bottom of the 16-bit range, and carry a
strong illumination gradient plus static stuck junk. Two things then wreck
detection:

  1. FASTrack reads frames with ``cv2.IMREAD_GRAYSCALE`` (16 -> 8 bit by a
     divide-by-256 bit-shift). Raw values of ~300-900 collapse to ~1-2 grey
     levels *before* detection, destroying the filament signal.
  2. A top->bottom illumination gradient (plus static junk) biases a global
     detector to the brightest region, so only the top of the field is found.

This tool corrects both, per movie:

    temporal-median background subtraction  (removes static bg / gradient / junk)
    -> illumination flat-field (divide by a smoothed gain field)   (equalises)
    -> mild Gaussian denoise
    -> robust, movie-consistent stretch to a full-range 8-bit (or 16-bit) image

It writes **drop-in** MicroManager movies: same folder layout and frame filenames
as the input, with ``metadata.txt`` copied through, so you can point FASTrack
straight at the output:

    python tools/preprocess_lowsnr.py -i RAW_DIR -o CORRECTED_DIR
    fast -d CORRECTED_DIR -f --detector ridge \
        --ridge-line-widths 3 5 --ridge-low-contrast 18 --ridge-high-contrast 50 \
        --ridge-min-len 12

Batch behaviour: every movie under ``-i`` is processed independently and
fail-isolated (one bad movie is logged and skipped, the rest continue). Re-runs
skip movies whose output already exists unless ``--force`` is given.

This is intentionally a separate script, not wired into the pipeline, while the
broader cross-pipeline preprocessing strategy is scoped (see
docs/preprocessing_notes.md).
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

try:
    import tifffile
    def _imread(p): return tifffile.imread(p)
    def _imwrite(p, a): tifffile.imwrite(p, a)
except ImportError:                                   # pragma: no cover
    import imageio.v3 as iio
    def _imread(p): return np.asarray(iio.imread(p))
    def _imwrite(p, a): iio.imwrite(p, a)

from scipy.ndimage import gaussian_filter

# Matches FASTrack's MicroManager frame convention (core/input/mm_dir.py).
_FRAME_RE = re.compile(r"^img_\d+_.*_000\.tif$")


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_movie_dirs(root: str) -> List[str]:
    """Return every directory under ``root`` that holds MicroManager frames."""
    hits = []
    for dp, _dirs, files in os.walk(root):
        if any(_FRAME_RE.match(f) for f in files):
            hits.append(dp)
    return sorted(hits)


def frame_paths(movie_dir: str) -> List[str]:
    """Frame files in acquisition order (by the numeric index in the name)."""
    fs = [p for p in glob.glob(os.path.join(movie_dir, "*.tif"))
          if _FRAME_RE.match(os.path.basename(p))]

    def _idx(p):
        try:
            return int(os.path.basename(p).split("_")[1])
        except (IndexError, ValueError):
            return 0

    return sorted(fs, key=_idx)


# --------------------------------------------------------------------------- #
# core preprocessing
# --------------------------------------------------------------------------- #
def preprocess_stack(stack: np.ndarray, *, background: str = "median",
                     flat_sigma: float = 50.0, denoise_sigma: float = 0.8,
                     clip_hi_pct: float = 99.9, bit_depth: int = 8) -> np.ndarray:
    """Correct + stretch a ``(T, H, W)`` float stack to a full-range uint stack.

    Parameters
    ----------
    background : "median" | "mean" | "none"
        Static-background model subtracted from every frame. ``median`` is the
        most robust to moving filaments and stuck junk.
    flat_sigma : float
        Gaussian sigma (px) for the smooth illumination gain field. 0 disables
        flat-fielding (subtraction only).
    denoise_sigma : float
        Gaussian denoise sigma (px) applied after correction. 0 disables it.
    clip_hi_pct : float
        Upper percentile (over the whole corrected movie) mapped to the max
        output value; everything at/below background maps to 0. A single
        movie-wide mapping keeps intensities comparable across frames.
    bit_depth : 8 | 16
        Output bit depth. 8 is recommended: FASTrack's reader is 8-bit anyway,
        and 8-bit sidesteps the 16->8 bit-shift that crushes low-valued data.
    """
    stack = stack.astype(np.float32, copy=False)

    if background == "median":
        bg = np.median(stack, axis=0)
    elif background == "mean":
        bg = stack.mean(axis=0)
    elif background == "none":
        bg = np.zeros(stack.shape[1:], np.float32)
    else:
        raise ValueError("background must be 'median', 'mean' or 'none'")

    corr = stack - bg[None]

    if flat_sigma and flat_sigma > 0:
        gain = gaussian_filter(stack.mean(axis=0), flat_sigma)
        gain = gain / max(float(gain.mean()), 1e-6)
        corr = corr / gain[None]

    if denoise_sigma and denoise_sigma > 0:
        corr = np.stack([gaussian_filter(c, denoise_sigma) for c in corr])

    corr = np.clip(corr, 0, None)                     # background -> 0
    hi = float(np.percentile(corr, clip_hi_pct))
    hi = max(hi, 1e-6)

    if bit_depth == 8:
        out = np.clip(corr / hi * 255.0, 0, 255).astype(np.uint8)
    elif bit_depth == 16:
        out = np.clip(corr / hi * 65535.0, 0, 65535).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")
    return out


# --------------------------------------------------------------------------- #
# per-movie driver
# --------------------------------------------------------------------------- #
def process_movie(movie_dir: str, in_root: str, out_root: str, params: dict,
                  force: bool) -> dict:
    """Preprocess one movie into a mirrored output directory. Never raises.

    Returns a picklable result dict (no logging inside) so it can run in a
    worker process; the parent formats and prints each result as it completes.
    """
    rel = os.path.relpath(movie_dir, in_root)
    dst = os.path.join(out_root, rel)
    result = {"movie": rel, "status": "ok", "frames": 0, "error": None,
              "seconds": 0.0}
    t0 = time.perf_counter()
    try:
        fs = frame_paths(movie_dir)
        if not fs:
            result["status"] = "skipped-empty"
            return result

        # Resume: skip if the output already has the same frame count.
        if not force and os.path.isdir(dst):
            existing = [p for p in glob.glob(os.path.join(dst, "*.tif"))
                        if _FRAME_RE.match(os.path.basename(p))]
            if len(existing) == len(fs):
                result["status"] = "skipped-exists"
                result["frames"] = len(fs)
                return result

        stack = np.stack([np.asarray(_imread(f)) for f in fs]).astype(np.float32)
        out = preprocess_stack(
            stack, background=params["background"], flat_sigma=params["flat_sigma"],
            denoise_sigma=params["denoise_sigma"], clip_hi_pct=params["clip_hi_pct"],
            bit_depth=params["bit_depth"])

        os.makedirs(dst, exist_ok=True)
        for src, frame in zip(fs, out):
            _imwrite(os.path.join(dst, os.path.basename(src)), frame)

        # Carry timing/metadata through so FASTrack reads frame intervals.
        for meta in ("metadata.txt", "display_and_comments.txt"):
            mp = os.path.join(movie_dir, meta)
            if os.path.isfile(mp):
                shutil.copy2(mp, os.path.join(dst, meta))

        result["frames"] = len(fs)
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    result["seconds"] = time.perf_counter() - t0
    return result


def _format_result(r: dict) -> str:
    """One-line log message for a completed movie result."""
    if r["status"] == "ok":
        return f"  [ok]   {r['movie']}  ({r['frames']} frames, {r['seconds']:.1f}s)"
    if r["status"].startswith("skipped"):
        return f"  [skip] {r['movie']}  ({r['status']})"
    return f"  [FAIL] {r['movie']}\n{r['error']}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Low-SNR / uneven-illumination preprocessing for gliding movies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="root folder of raw movies")
    p.add_argument("-o", "--output", required=True, help="output root (mirrors input)")
    p.add_argument("--background", choices=["median", "mean", "none"],
                   default="median", help="static-background model to subtract")
    p.add_argument("--flat-sigma", type=float, default=50.0,
                   help="illumination flat-field Gaussian sigma in px (0 = off)")
    p.add_argument("--denoise-sigma", type=float, default=0.8,
                   help="post-correction Gaussian denoise sigma in px (0 = off)")
    p.add_argument("--clip-hi-pct", type=float, default=99.9,
                   help="upper percentile mapped to the max output value")
    p.add_argument("--bit-depth", type=int, choices=[8, 16], default=8,
                   help="output bit depth (8 recommended)")
    p.add_argument("-f", "--force", action="store_true",
                   help="reprocess movies whose output already exists")
    p.add_argument("-j", "--workers", type=int, default=None,
                   help="parallel worker processes across movies "
                        "(default: min(CPUs, number of movies))")
    args = p.parse_args(argv)

    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)
    if not os.path.isdir(in_root):
        print(f"ERROR: input not found: {in_root}", file=sys.stderr)
        return 2

    def log(msg): print(msg, flush=True)

    movies = find_movie_dirs(in_root)
    if not movies:
        print(f"No MicroManager movies (img_*_000.tif) under {in_root}")
        return 1

    params = {"background": args.background, "flat_sigma": args.flat_sigma,
              "denoise_sigma": args.denoise_sigma, "clip_hi_pct": args.clip_hi_pct,
              "bit_depth": args.bit_depth}
    n_cpu = os.cpu_count() or 1
    workers = args.workers if (args.workers and args.workers > 0) else min(n_cpu, len(movies))
    workers = max(1, min(workers, len(movies)))

    log("=" * 70)
    log(f"Low-SNR preprocessing: {len(movies)} movie(s), {workers} worker(s)")
    log(f"  in : {in_root}")
    log(f"  out: {out_root}")
    log(f"  params: {params}")
    log("=" * 70)

    if workers == 1:
        results = []
        for m in movies:
            r = process_movie(m, in_root, out_root, params, args.force)
            log(_format_result(r))
            results.append(r)
    else:
        # Cap inner BLAS/OpenMP threads so `workers` processes don't oversubscribe
        # the cores (each movie's work is essentially single-threaded). Children
        # inherit these on spawn and import numpy with the limits applied.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_movie, m, in_root, out_root, params, args.force): m
                    for m in movies}
            for fut in as_completed(futs):
                r = fut.result()
                log(_format_result(r))
                results.append(r)

    ok = sum(r["status"] == "ok" for r in results)
    skipped = sum(r["status"].startswith("skipped") for r in results)
    failed = [r for r in results if r["status"] == "error"]
    log("-" * 70)
    log(f"Done: {ok} processed, {skipped} skipped, {len(failed)} failed.")
    for r in failed:
        log(f"  FAILED: {r['movie']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
