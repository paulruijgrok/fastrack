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

Optional auto-crop: with ``--crop-frac 0.8`` the tool samples a few corrected
frames, runs the ridge detector on decimated copies to measure the per-row
filament-track density, and vertically crops each movie to the shortest row band
holding ~80% of the tracks (retention closely matches ``--crop-frac``). This
shrinks the frames the downstream detector processes. How much it speeds things
up depends on the data: if tracks are spread across the field, an 80% crop only
trims the sparse edges (modest); lower ``--crop-frac`` crops harder, trading
tracks for speed. By default only rows are cropped (the usual top-bottom
illumination axis); ``--crop-cols`` additionally crops columns (a 2D bounding
box), which only helps when tracks are also concentrated horizontally. Needs
``ridge_detector`` (fastrack[ridge]); without it the crop falls back to a
brightness proxy that over-crops toward the high-SNR region.

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
import warnings
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
def correct_stack(stack: np.ndarray, *, background: str = "median",
                  flat_sigma: float = 50.0, denoise_sigma: float = 0.8) -> np.ndarray:
    """Background-subtract, flat-field and denoise a ``(T, H, W)`` stack.

    Returns a clipped (>= 0) float32 stack in corrected units (background -> 0).
    The final full-range stretch is applied separately (:func:`stretch_to_uint`),
    so callers can crop between correction and stretch.

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

    return np.clip(corr, 0, None)                     # background -> 0


def stretch_to_uint(corr: np.ndarray, *, clip_hi_pct: float = 99.9,
                    bit_depth: int = 8) -> np.ndarray:
    """Movie-wide contrast stretch of a corrected float stack to full-range uint.

    ``clip_hi_pct`` is the upper percentile (over the whole passed stack) mapped
    to the max output value; 0 maps to 0. A single movie-wide mapping keeps
    intensities comparable across frames. ``bit_depth`` 8 (recommended, sidesteps
    FASTrack's 16->8 read crush) or 16.
    """
    hi = max(float(np.percentile(corr, clip_hi_pct)), 1e-6)
    if bit_depth == 8:
        return np.clip(corr / hi * 255.0, 0, 255).astype(np.uint8)
    if bit_depth == 16:
        return np.clip(corr / hi * 65535.0, 0, 65535).astype(np.uint16)
    raise ValueError("bit_depth must be 8 or 16")


def _smallest_row_band(density: np.ndarray, target: float):
    """Shortest contiguous ``(start, stop)`` whose ``density`` sum >= ``target``."""
    n = len(density)
    left = 0
    cur = 0.0
    best = (0, n)
    best_len = n
    for right in range(n):
        cur += density[right]
        while cur >= target and left <= right:
            if (right - left + 1) < best_len:
                best_len = right - left + 1
                best = (left, right + 1)
            cur -= density[left]
            left += 1
    return best


def _row_col_density_bright(samples, fg_pct: float = 99.0):
    """Per-row and per-column bright-pixel counts. Fast but a poor proxy for
    tracks: it thresholds on brightness, so it favours the higher-SNR region and
    *over-crops*. Only used when the ridge detector isn't available."""
    H, W = samples[0].shape
    den_r = np.zeros(H, np.float64)
    den_c = np.zeros(W, np.float64)
    for f in samples:
        m = f > np.percentile(f, fg_pct)
        den_r += m.sum(axis=1)
        den_c += m.sum(axis=0)
    return den_r, den_c


def _row_col_density_ridge(samples_u8, H, W, *, downscale=2, line_widths=(3, 5),
                           low_contrast=18.0, high_contrast=50.0, min_len=12.0):
    """Per-row and per-column ridge-detection counts over the sampled frames.

    The *faithful* metric -- it uses the same detector the downstream analysis
    runs, so the bands reflect where tracks actually are. Frames are decimated by
    ``downscale`` first (the row/col distributions survive; detection is
    ~downscale^2 faster). Returns ``None`` if ``ridge_detector`` isn't installed.
    """
    try:
        from ridge_detector import RidgeDetector
    except Exception:
        return None
    s = max(1, int(downscale))
    lw = np.array([max(1, int(round(w / s))) for w in line_widths])
    den_r = np.zeros(H, np.float64)
    den_c = np.zeros(W, np.float64)
    for f in samples_u8:
        det = RidgeDetector(line_widths=lw, low_contrast=low_contrast,
                            high_contrast=high_contrast, min_len=max(3.0, min_len / s),
                            max_len=0, dark_line=False, estimate_width=False)
        det.detect_lines(f[::s, ::s].astype(np.float64))
        for c in (det.contours or []):
            if int(getattr(c, "num", 0)) >= 2:
                rr = int(min(H - 1, max(0, np.mean(np.asarray(c.row, float)) * s)))
                cc = int(min(W - 1, max(0, np.mean(np.asarray(c.col, float)) * s)))
                den_r[rr] += 1
                den_c[cc] += 1
    return den_r, den_c


def _band(density: np.ndarray, frac: float, length: int, margin: int):
    """Padded shortest contiguous band ``(a, b)`` holding ``frac`` of ``density``."""
    if frac >= 1.0:
        return (0, length)
    total = float(density.sum())
    if total <= 0:
        return (0, length)
    a, b = _smallest_row_band(density, frac * total)
    return (max(0, a - margin), min(length, b + margin))


def estimate_crop_box(corr: np.ndarray, *, frac: float = 0.8, n_sample: int = 8,
                      margin: int = 8, method: str = "ridge", downscale: int = 2,
                      clip_hi_pct: float = 99.9, crop_cols: bool = False):
    """Bounding box ``(r0, r1, c0, c1)`` holding ~``frac`` of the filament tracks.

    Samples ``n_sample`` corrected frames and builds per-row (and per-column)
    filament-density profiles, then takes the shortest contiguous band on each
    cropped axis, padded by ``margin`` px.

    Vertical-only by default (full width). With ``crop_cols=True`` both axes are
    trimmed and each targets ``sqrt(frac)``, so the joint box keeps ~``frac`` of
    the (roughly independent) tracks. ``method="ridge"`` (default, faithful) runs
    the real detector on decimated frames; without ``ridge_detector`` it warns and
    falls back to ``method="bright"`` (a brightness proxy that over-crops).
    """
    T, H, W = corr.shape
    idx = np.unique(np.linspace(0, T - 1, max(1, min(n_sample, T))).astype(int))

    dens = None
    if method == "ridge":
        hi = max(float(np.percentile(corr, clip_hi_pct)), 1e-6)
        samples_u8 = [np.clip(corr[i] / hi * 255.0, 0, 255).astype(np.uint8) for i in idx]
        dens = _row_col_density_ridge(samples_u8, H, W, downscale=downscale)
        if dens is None:
            warnings.warn(
                "ridge_detector not installed; crop metric falling back to a "
                "brightness proxy, which over-crops toward the high-SNR region. "
                "Install fastrack[ridge] for a faithful crop.", RuntimeWarning)
    if dens is None:
        dens = _row_col_density_bright([corr[i] for i in idx])
    den_r, den_c = dens

    axis_frac = float(np.sqrt(frac)) if crop_cols else frac
    r0, r1 = _band(den_r, axis_frac, H, margin)
    c0, c1 = _band(den_c, axis_frac, W, margin) if crop_cols else (0, W)
    return (r0, r1, c0, c1)


def estimate_crop_rows(corr: np.ndarray, *, frac: float = 0.8, n_sample: int = 8,
                       margin: int = 8, method: str = "ridge", downscale: int = 2,
                       clip_hi_pct: float = 99.9):
    """Vertical crop band ``(r0, r1)`` -- :func:`estimate_crop_box`, rows only."""
    r0, r1, _c0, _c1 = estimate_crop_box(
        corr, frac=frac, n_sample=n_sample, margin=margin, method=method,
        downscale=downscale, clip_hi_pct=clip_hi_pct, crop_cols=False)
    return (r0, r1)


def preprocess_stack(stack: np.ndarray, *, background: str = "median",
                     flat_sigma: float = 50.0, denoise_sigma: float = 0.8,
                     clip_hi_pct: float = 99.9, bit_depth: int = 8) -> np.ndarray:
    """Correct + stretch a ``(T, H, W)`` stack to a full-range uint stack (no crop).

    Thin wrapper over :func:`correct_stack` + :func:`stretch_to_uint` for direct
    use; the CLI path crops between the two when ``--crop-frac`` is set.
    """
    corr = correct_stack(stack, background=background, flat_sigma=flat_sigma,
                          denoise_sigma=denoise_sigma)
    return stretch_to_uint(corr, clip_hi_pct=clip_hi_pct, bit_depth=bit_depth)


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
              "seconds": 0.0, "crop": None}
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
        corr = correct_stack(stack, background=params["background"],
                             flat_sigma=params["flat_sigma"],
                             denoise_sigma=params["denoise_sigma"])

        # Optional auto-crop to the productive row band before the stretch, so the
        # kept region uses the full output range and downstream detection is faster.
        crop_frac = params.get("crop_frac")
        if crop_frac and 0 < crop_frac < 1:
            r0, r1, c0, c1 = estimate_crop_box(corr, frac=crop_frac,
                                               n_sample=params["crop_sample"],
                                               margin=params["crop_margin"],
                                               method=params["crop_metric"],
                                               downscale=params["crop_downscale"],
                                               clip_hi_pct=params["clip_hi_pct"],
                                               crop_cols=params["crop_cols"])
            corr = corr[:, r0:r1, c0:c1]
            result["crop"] = (r0, r1, c0, c1, stack.shape[1], stack.shape[2])

        out = stretch_to_uint(corr, clip_hi_pct=params["clip_hi_pct"],
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
        crop_msg = ""
        crop = r.get("crop")
        if crop:
            r0, r1, c0, c1, H, W = crop
            area = 100.0 * (r1 - r0) * (c1 - c0) / (H * W)
            cols = f" cols {c0}-{c1}/{W}" if (c0, c1) != (0, W) else ""
            crop_msg = f", rows {r0}-{r1}/{H}{cols} ({area:.0f}% area)"
        return (f"  [ok]   {r['movie']}  ({r['frames']} frames{crop_msg}, "
                f"{r['seconds']:.1f}s)")
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
    p.add_argument("--crop-frac", type=float, default=None,
                   help="auto-crop each movie (vertically) to the shortest row band "
                        "holding this fraction of the filament signal, e.g. 0.8; "
                        "off by default. Keeps most tracks, shrinks the frames the "
                        "detector processes")
    p.add_argument("--crop-sample-frames", type=int, default=8, dest="crop_sample",
                   help="number of frames sampled to decide the crop band")
    p.add_argument("--crop-margin", type=int, default=8,
                   help="pixels of padding added to each side of the crop band")
    p.add_argument("--crop-metric", choices=["ridge", "bright"], default="ridge",
                   help="crop density metric: 'ridge' (faithful; runs the detector "
                        "on sampled frames) or 'bright' (fast brightness proxy that "
                        "over-crops toward the high-SNR region)")
    p.add_argument("--crop-downscale", type=int, default=2,
                   help="decimation factor for the ridge crop metric (speed)")
    p.add_argument("--crop-cols", action="store_true",
                   help="also crop columns (2D bounding box), not just rows. Each "
                        "axis targets sqrt(crop-frac) so the box keeps ~crop-frac of "
                        "tracks. Only helps if tracks are concentrated horizontally")
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

    if args.crop_frac is not None and not (0 < args.crop_frac < 1):
        print("ERROR: --crop-frac must be between 0 and 1 (e.g. 0.8)", file=sys.stderr)
        return 2

    params = {"background": args.background, "flat_sigma": args.flat_sigma,
              "denoise_sigma": args.denoise_sigma, "clip_hi_pct": args.clip_hi_pct,
              "bit_depth": args.bit_depth, "crop_frac": args.crop_frac,
              "crop_sample": args.crop_sample, "crop_margin": args.crop_margin,
              "crop_metric": args.crop_metric, "crop_downscale": args.crop_downscale,
              "crop_cols": args.crop_cols}
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
