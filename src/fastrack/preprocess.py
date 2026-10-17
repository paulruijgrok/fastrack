"""Illumination / low-SNR preprocessing shared by the pipelines and the tools.

Some acquisitions are low-contrast (filaments only a few hundred counts over
background), sit at the bottom of the 16-bit range, and carry a strong
illumination gradient plus static stuck junk.  Two things then wreck detection:

1. The legacy frame read (``cv2.IMREAD_GRAYSCALE``) drops 16-bit data to 8-bit
   by ``>> 8``, so raw values of ~300-900 collapse to ~2 grey levels *before*
   detection ever runs.
2. A bright-top -> dark-bottom illumination gradient (plus static junk) biases a
   global detector to the brightest region, so only part of the field is found.

The correction, per movie::

    temporal-median background subtraction   (static bg / gradient / junk -> 0)
    -> illumination flat-field                (divide by a smoothed gain field)
    -> mild Gaussian denoise
    -> movie-wide stretch to a full-range 8-bit image

Two entry points share one implementation:

* :func:`correct_stack` / :func:`stretch_to_uint` / :func:`preprocess_stack`
  operate on a whole ``(T, H, W)`` stack in memory -- used by
  ``tools/preprocess_lowsnr.py`` (which writes corrected movies to disk) and by
  the FASTplus directional pipeline (which already holds the split stack).
* :func:`fit_model` + :class:`PreprocessModel` split the same maths into a
  *fit* (background, gain field and stretch ceiling, from all frames or an
  evenly spaced sample) and a per-frame *apply*.  This is what the in-pipeline
  :class:`~fastrack.core.input.preprocessed.PreprocessedSource` uses, so the
  strictly per-frame detection flow (and its multiprocessing workers) stays
  unchanged: the model is fitted once, saved next to the detection cache, and
  each worker just applies it to the frames it reads.

Fitting on every frame (``model_frames=0``) reproduces the stack functions
exactly.  ``model_frames=N`` fits on N evenly spaced frames instead, which caps
memory for long movies at the cost of a slightly noisier background estimate --
the fallback for stacks too large to hold in memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

#: Parameter names (and defaults) that define a preprocessing recipe.  Anything
#: that changes the corrected pixels is here, so hashing this dict is a complete
#: cache key (see :func:`params_from_settings`).
DEFAULT_PARAMS = {
    "background": "median",
    "model_frames": 0,
    "flat_sigma": 50.0,
    "denoise_sigma": 0.8,
    "clip_hi_pct": 99.9,
}

_BACKGROUNDS = ("median", "mean", "none")


# --------------------------------------------------------------------------- #
# stack functions (whole movie in memory)
# --------------------------------------------------------------------------- #
def estimate_background_gain(stack: np.ndarray, *, background: str = "median",
                             flat_sigma: float = 50.0):
    """Static background and normalised illumination gain of a ``(T, H, W)`` stack.

    Returns ``(bg, gain)`` as float32 ``(H, W)`` arrays; ``gain`` is ``None``
    when ``flat_sigma`` is 0 (flat-fielding disabled).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("expected a (T, H, W) stack, got shape %r" % (stack.shape,))
    if background not in _BACKGROUNDS:
        raise ValueError("background must be 'median', 'mean' or 'none'")
    stack = stack.astype(np.float32, copy=False)

    if background == "median":
        bg = np.median(stack, axis=0).astype(np.float32)
    elif background == "mean":
        bg = stack.mean(axis=0).astype(np.float32)
    else:
        bg = np.zeros(stack.shape[1:], np.float32)

    gain = None
    if flat_sigma and flat_sigma > 0:
        gain = gaussian_filter(stack.mean(axis=0), flat_sigma)
        gain = (gain / max(float(gain.mean()), 1e-6)).astype(np.float32)
    return bg, gain


def apply_correction(frame: np.ndarray, bg: np.ndarray, gain: Optional[np.ndarray],
                     denoise_sigma: float = 0.8) -> np.ndarray:
    """Correct one ``(H, W)`` frame with a fitted background/gain (float32, >= 0)."""
    corr = frame.astype(np.float32, copy=False) - bg
    if gain is not None:
        corr = corr / gain
    if denoise_sigma and denoise_sigma > 0:
        corr = gaussian_filter(corr, denoise_sigma)
    return np.clip(corr, 0, None)                     # background -> 0


def estimate_noise_sd(frames, bg: np.ndarray, gain: Optional[np.ndarray],
                      denoise_sigma: float = 0.8, max_pairs: int = 5) -> float:
    """Pixel noise (standard deviation) of corrected frames, in corrected units.

    Temporal-difference estimate: for pairs of frames, the robust (MAD) spread
    of ``(a - b) / gain`` after the pipeline's denoise, divided by sqrt(2).
    Static structure (background, residual gradient, stuck junk) cancels in the
    difference and the MAD ignores the few percent of pixels on moving
    filaments, so the estimate is the frame-to-frame noise of the image the
    detector sees -- unaffected by how crowded the field is.  It is measured
    before :func:`apply_correction` clips the background to 0 (which would hide
    half the noise).  Uses up to ``max_pairs`` evenly spaced pairs of
    neighbouring frames and returns their median; a single frame falls back to
    the spread of ``frame - background``.

    This is the yardstick that makes intensity thresholds comparable between
    movies: the movie-wide stretch maps a different corrected value to full
    scale in every movie, so the same camera noise lands at very different
    8-bit levels (~0.5 to ~5 grey levels on one dataset).
    """
    frames = list(frames)
    if not frames:
        raise ValueError("cannot estimate noise on zero frames")

    def _prep(f):
        u = np.asarray(f, dtype=np.float32)
        if gain is not None:
            u = u / gain
        if denoise_sigma and denoise_sigma > 0:
            u = gaussian_filter(u, denoise_sigma)
        return u

    def _mad(x):
        return 1.4826 * float(np.median(np.abs(x - np.median(x))))

    if len(frames) == 1:
        return _mad(_prep(np.asarray(frames[0], dtype=np.float32) - bg))
    starts = sample_indices(len(frames) - 1, max_pairs)
    return float(np.median([_mad(_prep(frames[i]) - _prep(frames[i + 1])) / np.sqrt(2.0)
                            for i in starts]))


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
    bg, gain = estimate_background_gain(stack, background=background,
                                        flat_sigma=flat_sigma)
    return np.stack([apply_correction(f, bg, gain, denoise_sigma) for f in stack])


def stretch_to_uint(corr: np.ndarray, *, clip_hi_pct: float = 99.9,
                    bit_depth: int = 8) -> np.ndarray:
    """Movie-wide contrast stretch of a corrected float stack to full-range uint.

    ``clip_hi_pct`` is the upper percentile (over the whole passed stack) mapped
    to the max output value; 0 maps to 0. A single movie-wide mapping keeps
    intensities comparable across frames. ``bit_depth`` 8 (recommended, sidesteps
    FASTrack's 16->8 read crush) or 16.
    """
    hi = stretch_ceiling(corr, clip_hi_pct)
    return _stretch(corr, hi, bit_depth)


def stretch_ceiling(corr: np.ndarray, clip_hi_pct: float = 99.9) -> float:
    """The corrected-units value mapped to full scale by :func:`stretch_to_uint`."""
    return max(float(np.percentile(corr, clip_hi_pct)), 1e-6)


def _stretch(corr: np.ndarray, hi: float, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        return np.clip(corr / hi * 255.0, 0, 255).astype(np.uint8)
    if bit_depth == 16:
        return np.clip(corr / hi * 65535.0, 0, 65535).astype(np.uint16)
    raise ValueError("bit_depth must be 8 or 16")


def preprocess_stack(stack: np.ndarray, *, background: str = "median",
                     flat_sigma: float = 50.0, denoise_sigma: float = 0.8,
                     clip_hi_pct: float = 99.9, bit_depth: int = 8) -> np.ndarray:
    """Correct + stretch a ``(T, H, W)`` stack to a full-range uint stack (no crop).

    Thin wrapper over :func:`correct_stack` + :func:`stretch_to_uint` for direct
    use; the standalone tool crops between the two when ``--crop-frac`` is set.
    """
    corr = correct_stack(stack, background=background, flat_sigma=flat_sigma,
                          denoise_sigma=denoise_sigma)
    return stretch_to_uint(corr, clip_hi_pct=clip_hi_pct, bit_depth=bit_depth)


# --------------------------------------------------------------------------- #
# fitted model (fit once, apply per frame)
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessModel:
    """A fitted correction: apply it to any frame of the movie it was fitted on."""
    background: np.ndarray            # (H, W) float32
    gain: Optional[np.ndarray]        # (H, W) float32 normalised to mean 1, or None
    hi: float                         # corrected-units value mapped to full scale
    denoise_sigma: float = 0.8
    noise_sd: Optional[float] = None  # pixel noise in corrected units (see estimate_noise_sd)

    def apply(self, frame: np.ndarray, bit_depth: int = 8) -> np.ndarray:
        """Corrected, full-range ``uint8`` (or ``uint16``) version of ``frame``."""
        corr = apply_correction(frame, self.background, self.gain, self.denoise_sigma)
        return _stretch(corr, self.hi, bit_depth)

    def apply_stack(self, stack: np.ndarray, bit_depth: int = 8) -> np.ndarray:
        return np.stack([self.apply(f, bit_depth) for f in stack])

    def output_noise_sd(self, bit_depth: int = 8) -> Optional[float]:
        """``noise_sd`` expressed in the units of :meth:`apply`'s output (or None)."""
        if self.noise_sd is None:
            return None
        full = 255.0 if bit_depth == 8 else 65535.0
        return float(self.noise_sd) / self.hi * full

    # -- persistence (so multiprocessing workers load instead of refit) ----- #
    def save(self, path: str) -> None:
        tmp = path + ".tmp%d" % os.getpid()
        np.savez(tmp, background=self.background,
                 gain=self.gain if self.gain is not None else np.zeros(0, np.float32),
                 hi=np.float64(self.hi), denoise_sigma=np.float64(self.denoise_sigma),
                 noise_sd=np.float64(np.nan if self.noise_sd is None else self.noise_sd))
        os.replace(tmp + ".npz", path)   # atomic: a reader never sees a partial file

    @classmethod
    def load(cls, path: str) -> "PreprocessModel":
        with np.load(path) as z:
            gain = z["gain"]
            # Models saved before noise_sd existed load with noise_sd=None.
            noise = float(z["noise_sd"]) if "noise_sd" in z.files else float("nan")
            return cls(background=z["background"],
                       gain=None if gain.size == 0 else gain,
                       hi=float(z["hi"]), denoise_sigma=float(z["denoise_sigma"]),
                       noise_sd=None if np.isnan(noise) else noise)


def sample_indices(n_frames: int, model_frames: int) -> list:
    """Indices of the frames used to fit the model: all, or ``model_frames`` evenly spaced."""
    if model_frames is None or model_frames <= 0 or model_frames >= n_frames:
        return list(range(n_frames))
    return [int(round(x)) for x in np.linspace(0, n_frames - 1, model_frames)]


def fit_model(frames: Sequence[np.ndarray] | np.ndarray, *, background: str = "median",
              flat_sigma: float = 50.0, denoise_sigma: float = 0.8,
              clip_hi_pct: float = 99.9, model_frames: int = 0) -> PreprocessModel:
    """Fit a :class:`PreprocessModel` on ``frames`` (a ``(T, H, W)`` stack or a list).

    With ``model_frames=0`` every frame is used and ``model.apply_stack(frames)``
    equals :func:`preprocess_stack` bit for bit.  With ``model_frames=N`` only N
    evenly spaced frames are loaded into the fit.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("cannot fit a preprocessing model on zero frames")
    idx = sample_indices(n, model_frames)
    sub = np.stack([np.asarray(frames[i], dtype=np.float32) for i in idx])
    bg, gain = estimate_background_gain(sub, background=background, flat_sigma=flat_sigma)
    corr = np.stack([apply_correction(f, bg, gain, denoise_sigma) for f in sub])
    return PreprocessModel(background=bg, gain=gain,
                           hi=stretch_ceiling(corr, clip_hi_pct),
                           denoise_sigma=float(denoise_sigma),
                           noise_sd=estimate_noise_sd(sub, bg, gain, denoise_sigma))


def fit_model_from_reader(read, frame_nos: Iterable[int], **params) -> PreprocessModel:
    """Fit from a ``read(frame_no) -> (H, W)`` callable, loading only the sampled frames."""
    frame_nos = list(frame_nos)
    idx = sample_indices(len(frame_nos), int(params.pop("model_frames", 0) or 0))
    frames = [read(frame_nos[i]) for i in idx]
    return fit_model(frames, model_frames=0, **params)


# --------------------------------------------------------------------------- #
# settings glue
# --------------------------------------------------------------------------- #
def params_from_settings(section) -> Optional[dict]:
    """The recipe dict for a ``PreprocessSettings`` section, or ``None`` when disabled.

    ``None`` is the "no preprocessing" value everywhere downstream (run kwargs,
    cache keys, task dicts), so a disabled run's names/caches are unchanged.
    """
    if section is None or not getattr(section, "enabled", False):
        return None
    return {k: getattr(section, k) for k in DEFAULT_PARAMS}


def params_hash(params: dict, length: int = 8) -> str:
    """Short stable hash of a recipe dict (for cache tags and output-dir names)."""
    from .io.detection_cache import params_hash as _h
    return _h({k: params.get(k, DEFAULT_PARAMS[k]) for k in DEFAULT_PARAMS}, length)


__all__ = [
    "DEFAULT_PARAMS", "PreprocessModel", "apply_correction", "correct_stack",
    "estimate_background_gain", "estimate_noise_sd", "fit_model", "fit_model_from_reader",
    "params_from_settings", "params_hash", "preprocess_stack", "sample_indices",
    "stretch_ceiling", "stretch_to_uint",
]
