"""Filament width measured directly from the image (focus metric).

Detector-reported ridge widths depend on the detector's own scale-space model
(``ridge`` and ``ridge-fast`` differ by ~60 nm on the same data), so they make
a poor, implementation-dependent focus gauge.  This measures the width from
the pixels instead: for each detected filament, intensity profiles are sampled
perpendicular to the centerline, re-centred to sub-pixel precision, combined
into one robust (median) profile, and fitted with a Gaussian on a sloped
background.  The result is the FWHM of the filament's cross-section -- for a
filament much thinner than the PSF, the FWHM of the line-spread function, which
grows with defocus.

The image should be the *raw* (or flat-field corrected, but not smoothed)
frame: any smoothing adds its own width.  Only the filament positions come
from the detector.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _gauss_slope(x, a, mu, sigma, c0, c1):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c0 + c1 * x


def _normals(contour, k=2):
    """Unit normals along an (N, 2) [row, col] contour (direction from i-k to i+k)."""
    c = np.asarray(contour, float)
    n = len(c)
    lo = np.clip(np.arange(n) - k, 0, n - 1)
    hi = np.clip(np.arange(n) + k, 0, n - 1)
    t = c[hi] - c[lo]
    norm = np.hypot(t[:, 0], t[:, 1])
    ok = norm > 0
    t[ok] /= norm[ok, None]
    return np.column_stack([-t[:, 1], t[:, 0]]), ok


def _id_map(shape, contours):
    """Label image of all centerlines, dilated by 1 px (0 = no filament)."""
    ids = np.zeros(shape, dtype=np.int32)
    for i, c in enumerate(contours, start=1):
        if len(c):
            r = np.rint(c[:, 0]).astype(int)
            q = np.rint(c[:, 1]).astype(int)
            for dr in (-1, 0, 1):
                for dq in (-1, 0, 1):
                    ids[np.clip(r + dr, 0, shape[0] - 1), np.clip(q + dq, 0, shape[1] - 1)] = i
    return ids


def _fit_sigma(img, ids, fid, c, half_width, step, trim, min_points, max_sigma):
    """Gaussian sigma (px) of one filament's median cross-section, or NaN."""
    x = np.arange(-half_width, half_width + 1e-9, step)
    if len(c) < min_points + 2 * trim:
        return np.nan
    nrm, ok = _normals(c)
    sel = np.arange(trim, len(c) - trim)
    sel = sel[ok[sel]]
    if len(sel) < min_points:
        return np.nan
    pts, nv = c[sel], nrm[sel]
    rr = pts[:, 0, None] + nv[:, 0, None] * x[None, :]
    cc = pts[:, 1, None] + nv[:, 1, None] * x[None, :]
    # drop profiles that cross another filament or leave the image
    hit = ids[np.clip(np.rint(rr).astype(int), 0, img.shape[0] - 1),
              np.clip(np.rint(cc).astype(int), 0, img.shape[1] - 1)]
    keep = (~np.any((hit != 0) & (hit != fid), axis=1)
            & ((rr >= 0) & (rr <= img.shape[0] - 1)
               & (cc >= 0) & (cc <= img.shape[1] - 1)).all(axis=1))
    if keep.sum() < min_points:
        return np.nan
    rr, cc = rr[keep], cc[keep]
    prof = map_coordinates(img, [rr.ravel(), cc.ravel()], order=3,
                           mode="nearest").reshape(len(rr), len(x))
    # re-centre each profile on its intensity-weighted centre near x=0
    # (contours are rounded to whole pixels: up to 0.5 px off-centre)
    centre = np.abs(x) <= 1.5
    p = prof[:, centre] - prof[:, centre].min(axis=1, keepdims=True)
    w = p.sum(axis=1)
    good = w > 0
    if good.sum() < min_points:
        return np.nan
    shift = (p[good] * x[centre][None, :]).sum(axis=1) / w[good]
    aligned = np.array([np.interp(x, x - sh, row) for sh, row in zip(shift, prof[good])])
    med = np.median(aligned, axis=0)
    edge = np.abs(x) > half_width - 2
    base = float(np.median(med[edge]))
    a0 = med[np.abs(x) <= step].max() - base
    if not np.isfinite(a0) or a0 <= 0:
        return np.nan
    try:
        par, _ = curve_fit(_gauss_slope, x, med, p0=[a0, 0.0, 1.5, base, 0.0], maxfev=2000,
                           bounds=([0, -1.5, 0.3, -np.inf, -np.inf],
                                   [np.inf, 1.5, max_sigma, np.inf, np.inf]))
    except (RuntimeError, ValueError):
        return np.nan
    return float(par[2])


def filament_profile_widths(img, contours, measure=None, half_width=6.0, step=0.25,
                            trim=2, min_points=5, max_sigma=8.0, max_half_width=16.0):
    """FWHM (pixels) of each filament's cross-section.

    Parameters
    ----------
    img : 2-D array, raw intensities (unsmoothed).
    contours : iterable of (N, 2) integer/float [row, col] centerlines -- all
        filaments in the image (used to reject profiles crossing a neighbour).
    measure : indices of the contours to measure (default: all).
    half_width : initial profile half-length, pixels.  When the fitted sigma
        is wide (window < 3.5 sigma), the fit is repeated with a window of
        3.5 sigma (up to ``max_half_width``) -- a window that is too narrow
        lets the background terms absorb the tails and under-reads blur.
    step : profile sampling step, pixels.
    trim : contour points dropped at each end (ends are tapered / curved).
    min_points : filaments with fewer usable profiles are skipped.

    Profiles that cross another contour (or leave the image) are dropped, so
    neighbours don't bias the fit.  Returns a list of FWHMs in pixels (NaN
    where too few clean profiles remain or the fit failed), one per measured contour.
    """
    img = np.asarray(img, dtype=np.float32)
    contours = [np.asarray(c, float).reshape(-1, 2) for c in contours]
    ids = _id_map(img.shape, contours)
    out = []
    for i in (range(len(contours)) if measure is None else measure):
        fid, c = i + 1, contours[i]
        hw = half_width
        sig = _fit_sigma(img, ids, fid, c, hw, step, trim, min_points, max_sigma)
        for _ in range(2):
            if not np.isfinite(sig) or 3.5 * sig <= hw or hw >= max_half_width:
                break
            hw = min(max_half_width, 3.5 * sig)
            sig = _fit_sigma(img, ids, fid, c, hw, step, trim, min_points, max_sigma)
        out.append(sig * FWHM_PER_SIGMA if np.isfinite(sig) else np.nan)
    return out
