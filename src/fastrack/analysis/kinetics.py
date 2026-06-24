"""Kinetic-model fitting of the per-frame velocity time course (FASTplus, req. 4).

After per-frame averaging, the signed mean velocity v(t) responds to external
perturbations (e.g. light pulses) applied at known times.  This module fits
exponential rise/decay models keyed to those perturbation times and reports the
time constants.

Models (t0 = perturbation onset; for t < t0, v = baseline):
    exp_rise(t)  = v0 + amp * (1 - exp(-(t - t0) / tau))
    exp_decay(t) = v0 + amp * exp(-(t - t0) / tau)

The fitter prefers ``scipy.optimize.curve_fit`` and falls back to an internal
Gauss-Newton solver so it works without scipy.  numpy only otherwise.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Model functions (vectorized; clamp t < t0 to the baseline)
# --------------------------------------------------------------------------- #
def exp_rise(t, v0, amp, tau, t0):
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, v0, dtype=float)
    m = t >= t0
    out[m] = v0 + amp * (1.0 - np.exp(-(t[m] - t0) / tau))
    return out


def exp_decay(t, v0, amp, tau, t0):
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, v0 + amp, dtype=float)
    m = t >= t0
    out[m] = v0 + amp * np.exp(-(t[m] - t0) / tau)
    return out


_MODELS = {"exp_rise": exp_rise, "exp_decay": exp_decay}


# --------------------------------------------------------------------------- #
# Minimal Gauss-Newton fallback for f(t; p) with fixed t0
# --------------------------------------------------------------------------- #
def _gauss_newton(model, t, y, p0, fixed, n_iter=200, lam=1e-3):
    p = np.array(p0, dtype=float)
    t = np.asarray(t, float); y = np.asarray(y, float)

    def resid(pp):
        return model(t, *pp, *fixed) - y

    for _ in range(n_iter):
        r = resid(p)
        J = np.zeros((len(t), len(p)))
        for k in range(len(p)):
            step = 1e-6 * (abs(p[k]) + 1e-6)
            pk = p.copy(); pk[k] += step
            J[:, k] = (resid(pk) - r) / step
        JTJ = J.T @ J + lam * np.eye(len(p))
        try:
            dp = np.linalg.solve(JTJ, -J.T @ r)
        except np.linalg.LinAlgError:
            break
        p_new = p + dp
        if np.sum(resid(p_new) ** 2) < np.sum(r ** 2):
            p = p_new; lam = max(lam * 0.7, 1e-9)
        else:
            lam = min(lam * 2.5, 1e6)
        if np.linalg.norm(dp) < 1e-9:
            break
    return p


def _fit_one(model_name: str, t: np.ndarray, y: np.ndarray, t0: float) -> Dict:
    """Fit one model with onset fixed at t0; returns parameter dict + quality."""
    model = _MODELS[model_name]
    y = np.asarray(y, float); t = np.asarray(t, float)
    base = float(np.nanmean(y[t < t0])) if np.any(t < t0) else float(y[0])
    span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    amp0 = (float(np.nanmean(y[t >= t0])) - base) if np.any(t >= t0) else span
    tau0 = max((t.max() - t0) / 3.0, 1e-3)
    p0 = [base, amp0, tau0]

    def f(tt, v0, amp, tau):
        return model(tt, v0, amp, tau, t0)

    ok = np.isfinite(y)
    tt, yy = t[ok], y[ok]
    if _HAVE_SCIPY and len(tt) >= 4:
        try:
            popt, _ = _scipy_curve_fit(f, tt, yy, p0=p0, maxfev=5000)
        except Exception:
            popt = _gauss_newton(model, tt, yy, p0, (t0,))
    else:
        popt = _gauss_newton(model, tt, yy, p0, (t0,))

    v0, amp, tau = float(popt[0]), float(popt[1]), float(abs(popt[2]))
    pred = f(tt, v0, amp, tau)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2)) or 1.0
    return {"model": model_name, "t0": t0, "v0": v0, "amp": amp,
            "tau": tau, "r2": 1.0 - ss_res / ss_tot, "n": int(len(tt))}


class KineticModelFitter:
    """Fit exponential rise/decay around external perturbation times."""

    def __init__(self, perturbation_times_s: Sequence[float] = ()):
        self.perturbations = [float(x) for x in perturbation_times_s]

    # ------------------------------------------------------------------ #
    def fit(self, time_s, mean_velocity, model: str = "exp_rise") -> Dict:
        """Fit a single model over the whole trace (onset = first perturbation).

        ``model`` is "exp_rise" or "exp_decay".
        """
        if model not in _MODELS:
            raise ValueError("unknown kinetic model %r" % model)
        t0 = self.perturbations[0] if self.perturbations else float(np.min(time_s))
        return _fit_one(model, np.asarray(time_s, float),
                        np.asarray(mean_velocity, float), t0)

    def fit_segments(self, time_s, mean_velocity) -> List[Dict]:
        """Fit one rise/decay per inter-perturbation segment.

        Alternates exp_rise / exp_decay starting from the first perturbation,
        the common pattern for a light-on / light-off train.
        """
        t = np.asarray(time_s, float); y = np.asarray(mean_velocity, float)
        edges = list(self.perturbations) + [float(t.max()) + 1.0]
        results = []
        for k in range(len(self.perturbations)):
            lo, hi = edges[k], edges[k + 1]
            seg = (t >= lo) & (t < hi)
            if seg.sum() < 4:
                continue
            model = "exp_rise" if k % 2 == 0 else "exp_decay"
            results.append(_fit_one(model, t[seg], y[seg], lo))
        return results

    @staticmethod
    def predict(model: str, time_s, params: Dict) -> np.ndarray:
        return _MODELS[model](time_s, params["v0"], params["amp"],
                              params["tau"], params["t0"])
