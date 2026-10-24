"""Curve-fitting helpers (Gaussian and length-velocity models).

Moved verbatim from the original ``motility.py``; the numerical behaviour is
unchanged.
"""
import numpy as np
from scipy.optimize import leastsq


def gaussian(X, amp, mu, stdev):
    return amp * np.exp(-(X - mu) ** 2 / (2 * stdev ** 2))


def fit_gaussian(bin_centers, bin_amps):
    err = lambda params: params[0] * np.exp(
        -(bin_centers - params[1]) ** 2 / (2 * params[2] ** 2)
    ) - bin_amps
    params = [50, 700, 100]
    best_params, success = leastsq(err, params, maxfev=1000)
    return best_params[0], best_params[1], best_params[2]


def fit_length_velocity(length, velocity, fil_weights, weighted=False):
    """Fit Uyeda's length-velocity relationship."""
    myosin_density = 1.0 / 36.0
    Neff = length * myosin_density

    weights = np.ones(len(length))
    if weighted:
        weights = fil_weights
    err = lambda params: weights * (
        params[0] * (1.0 - (1.0 - params[1]) ** Neff) - velocity
    )

    params = [700, 0.001]
    best_params, success = leastsq(err, params, maxfev=1000)
    # Unweighted residuals, computed directly (dividing the weighted error by
    # the weights gives 0/0 where a weight is 0).
    residuals = np.asarray(length_velocity(length, *best_params) - velocity)
    return best_params[0], best_params[1], residuals, success


def length_velocity(length, max_vel, f):
    """Uyeda's simple length-velocity relationship."""
    myosin_density = 1.0 / 36.0
    Neff = length * myosin_density
    return max_vel * (1.0 - (1.0 - f) ** Neff)


def coupling_velocity(length, max_vel, amp, tau):
    """Coupling relationship with single exponential decay."""
    return max_vel - amp * np.exp(-length / tau)


def fit_coupling_velocity(length, velocity, fil_weights, weighted=False):
    weights = np.ones(len(length))
    if weighted:
        weights = fil_weights
    err = lambda params: weights * (
        params[0] - params[1] * np.exp(-length / params[2]) - velocity
    )

    params = [700, 200, 500]
    best_params, success = leastsq(err, params, maxfev=1000)
    # Unweighted residuals, computed directly (dividing the weighted error by
    # the weights gives 0/0 where a weight is 0).
    residuals = np.asarray(coupling_velocity(length, *best_params) - velocity)
    return best_params[0], best_params[1], best_params[2], residuals, success


def fit_plateau(length, velocity, model="exp"):
    """Fit the length-velocity relation and return the plateau with its error.

    ``model`` is ``"exp"`` (:func:`coupling_velocity`, V = U - A exp(-L/tau))
    or ``"uyeda"`` (:func:`length_velocity`, V = U (1 - (1-f)^(L/36 nm))).
    Lengths in nm, velocities in nm/s -- normally one point per filament (its
    maximum-velocity window).  The plateau is ``U``, the velocity long
    filaments approach.

    Unlike :func:`fit_coupling_velocity` / :func:`fit_length_velocity` (kept
    for the original behaviour), this starts from data-based values, keeps the
    parameters physical (bounds), and returns standard errors from the fit
    covariance -- valid when the points are independent, i.e. one per filament.

    Returns ``dict(model, plateau, plateau_se, params, params_se, n, ok)``;
    on failure ``plateau`` is -1 and ``ok`` False.
    """
    from scipy.optimize import curve_fit

    L = np.asarray(length, dtype=float)
    V = np.asarray(velocity, dtype=float)
    keep = np.isfinite(L) & np.isfinite(V) & (L > 0) & (V > 0)
    L, V = L[keep], V[keep]
    out = dict(model=model, plateau=-1.0, plateau_se=np.nan, params=None,
               params_se=None, n=int(len(L)), ok=False)
    if model == "exp":
        func, npar = coupling_velocity, 3
    elif model == "uyeda":
        func, npar = length_velocity, 2
    else:
        raise ValueError("model must be 'exp' or 'uyeda', got %r" % (model,))
    if len(L) <= npar + 1:
        return out
    u0 = float(np.percentile(V, 90))
    if model == "exp":
        p0 = [u0, max(u0 - float(np.percentile(V, 10)), 1.0), float(np.median(L))]
        bounds = ([0.0, -np.inf, 1.0], [np.inf, np.inf, np.inf])
    else:
        p0 = [u0, 0.02]
        bounds = ([0.0, 1e-6], [np.inf, 1.0])
    try:
        params, cov = curve_fit(func, L, V, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    se = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(npar, np.nan)
    out.update(plateau=float(params[0]), plateau_se=float(se[0]),
               params=[float(p) for p in params], params_se=[float(s) for s in se], ok=True)
    return out
