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
    residuals = np.array(err(best_params) / weights)
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
    residuals = np.array(err(best_params) / weights)
    return best_params[0], best_params[1], best_params[2], residuals, success
