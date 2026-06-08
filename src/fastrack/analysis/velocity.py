"""Velocity-metric helpers (length-velocity binning).

Moved verbatim from the original ``motility.py``.
"""
import numpy as np


def bin_length_velocity(length, velocity, dx=100):
    max_len = np.max(length)
    bin_vel = []
    for i in np.arange(0, int(max_len / dx)):
        valid = (length > i * dx) * (length <= (i + 1) * dx)
        if np.sum(valid) > 0:
            mean_len = np.mean(length[valid])
            mean_vel = np.mean(velocity[valid])
            bin_vel.append([mean_len, mean_vel])
    return np.array(bin_vel)
