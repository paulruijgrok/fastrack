"""Contour geometry helpers.

Moved verbatim from the original ``motility.py``.
"""
import numpy as np


def contour2contour(contour1, contour2, fil_direction):
    """Find the distance between two contours."""
    short_contour = contour1
    long_contour = contour2

    if len(short_contour) > len(long_contour):
        short_contour = contour2
        long_contour = contour1

    short_len = len(short_contour)
    long_len = len(long_contour)

    multiplicate_measures = long_len - short_len + 1
    distance_score = 0
    for i in range(multiplicate_measures):
        long_short_diff = (
            long_contour[i : i + short_len, :][::fil_direction] - short_contour
        )
        distance_length = np.mean(np.sqrt(np.sum(long_short_diff ** 2, axis=1)))
        distance_score += distance_length

    distance_score /= multiplicate_measures
    return distance_score


def vec_length(vec):
    """Euclidian length of each row of a 2-D array."""
    return np.sqrt(np.sum(vec ** 2, axis=1))
