"""Atomic ``.npy`` writes for the detection / links caches.

Cached files are reused on the next run (unless forced), so a process killed
mid-write -- e.g. a preempted cluster job -- must not leave a truncated file
behind.  Write to a temp name next to the target, then ``os.replace`` it in.
"""
import os

import numpy as np


def save_npy_atomic(path, arr):
    """``np.save(path, arr)`` that never leaves a partial file at ``path``.

    ``path`` must end in ``.npy`` (unlike ``np.save``, nothing is appended).
    The temp name ends in ``.tmp`` so cache globs on ``*.npy`` never see it.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
