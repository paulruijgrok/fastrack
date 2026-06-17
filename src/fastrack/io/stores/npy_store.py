"""Per-frame ``.npy`` filament store -- the original ``filXYs`` layout.

One ``filXYs<tag>NNN.npy`` file per frame, byte-identical to the files the
original pipeline wrote (and that the committed golden-master baseline was
captured with).  This is the default layout.
"""
import glob
import os
import re

import numpy as np

from .base import STORES, FilamentStore


def _path(directory, cache_tag, frame_no):
    # Matches the original Frame.save_filXYs/read_filXYs naming exactly.
    return os.path.join(directory, "filXYs%s%03d.npy" % (cache_tag, int(frame_no)))


@STORES.register("npy")
class NpyFilamentStore(FilamentStore):
    def has(self, directory, cache_tag, frame_no):
        return os.path.isfile(_path(directory, cache_tag, frame_no))

    def write(self, directory, cache_tag, frame_no, filXYs):
        # np.save appends ".npy"; strip it from the target so we don't double it.
        target = _path(directory, cache_tag, frame_no)[: -len(".npy")]
        np.save(target, np.array(filXYs, dtype=object))

    def read(self, directory, cache_tag, frame_no):
        return np.load(_path(directory, cache_tag, frame_no), allow_pickle=True)

    def frames(self, directory, cache_tag):
        # filXYs<tag>NNN.npy -> NNN.  Anchor on the trailing digits so a non-empty
        # tag (e.g. "_ridge") doesn't swallow a leading digit.
        pat = re.compile(r"^filXYs%s(\d+)\.npy$" % re.escape(cache_tag))
        nums = []
        for p in glob.glob(os.path.join(directory, "filXYs%s*.npy" % cache_tag)):
            m = pat.match(os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
        return sorted(nums)
