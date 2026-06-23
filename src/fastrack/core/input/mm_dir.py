"""Micro-manager frame-folder source -- the original input layout.

A movie is a directory of single-page TIFFs named ``img_000000NNN_<tail>_000.tif``
plus an optional ``metadata.txt`` carrying per-frame ``ElapsedTime-ms``.  This
wraps exactly what ``Frame.read_frame`` / ``Motility.read_metadata`` did, so the
default path is unchanged.
"""
import glob
import os
import re

import cv2
import numpy as np

from .base import FrameSource, register_source, to_pipeline_image

_FRAME_RE = re.compile(r"^img_\d+_.*_000\.tif$")


@register_source("mm_dir")
class MicroManagerDirSource(FrameSource):
    def __init__(self, directory, header="img_000000", tail=None, frame_rate=1.0):
        self.directory = os.path.abspath(directory)
        self.header = header
        self.frame_rate = frame_rate
        self._frames = self._discover()
        self.tail = tail if tail is not None else self._derive_tail()

    # ----- discovery helpers --------------------------------------------- #
    @staticmethod
    def looks_like(directory):
        return any(_FRAME_RE.match(os.path.basename(p))
                   for p in glob.glob(os.path.join(directory, "*.tif")))

    def _discover(self):
        nums = []
        for p in glob.glob(os.path.join(self.directory, "*.tif")):
            b = os.path.basename(p)
            if _FRAME_RE.match(b):
                try:
                    nums.append(int(b.split("_")[1]))   # img_<NNNNNNNNN>_..._000.tif
                except (IndexError, ValueError):
                    continue
        return sorted(nums)

    def _derive_tail(self):
        tifs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.directory, "*.tif")))
        for b in tifs:
            if _FRAME_RE.match(b):
                parts = b.split("_")
                return parts[2] if len(parts) > 2 else ""
        return ""

    def _path(self, frame_no):
        return os.path.join(
            self.directory, "%s%03d_%s_000.tif" % (self.header, int(frame_no), self.tail))

    # ----- FrameSource API ----------------------------------------------- #
    @property
    def count(self):
        return len(self._frames)

    def frame_numbers(self):
        return list(self._frames)

    def read(self, frame_no):
        # IMREAD_GRAYSCALE reproduces the legacy 16->8-bit read; to_pipeline_image
        # then applies the x257 (img_as_uint) scaling -- identical to the old path.
        img8 = cv2.imread(self._path(frame_no), cv2.IMREAD_GRAYSCALE)
        if img8 is None:
            raise FileNotFoundError(self._path(frame_no))
        return to_pipeline_image(img8)

    def elapsed_times(self):
        meta = os.path.join(self.directory, "metadata.txt")
        if not os.path.isfile(meta):
            return None
        times = []
        with open(meta) as f:
            for line in f:
                m = re.search(r'ElapsedTime-ms":\s+(\d+),', line)
                if m:
                    times.append(float(m.group(1)))
        if not times:
            return None
        return np.sort(0.001 * np.array(times))

    @property
    def identity(self):
        return self.directory

    def descriptor(self):
        return {"kind": "mm_dir", "directory": self.directory,
                "header": self.header, "tail": self.tail, "frame_rate": self.frame_rate}

    @classmethod
    def from_descriptor(cls, d):
        return cls(d["directory"], header=d.get("header", "img_000000"),
                   tail=d.get("tail"), frame_rate=d.get("frame_rate", 1.0))
