"""Per-movie filament store -- one ``filXYs<tag>.npz`` (zip) per movie folder.

Collapses the per-frame ``.npy`` fan-out into a single file: each frame is a
zip member ``frameNNNNNN`` holding the same ``filXYs`` object array the ``npy``
backend stores.  Random access by member name keeps reads O(1) and memory
bounded (one frame at a time); writes stream member-by-member so the parent can
consolidate worker results without holding the whole movie in RAM.

Only the parent process writes (single writer), so there are no concurrent-write
races on the one file.  ``open_write`` opens the archive ('w' to rebuild, 'a' to
add missing frames on resume); ``write`` appends a member; ``close`` finalizes.
"""
import io
import os
import zipfile

import numpy as np

from .base import STORES, FilamentStore


def _path(directory, cache_tag):
    return os.path.join(directory, "filXYs%s.npz" % cache_tag)


def _member(frame_no):
    return "frame%06d" % int(frame_no)


@STORES.register("per-movie")
class NpzFilamentStore(FilamentStore):
    def __init__(self):
        self._zip = None
        self._path = None

    # --- streaming write lifecycle ------------------------------------- #
    def open_write(self, directory, cache_tag, force=False):
        self._path = _path(directory, cache_tag)
        # 'w' rebuilds from scratch (force, or no existing archive); 'a' adds
        # only the missing frames the parent dispatched on resume.
        mode = "w" if force or not os.path.isfile(self._path) else "a"
        self._zip = zipfile.ZipFile(self._path, mode, compression=zipfile.ZIP_STORED)

    def write(self, directory, cache_tag, frame_no, filXYs):
        if self._zip is None:
            # Allow a one-off write outside a session (e.g. tests).
            self.open_write(directory, cache_tag, force=False)
            own = True
        else:
            own = False
        buf = io.BytesIO()
        np.save(buf, np.array(filXYs, dtype=object))
        self._zip.writestr(_member(frame_no), buf.getvalue())
        if own:
            self.close()

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None
            self._path = None

    # --- reads ---------------------------------------------------------- #
    def has(self, directory, cache_tag, frame_no):
        path = _path(directory, cache_tag)
        if not os.path.isfile(path):
            return False
        with zipfile.ZipFile(path, "r") as z:
            return _member(frame_no) in z.namelist()

    def read(self, directory, cache_tag, frame_no):
        path = _path(directory, cache_tag)
        with zipfile.ZipFile(path, "r") as z:
            data = z.read(_member(frame_no))
        return np.load(io.BytesIO(data), allow_pickle=True)

    def frames(self, directory, cache_tag):
        path = _path(directory, cache_tag)
        if not os.path.isfile(path):
            return []
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
        nums = [int(n[len("frame"):]) for n in names if n.startswith("frame")]
        return sorted(nums)
