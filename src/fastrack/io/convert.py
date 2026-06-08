"""Batch TIFF-stack -> micro-manager frame-file conversion.

Walks a directory tree and runs :func:`fastrack.io.images.stack_to_tiffs` on
every ``.tif`` stack above a minimum size threshold.  (Port of ``bin/stack2tifs``.)
"""
import os
import sys
import warnings

from .images import stack_to_tiffs

warnings.filterwarnings("ignore")


def run(main_dir, min_size=6.0, frame_rate=1.0):
    if main_dir is None or not os.path.isdir(main_dir):
        sys.exit("Directory doesn't exist. Program is exiting.")

    for root, subFolders, files in os.walk(main_dir):
        for f in files:
            path = os.path.join(root, f)
            if os.path.splitext(f)[1] == ".tif" and os.path.getsize(path) * 1e-6 > min_size:
                stack_to_tiffs(path, frame_rate)
