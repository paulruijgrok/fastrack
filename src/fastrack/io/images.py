"""Image and frame-file I/O helpers.

``stack_to_tiffs`` converts a multi-page TIFF stack into micro-manager-style
single frame files; ``alpha_composite`` overlays one RGBA image on another
(the pure-Python replacement for the original ImageMagick ``composite`` call).
Both moved verbatim from the original ``motility.py``.
"""
import os

import imageio.v2 as imageio
from PIL import Image


def stack_to_tiffs(fname, frame_rate=1.0):
    """Read a multi-page TIFF stack and write individual micro-manager frames."""
    abs_path = os.path.abspath(fname)
    head, tail = os.path.split(abs_path)
    base, ext = os.path.splitext(tail)

    new_dir = os.path.join(head, ("_".join(base.split())).replace("#", ""))
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    tiff_frames = imageio.mimread(fname, memtest=False)
    num_frames = len(tiff_frames)

    with open(os.path.join(new_dir, "metadata.txt"), "w") as f:
        elapsed_time_ms = 0.0
        for i in range(num_frames):
            fout = os.path.join(new_dir, "img_000000%03d__000.tif" % (i))
            imageio.imwrite(fout, tiff_frames[i])
            f.write('  "ElapsedTime-ms": %d,\n' % (elapsed_time_ms))
            elapsed_time_ms += 1000 * 1.0 / frame_rate


def alpha_composite(fg_path, bg_path, out_path):
    """Composite ``fg_path`` (with alpha) over ``bg_path`` and write ``out_path``.

    Pure-Python replacement for the original ImageMagick ``composite`` shell
    call, so movie generation works on any platform without external tools.
    """
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")
    if bg.size != fg.size:
        bg = bg.resize(fg.size)
    Image.alpha_composite(bg, fg).save(out_path)
