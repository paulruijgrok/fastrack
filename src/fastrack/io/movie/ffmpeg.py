"""ffmpeg H.264/MP4 tracking-movie writer.

Lifted verbatim from ``Motility.make_movie``.  Encodes the ``skeletons_%03d.png``
sequence as H.264 in an MP4 container (yuv420p), which opens in QuickTime,
ImageJ, browsers and most players, then removes the intermediate PNGs.
"""
import glob
import os
import shutil
import subprocess

from .base import MOVIE_WRITERS, MovieWriter


@MOVIE_WRITERS.register("ffmpeg_h264")
class FFmpegH264Writer(MovieWriter):
    def write(self, directory, extra_fname=None,
              input_pattern="skeletons_%03d.png", output_name="filament_tracks.mp4",
              fps=1):
        # The input frames are named ``<prefix>NNN.png``; require at least one.
        prefix = input_pattern.split("%")[0]
        if not glob.glob(os.path.join(directory, prefix + "*.png")):
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg not found on PATH; skipping movie generation.")
            return

        cwd = os.getcwd()
        os.chdir(directory)
        try:
            # Encode H.264 in an MP4 container with the yuv420p pixel format.
            # The old default (mpeg4/FMP4 in an AVI container) is rejected by
            # QuickTime and ImageJ ("Unsupported compression: FMP4").  libx264
            # requires even frame dimensions, so pad up to the next even number.
            result = subprocess.run(
                [ffmpeg, "-y", "-r", str(fps), "-i", input_pattern,
                 "-r", str(fps),
                 "-c:v", "libx264",
                 "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-movflags", "+faststart",
                 output_name],
                check=False,
            )

            if result.returncode != 0 or not os.path.isfile(output_name):
                print("ffmpeg failed to encode the movie (is libx264 available "
                      "in your ffmpeg build?); skipping movie generation.")
                return

            # Encoding happens in ``directory`` (where the frame PNGs live), but
            # when an output destination is given we MOVE the movie there so the
            # input data folder is left clean -- the movie belongs with the other
            # outputs, not next to the raw frames.
            if extra_fname is not None:
                dest = extra_fname + output_name
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                shutil.move(output_name, dest)
                print("Saved movie: %s" % os.path.abspath(dest))
            else:
                print("Saved movie: %s" % os.path.abspath(output_name))

            for png in glob.glob(prefix + "*.png"):
                os.remove(png)
        finally:
            os.chdir(cwd)
