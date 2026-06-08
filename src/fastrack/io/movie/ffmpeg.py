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
    def write(self, directory, extra_fname=None):
        if not os.path.isfile(os.path.join(directory, "paths_2D.png")):
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
            movie_name = "filament_tracks.mp4"
            result = subprocess.run(
                [ffmpeg, "-y", "-r", "1", "-i", "skeletons_%03d.png",
                 "-r", "1",
                 "-c:v", "libx264",
                 "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-movflags", "+faststart",
                 movie_name],
                check=False,
            )

            if result.returncode != 0 or not os.path.isfile(movie_name):
                print("ffmpeg failed to encode the movie (is libx264 available "
                      "in your ffmpeg build?); skipping movie generation.")
                return

            if extra_fname is not None:
                shutil.copy(movie_name, extra_fname + movie_name)

            for png in glob.glob("skeletons_*.png"):
                os.remove(png)
        finally:
            os.chdir(cwd)
