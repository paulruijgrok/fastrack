"""Motility: the per-movie analysis driver.

Orchestrates detection (delegated to a :mod:`fastrack.core.detection` strategy),
frame-to-frame linking (delegated to a :mod:`fastrack.core.tracking` strategy),
path construction, velocity statistics, plotting and movie output.

NOTE (refactor in progress): the plotting methods and the movie/compositing
helpers still live here for now; they move to ``fastrack.viz`` and
``fastrack.io`` in subsequent phases.  Detection and linking have already been
lifted behind their interfaces -- see ``read_frame`` and ``make_frame_links``.
The numerical behaviour is unchanged from the original ``motility.py``.
"""
import os
import re

import cv2
import numpy as np
from numpy import ma

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as py  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from skimage.morphology import dilation, disk, skeletonize  # noqa: E402
from ..analysis import (  # noqa: E402
    coupling_velocity,
    fit_coupling_velocity,
    fit_length_velocity,
    length_velocity,
)
from ..io.images import alpha_composite  # noqa: E402
from ..io.movie import MOVIE_WRITERS  # noqa: E402
from ..io.stores import STORES  # noqa: E402
from ..viz.plots import MotilityPlots  # noqa: E402
from .detection import DETECTORS  # noqa: E402
from .filament import Filament  # noqa: E402
from .frame import Frame  # noqa: E402
from .link import Link, Path  # noqa: E402
from .tracking import LINKERS  # noqa: E402


class Motility(MotilityPlots):
    def __init__(self):
        self.elapsed_times = []
        self.dt = 1.0
        self.dx = 80.65
        self.frame = None
        self.frame1 = None
        self.frame2 = None
        self.frame_links = []

        self.min_velocity = 80
        self.max_velocity = 25
        self.min_fil_length = 0
        self.max_fil_length = 125
        self.max_fil_width = 25

        self.max_length_dif = 5
        self.max_velocity_dif = 5

        self.directory = ""
        self.header = ""
        self.tail = ""
        self.num_frames = 0

        self.norm_len_vel = []
        self.full_len_vel = []
        self.max_len_vel = []

        self.corr_lens = []

        self.paths = []
        self.path_img = None

        self.width = 1002
        self.height = 1004

        self.overlap_score_cutoff = 0.4
        self.log_area_score_cutoff = 1.0
        self.dif_log_area_score_cutoff = 0.5
        # Original code also reads ``diff_log_area_score_cutoff`` from the driver.
        self.diff_log_area_score_cutoff = 0.5

        self.force_analysis = False

        # When True, reproduce the original Python 2 ``make_frame_links``
        # behaviour, where the partner filament's identity/geometry was taken
        # from the leftover inner-loop variable (the last *unsorted* candidate)
        # rather than the sorted best candidate whose scores are used.  This is
        # a latent bug in the original, but it is the behaviour that generated
        # the published results, so it is offered for bit-for-bit reproduction.
        # Default (False) uses the corrected, self-consistent linking.
        self.legacy_linking = False

        # Propagated to each Frame: run the full-frame percentile filters on an
        # 8-bit rescaling for speed (see Frame.fast_rank).  ON by default.
        self.fast_rank = True

        # Propagated to each Frame: one-pass morphological-gradient contrast map
        # instead of two percentile passes (see Frame.morph_contrast).  OFF by
        # default.
        self.morph_contrast = False

        # Names of the pluggable strategies to use (looked up in the registries).
        self.detection_algorithm = "entropy"
        self.tracking_algorithm = "greedy"
        self.movie_format = "ffmpeg_h264"

        # Extra keyword arguments for non-entropy detectors (e.g. ridge params).
        self.detection_params = {}
        # Suffix for cached filXYs files so detectors don't share a cache
        # ("" for entropy keeps the original filenames).
        self.cache_tag = ""

        # Intermediate filXYs cache layout: "per-frame" (one .npy per frame,
        # the original/default) or "per-movie" (one .npz per movie).
        self.cache_layout = "per-frame"
        self._store = None

    # ----- pluggable strategy factories ----------------------------------- #
    def get_store(self):
        """Build (once) the configured filament store from ``cache_layout``."""
        if self._store is None:
            name = "per-movie" if self.cache_layout == "per-movie" else "npy"
            self._store = STORES.create(name)
        return self._store

    def get_detector(self):
        """Build the configured filament detector from the current settings.

        The entropy detector takes the fast_rank / morph_contrast options;
        other detectors (e.g. ridge) receive ``detection_params``.
        """
        if self.detection_algorithm == "entropy":
            return DETECTORS.create(
                "entropy",
                fast_rank=self.fast_rank,
                morph_contrast=self.morph_contrast,
            )
        return DETECTORS.create(self.detection_algorithm, **self.detection_params)

    def get_linker(self):
        """Build the configured frame-to-frame linker from the current settings.

        Note: ``dif_log_area_score_cutoff`` (single 'f') is the value the
        original ``make_frame_links`` actually consulted; the driver's
        ``diff_log_area_score_cutoff`` was never read.  That behaviour is
        preserved here so results match the reference exactly.
        """
        return LINKERS.create(
            self.tracking_algorithm,
            min_fil_length=self.min_fil_length,
            max_velocity=self.max_velocity,
            max_length_dif=self.max_length_dif,
            overlap_score_cutoff=self.overlap_score_cutoff,
            log_area_score_cutoff=self.log_area_score_cutoff,
            dif_log_area_score_cutoff=self.dif_log_area_score_cutoff,
            legacy=self.legacy_linking,
        )

    # ----- velocity / length data helpers --------------------------------- #
    def min_length_filter(self, min_filament_length):
        valid_length = np.nonzero(self.full_len_vel[:, 0] >= min_filament_length)[0]
        self.full_len_vel = self.full_len_vel[valid_length, :]

    def read_metadata(self):
        fname = self.directory + "/metadata.txt"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                lines = f.readlines()
            filtered_lines = [x for x in lines if x.find('"ElapsedTime-ms"') > 0]

            self.elapsed_times = []
            for line in filtered_lines:
                m = re.search(r'ElapsedTime-ms":\s+(\d+),', line)
                if m is not None:
                    self.elapsed_times.append(float(m.group(1)))

            self.elapsed_times = 0.001 * np.array(self.elapsed_times)
            self.elapsed_times = np.sort(self.elapsed_times)

    def calc_persistence_len(self):
        self.final_corr_len = np.zeros(1000)
        self.final_corr_weight = np.zeros(1000)

        for corr_len in self.corr_lens:
            if len(corr_len) == 0:
                continue
            new_corr_len = np.zeros(1000)
            new_corr_weight = np.zeros(1000)

            new_corr_len[np.arange(corr_len.shape[0], dtype=int)] = corr_len[:, 1]
            new_corr_weight[np.arange(corr_len.shape[0], dtype=int)] = corr_len[:, 0]

            self.final_corr_len += new_corr_len * new_corr_weight
            self.final_corr_weight += new_corr_weight

        valid = self.final_corr_weight > 0
        self.final_corr_len[valid] = self.final_corr_len[valid] / self.final_corr_weight[valid]
        self.final_corr_len = self.final_corr_len[valid]
        self.final_corr_weight = self.final_corr_weight[valid]

    def wire_frame_links(self, depth=5):
        """Wire disconnected frame links to create contiguous paths."""
        num_frames = len(self.frame_links)
        for d in range(1, depth + 1):
            for f1 in range(len(self.frame_links)):
                possible_links = [
                    link for link in self.frame_links[f1] if link.forward_link is None
                ]
                for l1 in range(len(possible_links)):
                    link1 = possible_links[l1]
                    new_filament1 = Filament()
                    new_filament1.contour = link1.filament2_contour
                    new_filament1.fil_length = link1.filament2_length
                    new_filament1.cm = link1.filament2_cm
                    new_filament1.time = link1.filament2_time

                    avg_distance_score_1 = (
                        possible_links[l1].distance_score / possible_links[l1].dt
                    )

                    if f1 + d < num_frames:
                        forward_links = [
                            link
                            for link in self.frame_links[f1 + d]
                            if link.reverse_link is None
                            and np.sqrt(np.sum((link.filament1_cm - new_filament1.cm) ** 2))
                            < d * self.max_velocity
                            and np.fabs(link.filament1_length - new_filament1.fil_length)
                            < self.max_length_dif
                        ]
                        for l2 in range(len(forward_links)):
                            link2 = forward_links[l2]
                            new_filament2 = Filament()
                            new_filament2.contour = link2.filament1_contour
                            new_filament2.fil_length = link2.filament1_length
                            new_filament2.cm = link2.filament1_cm
                            new_filament2.time = link2.filament1_time

                            avg_distance_score_2 = (
                                forward_links[l2].distance_score / forward_links[l2].dt
                            )
                            avg_distance_score = 0.5 * (
                                avg_distance_score_1 + avg_distance_score_2
                            )

                            dt = new_filament2.time - new_filament1.time

                            (
                                overlap_score,
                                area_score,
                                distance_score,
                                fil_direction,
                                mov_direction,
                            ) = new_filament1.sim_score(new_filament2)
                            if (
                                np.fabs(overlap_score) > self.overlap_score_cutoff
                                and np.log10(area_score) < self.log_area_score_cutoff
                                and distance_score / dt < 2 * avg_distance_score
                                and distance_score / dt > 0.5 * avg_distance_score
                            ):
                                new_link = Link()
                                new_link.frame1_no = link1.frame2_no
                                new_link.frame2_no = link2.frame1_no
                                new_link.filament1_label = link1.filament2_label
                                new_link.filament2_label = link2.filament1_label
                                new_link.filament1_cm = link1.filament2_cm
                                new_link.filament2_cm = link2.filament1_cm
                                new_link.filament1_length = link1.filament2_length
                                new_link.filament2_length = link2.filament1_length
                                new_link.filament1_contour = link1.filament2_contour
                                new_link.filament2_contour = link2.filament1_contour
                                new_link.filament1_midpoint = link1.filament2_midpoint
                                new_link.filament2_midpoint = link2.filament1_midpoint
                                new_link.filament1_time = link1.filament2_time
                                new_link.filament2_time = link2.filament1_time

                                new_link.fil_direction = fil_direction
                                new_link.mov_direction = mov_direction

                                new_link.overlap_score = overlap_score
                                new_link.area_score = area_score
                                new_link.distance_score = distance_score

                                new_link.average_length = 0.5 * (
                                    new_link.filament1_length + new_link.filament2_length
                                )
                                new_link.instant_velocity = new_link.distance_score / dt
                                new_link.dt = dt

                                new_link.direct_link = False

                                self.frame_links[new_link.frame1_no].append(new_link)

                                link1.forward_link = new_link
                                link2.reverse_link = new_link

                                new_link.reverse_link = link1
                                new_link.forward_link = link2

    def read_frame_links(self):
        """Read frame links saved as a pickled .npy object array."""
        links_file = self.directory + "/links%s.npy" % self.cache_tag
        if not self.force_analysis and os.path.exists(links_file):
            try:
                self.frame_links = list(
                    np.load(links_file, allow_pickle=True)
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                print(
                    "Movie analysed previously with an old version of motility."
                    " Links will be regenerated."
                )
                return False
            return True
        return False

    def reconstruct_skeleton_images(self, frame_label=False, time_label=False,
                                    frame_interval_s=1.0, font_scale=0.6):
        if not os.path.isfile(self.directory + "/paths_2D.png"):
            return

        # Prefer acquisition metadata for the time label; load it if needed.
        if time_label and len(self.elapsed_times) == 0:
            try:
                self.read_metadata()
            except Exception:
                pass

        ratio = self.width / 1002.0

        for i in range(len(self.frame_links)):
            new_frame = Frame()
            new_frame.img_skeletons = np.zeros((self.width, self.height), dtype=bool)
            for link in self.frame_links[i]:
                new_frame.img_skeletons[
                    link.filament1_contour[:, 0], link.filament1_contour[:, 1]
                ] = True

            new_frame.img_skeletons = dilation(
                new_frame.img_skeletons, footprint=disk(int(round(ratio * 6)))
            )
            new_frame.img_skeletons = ma.masked_where(
                new_frame.img_skeletons == 0, new_frame.img_skeletons
            )

            py.figure()
            py.imshow(new_frame.img_skeletons, cmap=cm.gray, interpolation="nearest", alpha=1.0)

            arrow_length = ratio * 1.0
            for link in self.frame_links[i]:
                velocity = link.instant_velocity
                mp_1 = link.filament1_midpoint
                mp_2 = link.filament2_midpoint
                mp_diff = mp_2 - mp_1

                py.arrow(
                    mp_1[1],
                    mp_1[0],
                    arrow_length * mp_diff[1],
                    arrow_length * mp_diff[0],
                    color="r",
                    head_width=ratio * 20,
                    head_length=ratio * 30,
                    alpha=1.0,
                )
                py.text(mp_1[1], mp_1[0], "%.f" % (velocity), fontsize=10, color="k", alpha=1.0)

            ax = py.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            if frame_label or time_label:
                links_i = self.frame_links[i]
                fno = int(links_i[0].frame1_no) if len(links_i) else i
                fig = py.gcf()
                label_fs = 9.0 * (font_scale / 0.6)
                box = dict(facecolor="black", edgecolor="none", pad=1.5, alpha=0.6)
                # monospace + right-alignment keeps the digits steady frame-to-frame.
                if frame_label:
                    fig.text(0.13, 0.015, str(fno), ha="right", va="bottom",
                             fontsize=label_fs, color="white", family="monospace", bbox=box)
                if time_label:
                    if fno < len(self.elapsed_times):
                        t = float(self.elapsed_times[fno])
                    else:
                        t = fno * float(frame_interval_s)
                    mm, ss = int(t // 60), int(round(t % 60))
                    if ss == 60:
                        mm, ss = mm + 1, 0
                    fig.text(0.985, 0.015, "%02d:%02d" % (mm, ss), ha="right", va="bottom",
                             fontsize=label_fs, color="white", family="monospace", bbox=box)

            skeleton_fname = os.path.join(self.directory, "skeletons_%03d.png" % (i))
            paths_fname = os.path.join(self.directory, "paths_2D.png")
            py.savefig(skeleton_fname, dpi=400, transparent=True)
            py.close()

            # Overlay the skeleton (transparent background) on the paths image.
            alpha_composite(skeleton_fname, paths_fname, skeleton_fname)

    def make_forward_links(self):
        for i in range(len(self.frame_links) - 1, -1, -1):
            for link in self.frame_links[i]:
                prev_link = link.reverse_link
                current_link = link
                if link.forward_link is None:
                    while prev_link is not None:
                        prev_link.forward_link = current_link
                        current_link = prev_link
                        prev_link = prev_link.reverse_link

    def create_paths(self):
        self.paths = []
        for i in range(len(self.frame_links) - 1, -1, -1):
            for link in self.frame_links[i]:
                new_path = Path()
                prev_link = link.reverse_link
                current_link = link
                if link.forward_link is None:
                    new_path.links.append(current_link)
                    while prev_link is not None:
                        current_link = prev_link
                        new_path.links.append(current_link)
                        prev_link = prev_link.reverse_link

                if len(new_path.links) > 0:
                    self.paths.append(new_path)

        # Correct velocity values with respect to filament direction.
        for path in self.paths:
            fil_direction = path.links[0].mov_direction
            for i in range(1, len(path.links)):
                fil_direction *= path.links[i - 1].fil_direction
                mov_direction = fil_direction * path.links[i].mov_direction
                path.links[i].instant_velocity = (
                    path.links[i].instant_velocity * mov_direction
                )

    def path_velocities(self, num_points=1):
        self.full_len_vel = []
        self.max_len_vel = []

        for path in self.paths:
            if len(path.links) < num_points:
                continue

            mp_diff = (
                path.links[-1].filament1_midpoint - path.links[0].filament2_midpoint
            )
            time_diff = np.fabs(
                path.links[-1].filament1_time - path.links[0].filament2_time
            )
            dist = np.sqrt(np.sum(mp_diff ** 2))

            if self.dx * dist / time_diff < self.min_velocity:
                path.stuck = True

            for link in path.links:
                link.instant_velocity *= self.dx
                link.average_length *= self.dx

            array_vel = np.array([np.fabs(link.instant_velocity) for link in path.links])
            array_len = np.array([np.fabs(link.average_length) for link in path.links])
            path_length = len(array_vel)

            array_smooth = []
            for i in range(len(path.links) - num_points + 1):
                ave_len = np.mean(array_len[i : i + num_points])
                if path.stuck:
                    ave_vel = 0
                    std_vel = 0
                else:
                    ave_vel = np.mean(array_vel[i : i + num_points])
                    std_vel = np.std(array_vel[i : i + num_points])

                array_smooth.append([ave_len, ave_vel, std_vel, path_length])
                self.full_len_vel.append([ave_len, ave_vel, std_vel, path_length])

            array_smooth = np.array(array_smooth)
            max_i = np.argmax(array_smooth[:, 1])
            self.max_len_vel.append(array_smooth[max_i, :])

        self.full_len_vel = np.array(self.full_len_vel)
        self.max_len_vel = np.array(self.max_len_vel)

        if len(self.full_len_vel) == 0:
            return

        sort_i = np.argsort(self.full_len_vel[:, 0])
        self.full_len_vel = self.full_len_vel[sort_i, :]

        sort_i = np.argsort(self.max_len_vel[:, 0])
        self.max_len_vel = self.max_len_vel[sort_i, :]

    def make_frame_links(self):
        """Link two adjacent frames (delegated to the configured linker)."""
        new_frame_links, self.dt = self.get_linker().link(
            self.frame1, self.frame2, self.dt, self.elapsed_times
        )
        self.frame_links.append(new_frame_links)

    def write_path_data(self, extra_fname=None):
        with open(self.directory + "/paths.txt", "w") as f:
            for data in self.path_data:
                f.write("%8d\t%8d" % (data[0], data[1]))
                for vel in data[2]:
                    f.write("\t%8.f" % (vel))
                f.write("\n")

        if extra_fname is not None:
            with open(extra_fname + ".txt", "w") as f:
                for data in self.path_data:
                    f.write("%8d\t%8d" % (data[0], data[1]))
                    for vel in data[2]:
                        f.write("\t%8.f" % (vel))
                    f.write("\n")

    def process_frame_links(self, num_points=5):
        """Process frame links into paths and per-path velocities."""
        self.make_forward_links()
        self.wire_frame_links()
        self.create_paths()
        self.path_velocities(num_points)

    def make_movie(self, extra_fname=None, fps=1):
        """Assemble per-frame skeleton PNGs into a tracking movie.

        Delegated to the configured movie writer (default: H.264/MP4 via ffmpeg).
        """
        MOVIE_WRITERS.create(self.movie_format).write(self.directory, extra_fname, fps=fps)

    @staticmethod
    def _to_uint8_bgr(img):
        """Linearly rescale a (possibly 16-bit) grayscale image to an 8-bit BGR image."""
        a = np.asarray(img).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            a = (a - lo) * (255.0 / (hi - lo))
        gray8 = a.astype(np.uint8)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _draw_label(img, text, org, font, scale, thick, color=(255, 255, 255)):
        """Draw text with a thin black outline so it's legible on any background."""
        cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)

    def make_overlay_movie(self, extra_fname=None, fps=10.0, frame_label=True,
                           time_label=True, frame_interval_s=1.0, font_scale=0.6):
        """Render an overlay movie: each original frame with its tracked
        filaments drawn on top, colored by whether the path is stuck.

        Moving filaments are green, stuck filaments red.  Optional labels show
        the frame number (bottom-left, right-aligned so it stays steady) and the
        elapsed time in mm:ss (bottom-right).  The time comes from the
        acquisition metadata (ElapsedTime-ms) when available, else from
        ``frame_interval_s``.  Produces ``overlay_tracks.mp4`` at ``fps``.
        Requires that ``process_frame_links`` has already built ``self.paths``.
        """
        if not self.paths:
            return

        # Collect, per frame number, the filament contours to draw and whether
        # each belongs to a stuck path.  A link connects a filament in frame1_no
        # to its partner in frame2_no, so contribute both endpoints.
        per_frame = {}
        for path in self.paths:
            stuck = bool(getattr(path, "stuck", False))
            for link in path.links:
                for fno, contour in (
                    (int(link.frame1_no), link.filament1_contour),
                    (int(link.frame2_no), link.filament2_contour),
                ):
                    per_frame.setdefault(fno, []).append((contour, stuck))
        if not per_frame:
            return

        # Prefer acquisition metadata for the time label; load it if needed.
        if time_label and len(self.elapsed_times) == 0:
            try:
                self.read_metadata()
            except Exception:
                pass

        green = (0, 255, 0)   # moving (BGR)
        red = (0, 0, 255)     # stuck (BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 1
        margin = 8

        lo, hi = min(per_frame), max(per_frame)
        # Fixed field width for the right-aligned frame number (steady as digits grow).
        (num_field_w, _nh), _ = cv2.getTextSize(str(hi), font, font_scale, thick)

        out_index = 0
        for fno in range(lo, hi + 1):
            frame = Frame()
            frame.directory = self.directory
            frame.header = self.header
            frame.tail = self.tail
            if not frame.read_frame(fno):
                continue
            canvas = self._to_uint8_bgr(frame.img)
            h, w = canvas.shape[:2]
            for contour, stuck in per_frame.get(fno, []):
                pts = np.asarray(contour)
                if pts.ndim != 2 or len(pts) < 2:
                    continue
                # contour is [row, col]; OpenCV wants [x=col, y=row].
                xy = np.clip(pts[:, ::-1].astype(np.int32), [0, 0], [w - 1, h - 1])
                cv2.polylines(canvas, [xy.reshape(-1, 1, 2)], False,
                              red if stuck else green, 1, lineType=cv2.LINE_AA)

            baseline_y = h - margin
            if frame_label:
                txt = str(fno)
                (tw, _th), _ = cv2.getTextSize(txt, font, font_scale, thick)
                # Right-align the number's right edge to a fixed column.
                x = margin + num_field_w - tw
                self._draw_label(canvas, txt, (x, baseline_y), font, font_scale, thick)
            if time_label:
                if fno < len(self.elapsed_times):
                    t = float(self.elapsed_times[fno])
                else:
                    t = fno * float(frame_interval_s)
                mm, ss = int(t // 60), int(round(t % 60))
                if ss == 60:
                    mm, ss = mm + 1, 0
                tstr = "%02d:%02d" % (mm, ss)
                (tw, _th), _ = cv2.getTextSize(tstr, font, font_scale, thick)
                self._draw_label(canvas, tstr, (w - margin - tw, baseline_y),
                                 font, font_scale, thick)

            cv2.imwrite(os.path.join(self.directory, "overlay_%03d.png" % out_index), canvas)
            out_index += 1

        if out_index == 0:
            print("No original frames could be read for the overlay movie "
                  "(check header/tail); skipping overlay_tracks.mp4.")
            return

        print("Overlay movie: wrote %d frames in %s; encoding overlay_tracks.mp4 ..."
              % (out_index, self.directory))
        MOVIE_WRITERS.create(self.movie_format).write(
            self.directory, extra_fname,
            input_pattern="overlay_%03d.png", output_name="overlay_tracks.mp4",
            fps=fps,
        )

    def read_frame(self, num_frame, force_read=False):
        """Extract filaments from a single frame (or load cached result)."""
        print("Reading frame: %d" % (num_frame))
        self.frame = Frame()
        self.frame.directory = self.directory
        self.frame.header = self.header
        self.frame.tail = self.tail
        self.frame.frame_no = num_frame
        self.frame.cache_tag = self.cache_tag

        store = self.get_store()
        if not force_read and store.has(self.directory, self.cache_tag, num_frame):
            self.frame.filXYs = store.read(self.directory, self.cache_tag, num_frame)
            self.frame.filXY2filaments()
            return 1

        if not self.frame.read_frame(num_frame):
            raise FileNotFoundError("File not found!")

        # Detection is delegated to the configured strategy (default: the
        # entropy/watershed detector).  It runs low_pass -> entropy_clusters ->
        # filter_islands -> skeletonize_islands -> filaments2filXYs and carries
        # the fast_rank / morph_contrast options.
        self.get_detector().detect(self.frame)
        return 0

    def save_frame(self):
        self.get_store().write(
            self.directory, self.cache_tag, self.frame.frame_no, self.frame.filXYs
        )

    def load_frame1(self, frame_no):
        self.frame1 = Frame()
        self.frame1.directory = self.directory
        self.frame1.header = self.header
        self.frame1.tail = self.tail
        self.frame1.frame_no = frame_no
        self.frame1.cache_tag = self.cache_tag
        self.frame1.filXYs = self.get_store().read(self.directory, self.cache_tag, frame_no)
        self.frame1.filXY2filaments()

    def load_frame2(self, frame_no):
        self.frame2 = Frame()
        self.frame2.directory = self.directory
        self.frame2.header = self.header
        self.frame2.tail = self.tail
        self.frame2.frame_no = frame_no
        self.frame2.cache_tag = self.cache_tag
        self.frame2.filXYs = self.get_store().read(self.directory, self.cache_tag, frame_no)
        self.frame2.filXY2filaments()

    def write_length_velocity(self, header="", extra_fname=None):
        np.savetxt(self.directory + "/" + header + "full_length_velocity.txt", self.full_len_vel)
        np.savetxt(self.directory + "/" + header + "max_length_velocity.txt", self.max_len_vel)

        if extra_fname is not None:
            np.savetxt(extra_fname + "full_length_velocity.txt", self.full_len_vel)
            np.savetxt(extra_fname + "max_length_velocity.txt", self.max_len_vel)

    def save_links(self):
        np.save(
            self.directory + "/links%s.npy" % self.cache_tag,
            np.array(self.frame_links, dtype=object),
        )

    def load_links(self):
        self.frame_links = list(
            np.load(self.directory + "/links%s.npy" % self.cache_tag, allow_pickle=True)
        )

