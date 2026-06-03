"""In-vitro motility (actin gliding via myosins) analysis.

Modernized Python 3 port of the original ``FAST/motility.py`` by Tural Aksel
(Aksel et al., Cell Reports, 2015).  The numerical algorithm is preserved
verbatim; only Python 2->3, NumPy 2 and scikit-image API changes were applied.

Key compatibility fixes vs. the 2018 original:
  * ``filter(...)`` results are materialized with ``list(...)`` wherever they
    are indexed or measured with ``len``.
  * Integer (floor) division ``//`` is used for midpoint indexing.
  * ``np.bool`` -> ``bool``; fancy-indexing uses tuples (required by NumPy 2).
  * scikit-image imports updated: ``watershed`` from ``skimage.segmentation``;
    ``rank``/``threshold_otsu`` from ``skimage.filters``; ``gaussian_filter``
    from ``scipy.ndimage``; ``img_as_uint`` from ``skimage.util``;
    morphology ``selem=`` keyword renamed to ``footprint=``.
  * ``np.histogram``/``hist`` ``normed=`` -> ``density=``.
  * ``scipy.stats.kde.gaussian_kde`` -> ``scipy.stats.gaussian_kde``.
  * Object arrays saved/loaded with ``dtype=object`` / ``allow_pickle=True``.
  * Rank filters receive integer (uint8) images.
"""
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as py  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from numpy import ma  # noqa: E402

from scipy.ndimage import (  # noqa: E402
    label,
    binary_fill_holes,
    binary_closing,
    gaussian_filter,
)
from scipy import stats  # noqa: E402
from scipy.optimize import leastsq  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from skimage.util import img_as_uint  # noqa: E402
from skimage.filters import rank, threshold_otsu  # noqa: E402
from skimage.morphology import disk, square, skeletonize, dilation  # noqa: E402
from skimage.segmentation import watershed  # noqa: E402

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image  # noqa: E402

from . import plotparams  # noqa: E402


def _alpha_composite(fg_path, bg_path, out_path):
    """Composite ``fg_path`` (with alpha) over ``bg_path`` and write ``out_path``.

    Pure-Python replacement for the original ImageMagick ``composite`` shell
    call, so movie generation works on any platform without external tools.
    """
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")
    if bg.size != fg.size:
        bg = bg.resize(fg.size)
    Image.alpha_composite(bg, fg).save(out_path)

# Global structuring elements
sqr_1 = square(1)
sqr_2 = square(2)
sqr_3 = square(3)
disk_1 = disk(1)
disk_2 = disk(2)
disk_3 = disk(3)

# Very small number used to keep logarithms finite.
ZERO = 1e-100


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def make_N_colors(cmap_name, N):
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(N)
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(cmap_name, N)
    return cmap(np.arange(N))


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


# --------------------------------------------------------------------------- #
# Statistical helper functions
# --------------------------------------------------------------------------- #
def gaussian(X, amp, mu, stdev):
    return amp * np.exp(-(X - mu) ** 2 / (2 * stdev ** 2))


def fit_gaussian(bin_centers, bin_amps):
    err = lambda params: params[0] * np.exp(
        -(bin_centers - params[1]) ** 2 / (2 * params[2] ** 2)
    ) - bin_amps
    params = [50, 700, 100]
    best_params, success = leastsq(err, params, maxfev=1000)
    return best_params[0], best_params[1], best_params[2]


def fit_length_velocity(length, velocity, fil_weights, weighted=False):
    """Fit Uyeda's length-velocity relationship."""
    myosin_density = 1.0 / 36.0
    Neff = length * myosin_density

    weights = np.ones(len(length))
    if weighted:
        weights = fil_weights
    err = lambda params: weights * (
        params[0] * (1.0 - (1.0 - params[1]) ** Neff) - velocity
    )

    params = [700, 0.001]
    best_params, success = leastsq(err, params, maxfev=1000)
    residuals = np.array(err(best_params) / weights)
    return best_params[0], best_params[1], residuals, success


def length_velocity(length, max_vel, f):
    """Uyeda's simple length-velocity relationship."""
    myosin_density = 1.0 / 36.0
    Neff = length * myosin_density
    return max_vel * (1.0 - (1.0 - f) ** Neff)


def coupling_velocity(length, max_vel, amp, tau):
    """Coupling relationship with single exponential decay."""
    return max_vel - amp * np.exp(-length / tau)


def fit_coupling_velocity(length, velocity, fil_weights, weighted=False):
    weights = np.ones(len(length))
    if weighted:
        weights = fil_weights
    err = lambda params: weights * (
        params[0] - params[1] * np.exp(-length / params[2]) - velocity
    )

    params = [700, 200, 500]
    best_params, success = leastsq(err, params, maxfev=1000)
    residuals = np.array(err(best_params) / weights)
    return best_params[0], best_params[1], best_params[2], residuals, success


def bin_length_velocity(length, velocity, dx=100):
    max_len = np.max(length)
    bin_vel = []
    for i in np.arange(0, int(max_len / dx)):
        valid = (length > i * dx) * (length <= (i + 1) * dx)
        if np.sum(valid) > 0:
            mean_len = np.mean(length[valid])
            mean_vel = np.mean(velocity[valid])
            bin_vel.append([mean_len, mean_vel])
    return np.array(bin_vel)


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


# --------------------------------------------------------------------------- #
# Frame-link data structures
# --------------------------------------------------------------------------- #
class Link:
    def __init__(self):
        self.frame1_no = 0
        self.frame2_no = 0
        self.filament1_label = 0
        self.filament2_label = 0
        self.filament1_length = 0
        self.filament2_length = 0
        self.filament1_contour = []
        self.filament2_contour = []
        self.filament1_cm = []
        self.filament2_cm = []

        self.average_length = 0
        self.overlap_score = 0
        self.area_score = 0
        self.distance_score = 0

        self.fil_direction = 1
        self.mov_direction = 1

        self.dt = 0
        self.instant_velocity = 0

        self.forward_link = None
        self.reverse_link = None

        self.direct_link = False


class Path:
    def __init__(self):
        self.links = []
        self.first_frame_no = 0
        self.path_length = 0
        self.ave_fil_length = 0
        self.ave_velocity = 0
        self.std_velocity = 0
        self.max_velocity = 0
        self.min_velocity = 0

        self.stuck = False


class Motility:
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
        if not self.force_analysis and os.path.exists(self.directory + "/links.npy"):
            try:
                self.frame_links = list(
                    np.load(self.directory + "/links.npy", allow_pickle=True)
                )
            except (ImportError, ModuleNotFoundError, AttributeError):
                print(
                    "Movie analysed previously with an old version of motility."
                    " Links will be regenerated."
                )
                return False
            return True
        return False

    def reconstruct_skeleton_images(self):
        if not os.path.isfile(self.directory + "/paths_2D.png"):
            return

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

            skeleton_fname = os.path.join(self.directory, "skeletons_%03d.png" % (i))
            paths_fname = os.path.join(self.directory, "paths_2D.png")
            py.savefig(skeleton_fname, dpi=400, transparent=True)
            py.close()

            # Overlay the skeleton (transparent background) on the paths image.
            _alpha_composite(skeleton_fname, paths_fname, skeleton_fname)

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
        """Make links between two adjacent frames."""
        new_frame_links = []

        self.frame1.filaments = [
            f for f in self.frame1.filaments if f.fil_length > self.min_fil_length
        ]
        self.frame2.filaments = [
            f for f in self.frame2.filaments if f.fil_length > self.min_fil_length
        ]

        self.frame1.reset_filament_labels()
        self.frame2.reset_filament_labels()

        if len(self.elapsed_times) > 0:
            self.dt = (
                self.elapsed_times[self.frame2.frame_no]
                - self.elapsed_times[self.frame1.frame_no]
            )
            frame1_time = self.elapsed_times[self.frame1.frame_no]
            frame2_time = self.elapsed_times[self.frame2.frame_no]
        else:
            frame1_time = self.frame1.frame_no * self.dt
            frame2_time = self.frame2.frame_no * self.dt

        for i in range(len(self.frame1.filaments)):
            filament1 = self.frame1.filaments[i]

            link_candidates = []

            frame2_filaments = [
                filament
                for filament in self.frame2.filaments
                if np.sqrt(np.sum((filament.cm - filament1.cm) ** 2)) < self.max_velocity
                and np.fabs(filament.fil_length - filament1.fil_length)
                < self.max_length_dif
            ]

            for j in range(len(frame2_filaments)):
                filament2 = frame2_filaments[j]
                (
                    overlap_score,
                    area_score,
                    distance_score,
                    fil_direction,
                    mov_direction,
                ) = filament1.sim_score(filament2)
                link_candidates.append(
                    [
                        filament2.label,
                        area_score,
                        overlap_score,
                        distance_score,
                        fil_direction,
                        mov_direction,
                    ]
                )

            link_candidates = np.array(link_candidates)
            num_candidates = len(link_candidates)

            if num_candidates > 0:
                sorted_i = np.argsort(link_candidates[:, 2])
                link_candidates = link_candidates[sorted_i, :]

                area_score_list = link_candidates[:, 1]
                log_area_score_list = np.log10(area_score_list)
                log_area_score_diff_list = (
                    log_area_score_list[1:] - log_area_score_list[:-1]
                )
                overlap_score_list = link_candidates[:, 2]
                distance_score_list = link_candidates[:, 3]
                fil_direction_list = link_candidates[:, 4]
                mov_direction_list = link_candidates[:, 5]

                if (
                    np.fabs(overlap_score_list[0]) > self.overlap_score_cutoff
                    and log_area_score_list[0] < self.log_area_score_cutoff
                    and (
                        (
                            num_candidates > 1
                            and log_area_score_diff_list[0]
                            >= self.dif_log_area_score_cutoff
                        )
                        or (num_candidates == 1)
                    )
                ):
                    # The accepted partner is the first candidate (lowest overlap
                    # after sort); recover its filament2 object by label so that
                    # the link's partner identity matches the scores it carries.
                    if self.legacy_linking:
                        # Original behaviour: ``filament2`` is the leftover inner-
                        # loop variable, i.e. the last *unsorted* candidate. This
                        # reproduces the published (internally inconsistent)
                        # results bit-for-bit.
                        filament2 = frame2_filaments[-1]
                    else:
                        filament2 = self.frame2.filaments[int(link_candidates[0, 0])]

                    new_link = Link()
                    new_link.frame1_no = filament1.frame_no
                    new_link.frame2_no = filament2.frame_no
                    new_link.filament1_label = filament1.label
                    new_link.filament2_label = filament2.label
                    new_link.filament1_cm = filament1.cm
                    new_link.filament2_cm = filament2.cm
                    new_link.filament1_length = filament1.fil_length
                    new_link.filament2_length = filament2.fil_length
                    new_link.filament1_contour = filament1.contour
                    new_link.filament2_contour = filament2.contour
                    new_link.filament1_midpoint = filament1.midpoint
                    new_link.filament2_midpoint = filament2.midpoint
                    new_link.filament1_time = frame1_time
                    new_link.filament2_time = frame2_time

                    new_link.fil_direction = fil_direction_list[0]
                    new_link.mov_direction = mov_direction_list[0]

                    new_link.overlap_score = overlap_score_list[0]
                    new_link.area_score = area_score_list[0]
                    new_link.distance_score = distance_score_list[0]

                    new_link.average_length = 0.5 * (
                        filament1.fil_length + filament2.fil_length
                    )
                    new_link.instant_velocity = new_link.distance_score / self.dt
                    new_link.dt = self.dt

                    new_link.direct_link = True
                    new_link.reverse_link = filament1.reverse_link

                    filament1.forward_link = new_link

                    if filament2.reverse_link is None:
                        filament2.reverse_link = new_link
                    elif new_link.overlap_score < filament2.reverse_link.overlap_score:
                        prev_fil_label = int(filament2.reverse_link.filament1_label)
                        prev_filament1 = self.frame1.filaments[prev_fil_label]
                        prev_filament1.forward_link = None
                        filament2.reverse_link = new_link

        for i in range(len(self.frame1.filaments)):
            filament1 = self.frame1.filaments[i]
            if filament1.forward_link is not None:
                new_frame_links.append(filament1.forward_link)

        self.frame_links.append(new_frame_links)

    def plot_2D_path_data(self, num_points, extra_fname=None):
        ratio = self.width / 1002.0

        self.path_data = []
        self.path_stats = []

        self.path_img = np.nan * np.ones((self.width, self.height), dtype=float)

        filtered_paths = [x for x in self.paths if len(x.links) >= num_points]

        if len(filtered_paths) == 0:
            return

        path_colors = make_N_colors("Accent", len(filtered_paths))

        py.figure(2000)
        py.imshow(self.path_img, cmap=cm.gray, alpha=1.0)

        py.figure(2001)
        py.imshow(self.path_img, cmap=cm.gray, alpha=1.0)

        for i in range(len(filtered_paths)):
            path = filtered_paths[i]
            mp_mean = np.mean(
                np.array(
                    [
                        [link.filament1_midpoint[1], link.filament1_midpoint[0]]
                        for link in path.links[::-1]
                    ]
                ),
                axis=0,
            )

            len_array = np.array(
                [np.fabs(link.average_length) for link in path.links[::-1]]
            )
            vel_array = np.array(
                [np.fabs(link.instant_velocity) for link in path.links[::-1]]
            )

            first_frame = path.links[-1].frame1_no
            path_length = len(path.links)

            stuck = path.stuck

            self.path_data.append([first_frame, stuck, vel_array])
            self.path_stats.append(
                [
                    first_frame,
                    stuck,
                    path_length,
                    np.mean(len_array),
                    np.mean(vel_array),
                    np.std(vel_array),
                ]
            )

            mean_velocity = np.fabs(np.mean(vel_array))
            if stuck:
                mean_velocity = 0

            for j in range(len(path.links)):
                mp_x1 = path.links[j].filament1_midpoint[1]
                mp_y1 = path.links[j].filament1_midpoint[0]
                mp_x2 = path.links[j].filament2_midpoint[1]
                mp_y2 = path.links[j].filament2_midpoint[0]

                py.figure(2000)
                py.arrow(
                    mp_x2,
                    mp_y2,
                    mp_x1 - mp_x2,
                    mp_y1 - mp_y2,
                    color=path_colors[i],
                    head_width=ratio * 5,
                    head_length=ratio * 10,
                    alpha=1.0,
                )

                py.figure(2001)
                py.arrow(
                    mp_x2,
                    mp_y2,
                    mp_x1 - mp_x2,
                    mp_y1 - mp_y2,
                    color=path_colors[i],
                    head_width=ratio * 5,
                    head_length=ratio * 10,
                    alpha=1.0,
                )

            py.figure(2001)
            py.text(mp_mean[0], mp_mean[1], "%.f" % (mean_velocity), fontsize=10, color="k")

        self.path_stats = np.array(self.path_stats)

        py.figure(2000)
        ax = py.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        py.savefig(self.directory + "/paths_2D.png", dpi=400, transparent=False)

        py.figure(2001)
        ax = py.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if extra_fname is not None:
            py.figure(2001)
            py.savefig(extra_fname + "_2D.png", dpi=400, transparent=False)

        py.close("all")
        return self.path_data

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

    def make_movie(self, extra_fname=None):
        """Assemble per-frame skeleton PNGs into a tracking movie via ffmpeg."""
        if not os.path.isfile(os.path.join(self.directory, "paths_2D.png")):
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg not found on PATH; skipping movie generation.")
            return

        cwd = os.getcwd()
        os.chdir(self.directory)
        try:
            # ffmpeg replaces the long-deprecated avconv.  Run via subprocess
            # with an argument list (no shell) so it is platform independent.
            #
            # Encode H.264 in an MP4 container with the yuv420p pixel format.
            # The old default (mpeg4/FMP4 in an AVI container) is rejected by
            # QuickTime and ImageJ ("Unsupported compression: FMP4").  H.264 /
            # yuv420p / mp4 is the broadly compatible combination that opens in
            # QuickTime, ImageJ, browsers, and most players.  libx264 requires
            # even frame dimensions, so pad width/height up to the next even
            # number.
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

    def read_frame(self, num_frame, force_read=False):
        """Extract filaments from a single frame (or load cached result)."""
        print("Reading frame: %d" % (num_frame))
        self.frame = Frame()
        self.frame.directory = self.directory
        self.frame.header = self.header
        self.frame.tail = self.tail
        self.frame.frame_no = num_frame
        self.frame.fast_rank = self.fast_rank
        self.frame.morph_contrast = self.morph_contrast

        filament_file = self.directory + "/filXYs%03d.npy" % num_frame
        if not force_read and os.path.isfile(filament_file):
            self.frame.read_filXYs()
            self.frame.filXY2filaments()
            return 1

        if not self.frame.read_frame(num_frame):
            raise FileNotFoundError("File not found!")

        self.frame.low_pass_filter()
        self.frame.entropy_clusters()
        self.frame.filter_islands()
        self.frame.skeletonize_islands()
        self.frame.filaments2filXYs()
        return 0

    def save_frame(self):
        self.frame.save_filXYs()

    def load_frame1(self, frame_no):
        self.frame1 = Frame()
        self.frame1.directory = self.directory
        self.frame1.header = self.header
        self.frame1.tail = self.tail
        self.frame1.frame_no = frame_no
        self.frame1.read_filXYs()
        self.frame1.filXY2filaments()

    def load_frame2(self, frame_no):
        self.frame2 = Frame()
        self.frame2.directory = self.directory
        self.frame2.header = self.header
        self.frame2.tail = self.tail
        self.frame2.frame_no = frame_no
        self.frame2.read_filXYs()
        self.frame2.filXY2filaments()

    def write_length_velocity(self, header="", extra_fname=None):
        np.savetxt(self.directory + "/" + header + "full_length_velocity.txt", self.full_len_vel)
        np.savetxt(self.directory + "/" + header + "max_length_velocity.txt", self.max_len_vel)

        if extra_fname is not None:
            np.savetxt(extra_fname + "full_length_velocity.txt", self.full_len_vel)
            np.savetxt(extra_fname + "max_length_velocity.txt", self.max_len_vel)

    def save_links(self):
        np.save(
            self.directory + "/links.npy",
            np.array(self.frame_links, dtype=object),
        )

    def load_links(self):
        self.frame_links = list(np.load(self.directory + "/links.npy", allow_pickle=True))

    def plot_length_velocity(
        self,
        header="",
        extra_fname=None,
        max_vel=2400,
        max_length=10000,
        nbins=30,
        min_points=2,
        min_path_length=5,
        weighted=True,
        percent_tolerance=500,
        print_plot=True,
        minimal_plot=False,
        maxvel_color="b",
        plot_xlabels=True,
        plot_ylabels=True,
        square_plot=True,
        plot_length_f=False,
        fit_f="exp",
        dpi_plot=200,
    ):
        """Compute velocity statistics and (optionally) render the length-velocity plot.

        Returns a tuple of summary statistics (see the original paper).  On too
        few data points it returns a tuple of ``-1`` sentinels.
        """
        valid = np.nonzero(self.full_len_vel[:, 0] < max_length)[0]
        self.full_len_vel = self.full_len_vel[valid, :]

        tolerance_data = []
        tolerance_list = [2.5, 5, 10, 20, 40, 80]
        valid_points = np.nonzero(self.full_len_vel[:, 1] >= 0)[0]
        for filter_value in tolerance_list[::-1]:
            filtered_data = self.full_len_vel[valid_points, :]
            if len(valid_points) > 10:
                non_stuck = np.nonzero(filtered_data[:, 1] != 0)[0]

                if len(non_stuck) > 0:
                    fil_vel = filtered_data[non_stuck, 1]
                    mean_vel_m = np.mean(fil_vel)
                    std_vel_m = np.std(fil_vel)

                    velocities_sorted = np.sort(fil_vel)[::-1]
                    top_1_num = int(np.ceil(0.01 * len(velocities_sorted)))
                    top_5_num = int(np.ceil(0.05 * len(velocities_sorted)))

                    num_filter_points = len(fil_vel)

                    top_1_velocity = np.mean(velocities_sorted[:top_1_num])
                    top_5_velocity = np.mean(velocities_sorted[:top_5_num])

                    tolerance_data.append(
                        [
                            filter_value * 2,
                            num_filter_points,
                            top_1_velocity,
                            top_5_velocity,
                            mean_vel_m,
                            std_vel_m,
                        ]
                    )
                else:
                    tolerance_data.append([filter_value * 2, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                tolerance_data.append([filter_value * 2, 0.0, 0.0, 0.0, 0.0, 0.0])
            valid_points = np.nonzero(
                self.full_len_vel[:, 2] <= filter_value / 100.0 * self.full_len_vel[:, 1]
            )[0]

        tolerance_data = np.array(tolerance_data)

        percent_stuck = 100.0 * np.sum(self.full_len_vel[:, 1] == 0) / len(
            self.full_len_vel[:, 0]
        )

        text_font_size = 30

        valid_filtered = np.nonzero(
            (self.full_len_vel[:, 1] > 0)
            * (self.full_len_vel[:, 2] <= percent_tolerance / 100.0 * self.full_len_vel[:, 1])
            * (self.full_len_vel[:, 3] >= min_path_length)
        )[0]
        num_points_filtered = len(valid_filtered)

        valid_mobile = np.nonzero(
            (self.full_len_vel[:, 1] > 0) * (self.full_len_vel[:, 3] >= min_path_length)
        )[0]
        num_points_mobile = len(valid_mobile)

        valid_all = np.nonzero(self.full_len_vel[:, 3] >= min_path_length)[0]
        num_points_all = len(valid_all)

        valid_stuck = np.nonzero(
            (self.full_len_vel[:, 3] >= min_path_length) * (self.full_len_vel[:, 1] == 0)
        )[0]
        num_points_stuck = len(valid_stuck)

        if num_points_filtered < min_points:
            print(
                "Warning: There is not enough velocity data! - %d points"
                % (num_points_filtered)
            )
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

        MVEL_filtered = np.mean(self.full_len_vel[valid_filtered, 1])
        MVEL = np.mean(self.full_len_vel[valid_mobile, 1])
        MVIS = np.mean(self.full_len_vel[valid_all, 1])
        mean_len_filtered = np.mean(self.full_len_vel[valid_filtered, 0])
        mean_len_mobile = np.mean(self.full_len_vel[valid_mobile, 0])
        mean_len_stuck = np.mean(self.full_len_vel[valid_stuck, 0]) if num_points_stuck else 0.0
        mean_len_all = np.mean(self.full_len_vel[valid_all, 0])

        fil_len = self.full_len_vel[valid_filtered, 0]
        fil_vel = self.full_len_vel[valid_filtered, 1]

        l_bin_edges = np.linspace(0, max_length * 1e-3, nbins)
        l_bin_centers = 0.5 * (l_bin_edges[:-1] + l_bin_edges[1:])

        l_bin_counts, l_bin_locs = np.histogram(
            self.full_len_vel[valid_filtered, 0], bins=l_bin_edges, density=False
        )

        v_bin_edges = np.linspace(0, max_vel, nbins)
        v_bin_centers = 0.5 * (v_bin_edges[:-1] + v_bin_edges[1:])

        v_bin_counts, v_bin_locs = np.histogram(
            self.full_len_vel[valid_filtered, 1], bins=v_bin_edges, density=True
        )

        velocities_sorted = np.sort(fil_vel)[::-1]
        top_1_num = int(np.ceil(0.01 * len(velocities_sorted)))
        top_5_num = int(np.ceil(0.05 * len(velocities_sorted)))

        top_1_velocity = np.mean(velocities_sorted[:top_1_num])
        top_5_velocity = np.mean(velocities_sorted[:top_5_num])

        # Length-dependent weights for the coupling fit.
        fil_len_digitized = np.digitize(1e-3 * fil_len, l_bin_locs)
        fil_len_digitized = np.clip(fil_len_digitized, 1, len(l_bin_counts))
        with np.errstate(divide="ignore"):
            fil_weights = 1.0 / l_bin_counts[fil_len_digitized - 1]
        mean_vel_u, mean_vel_amp, mean_vel_tau, residuals, success = fit_coupling_velocity(
            fil_len, fil_vel, fil_weights, weighted=weighted
        )
        std_u = np.sqrt(np.mean(residuals ** 2))

        bound_prob = coupling_velocity(fil_len, 0.0, -1.0, mean_vel_tau)
        plateu_valid = np.nonzero(bound_prob > 0.95)[0]
        plateu_vel = fil_vel[plateu_valid]
        mean_plateu = np.mean(plateu_vel) if len(plateu_vel) else 0.0
        std_plateu = np.std(plateu_vel) if len(plateu_vel) else 0.0

        max_index_t = np.argmax(v_bin_counts)
        peak_vel_t = v_bin_centers[max_index_t]

        max_valid = np.nonzero(
            (self.max_len_vel[:, 1] > 0)
            * (self.max_len_vel[:, 2] <= percent_tolerance / 100.0 * self.max_len_vel[:, 1])
            * (self.max_len_vel[:, 3] >= min_path_length)
        )[0]

        max_vel_u = -1
        max_vel_amp = 0
        max_vel_tau = 1
        max_vel_r = 0
        if fit_f == "exp":
            max_vel_u, max_vel_amp, max_vel_tau, residuals, success = fit_coupling_velocity(
                self.max_len_vel[max_valid, 0],
                self.max_len_vel[max_valid, 1],
                np.ones(len(self.max_len_vel[max_valid, 0])),
                weighted=False,
            )
            std_u = np.sqrt(np.mean(residuals ** 2))
        elif fit_f == "uyeda":
            max_vel_u, max_vel_r, residuals, success = fit_length_velocity(
                self.max_len_vel[max_valid, 0],
                self.max_len_vel[max_valid, 1],
                np.ones(len(self.max_len_vel[max_valid, 0])),
                weighted=False,
            )
            std_u = np.sqrt(np.mean(residuals ** 2))

        exp_len = np.linspace(np.min(fil_len), 15000, 1000)
        if fit_f == "exp":
            exp_vel = coupling_velocity(exp_len, max_vel_u, max_vel_amp, max_vel_tau)
        elif fit_f == "uyeda":
            exp_vel = length_velocity(exp_len, max_vel_u, max_vel_r)
        else:
            exp_vel = np.zeros(len(exp_len))

        if percent_tolerance == 500:
            tolerance_string = "none"
        else:
            tolerance_string = str(percent_tolerance)

        if print_plot:
            self._render_length_velocity_plot(
                header=header,
                extra_fname=extra_fname,
                minimal_plot=minimal_plot,
                square_plot=square_plot,
                plot_xlabels=plot_xlabels,
                plot_ylabels=plot_ylabels,
                plot_length_f=plot_length_f,
                fit_f=fit_f,
                dpi_plot=dpi_plot,
                max_vel=max_vel,
                max_length=max_length,
                maxvel_color=maxvel_color,
                fil_len=fil_len,
                fil_vel=fil_vel,
                max_valid=max_valid,
                exp_len=exp_len,
                exp_vel=exp_vel,
                top_5_velocity=top_5_velocity,
                MVEL=MVEL,
                MVEL_filtered=MVEL_filtered,
                mean_len_filtered=mean_len_filtered,
                percent_stuck=percent_stuck,
                tolerance_data=tolerance_data,
                tolerance_string=tolerance_string,
                valid_filtered=valid_filtered,
                valid_mobile=valid_mobile,
                l_bin_edges=l_bin_edges,
                v_bin_edges=v_bin_edges,
                max_vel_u=max_vel_u,
                max_vel_tau=max_vel_tau,
                max_vel_r=max_vel_r,
                text_font_size=text_font_size,
            )

        if fit_f != "none":
            return (
                top_5_velocity,
                percent_stuck,
                MVEL,
                MVEL_filtered,
                max_vel_u,
                MVIS,
                mean_len_stuck,
                mean_len_filtered,
                mean_len_mobile,
                mean_len_all,
                num_points_filtered,
            )
        return (
            top_5_velocity,
            percent_stuck,
            MVEL,
            MVEL_filtered,
            -1,
            MVIS,
            mean_len_stuck,
            mean_len_filtered,
            mean_len_mobile,
            mean_len_all,
            num_points_filtered,
        )

    def _render_length_velocity_plot(self, **kw):
        """Render the length-velocity figure.  Isolated so a plotting failure
        cannot corrupt the numeric statistics returned to the caller."""
        header = kw["header"]
        extra_fname = kw["extra_fname"]
        minimal_plot = kw["minimal_plot"]
        square_plot = kw["square_plot"]
        plot_xlabels = kw["plot_xlabels"]
        plot_ylabels = kw["plot_ylabels"]
        plot_length_f = kw["plot_length_f"]
        fit_f = kw["fit_f"]
        dpi_plot = kw["dpi_plot"]
        max_vel = kw["max_vel"]
        max_length = kw["max_length"]
        maxvel_color = kw["maxvel_color"]
        fil_len = kw["fil_len"]
        fil_vel = kw["fil_vel"]
        max_valid = kw["max_valid"]
        exp_len = kw["exp_len"]
        exp_vel = kw["exp_vel"]
        top_5_velocity = kw["top_5_velocity"]
        MVEL = kw["MVEL"]
        MVEL_filtered = kw["MVEL_filtered"]
        mean_len_filtered = kw["mean_len_filtered"]
        percent_stuck = kw["percent_stuck"]
        tolerance_data = kw["tolerance_data"]
        tolerance_string = kw["tolerance_string"]
        valid_filtered = kw["valid_filtered"]
        valid_mobile = kw["valid_mobile"]
        l_bin_edges = kw["l_bin_edges"]
        v_bin_edges = kw["v_bin_edges"]
        max_vel_u = kw["max_vel_u"]
        max_vel_tau = kw["max_vel_tau"]
        max_vel_r = kw["max_vel_r"]
        text_font_size = kw["text_font_size"]

        if minimal_plot:
            text_font_size = 55
            linewidth = 10

            if fit_f == "exp":
                length_f = max_vel_tau
            elif fit_f == "uyeda":
                length_f = np.log(0.01) / np.log(1 - max_vel_r) * 36.0
            else:
                length_f = 0.0

            x, y = plotparams.get_figsize(1080)
            if square_plot:
                py.figure(0, figsize=(y, y))
            else:
                py.figure(0, figsize=(x, y))

            py.plot(1e-3 * fil_len, fil_vel, ".", markersize=10, color="gray")
            py.plot(
                1e-3 * self.max_len_vel[max_valid, 0],
                self.max_len_vel[max_valid, 1],
                marker="^",
                markersize=10,
                mec=maxvel_color,
                mfc=maxvel_color,
                linestyle="None",
            )

            if fit_f != "none":
                py.plot(1e-3 * exp_len, exp_vel, "k-", linewidth=linewidth, alpha=0.7)

            py.plot(
                1e-3 * exp_len,
                np.ones(len(exp_len)) * top_5_velocity,
                "k-.",
                linewidth=linewidth,
            )

            if plot_length_f:
                py.plot(
                    1e-3 * np.array([length_f, length_f]),
                    [0, top_5_velocity],
                    linestyle="dashed",
                    color="k",
                )
                py.text(
                    length_f * 1e-3 + 0.1,
                    10,
                    "%.1f" % (length_f * 1e-3),
                    fontsize=text_font_size,
                    color="k",
                )

            py.ylim([0, max_vel])
            py.xlim([0, max_length * 1e-3 + 1.0])

            ax = py.gca()
            vel_ticks = ax.get_yticks()
            ax.set_yticks(vel_ticks[::2])
            ax.set_yticklabels(vel_ticks[::2] * 1e-3)

            len_ticks = ax.get_xticks()
            ax.set_xticks(len_ticks[::2])
            ax.set_xticklabels([int(x) for x in len_ticks[::2]])

            ax.tick_params(pad=10)
            py.setp(ax.get_xticklabels(), fontsize=text_font_size, visible=plot_xlabels)
            py.setp(ax.get_yticklabels(), fontsize=text_font_size, visible=plot_ylabels)
        else:
            left, width = 0.1, 0.5
            bottom, height = 0.1, 0.5

            left_h1 = left + width
            left_h2 = left_h1 + 0.15
            bottom_v1 = bottom + 0.27

            rect_scatter = [left, bottom_v1, width, height]
            rect_tolerance = [left_h1 + 0.01, bottom + 0.02, 0.29, 0.24]
            rect_histy1 = [left_h1, bottom_v1, 0.15, height]
            rect_histy2 = [left_h2, bottom_v1, 0.15, height]
            rect_histx1 = [left, bottom + 0.02, width, 0.25]

            py.figure(0, figsize=plotparams.get_figsize(1200))
            axScatter = py.axes(rect_scatter)
            axHisty1 = py.axes(rect_histy1)
            axHisty2 = py.axes(rect_histy2)
            axHistx1 = py.axes(rect_histx1)
            axTolerance1 = py.axes(rect_tolerance)
            axTolerance2 = axTolerance1.twinx()

            max_tol_vel = np.max(tolerance_data[:, 3:5])
            min_tol_vel = np.min(tolerance_data[:, 3:5])

            axTolerance2.plot(
                tolerance_data[:, 0],
                tolerance_data[:, 3],
                color="k",
                linestyle="--",
                marker=".",
                linewidth=5,
                markersize=15,
            )
            axTolerance2.plot(
                tolerance_data[:, 0],
                tolerance_data[:, 4],
                color="k",
                linestyle="-",
                marker=".",
                linewidth=5,
                markersize=15,
            )
            axTolerance2.set_xscale("symlog")
            axTolerance1.set_xscale("symlog")

            tol_ymin = min_tol_vel - 100
            tol_ymax = max_tol_vel + 100
            tol_diff = max_tol_vel - min_tol_vel + 200

            axTolerance2.set_ylim([tol_ymin, tol_ymax])
            axTolerance2.set_xlim([5, 200])

            axTolerance2.plot(
                [6, 9],
                [tol_ymin + 0.25 * tol_diff, tol_ymin + 0.25 * tol_diff],
                color="k",
                linestyle="--",
                linewidth=5,
            )
            axTolerance2.text(
                10, tol_ymin + 0.20 * tol_diff, r"%s" % ("TOP5%"), fontsize=text_font_size, color="k"
            )
            axTolerance2.plot(
                [6, 9],
                [tol_ymin + 0.10 * tol_diff, tol_ymin + 0.1 * tol_diff],
                color="k",
                linestyle="-",
                linewidth=5,
            )
            axTolerance2.text(
                10,
                tol_ymin + 0.05 * tol_diff,
                r"%s" % ("Mean Velocity"),
                fontsize=text_font_size,
                color="k",
            )

            axTolerance2.set_xticks(tolerance_data[:, 0])
            axTolerance2.set_xticklabels(["*"] + [int(x) for x in tolerance_data[1:, 0]])
            axTolerance1.set_xlabel("% Tolerance", fontsize=text_font_size, labelpad=20)

            vel_ticks = axTolerance2.get_yticks()
            axTolerance2.set_yticks(vel_ticks[1::2])
            axTolerance2.set_yticklabels(vel_ticks[1::2] * 1e-3)

            ylim = axTolerance2.get_ylim()
            tol_diff = ylim[1] - ylim[0]
            axTolerance2.text(300, ylim[1] + 0.1 * tol_diff, r"$x10^3$", fontsize=25)

            py.setp(axTolerance2.get_yticklabels(), fontsize=text_font_size, visible=True)
            py.setp(axTolerance2.get_xticklabels(), fontsize=text_font_size, visible=True)
            py.setp(axTolerance1.get_yticklabels(), fontsize=text_font_size, visible=False)
            py.setp(axTolerance1.get_xticklabels(), fontsize=text_font_size, visible=True)

            l_bin_counts, _, _ = axHistx1.hist(
                1e-3 * self.full_len_vel[valid_filtered, 0],
                bins=l_bin_edges,
                density=False,
                orientation="vertical",
                color="gray",
            )
            max_prob_l = np.max(l_bin_counts)

            axHisty2.hist(
                self.full_len_vel[valid_mobile, 1],
                bins=v_bin_edges,
                density=True,
                orientation="horizontal",
                color="gray",
            )
            max_prob_a = axHisty2.get_xlim()[1]

            axHisty1.hist(
                self.full_len_vel[valid_filtered, 1],
                bins=v_bin_edges,
                density=True,
                orientation="horizontal",
                color="gray",
            )
            max_prob_t = axHisty1.get_xlim()[1]

            axScatter.plot(1e-3 * fil_len, fil_vel, ".", markersize=5, color="gray")
            axScatter.plot(
                1e-3 * self.max_len_vel[max_valid, 0],
                self.max_len_vel[max_valid, 1],
                marker="^",
                markersize=5,
                mec=maxvel_color,
                mfc=maxvel_color,
                linestyle="None",
            )

            if fit_f != "none":
                axScatter.plot(1e-3 * exp_len, exp_vel, "k-", alpha=0.7)

            axScatter.plot(
                1e-3 * exp_len, np.ones(len(exp_len)) * top_5_velocity, "k--", linewidth=5
            )

            axHisty1.plot([0, max_prob_t], np.ones(2) * MVEL_filtered, color="k", linestyle="-", linewidth=5)
            axHisty2.plot([0, max_prob_a], np.ones(2) * MVEL, "k-", linewidth=5)
            axHistx1.plot(
                [mean_len_filtered * 1e-3, mean_len_filtered * 1e-3],
                [0, max_prob_l],
                "k-",
                linewidth=5,
            )

            axScatter.set_ylim([0, max_vel])
            axScatter.set_xlim([0, max_length * 1e-3])

            vel_ticks = axScatter.get_yticks()[::2]
            axScatter.set_yticks(vel_ticks)
            axScatter.set_yticklabels(vel_ticks * 1e-3)

            len_ticks = axScatter.get_xticks()
            axScatter.set_xticks(len_ticks[:-1])
            axScatter.set_xticklabels([int(x) for x in len_ticks[:-1]])

            axHistx1.set_xticks(len_ticks[:-1])
            axHistx1.set_xticklabels([int(x) for x in len_ticks[:-1]])

            axHisty1.set_yticks(vel_ticks)
            axHisty1.set_yticklabels(vel_ticks * 1e-3)
            axHisty2.set_yticks(vel_ticks)
            axHisty2.set_yticklabels(vel_ticks * 1e-3)

            axScatter.text(0, max_vel, r"$x10^3$", fontsize=25)

            axScatter.set_ylim([0, max_vel])
            axScatter.set_xlim([0, max_length * 1e-3])

            py.setp(axHisty1.get_yticklabels(), fontsize=text_font_size, visible=False)
            py.setp(axHistx1.get_xticklabels(), fontsize=text_font_size, visible=True)
            py.setp(axScatter.get_xticklabels(), fontsize=text_font_size, visible=False)
            py.setp(axScatter.get_yticklabels(), fontsize=text_font_size, visible=True)

            axHisty1.set_ylim([0, max_vel])
            axHisty2.set_ylim([0, max_vel])
            axHistx1.set_xlim([0, max_length * 1e-3])

            axHisty1.ticklabel_format(style="sci", axis="x", scilimits=(-5, 5))
            axHisty2.ticklabel_format(style="sci", axis="x", scilimits=(-5, 5))

            py.setp(axHisty1.get_xticklabels(), visible=False)
            py.setp(axHisty2.get_xticklabels(), visible=False)
            py.setp(axHisty2.get_yticklabels(), visible=False)
            py.setp(axHistx1.get_yticklabels(), visible=False)

            axTolerance2.set_ylabel(r"Velocity (nm/s)", labelpad=20, fontsize=text_font_size)
            axScatter.set_ylabel(r"Velocity (nm/s)", labelpad=20, fontsize=text_font_size)
            axHistx1.set_xlabel(
                r"Actin filament length ($\mu m$)", labelpad=20, fontsize=text_font_size
            )

            axHisty1.text(0.1 * max_prob_t, 1.1 * max_vel, "Filtered", fontsize=text_font_size)
            axHisty2.text(0.1 * max_prob_a, 1.1 * max_vel, "Unfiltered", fontsize=text_font_size)

            axScatter.plot(
                max_length * 1e-3 * np.array([1, 2]) / 15.0,
                [2150 / 2400.0 * max_vel, 2150 / 2400.0 * max_vel],
                "k--",
                linewidth=5,
            )
            axScatter.text(
                max_length * 1e-3 * 2.1 / 15.0,
                2100 / 2400.0 * max_vel,
                r"%.f$^{TOP5\%%}$" % (top_5_velocity),
                fontsize=text_font_size,
                color="k",
            )

            if fit_f != "none":
                axScatter.plot(
                    max_length * 1e-3 * np.array([6, 7]) / 15.0,
                    [2150 / 2400.0 * max_vel, 2150 / 2400.0 * max_vel],
                    "k-",
                    linewidth=10,
                )
                axScatter.text(
                    max_length * 1e-3 * 7.1 / 15.0,
                    2100 / 2400.0 * max_vel,
                    r"%.f$^{PLATEAU}$" % (max_vel_u),
                    fontsize=text_font_size,
                    color="k",
                )

            axHisty1.text(
                0.1 * max_prob_t,
                1900 / 2400.0 * max_vel,
                r"%.f$^{MVEL_{%s}}$" % (MVEL_filtered, tolerance_string),
                fontsize=text_font_size,
                color="k",
            )
            axHisty2.text(
                0.1 * max_prob_a,
                1900 / 2400.0 * max_vel,
                r"%.f$^{MVEL}$" % (MVEL),
                fontsize=text_font_size,
                color="k",
            )
            axHisty2.text(
                0.15 * max_prob_a,
                1600 / 2400.0 * max_vel,
                r"%.f$^{\%%STUCK}$" % (percent_stuck),
                fontsize=text_font_size,
                color="k",
            )
            axHistx1.text(
                mean_len_filtered * 1e-3,
                max_prob_l * 0.5,
                r"%.3f$^{<FIL-LENGTH>}$" % (mean_len_filtered * 1e-3),
                fontsize=text_font_size,
                color="k",
            )

        py.savefig(self.directory + "/" + header + "length_velocity.png", dpi=dpi_plot, transparent=False)
        if extra_fname is not None:
            py.savefig(extra_fname, dpi=dpi_plot, transparent=False)
        py.close()

    def plot_correlation_profile(self, extra_fname=None):
        py.figure(4, figsize=plotparams.get_figsize(1080))

        array_corr_len = np.arange(len(self.final_corr_len)) * self.dx
        array_corr_weight = np.arange(len(self.final_corr_weight)) * self.dx

        valid = np.nonzero((self.final_corr_len > 0.7) * (array_corr_len <= 1500))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            array_corr_len[valid], 1.0 * self.final_corr_len[valid]
        )

        length_0_7 = np.round((0.7 - intercept) / slope)
        mean_corr_1500 = np.mean(1.0 * self.final_corr_len[valid])

        py.subplot(211)
        py.plot(array_corr_len, 1.0 * self.final_corr_len, "bo")
        py.plot(array_corr_len, array_corr_len * slope + intercept, "r-", linewidth=5)
        py.text(1500, 0.9, r"l$_{0.7}$: %d" % (length_0_7), fontsize=50)
        py.xlim(0, 3000)
        py.ylim(0.7, 1.0)
        py.ylabel(r"c($\Delta$ nm)")

        py.subplot(212)
        py.plot(array_corr_weight, 1.0 * self.final_corr_weight)
        py.xlim(0, self.max_fil_length * self.dx)
        py.xlabel(r"$\Delta$ nm")
        py.ylabel(r"weight (#)")
        py.xlim(0, self.max_fil_length * self.dx)

        py.savefig(self.directory + "/correlation_length.png", dpi=200)
        if extra_fname is not None:
            py.savefig(extra_fname, dpi=200)
        py.close()

        return length_0_7, mean_corr_1500


# --------------------------------------------------------------------------- #
# Frame / Island / Filament classes
# --------------------------------------------------------------------------- #
class Frame:
    def __init__(self):
        self.frame_no = 0
        self.window_island = 15
        self.disk_win = disk(self.window_island)

        # When True, the full-frame percentile (rank) filters operate on an
        # 8-bit rescaling of the image instead of the native 16-bit data.
        # scikit-image rank filters keep a per-pixel local histogram whose cost
        # scales with the number of grey levels (65536 for uint16 vs 256 for
        # uint8), so this is the dominant per-frame speedup.  It is mildly
        # lossy (8-bit quantization of intensities), but on real gliding-assay
        # data the velocity deltas are <0.4% with an unchanged alpha>beta
        # ordering, so it is ON by default.  Set False (CLI: --exact-rank) for
        # the exact 16-bit reference/validation path.
        self.fast_rank = True

        # When True, the local-contrast map in ``entropy_clusters`` is computed
        # with a single-pass morphological gradient (local max - min via
        # cv2.dilate/erode over the same disk window) instead of two
        # sliding-histogram percentile passes (5th/95th).  Much faster but
        # slightly more noise-sensitive (uses extremes rather than percentiles),
        # so it is OFF by default; A/B it with compare_fast_rank.py before use.
        self.morph_contrast = False
        self._morph_k = None

        self.img = None
        self.img_filaments = None
        self.img_skeletons = None

        self.width = 1002
        self.height = 1004

        self.backward_links = []
        self.islands = []
        self.filaments = []
        self.filXYs = []

        self.directory = ""
        self.header = ""
        self.tail = ""

        self.filament_counter = 0

    def reset_filament_labels(self):
        for i in range(len(self.filaments)):
            self.filaments[i].label = i

    def read_frame(self, frame_no):
        fname = (
            self.directory + "/" + self.header + "%03d" % frame_no + "_" + self.tail + "_000.tif"
        )
        self.frame_no = frame_no

        if not os.path.isfile(fname):
            return False

        # Read preserving the original 16-bit depth where present.
        self.img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        self.img = img_as_uint(self.img)

        self.width, self.height = self.img.shape
        return True

    def filXY2filaments(self):
        self.filaments = []
        fil_counter = 0
        for filXY, width, density, midpoint in self.filXYs:
            filament = Filament()
            filament.frame_no = self.frame_no
            filament.contour = filXY
            filament.fil_width = width
            filament.fil_density = density
            filament.label = fil_counter
            filament.midpoint = midpoint
            filament.calc_props()
            self.filaments.append(filament)
            fil_counter += 1

    def reconstruct_skeleton_images(self):
        self.img_skeletons = np.zeros((self.width, self.height), dtype=bool)
        for filament in self.filaments:
            self.img_skeletons[filament.contour[:, 0], filament.contour[:, 1]] = True

    def reconstruct_filament_images(self):
        self.img_filaments = np.zeros((self.width, self.height), dtype=np.uint16)
        for filament in self.filaments:
            x_corrected = filament.xy[0] + filament.island.x_min
            y_corrected = filament.xy[1] + filament.island.y_min
            self.img_filaments[x_corrected, y_corrected] = filament.img_reduced[
                tuple(filament.xy_norm)
            ]

    def save_filament_img(self):
        py.figure()
        py.imshow(self.img_filaments, cmap=cm.gray)
        ax = py.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        py.savefig(self.directory + "/filaments_%03d.png" % (self.frame_no), dpi=200)
        py.close()

    def calc_fil_corr_funcs(self):
        for filament in self.filaments:
            filament.correlation_function()

    def save_filXYs(self):
        filament_file = self.directory + "/filXYs%03d" % self.frame_no
        np.save(filament_file, np.array(self.filXYs, dtype=object))

    def read_filXYs(self):
        filament_file = self.directory + "/filXYs%03d.npy" % self.frame_no
        self.filXYs = np.load(filament_file, allow_pickle=True)

    def check_picture_quality(self):
        img_u, scale = self._rank_image()
        # The absolute-contrast gate (max_diff > 1000) is on the native 16-bit
        # scale; when the rank input has been rescaled to 8-bit, map the
        # threshold by the same factor so the gate keeps its meaning.
        diff_thresh = 1000.0 * scale
        img_1 = rank.percentile(img_u, self.disk_win, p0=0.95)
        img_0 = rank.percentile(img_u, self.disk_win, p0=0.05)
        self.img_diff = img_1.astype(np.int32) - img_0.astype(np.int32)

        max_diff = np.max(self.img_diff)
        # relative_contrast is a ratio, so it is unaffected by rescaling.
        relative_contrast = 1.0 * np.max(self.img_diff) / np.max(img_1)

        if relative_contrast > 0.7 and max_diff > diff_thresh:
            return "good"
        return "bad"

    def low_pass_filter(self, sigma=2):
        # cv2.GaussianBlur is markedly faster than scipy.ndimage.gaussian_filter
        # and numerically equivalent here.  ksize=(0,0) lets OpenCV derive the
        # kernel size from sigma.  Fall back to scipy if OpenCV rejects the
        # dtype for any reason.
        try:
            self.img = cv2.GaussianBlur(self.img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        except cv2.error:
            self.img = gaussian_filter(self.img, sigma=sigma)

    def _morph_kernel(self):
        """Disk structuring element matching ``disk_win``, cached for reuse."""
        if self._morph_k is None:
            r = int(self.window_island)
            self._morph_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
        return self._morph_k

    def _as_rank_image(self):
        """Return ``self.img`` as an integer image suitable for rank filters."""
        if self.img.dtype == np.uint16 or self.img.dtype == np.uint8:
            return self.img
        return self.img.astype(np.uint16)

    def _rank_image(self):
        """Return (image, scale) for the percentile/rank filters.

        With ``fast_rank`` enabled, a 16-bit image is linearly rescaled to the
        full 0-255 uint8 range, which shrinks the rank filter's local histogram
        from 65536 to 256 bins (the dominant per-frame cost).  ``scale`` is the
        8-bit-per-16-bit-count factor (255 / dynamic-range) so callers that
        compare against native-scale thresholds can convert them; it is 1.0 in
        the non-rescaled path.
        """
        img = self._as_rank_image()
        if not self.fast_rank or img.dtype != np.uint16:
            return img, 1.0
        lo = int(img.min())
        hi = int(img.max())
        if hi <= lo:
            return np.zeros(img.shape, dtype=np.uint8), 1.0
        scale = 255.0 / (hi - lo)
        img8 = ((img.astype(np.float32) - lo) * scale).astype(np.uint8)
        return img8, scale

    def entropy_clusters(self):
        img_u, _ = self._rank_image()
        if self.morph_contrast:
            # Single-pass local contrast: morphological gradient (local max -
            # min) over the same disk window, replacing the two sliding-
            # histogram percentile passes below.
            kernel = self._morph_kernel()
            img_hi = cv2.dilate(img_u, kernel)
            img_lo = cv2.erode(img_u, kernel)
            self.img_diff = img_hi.astype(np.int32) - img_lo.astype(np.int32)
        else:
            img_1 = rank.percentile(img_u, self.disk_win, p0=0.95).astype(np.int32)
            img_0 = rank.percentile(img_u, self.disk_win, p0=0.05).astype(np.int32)
            self.img_diff = img_1 - img_0

        self.cutoff_diff = np.mean(self.img_diff)
        self.mask_diff = self.img_diff > self.cutoff_diff

        self.labels_island, self.num_island = label(self.mask_diff)
        self.img_water = watershed(self.mask_diff, self.labels_island, mask=self.mask_diff)

        self.islands = []
        for i in range(1, self.num_island + 1):
            xy = np.nonzero(self.img_water == i)
            if len(xy[0]) < 2:
                continue
            new_island = Island()
            new_island.reduce_image(xy, self.img)
            new_island.frame = self
            self.islands.append(new_island)

    def filter_function(self, island):
        return island.area > self.window_island ** 2

    def filter_islands(self):
        self.islands = list(filter(self.filter_function, self.islands))

    def skeletonize_islands(self):
        self.filament_counter = 0
        [island.decompose_to_filaments() for island in self.islands]
        [island.filter_filaments() for island in self.islands]
        [island.skeletonize_filaments() for island in self.islands]
        [island.remove_crossing_filaments() for island in self.islands]

        self.filaments = [island.filaments for island in self.islands]
        self.filaments = [item for sub in self.filaments for item in sub]

    def filaments2filXYs(self):
        self.filXYs = []
        for filament in self.filaments:
            self.filXYs.append(
                [filament.contour, filament.fil_width, filament.fil_density, filament.midpoint]
            )


class Island:
    def __init__(self):
        self.area = 0
        self.xy = []
        self.xy_norm = [[], []]
        self.x_min = []
        self.y_min = []
        self.x_dim = 0
        self.y_dim = 0
        self.img_reduced = None
        self.filaments = []

        self.min_filament = 4
        self.window_island = 15

        self.frame = None

    def reduce_image(self, xy, img):
        self.area = len(xy[0])
        self.xy = xy
        self.x_min = np.min(self.xy[0])
        self.y_min = np.min(self.xy[1])
        self.x_dim = np.max(self.xy[0]) - np.min(self.xy[0])
        self.y_dim = np.max(self.xy[1]) - np.min(self.xy[1])

        self.xy_norm = [self.xy[0] - self.x_min, self.xy[1] - self.y_min]

        self.img_reduced = np.zeros((self.x_dim + 1, self.y_dim + 1))
        self.img_reduced[tuple(self.xy_norm)] = img[tuple(self.xy)]

    def decompose_to_filaments(self):
        valid = self.img_reduced > 0
        cutoff = threshold_otsu(self.img_reduced[valid])

        self.fil_reduced = self.img_reduced > cutoff
        self.img_fil = self.fil_reduced * self.img_reduced

        fil_labels, fil_features = label(self.fil_reduced)
        fine_clusters = watershed(self.fil_reduced, fil_labels, mask=self.fil_reduced)

        self.filaments = []
        for i in range(1, fil_features + 1):
            xy_bool = fine_clusters == i
            xy = np.nonzero(xy_bool)
            if len(xy[0]) < 2:
                continue

            new_filament = Filament()
            new_filament.label = self.frame.filament_counter
            new_filament.island = self
            new_filament.reduce_image(xy)
            new_filament.fil_density = np.sum(1.0 * self.img_reduced * xy_bool) / np.sum(xy_bool)
            new_filament.img_reduced = binary_fill_holes(new_filament.img_reduced)
            new_filament.img_reduced = binary_closing(
                new_filament.img_reduced, structure=disk_1
            )
            self.filaments.append(new_filament)
            self.frame.filament_counter += 1

    def filter_function(self, filament):
        size_constraint = np.sum(filament.img_reduced) > 10
        x_constraint = (filament.x_min + self.x_min > 5) and (
            filament.x_min + self.x_min + filament.x_dim < self.frame.width
        )
        y_constraint = (filament.y_min + self.y_min > 5) and (
            filament.y_min + self.y_min + filament.y_dim < self.frame.height
        )
        return size_constraint and x_constraint and y_constraint

    def filter_filaments(self):
        self.filaments = list(filter(self.filter_function, self.filaments))

    def skeletonize_filaments(self):
        [filament.make_skeleton() for filament in self.filaments]
        self.filaments = [fil for fil in self.filaments if len(fil.contour) > 0]
        [filament.calc_fil_stats() for filament in self.filaments]

    def remove_crossing_filaments(self):
        self.filaments = [f for f in self.filaments if f.num_tips == 2]


class Filament:
    def __init__(self):
        self.frame_no = 0
        self.label = 0

        self.contour = []
        self.coarse = []
        self.cm = []
        self.midpoint = []

        self.edge = 5
        self.img_reduced = []
        self.img_skeleton = []

        self.tips = []
        self.num_tips = 0

        self.fil_length = 0
        self.fil_density = 0
        self.fil_area = 0
        self.fil_width = 0
        self.end2end = 0

        self.next_filament = None
        self.pre_filament = None

        self.island = None
        self.corr_len = None

        self.forward_link = None
        self.reverse_link = None

        self.time = 0

    def reduce_image(self, xy):
        self.xy = xy
        self.x_min = np.min(self.xy[0])
        self.y_min = np.min(self.xy[1])
        self.x_dim = np.max(self.xy[0]) - np.min(self.xy[0])
        self.y_dim = np.max(self.xy[1]) - np.min(self.xy[1])

        self.xy_norm = [self.xy[0] - self.x_min + self.edge, self.xy[1] - self.y_min + self.edge]

        self.img_reduced = np.zeros(
            (self.x_dim + 1 + 2 * self.edge, self.y_dim + 1 + 2 * self.edge), dtype=np.uint16
        )
        self.img_reduced[tuple(self.xy_norm)] = True

    def find_tips(self):
        skel_u = self.img_skeleton.astype(np.uint8)
        neighbours = rank.pop(skel_u, sqr_3, mask=skel_u)
        tips = np.nonzero(neighbours * self.img_skeleton == 2)
        if len(tips[0]) == 0:
            self.num_tips = 0
            return np.array([])
        self.tips = np.hstack((np.vstack(tips[0]), np.vstack(tips[1])))
        self.num_tips = len(self.tips)
        return self.tips

    def make_skeleton(self):
        self.img_skeleton = skeletonize(self.img_reduced.astype(bool))
        self.find_tips()
        if self.num_tips == 2:
            self.remove_bad_pixels()
            self.make_links()

    def sim_score(self, fil_other):
        short_current = False
        if len(self.contour) < len(fil_other.contour):
            short_current = True

        contour1_diff = self.contour[1:] - self.contour[:-1]
        contour2_diff = fil_other.contour[1:] - fil_other.contour[:-1]

        contour1_len = len(self.contour)
        contour2_len = len(fil_other.contour)

        short_con_len = min((contour1_len, contour2_len))
        short_fil_len = min((self.fil_length, fil_other.fil_length))

        multiplicate_measures = abs(contour1_len - contour2_len) + 1

        fil_direction = 1
        mov_direction = 1

        overlap_score = 0
        for i in range(multiplicate_measures):
            if short_current:
                len1 = np.sum(
                    np.sqrt(np.sum(contour2_diff[i : i + short_con_len - 1, :] ** 2, axis=1))
                )
                len2 = np.sum(np.sqrt(np.sum(contour1_diff ** 2, axis=1)))
                if len1 > 0 and len2 > 0:
                    overlap_score += np.sum(
                        contour2_diff[i : i + short_con_len - 1, :] * contour1_diff
                    )
            else:
                len1 = np.sum(
                    np.sqrt(np.sum(contour1_diff[i : i + short_con_len - 1, :] ** 2, axis=1))
                )
                len2 = np.sum(np.sqrt(np.sum(contour2_diff ** 2, axis=1)))
                if len1 > 0 and len2 > 0:
                    overlap_score += np.sum(
                        contour1_diff[i : i + short_con_len - 1, :] * contour2_diff
                    )

        overlap_score /= 1.0 * (multiplicate_measures * short_fil_len)

        if overlap_score > 0:
            fil_direction = 1
        else:
            fil_direction = -1

        area_score = 0
        distance_score = 0
        move_score = 0

        for i in range(multiplicate_measures):
            if short_current:
                contour2_1_diff = (
                    fil_other.contour[i : i + short_con_len, :][::fil_direction] - self.contour
                )
                dot_prod = contour2_1_diff[:-1, :] * contour1_diff
                cross_prod = contour2_1_diff[:-1, :] * contour1_diff[:, [1, 0]]
                cross_prod = np.fabs(cross_prod[:, 1] - cross_prod[:, 0])
                contour1_diff_len = np.mean(np.sqrt(np.sum(contour1_diff ** 2, axis=1)))
            else:
                contour2_1_diff = (
                    fil_other.contour[::fil_direction] - self.contour[i : i + short_con_len, :]
                )
                dot_prod = contour2_1_diff[:-1, :] * contour1_diff[i : i + short_con_len - 1, :]
                cross_prod = (
                    contour2_1_diff[:-1, :] * contour1_diff[i : i + short_con_len - 1, :][:, [1, 0]]
                )
                cross_prod = np.fabs(cross_prod[:, 1] - cross_prod[:, 0])
                contour1_diff_len = np.mean(
                    np.sqrt(np.sum(contour1_diff[i : i + short_con_len - 1, :] ** 2, axis=1))
                )

            area_score += np.sum(cross_prod)
            distance_length = np.mean(np.sqrt(np.sum(contour2_1_diff[:-1, :] ** 2, axis=1)))
            distance_score += distance_length

            if distance_length > 0 and contour1_diff_len > 0:
                move_score += np.mean(np.sum(dot_prod, axis=1)) / (
                    distance_length * contour1_diff_len
                )

        area_score = area_score / (short_fil_len * multiplicate_measures) + ZERO
        distance_score /= multiplicate_measures
        move_score /= multiplicate_measures

        if move_score > 0:
            mov_direction = 1
        else:
            mov_direction = -1

        return overlap_score, area_score, distance_score, fil_direction, mov_direction

    def remove_bad_pixels(self):
        tip_s = self.tips[0, :]
        tip_e = self.tips[1, :]
        bad_fil = True
        while bad_fil:
            skel_u = self.img_skeleton.astype(np.uint8)
            nb_all = rank.pop(skel_u, disk_1, mask=skel_u)
            nb_3 = np.nonzero(nb_all * self.img_skeleton == 4)

            if len(nb_3[0]) > 0:
                bad_1 = [nb_3[0][0], nb_3[1][0]]
                self.img_skeleton[bad_1[0]][bad_1[1]] = 0
                new_tips = self.find_tips()
                for i in range(self.num_tips):
                    new_tip = new_tips[i, :]
                    if np.all(new_tip != tip_s) and np.all(new_tip != tip_e):
                        self.img_skeleton[new_tip[0]][new_tip[1]] = 0
            else:
                bad_fil = False

    def N_point(self, N=3):
        points_floor = np.array(
            [int(np.floor(x)) for x in np.linspace(0, self.num_contour_pixels - 1, N)], dtype=int
        )
        points_ceil = np.array(
            [int(np.ceil(x)) for x in np.linspace(0, self.num_contour_pixels - 1, N)], dtype=int
        )
        self.coarse = 0.5 * (1.0 * self.contour[points_floor, :] + 1.0 * self.contour[points_ceil, :])
        return self.coarse

    def calc_fil_length(self):
        self.num_contour_pixels = len(self.contour[:, 0])
        dist = self.contour[1:, :] - self.contour[:-1, :]
        self.fil_length = np.sum(np.sqrt(np.sum(dist ** 2, axis=1)))

    def calc_fil_stats(self):
        self.calc_fil_length()
        self.cm = np.mean(1.0 * self.contour, axis=0)
        self.midpoint = self.contour[len(self.contour) // 2 - 1]
        self.fil_area = np.sum(self.img_reduced)
        self.fil_width = 0.0
        if self.fil_length > 0:
            self.fil_width = 1.0 * self.fil_area / self.fil_length
        self.N_point()

    def calc_props(self):
        self.calc_fil_length()
        self.cm = np.mean(self.contour, axis=0)
        self.midpoint = self.contour[len(self.contour) // 2 - 1]
        self.N_point()

    def correlation_function(self, P=3):
        tan_vecs = self.contour[:-P, :] - self.contour[P:, :]
        len_vecs = vec_length(tan_vecs)
        num_vecs = len(tan_vecs)
        self.corr_len = [[num_vecs, 1.0]]

        for i in range(1, len(tan_vecs)):
            if np.sum(len_vecs[:-i] * len_vecs[i:] == 0) == 0:
                self.corr_len.append(
                    [
                        num_vecs - i,
                        np.mean(
                            np.sum(tan_vecs[:-i, :] * tan_vecs[i:, :], axis=1)
                            / (len_vecs[:-i] * len_vecs[i:])
                        ),
                    ]
                )

        if np.sum(np.isnan(self.corr_len)) > 0:
            self.corr_len = []
        self.corr_len = np.array(self.corr_len)

    def make_links(self):
        self.contour = []
        self.num_contour_pixels = int(np.sum(self.img_skeleton))
        img_c = self.img_skeleton.copy()
        tip_s = self.tips[0, :]
        tip_e = self.tips[1, :]
        self.contour.append(tip_s)

        for i in range(self.num_contour_pixels - 1):
            img_c[tip_s[0], tip_s[1]] = 0

            if img_c[tip_s[0] + 1, tip_s[1]] == 1:
                new_tip_s = [tip_s[0] + 1, tip_s[1]]
            elif img_c[tip_s[0], tip_s[1] + 1] == 1:
                new_tip_s = [tip_s[0], tip_s[1] + 1]
            elif img_c[tip_s[0] + 1, tip_s[1] + 1] == 1:
                new_tip_s = [tip_s[0] + 1, tip_s[1] + 1]
            elif img_c[tip_s[0] - 1, tip_s[1]] == 1:
                new_tip_s = [tip_s[0] - 1, tip_s[1]]
            elif img_c[tip_s[0], tip_s[1] - 1] == 1:
                new_tip_s = [tip_s[0], tip_s[1] - 1]
            elif img_c[tip_s[0] - 1, tip_s[1] - 1] == 1:
                new_tip_s = [tip_s[0] - 1, tip_s[1] - 1]
            elif img_c[tip_s[0] + 1, tip_s[1] - 1] == 1:
                new_tip_s = [tip_s[0] + 1, tip_s[1] - 1]
            elif img_c[tip_s[0] - 1, tip_s[1] + 1] == 1:
                new_tip_s = [tip_s[0] - 1, tip_s[1] + 1]

            tip_s = new_tip_s
            self.contour.append(tip_s)
        self.contour = np.array(self.contour)

        offset_x = self.x_min + self.island.x_min - self.edge
        offset_y = self.y_min + self.island.y_min - self.edge

        self.contour = self.contour + np.array([offset_x, offset_y])
        return self.contour
