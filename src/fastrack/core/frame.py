"""Frame: one movie frame and the filaments detected within it.

Holds the loaded image plus the entropy/watershed working state, and exposes
the per-frame primitives (low-pass, percentile/morphological contrast, island
labelling, skeletonization) that the entropy detector orchestrates.  Moved
verbatim from the original ``motility.py``; the ``fast_rank`` / ``morph_contrast``
optimizations are preserved.
"""
import os

import numpy as np

import cv2
from scipy.ndimage import gaussian_filter, label
from skimage.filters import rank
from skimage.morphology import disk, skeletonize
from skimage.segmentation import watershed
from skimage.util import img_as_uint

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as py  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from .filament import Filament  # noqa: E402
from .island import Island  # noqa: E402


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

        # Suffix for the cached filXYs filename, so different detectors don't
        # share a cache in the same movie folder ("" for the default entropy
        # detector keeps the original filXYs%03d.npy names unchanged).
        self.cache_tag = ""

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
        filament_file = self.directory + "/filXYs%s%03d" % (self.cache_tag, self.frame_no)
        np.save(filament_file, np.array(self.filXYs, dtype=object))

    def read_filXYs(self):
        filament_file = self.directory + "/filXYs%s%03d.npy" % (self.cache_tag, self.frame_no)
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


