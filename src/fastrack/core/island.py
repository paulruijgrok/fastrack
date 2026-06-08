"""Island: a connected bright region that decomposes into one or more filaments.

Moved verbatim from the original ``motility.py``.  The Otsu/watershed
decomposition and per-filament morphology are unchanged.
"""
import numpy as np

from scipy.ndimage import binary_closing, binary_fill_holes, label
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from ._kernels import disk_1
from .filament import Filament


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


