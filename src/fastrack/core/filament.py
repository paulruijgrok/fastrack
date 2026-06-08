"""Filament: a single detected actin filament.

Holds the reduced image, skeleton, tips, contour and the per-filament
measurements, plus ``sim_score`` (used by the tracker) and the local link
bookkeeping.  Moved verbatim from the original ``motility.py``; numerical
behaviour is unchanged.
"""
import numpy as np

from scipy.ndimage import label
from skimage.filters import rank
from skimage.morphology import skeletonize

from ._kernels import ZERO, disk_1, sqr_3
from ..analysis.geometry import vec_length


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
