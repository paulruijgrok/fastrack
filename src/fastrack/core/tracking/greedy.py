"""Greedy, score-based frame-to-frame linker.

This is the original FAST linking algorithm, lifted verbatim out of
``Motility.make_frame_links`` and parameterized.  For each filament in frame N
it scores candidate partners in frame N+1 (overlap / area / distance), sorts by
overlap, and accepts the best candidate that passes the cutoffs.

The ``legacy`` flag selects the original Python-2 partner-recovery behaviour
(the leftover inner-loop variable) for bit-for-bit reproduction of published
results; the default recovers the partner by label so identity matches scores.
"""
import numpy as np

from ..link import Link
from .base import LINKERS, Linker


@LINKERS.register("greedy")
class GreedyLinker(Linker):
    def __init__(
        self,
        min_fil_length=0,
        max_velocity=25,
        max_length_dif=5,
        overlap_score_cutoff=0.4,
        log_area_score_cutoff=1.0,
        dif_log_area_score_cutoff=0.5,
        legacy=False,
    ):
        self.min_fil_length = min_fil_length
        self.max_velocity = max_velocity
        self.max_length_dif = max_length_dif
        self.overlap_score_cutoff = overlap_score_cutoff
        self.log_area_score_cutoff = log_area_score_cutoff
        self.dif_log_area_score_cutoff = dif_log_area_score_cutoff
        self.legacy = legacy

    def link(self, frame1, frame2, dt, elapsed_times):
        new_frame_links = []
        frame1.filaments = [
            f for f in frame1.filaments if f.fil_length > self.min_fil_length
        ]
        frame2.filaments = [
            f for f in frame2.filaments if f.fil_length > self.min_fil_length
        ]

        frame1.reset_filament_labels()
        frame2.reset_filament_labels()

        if len(elapsed_times) > 0:
            dt = (
                elapsed_times[frame2.frame_no]
                - elapsed_times[frame1.frame_no]
            )
            frame1_time = elapsed_times[frame1.frame_no]
            frame2_time = elapsed_times[frame2.frame_no]
        else:
            frame1_time = frame1.frame_no * dt
            frame2_time = frame2.frame_no * dt

        for i in range(len(frame1.filaments)):
            filament1 = frame1.filaments[i]

            link_candidates = []

            frame2_filaments = [
                filament
                for filament in frame2.filaments
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
                    if self.legacy:
                        # Original behaviour: ``filament2`` is the leftover inner-
                        # loop variable, i.e. the last *unsorted* candidate. This
                        # reproduces the published (internally inconsistent)
                        # results bit-for-bit.
                        filament2 = frame2_filaments[-1]
                    else:
                        filament2 = frame2.filaments[int(link_candidates[0, 0])]

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
                    new_link.instant_velocity = new_link.distance_score / dt
                    new_link.dt = dt

                    new_link.direct_link = True
                    new_link.reverse_link = filament1.reverse_link

                    filament1.forward_link = new_link

                    if filament2.reverse_link is None:
                        filament2.reverse_link = new_link
                    elif new_link.overlap_score < filament2.reverse_link.overlap_score:
                        prev_fil_label = int(filament2.reverse_link.filament1_label)
                        prev_filament1 = frame1.filaments[prev_fil_label]
                        prev_filament1.forward_link = None
                        filament2.reverse_link = new_link

        for i in range(len(frame1.filaments)):
            filament1 = frame1.filaments[i]
            if filament1.forward_link is not None:
                new_frame_links.append(filament1.forward_link)

        return new_frame_links, dt
