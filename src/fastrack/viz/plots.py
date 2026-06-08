"""Plotting for the motility analysis.

The plotting routines are large and self-contained, so they live here rather
than in :mod:`fastrack.core.motility`.  They are provided as a ``MotilityPlots``
mixin whose methods operate on the ``Motility`` instance exactly as before, so
the behaviour and output files are unchanged -- only the code location moved.
``make_N_colors`` is the colour-ramp helper used by these plots.
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as py  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from scipy import stats  # noqa: E402

from . import plotparams  # noqa: E402
from ..analysis import (  # noqa: E402
    coupling_velocity,
    fit_coupling_velocity,
    fit_length_velocity,
    length_velocity,
)


def make_N_colors(cmap_name, N):
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(N)
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(cmap_name, N)
    return cmap(np.arange(N))


class MotilityPlots:
    """Plotting methods mixed into :class:`fastrack.core.motility.Motility`."""

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
