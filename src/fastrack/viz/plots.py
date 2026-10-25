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
    fit_plateau,
    length_velocity,
)

# Semi-transparent white backing for value labels drawn over data (TOP5%,
# MVEL, %STUCK, <FIL-LENGTH>) in the length-velocity plot, so they stay
# readable over histogram bars and scatter points.
_LABEL_BOX = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0)


def auto_velocity_limit(velocities, top_5_velocity=0.0, step=250.0, margin=1.1):
    """Velocity axis limit (nm/s) for the length-velocity figure, from the data.

    The 99.5th percentile of the moving (velocity > 0) points, or the TOP5%
    velocity if that is higher (so its marker line is always on the plot),
    plus a 10 % margin, rounded up to a multiple of ``step``.  The fixed
    default (1500 nm/s) cut off fast assays: Taka's filaments glide at up to
    ~2.5 um/s, so their TOP5% line sat above the plot.
    """
    v = np.asarray(velocities, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    top = max(float(np.percentile(v, 99.5)) if len(v) else 0.0,
              float(top_5_velocity) if np.isfinite(top_5_velocity) else 0.0)
    if top <= 0:
        return 1500.0
    return float(step * np.ceil(margin * top / step))


def track_midpoints(path):
    """A track's filament midpoints as (x=col, y=row), earliest frame first.

    ``Path.links`` is stored latest link first; in each link ``filament1`` is
    the earlier frame and ``filament2`` the later one.
    """
    links = path.links[::-1]
    pts = [links[0].filament1_midpoint] + [lk.filament2_midpoint for lk in links]
    return np.array([[p[1], p[0]] for p in pts], dtype=float)


def make_N_colors(cmap_name, N):
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(N)
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(cmap_name, N)
    return cmap(np.arange(N))


class MotilityPlots:
    """Plotting methods mixed into :class:`fastrack.core.motility.Motility`."""

    def plot_2D_path_data(self, num_points, extra_fname=None, style="velocity",
                          show_labels=False, scale_bar_um=10.0):
        """Draw the tracks of all filaments followed for >= ``num_points`` links.

        Writes ``<directory>/paths_2D.png`` (no labels) and, with
        ``extra_fname``, ``<extra_fname>_2D.png`` (labels if ``show_labels``).

        ``style="velocity"`` (default): one line per track through the
        filament midpoints, coloured by the track's mean velocity (colourbar,
        nm/s), stuck tracks in grey, one arrowhead at the latest position
        pointing along the motion, a scale bar of ``scale_bar_um`` um (0 = none)
        and no margins.  ``show_labels`` writes each track's mean velocity.

        ``style="legacy"``: the original FAST figure (arrows per step in an
        arbitrary qualitative palette, velocity numbers always on the
        ``extra_fname`` copy).  Note its arrowheads point from the later to the
        earlier position, i.e. against the motion.
        """
        if style == "legacy":
            return self._plot_2D_path_data_legacy(num_points, extra_fname=extra_fname)
        if style != "velocity":
            raise ValueError("paths style must be 'velocity' or 'legacy', got %r" % (style,))
        return self._plot_2D_path_data_velocity(num_points, extra_fname=extra_fname,
                                                show_labels=show_labels,
                                                scale_bar_um=scale_bar_um)

    def _plot_2D_path_data_velocity(self, num_points, extra_fname=None, show_labels=False,
                                    scale_bar_um=10.0):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        filtered_paths = [x for x in self.paths if len(x.links) >= num_points]
        self.path_data, self.path_stats = [], []
        if len(filtered_paths) == 0:
            return

        tracks, vels, stuck_flags = [], [], []
        for path in filtered_paths:
            links = path.links[::-1]                       # earliest link first
            xy = track_midpoints(path)
            len_array = np.array([np.fabs(lk.average_length) for lk in links])
            vel_array = np.array([np.fabs(lk.instant_velocity) for lk in links])
            stuck = bool(path.stuck)
            # colour by the median: robust to the odd mislinked step, whose
            # instant velocity can be 100x the track's real speed
            track_v = 0.0 if stuck else float(np.median(vel_array))
            # same bookkeeping as the legacy plot (other code may read these)
            self.path_data.append([links[0].frame1_no, stuck, vel_array])
            self.path_stats.append([links[0].frame1_no, stuck, len(path.links),
                                    np.mean(len_array), np.mean(vel_array), np.std(vel_array)])
            tracks.append(xy)
            vels.append(track_v)
            stuck_flags.append(stuck)
        self.path_stats = np.array(self.path_stats)
        vels = np.asarray(vels)
        stuck_flags = np.asarray(stuck_flags)

        # scale from the bulk of the moving tracks; faster ones saturate
        # (top tick reads ">= vmax") instead of compressing everything else
        mv = vels[(~stuck_flags) & np.isfinite(vels) & (vels > 0)]
        vmax = auto_velocity_limit([np.percentile(mv, 95)] if mv.size else [], 0.0, margin=1.0)
        norm = Normalize(0, vmax)
        cmap = matplotlib.colormaps["viridis"]
        W, H = float(self.width), float(self.height)

        def draw(labels, fname):
            # figure exactly the image's aspect, axes filling it: no margins;
            # the colourbar sits outside and bbox_inches="tight" keeps it
            fig_w = 8.0
            fig = py.figure(figsize=(fig_w, fig_w * H / W))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)                              # image orientation
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            lw = 1.2
            moving = [t for t, s_ in zip(tracks, stuck_flags) if not s_ and len(t) > 1]
            moving_v = [v for v, s_, t in zip(vels, stuck_flags, tracks) if not s_ and len(t) > 1]
            static = [t for t, s_ in zip(tracks, stuck_flags) if s_ and len(t) > 1]
            if static:
                ax.add_collection(LineCollection(static, colors="0.6", linewidths=lw))
            if moving:
                lc = LineCollection(moving, cmap=cmap, norm=norm, linewidths=lw)
                lc.set_array(np.asarray(moving_v))
                ax.add_collection(lc)
            # one arrowhead per moving track, at its latest point, along the motion
            for t, v in zip(moving, moving_v):
                d = t[-1] - t[-2]
                if not np.any(d):
                    continue
                ax.annotate("", xy=t[-1], xytext=t[-1] - 0.5 * d,
                            arrowprops=dict(arrowstyle="-|>", color=cmap(norm(v)),
                                            lw=lw, mutation_scale=9, shrinkA=0, shrinkB=0))
            if labels:
                for t, v in zip(tracks, vels):
                    c = t.mean(axis=0)
                    ax.text(c[0], c[1], "%.f" % v, fontsize=6, color="k")
            if scale_bar_um and self.dx:
                bar_px = scale_bar_um * 1000.0 / float(self.dx)
                x0, y0 = 0.04 * W, 0.95 * H
                ax.plot([x0, x0 + bar_px], [y0, y0], "-", color="k", linewidth=4,
                        solid_capstyle="butt")
                ax.text(x0 + bar_px / 2, y0 - 0.015 * H, "%g $\\mu$m" % scale_bar_um,
                        ha="center", va="bottom", fontsize=12)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cax = fig.add_axes([1.02, 0.15, 0.03, 0.7])
            cb = fig.colorbar(sm, cax=cax)
            cb.formatter.set_useOffset(False)
            if np.any(vels > vmax):
                # faster tracks share the top colour; say so on the top tick
                fig.canvas.draw()
                ticks = cb.get_ticks()
                cb.set_ticks(ticks)
                cb.set_ticklabels(["%g" % t if t < vmax else "\u2265%g" % t for t in ticks])
            cb.set_label("Track velocity (median, nm/s)", fontsize=12)
            cb.ax.tick_params(labelsize=10)
            if len(static):
                cax.text(0.5, -0.06, "grey: stuck", transform=cax.transAxes, ha="center",
                         va="top", fontsize=10, color="0.45")
            fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
            py.close(fig)

        draw(False, self.directory + "/paths_2D.png")
        if extra_fname is not None:
            draw(show_labels, extra_fname + "_2D.png")

    def _plot_2D_path_data_legacy(self, num_points, extra_fname=None):
        """The original FAST paths figure, unchanged (``style="legacy"``)."""
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
        max_vel=0,
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
        label_position="bottom",
        label_units=False,
    ):
        """Compute velocity statistics and (optionally) render the length-velocity plot.

        Returns a tuple of summary statistics (see the original paper).  On too
        few data points it returns a tuple of ``-1`` sentinels.

        ``weighted`` is accepted for backward compatibility and ignored: it
        controlled a length-weighted coupling fit whose results were never used
        (and which, because its length histogram mixed nm and um, never ran
        successfully).  The fit that matters is selected by ``fit_f``.

        ``label_position`` ("bottom", default, or "top") places the value labels
        of the three top panels (TOP5%, MVEL, %STUCK, ...) on shared rows near
        the bottom or near the top of the panels.

        ``max_vel`` is the velocity axis limit (nm/s) of the figure; 0 or None
        picks it from the data (see :func:`auto_velocity_limit`).  It only
        affects the figure, not the returned statistics.
        """
        if label_position not in ("top", "bottom"):
            raise ValueError("label_position must be 'top' or 'bottom', got %r" % (label_position,))
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

        # Histogram edges for the figure (lengths in um, velocities in nm/s).
        l_bin_edges = np.linspace(0, max_length * 1e-3, nbins)
        v_bin_edges = np.linspace(0, max_vel, nbins)

        velocities_sorted = np.sort(fil_vel)[::-1]
        top_5_num = int(np.ceil(0.05 * len(velocities_sorted)))
        top_5_velocity = np.mean(velocities_sorted[:top_5_num])
        if not max_vel or max_vel <= 0:
            max_vel = auto_velocity_limit(self.full_len_vel[:, 1], top_5_velocity)
            v_bin_edges = np.linspace(0, max_vel, nbins)


        # One point per filament (its fastest window).  Same length cut as the
        # scatter/histograms: a few merged clumps or crossings come out tens of
        # um "long", and one 224 um point dragged the plateau fit to
        # 215,000 nm/s (the original code has the same issue).
        max_valid = np.nonzero(
            (self.max_len_vel[:, 1] > 0)
            * (self.max_len_vel[:, 2] <= percent_tolerance / 100.0 * self.max_len_vel[:, 1])
            * (self.max_len_vel[:, 3] >= min_path_length)
            * (self.max_len_vel[:, 0] < max_length)
        )[0]

        max_vel_u = -1
        max_vel_amp = 0
        max_vel_tau = 1
        max_vel_r = 0
        # Plateau fit with a standard error (fit_plateau: data-based start,
        # bounded parameters); result kept on self.fit_result for callers.
        self.fit_result = None
        max_vel_u_se = np.nan
        if fit_f in ("exp", "uyeda"):
            fr = fit_plateau(self.max_len_vel[max_valid, 0], self.max_len_vel[max_valid, 1], fit_f)
            self.fit_result = fr
            if fr["ok"]:
                max_vel_u, max_vel_u_se = fr["plateau"], fr["plateau_se"]
                if fit_f == "exp":
                    _, max_vel_amp, max_vel_tau = fr["params"]
                else:
                    _, max_vel_r = fr["params"]

        # Standard error of the mean filament length, over filaments (the
        # windowed points along one track are not independent): sets how many
        # decimals the <FIL-LENGTH> label shows.
        n_fil = max(len(max_valid), 1)
        mean_len_se = (float(np.std(fil_len)) / np.sqrt(n_fil)) if len(fil_len) > 1 else np.nan

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
                label_position=label_position,
                max_vel_u_se=max_vel_u_se,
                mean_len_se=mean_len_se,
                label_units=label_units,
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
        label_position = kw.get("label_position", "bottom")
        max_vel_u_se = kw.get("max_vel_u_se", np.nan)
        mean_len_se = kw.get("mean_len_se", np.nan)
        label_units = kw.get("label_units", False)
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
            # same left/right edges as the two velocity histograms above it and
            # the same bottom/height as the length histogram beside it
            rect_tolerance = [left_h1, bottom + 0.02, 0.30, 0.25]
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
            # extra headroom above the curves: the legend goes there
            tol_ymax = max_tol_vel + 100 + 0.5 * (max_tol_vel - min_tol_vel + 200)

            axTolerance2.set_ylim([tol_ymin, tol_ymax])
            # start just below the first tolerance (5 %) so its marker isn't
            # cut in half by the axis
            axTolerance2.set_xlim([4, 200])

            # a real legend in the headroom above the curves (the hand-drawn
            # one sat at a fixed spot, on top of low curves)
            from matplotlib.lines import Line2D
            axTolerance2.legend(
                handles=[Line2D([], [], color="k", linestyle="--", linewidth=5),
                         Line2D([], [], color="k", linestyle="-", linewidth=5)],
                labels=["TOP5%", "Mean Velocity"], loc="upper left", ncol=2,
                fontsize=0.7 * text_font_size, handlelength=1.6, columnspacing=1.0,
                handletextpad=0.4, borderaxespad=0.3,
                frameon=True, framealpha=0.85, edgecolor="none",
            )

            axTolerance2.set_xticks(tolerance_data[:, 0])
            axTolerance2.set_xticklabels(["*"] + [int(x) for x in tolerance_data[1:, 0]])
            axTolerance1.set_xlabel("% Tolerance", fontsize=text_font_size, labelpad=20)

            # ticks only over the range the curves span (not in the legend
            # headroom, where a top tick collided with the x10^3 marker)
            from matplotlib.ticker import MaxNLocator
            vel_ticks = np.array([t for t in MaxNLocator(nbins=2, steps=[1, 2, 5, 10]).tick_values(min_tol_vel, max_tol_vel)
                                  if tol_ymin <= t <= max_tol_vel + 100])
            if len(vel_ticks) > 2:
                vel_ticks = vel_ticks[::2]          # keep the labels apart
            axTolerance2.set_yticks(vel_ticks)
            axTolerance2.set_yticklabels(["%.1f" % (v * 1e-3) for v in vel_ticks])

            axTolerance2.annotate(r"$x10^3$", xy=(1, 1), xycoords="axes fraction", xytext=(6, 8),
                                  textcoords="offset points", ha="left", va="bottom", fontsize=25)

            py.setp(axTolerance2.get_yticklabels(), fontsize=text_font_size, visible=True)
            py.setp(axTolerance2.get_xticklabels(), fontsize=text_font_size, visible=True)
            py.setp(axTolerance1.get_yticklabels(), fontsize=text_font_size, visible=False)
            # the left axis of the twin pair is unused: no tick marks either,
            # and no symlog minor ticks between the tolerance values
            axTolerance1.tick_params(axis="y", left=False, which="both")
            from matplotlib.ticker import NullLocator
            axTolerance2.xaxis.set_minor_locator(NullLocator())
            py.setp(axTolerance1.get_xticklabels(), fontsize=text_font_size, visible=True)

            l_bin_counts, _, _ = axHistx1.hist(
                1e-3 * self.full_len_vel[valid_filtered, 0],
                bins=l_bin_edges,
                density=False,
                orientation="vertical",
                color="gray",
                edgecolor="k",
                linewidth=0.6,
            )
            max_prob_l = np.max(l_bin_counts)

            axHisty2.hist(
                self.full_len_vel[valid_mobile, 1],
                bins=v_bin_edges,
                density=True,
                orientation="horizontal",
                color="gray",
                edgecolor="k",
                linewidth=0.6,
            )
            max_prob_a = axHisty2.get_xlim()[1]

            axHisty1.hist(
                self.full_len_vel[valid_filtered, 1],
                bins=v_bin_edges,
                density=True,
                orientation="horizontal",
                color="gray",
                edgecolor="k",
                linewidth=0.6,
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

            if fit_f != "none" and max_vel_u > 0:
                axScatter.plot(1e-3 * exp_len, exp_vel, "k-", linewidth=3, alpha=0.8)

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

            from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
            # velocity: ~3 round major ticks over [0, max_vel], 4 sub-ticks between
            vel_ticks = np.array([t for t in MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(0, max_vel)
                                  if 0 <= t <= max_vel])
            axScatter.set_yticks(vel_ticks)
            axScatter.set_yticklabels(["%.1f" % (v * 1e-3) for v in vel_ticks])
            axScatter.yaxis.set_minor_locator(AutoMinorLocator(5))

            # length: majors every 2 um (the old code labelled 2.5 / 7.5 um ticks
            # as "2" / "7" by truncating), 1 um sub-ticks; the right edge (shared
            # with the tolerance panel) gets no label
            xmax_um = max_length * 1e-3
            len_ticks = np.arange(0, xmax_um - 1e-9, 2.0)
            for _ax in (axScatter, axHistx1):
                _ax.set_xticks(len_ticks)
                _ax.set_xticklabels(["%d" % x for x in len_ticks])
                _ax.xaxis.set_minor_locator(MultipleLocator(1.0))

            axHisty1.set_yticks(vel_ticks)
            axHisty1.set_yticklabels(vel_ticks * 1e-3)
            axHisty2.set_yticks(vel_ticks)
            axHisty2.set_yticklabels(vel_ticks * 1e-3)

            # axis multiplier, lifted a little off the axis line
            axScatter.annotate(r"$x10^3$", xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                               textcoords="offset points", ha="left", va="bottom", fontsize=25)

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
            # the histograms share the scatter's velocity axis and have no
            # labels of their own: drop their tick marks, which poked into the
            # neighbouring panels
            for _ax in (axHisty1, axHisty2):
                _ax.tick_params(left=False, bottom=False, which="both")
            py.setp(axHisty2.get_yticklabels(), visible=False)
            # length histogram: labelled count axis; headroom so its top tick
            # label stays clear of the scatter's "0.0" just above
            axHistx1.set_ylim(0, 1.15 * max_prob_l)
            axHistx1.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, prune="upper"))
            py.setp(axHistx1.get_yticklabels(), fontsize=0.8 * text_font_size, visible=True)
            axHistx1.set_ylabel("Number of events", labelpad=20, fontsize=0.8 * text_font_size)

            axTolerance2.set_ylabel(r"Velocity (nm/s)", labelpad=20, fontsize=text_font_size)
            axScatter.set_ylabel(r"Velocity (nm/s)", labelpad=20, fontsize=text_font_size)
            axHistx1.set_xlabel(
                r"Actin filament length ($\mu m$)", labelpad=20, fontsize=text_font_size
            )
            py.gcf().align_ylabels([axScatter, axHistx1])

            axHisty1.text(0.1 * max_prob_t, 1.1 * max_vel, "Filtered", fontsize=text_font_size)
            axHisty2.text(0.1 * max_prob_a, 1.1 * max_vel, "Unfiltered", fontsize=text_font_size)

            # The value labels of the three top panels share two rows, so they
            # line up across the figure: row 1 = TOP5% | MVEL filtered | MVEL,
            # row 2 = PLATEAU | %STUCK.  label_position puts the rows near the
            # top of the panels or near the bottom (slow, long filaments are
            # rare, so the bottom of the scatter and histograms is often empty).
            gap = 0.16 if label_units else 0.13      # stacked units need more room
            if label_position == "bottom":
                row2 = 0.06 * max_vel
                row1 = row2 + gap * max_vel
            else:
                row1 = 0.86 * max_vel
                row2 = row1 - gap * max_vel
            key_dy = 0.02 * max_vel          # key line at mid-height of the digits
            # label_units: the unit goes *under* the name (subscript stacked
            # with the superscript), so a label is no wider than without it
            unit_v = r"_{\mathregular{nm/s}}" if label_units else ""
            unit_l = r"_{\mu\mathregular{m}}" if label_units else ""
            # mean length shown to the decimal of its standard error (1st
            # significant digit), between 1 and 3 decimals
            se_um = mean_len_se * 1e-3
            len_decimals = (int(np.clip(-np.floor(np.log10(se_um)), 1, 3))
                            if np.isfinite(se_um) and se_um > 0 else 3)
            # value keys right of centre: long filaments are rare, so that part of
            # the panel is usually empty (the left sits on the short-filament cloud)
            axScatter.plot(
                max_length * 1e-3 * np.array([6, 7]) / 15.0,
                [row1 + key_dy, row1 + key_dy],
                "k--",
                linewidth=5,
            )
            axScatter.text(
                max_length * 1e-3 * 7.1 / 15.0,
                row1,
                r"%.f$^{TOP5\%%}%s$" % (top_5_velocity, unit_v),
                fontsize=text_font_size,
                color="k",
                bbox=_LABEL_BOX,
            )

            if fit_f != "none":
                axScatter.plot(
                    max_length * 1e-3 * np.array([6, 7]) / 15.0,
                    [row2 + key_dy, row2 + key_dy],
                    "k-",
                    linewidth=3,
                )
                axScatter.text(
                    max_length * 1e-3 * 7.1 / 15.0,
                    row2,
                    (r"%.f$\pm$%.f$^{PLATEAU}%s$" % (max_vel_u, max_vel_u_se, unit_v)
                     if np.isfinite(max_vel_u_se) else r"%.f$^{PLATEAU}%s$" % (max_vel_u, unit_v)),
                    fontsize=text_font_size,
                    color="k",
                bbox=_LABEL_BOX,
                )

            axHisty1.text(
                0.03 * max_prob_t,
                row1,
                r"%.f$^{MVEL_{%s}}%s$" % (MVEL_filtered, tolerance_string, unit_v),
                fontsize=text_font_size,
                color="k",
                bbox=_LABEL_BOX,
            )
            axHisty2.text(
                0.03 * max_prob_a,
                row1,
                r"%.f$^{MVEL}%s$" % (MVEL, unit_v),
                fontsize=text_font_size,
                color="k",
                bbox=_LABEL_BOX,
            )
            axHisty2.text(
                0.03 * max_prob_a,
                row2,
                r"%.f$^{\%%STUCK}$" % (percent_stuck),
                fontsize=text_font_size,
                color="k",
                bbox=_LABEL_BOX,
            )
            axHistx1.text(
                mean_len_filtered * 1e-3,
                max_prob_l * 0.5,
                r"%.*f$^{<FIL-LENGTH>}%s$" % (len_decimals, mean_len_filtered * 1e-3, unit_l),
                fontsize=text_font_size,
                color="k",
                bbox=_LABEL_BOX,
            )

        # bbox_inches="tight": grow the saved image to include every label and
        # annotation (axis labels and the x10^3 markers sit outside the axes and
        # were cut off at the figure edge)
        py.savefig(self.directory + "/" + header + "length_velocity.png", dpi=dpi_plot,
                   transparent=False, bbox_inches="tight", pad_inches=0.3)
        if extra_fname is not None:
            py.savefig(extra_fname, dpi=dpi_plot, transparent=False,
                       bbox_inches="tight", pad_inches=0.3)
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
