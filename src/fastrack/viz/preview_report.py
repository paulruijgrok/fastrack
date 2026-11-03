"""The fast-preview report: a portrait, multi-page document per movie.

Page 1 -- the main findings:
    title (movie) and run line; summary (verdict, mean / top-5 % velocity,
    stuck %, length, density); a row of three small plots (velocity histogram,
    length vs velocity, length histogram); the quality scores as slider rows
    with the suggested fix; the frame with the tracks overlaid when it fits at
    >= 2/3 of the page width.
Page 2 -- supporting data:
    the overlay (if it did not fit on page 1), floppy / pinned track examples,
    tracking statistics.

Written as one PDF (``<movie>_preview.pdf``) and one PNG per page
(``<movie>_preview_p1.png``, ...).  Figure ``preview_report`` in the plot
theme; its options (page size, overlay placement, slider colouring) are in
``[figures.preview_report]``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

import numpy as np

PAGE_SIZES = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}

# display names for the quality metrics (QUALITY_SPECS labels are plain ASCII
# for the terminal report)
_LABELS = {
    "snr": "Filament signal-to-noise",
    "focus_width_nm": "Filament width (focus)",
    "flatness": "Illumination flatness",
    "saturation_pct": "Saturated pixels",
    "background_to_signal": "Background / filament signal",
    "density_low": "Filament density (too few)",
    "density_high": "Filament density (crowded)",
    "stuck_pct": "Stuck filaments",
    "floppy_pct": "Floppy tracks",
    "pinned_pct": "Pinned filaments",
    "bleach_pct_per_frame": "Bleaching",
    "step_over_length": "Step per frame / filament length",
}
_DECIMALS = {"snr": 1, "focus_width_nm": 0, "flatness": 2, "saturation_pct": 3,
             "background_to_signal": 1, "density_low": 2, "density_high": 2,
             "stuck_pct": 1, "floppy_pct": 1, "pinned_pct": 1,
             "bleach_pct_per_frame": 2, "step_over_length": 2}
_UNITS = {"density_low": "/100 µm²", "density_high": "/100 µm²",
          "bleach_pct_per_frame": "%/frame"}


def _num(v, nd=0, thousands=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return ("{:,.%df}" % nd).format(v) if thousands else "%.*f" % (nd, v)


def metric_text(q: Dict[str, Any]) -> str:
    """Value + unit of a quality row, e.g. '48.1 %', '264 nm', '0.54 /100 µm²'."""
    key = q.get("metric", "")
    unit = _UNITS.get(key, q.get("unit", ""))
    val = _num(q.get("value"), _DECIMALS.get(key, 2), thousands=False)
    if not unit:
        return val
    return val + (" " if unit.startswith("%") else " ") + unit


# --------------------------------------------------------------------------- #
# layout helper: place axes / text in inches from the top-left of a page
# --------------------------------------------------------------------------- #
class _Page:
    def __init__(self, plt, size, fx, margin=0.6):
        self.W, self.H = size
        self.fig = plt.figure(figsize=size)
        self.fx = fx
        self.m = margin
        self.y = margin                      # cursor, inches from the top

    @property
    def bottom(self):
        """Lowest y (inches from the top) content may reach (above the footer)."""
        return self.H - self.m - 0.15

    @property
    def text_w(self):
        return self.W - 2 * self.m

    def ax(self, x, y, w, h, **kw):
        """Axes at (x, y) inches from the top-left, size w x h inches."""
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def rule(self, y, color="0.82", lw=0.6):
        a = self.ax(self.m, y, self.text_w, 0.001)
        a.axis("off")
        a.plot([0, 1], [0.5, 0.5], "-", color=color, lw=lw, transform=a.transAxes, clip_on=False)

    def section(self, title, y=None):
        y = self.y if y is None else y
        self.text(self.m, y, title, fontsize=self.fx.fs(12.5), fontweight="bold",
                  color=self.fx.color("report_ink"))
        self.y = y + 0.30
        return self.y


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def render_report(name: str, r: Dict[str, Any], path: str, fx) -> List[str]:
    """Draw the report for one movie and save it (PDF + PNG per page by default)."""
    import matplotlib.pyplot as plt

    opt = fx.opt
    size = PAGE_SIZES.get(str(opt.get("page_size", "letter")).lower(), PAGE_SIZES["letter"])
    P = r["_plot"]

    p1 = _Page(plt, size, fx)
    _header(p1, name, r)
    _summary(p1, r)
    _plot_row(p1, r, P)
    _quality(p1, r, slider=opt.get("slider", "status"))
    pages = [p1]

    def room(pg):
        return pg.bottom - pg.y

    def page_for(need):
        """Current page if ``need`` inches fit, else a new page."""
        pg = pages[-1]
        if len(pages) > 1 and room(pg) >= need:
            return pg
        pg = _Page(plt, size, fx)
        _header(pg, name, r, small=True)
        pages.append(pg)
        return pg

    # overlay on page 1 if it fits at >= overlay_min_width of the page width
    img = P["img"]
    aspect = img.shape[0] / img.shape[1]
    where = str(opt.get("overlay_page", "auto"))
    fit_w = min(p1.text_w, (room(p1) - _OVERLAY_EXTRA) / aspect)
    min_w = float(opt.get("overlay_min_width", 0.667)) * p1.W
    if (where == "1" and fit_w >= 1.5) or (where == "auto" and fit_w >= min_w):
        _overlay(p1, r, P, width=fit_w)
    else:
        w = min(float(opt.get("overlay_width", 0.68)) * p1.W, p1.text_w)
        _overlay(page_for(w * aspect + _OVERLAY_EXTRA), r, P, width=w)
    _examples(page_for(_examples_height(p1)), r, P)
    rows = _tracking_rows(r)
    _table(page_for(_table_height(len(rows))), "Tracking statistics", rows)
    rows = _threshold_rows(r)
    _table(page_for(_table_height(len(rows) + 1)), "Quality metrics and thresholds", rows,
           header=("Metric", "Value", "Good", "Poor"), cols=(0.0, 3.0, 4.6, 5.8))
    for i, pg in enumerate(pages):
        _footer(pg, i + 1, len(pages))
    files = fx.save_pages([pg.fig for pg in pages], path)
    for pg in pages:
        plt.close(pg.fig)
    return files


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _header(pg, name, r, small=False):
    fx = pg.fx
    ink = fx.color("report_ink")
    if small:
        pg.text(pg.m, pg.y, name, fontsize=fx.fs(11), fontweight="bold", color=ink)
        pg.text(pg.W - pg.m, pg.y, "fast-preview report – supporting data", ha="right",
                fontsize=fx.fs(8.5), color=fx.color("report_muted"))
        pg.y += 0.30
        pg.rule(pg.y)
        pg.y += 0.25
        return
    pg.text(pg.m, pg.y, "FAST-PREVIEW REPORT", fontsize=fx.fs(8), color=fx.color("report_muted"),
            fontweight="bold")
    pg.text(pg.m, pg.y + 0.20, name, fontsize=fx.fs(20), fontweight="bold", color=ink)
    tr = r.get("tracking") or {}
    dt = tr.get("frame_interval_s")
    run = ["%d frames" % r["n_frames"]
           + (" at %s s/frame" % _num(dt, 2, False) if dt else ""),
           "%d×%d binning" % (r["bin"], r["bin"]),
           r["detector"],
           "%s nm/px" % _num(r["pixel_size_nm"] * r["bin"], 0),
           "analysed in %s s" % _num(r["seconds"]["total"], 1),
           _dt.datetime.now().strftime("%Y-%m-%d %H:%M")]
    pg.text(pg.m, pg.y + 0.62, "  ·  ".join(run), fontsize=fx.fs(8.5),
            color=fx.color("report_muted"))
    pg.y += 0.92
    pg.rule(pg.y)
    pg.y += 0.22


def _summary(pg, r):
    """Verdict badge + headline numbers + one sentence."""
    fx = pg.fx
    ink, muted = fx.color("report_ink"), fx.color("report_muted")
    status = {"GOOD": "good", "MARGINAL": "marginal", "POOR": "poor"}.get(r["verdict"], "marginal")
    v = r["velocity_nm_s"]
    L = r["length_um"]
    # verdict badge
    bw, bh = 1.45, 0.72
    a = pg.ax(pg.m, pg.y, bw, bh)
    a.axis("off")
    from matplotlib.patches import FancyBboxPatch
    a.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.10",
                               transform=a.transAxes, facecolor=fx.color(status), edgecolor="none"))
    a.text(0.5, 0.70, "QUALITY", transform=a.transAxes, ha="center", va="center",
           fontsize=fx.fs(7.5), color="white", fontweight="bold", alpha=0.9)
    a.text(0.5, 0.36, r["verdict"], transform=a.transAxes, ha="center", va="center",
           fontsize=fx.fs(17), color="white", fontweight="bold")
    # headline numbers
    tiles = [
        ("MEAN VELOCITY", _num(v.get("mean_moving"), 0), "nm/s",
         "top 5 %%: %s nm/s" % _num(v.get("top5"), 0)),
        ("STUCK", _num(r.get("stuck_pct"), 1), "%",
         "%d static / %s moving per frame" % (r["static_filaments"], _num(r["moving_per_frame"], 0))),
        ("FILAMENT LENGTH", _num(L.get("median"), 2), "µm",
         "median (p10–p90: %s–%s)" % (_num(L.get("p10"), 1), _num(L.get("p90"), 1))),
    ]
    x0 = pg.m + bw + 0.35
    tw = (pg.W - pg.m - x0) / len(tiles)
    for i, (lab, val, unit, sub) in enumerate(tiles):
        x = x0 + i * tw
        pg.text(x, pg.y + 0.02, lab, fontsize=fx.fs(7.5), fontweight="bold", color=muted)
        t = pg.text(x, pg.y + 0.20, val, fontsize=fx.fs(19), fontweight="bold", color=ink)
        _after(pg, t, " " + unit, fontsize=fx.fs(10), color=ink, dy=0.13)
        pg.text(x, pg.y + 0.56, sub, fontsize=fx.fs(8), color=muted)
    pg.y += bh + 0.18
    # one-sentence summary
    q = r["quality"]
    bad = [x for x in q if x["status"] != "good"]
    if bad:
        w = bad[0]
        s = ("Main issue: %s (%s). %d of %d quality checks need attention."
             % (_LABELS.get(w["metric"], w["label"]).lower(), metric_text(w), len(bad), len(q)))
    else:
        s = "All %d quality checks are in the good range." % len(q)
    pg.text(pg.m, pg.y, s, fontsize=fx.fs(10), color=ink)
    pg.y += 0.38


def _after(pg, t, s, dy=0.0, **kw):
    """Draw ``s`` right after text ``t`` (same baseline region)."""
    fig = pg.fig
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x_in = bb.x1 / fig.dpi
    y_top_in = pg.H - bb.y1 / fig.dpi
    return pg.text(x_in, y_top_in + dy, s, **kw)


def _style_small_axes(a, fx):
    a.tick_params(labelsize=fx.fs(7.5), length=3, width=0.6, pad=2)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_linewidth(0.7)
    a.xaxis.label.set_size(fx.fs(8.5))
    a.yaxis.label.set_size(fx.fs(8.5))


def _plot_row(pg, r, P):
    """Velocity histogram | length vs velocity | length histogram."""
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    fx = pg.fx
    gap, h = 0.62, 1.75
    w = (pg.text_w - 2 * gap) / 3.0
    x = [pg.m + 0.18 + i * (w + gap) for i in range(3)]
    w = w - 0.18 / 3
    y = pg.y + 0.22
    moving_c, maxvel_c, edge = fx.color("moving"), fx.color("maxvel"), "k"
    lv = np.asarray(P.get("len_vel", np.zeros((0, 2))), float)
    mx = np.asarray(P.get("max_len_vel", np.zeros((0, 2))), float)
    mx = mx[mx[:, 1] > 0] if len(mx) else mx
    vel = np.asarray(P.get("velocities", []), float)
    lens = np.asarray(P.get("lengths_um", []), float)
    slens = np.asarray(P.get("static_lengths_um", []), float)
    vmax = max(500.0, float(np.percentile(vel, 99.5)) * 1.1) if len(vel) else 2500.0
    vmax = 250.0 * np.ceil(vmax / 250.0)
    lmax = max(4.0, float(np.percentile(lens, 99)) * 1.05) if len(lens) else 6.0
    lmax = float(np.ceil(lmax))
    titles = ("Velocity", "Length vs velocity", "Filament length")
    for xi, t in zip(x, titles):
        pg.text(xi - 0.18, pg.y, t, fontsize=fx.fs(9.5), fontweight="bold",
                color=fx.color("report_ink"))

    # 1. velocity histogram (moving windows)
    a = pg.ax(x[0], y, w, h)
    if len(vel):
        a.hist(vel, bins=np.linspace(0, vmax, 26), color=moving_c, edgecolor=edge, linewidth=0.5)
        mv = r["velocity_nm_s"].get("mean_moving")
        if mv:
            a.axvline(mv, color=fx.color("report_ink"), ls="--", lw=1.0)
            a.text(mv, 1.02, "mean", transform=a.get_xaxis_transform(), fontsize=fx.fs(7),
                   va="bottom", ha="center", color=fx.color("report_ink"))
    a.set_xlim(0, vmax)
    a.set_xlabel("Velocity (nm/s)")
    a.set_ylabel("Count")
    a.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    a.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    _style_small_axes(a, fx)

    # 2. length vs velocity (all 5-step windows grey, fastest per track blue)
    a = pg.ax(x[1], y, w, h)
    if len(lv):
        a.plot(lv[:, 0] * 1e-3, lv[:, 1], ".", ms=2.2, color=moving_c, alpha=0.6,
               label="all windows", rasterized=True)
    if len(mx):
        a.plot(mx[:, 0] * 1e-3, mx[:, 1], "^", ms=2.6, mec="none", mfc=maxvel_c, ls="none",
               label="fastest per track")
    a.set_xlim(0, lmax)
    a.set_ylim(0, vmax)
    a.set_xlabel("Length (µm)")
    a.set_ylabel("Velocity (nm/s)")
    a.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    a.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    if len(lv) or len(mx):
        a.legend(loc="lower right", fontsize=fx.fs(6.5), frameon=False, handletextpad=0.2,
                 borderaxespad=0.2, markerscale=1.5)
    _style_small_axes(a, fx)

    # 3. length histogram (moving filaments; stuck as an outline)
    a = pg.ax(x[2], y, w, h)
    bins = np.linspace(0, lmax, 25)
    if len(lens):
        a.hist(lens, bins=bins, color=moving_c, edgecolor=edge, linewidth=0.5, label="moving")
    if len(slens):
        a.hist(slens, bins=bins, histtype="step", color=fx.color("preview_stuck"), lw=1.2,
               label="stuck")
        a.legend(loc="upper right", fontsize=fx.fs(6.5), frameon=False)
    a.set_xlim(0, lmax)
    a.set_xlabel("Length (µm)")
    a.set_ylabel("Count")
    a.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    a.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    _style_small_axes(a, fx)
    pg.y = y + h + 0.62


def _quality(pg, r, slider="status"):
    """Quality scores as slider rows: name + value | light track + marker."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fx = pg.fx
    ink, muted = fx.color("report_ink"), fx.color("report_muted")
    y0 = pg.section("Quality scores")
    q = r["quality"]
    row_h = 0.235
    name_x = pg.m
    val_x = pg.m + 3.05                       # right edge of the value column
    bar_x0, bar_x1 = pg.m + 3.35, pg.W - pg.m
    bar_w = bar_x1 - bar_x0
    pg.text(bar_x0, y0 - 0.20, "poor", fontsize=fx.fs(7.5), color=muted)
    pg.text(bar_x1, y0 - 0.20, "good", fontsize=fx.fs(7.5), color=muted, ha="right")
    a = pg.ax(bar_x0, y0, bar_w, row_h * len(q))
    a.set_xlim(0, 1)
    a.set_ylim(len(q), 0)
    a.axis("off")
    zone = [(0.0, 0.35, fx.color("poor")), (0.35, 0.70, fx.color("marginal")),
            (0.70, 1.0, fx.color("good"))]
    for i, x in enumerate(q):
        yc = i + 0.5
        pg.text(name_x, y0 + i * row_h + 0.035, _LABELS.get(x["metric"], x["label"]),
                fontsize=fx.fs(10), color=ink)
        pg.text(val_x, y0 + i * row_h + 0.035, metric_text(x), fontsize=fx.fs(10),
                color=ink, ha="right", fontweight="bold")
        th = 0.30                                         # track height (rows)
        if slider == "zones":
            for z0, z1, c in zone:
                a.add_patch(Rectangle((z0, yc - th / 2), z1 - z0, th, facecolor=c, alpha=0.18,
                                      edgecolor="none"))
        else:
            a.add_patch(FancyBboxPatch((0, yc - th / 2), 1, th,
                                       boxstyle="round,pad=0,rounding_size=0.004",
                                       facecolor=fx.color("report_track"), edgecolor="none",
                                       mutation_aspect=40))
            for z in (0.35, 0.70):                       # faint status boundaries
                a.plot([z, z], [yc - th / 2, yc + th / 2], "-", color="white", lw=1.2)
        col = fx.color(x["status"]) if slider != "zones" else ink
        mw = 0.016
        s = float(np.clip(x["score"], 0, 1))
        a.add_patch(Rectangle((s - mw / 2 if 0 < s < 1 else (0 if s <= 0 else 1 - mw),
                               yc - 0.32), mw, 0.64, facecolor=col, edgecolor="white", lw=0.6,
                              zorder=3))
    pg.y = y0 + row_h * len(q) + 0.12
    # suggested fix
    bad = [x for x in q if x["status"] != "good"]
    if bad:
        w = bad[0]
        lines = [("Suggested fix", "%s – %s." % (_LABELS.get(w["metric"], w["label"]),
                                                     _cap(w["advice"])))]
        for x in bad[1:3]:
            lines.append(("Also", "%s – %s." % (_LABELS.get(x["metric"], x["label"]),
                                                     _cap(x["advice"]))))
    else:
        lines = [("No action needed", "every metric is in the good range.")]
    wrapped = []
    for head, body in lines:
        head_w = _text_width(pg, head + ":  ", fx.fs(10), bold=True)
        parts = _wrap(pg, body, pg.text_w - 0.35 - head_w, fx.fs(10))
        wrapped.append((head, parts[0]))
        wrapped += [("", p) for p in parts[1:]]
    lines = wrapped
    h = 0.10 + 0.24 * len(lines)
    box = pg.ax(pg.m, pg.y, pg.text_w, h)
    box.axis("off")
    from matplotlib.patches import FancyBboxPatch as FB
    c = fx.color(bad[0]["status"]) if bad else fx.color("good")
    box.add_patch(FB((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                     transform=box.transAxes, facecolor=c, alpha=0.10, edgecolor="none",
                     mutation_aspect=pg.text_w / h))
    box.add_patch(Rectangle((0, 0), 0.006, 1, transform=box.transAxes, facecolor=c,
                            edgecolor="none"))
    head_indent = 0.0
    for i, (head, body) in enumerate(lines):
        yy = pg.y + 0.07 + 0.24 * i
        if head:
            head_indent = _text_width(pg, head + ":  ", fx.fs(10), bold=True)
        if head:
            t = pg.text(pg.m + 0.15, yy, head + ":", fontsize=fx.fs(10), fontweight="bold",
                        color=ink)
            _after(pg, t, "  " + body, fontsize=fx.fs(10), color=ink)
        else:
            pg.text(pg.m + 0.15 + head_indent, yy, body, fontsize=fx.fs(10), color=ink)
    pg.y += h + 0.25


def _text_width(pg, s, size, bold=False):
    """Width of ``s`` in inches when drawn on this page."""
    t = pg.fig.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(renderer=pg.fig.canvas.get_renderer()).width / pg.fig.dpi
    t.remove()
    return w


def _wrap(pg, s, width_in, size):
    """Greedy word wrap of ``s`` to ``width_in`` inches (measured, not guessed)."""
    lines, cur = [], ""
    for word in s.split():
        cand = (cur + " " + word).strip()
        if cur and _text_width(pg, cand, size) > width_in:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur] if cur or not lines else lines


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


_OVERLAY_EXTRA = 0.30 + 0.72          # section title + legend under the image
_ROW_H = 0.20


def _examples_height(pg, n=6):
    size = (pg.text_w - (n - 1) * 0.12) / n
    return 0.30 + 0.22 + 0.18 + size + 0.40


def _table_height(n_rows):
    return 0.30 + n_rows * _ROW_H + 0.25


def _overlay(pg, r, P, width):
    """Middle frame with the analysed tracks and the frame's filaments."""
    fx = pg.fx
    y0 = pg.section("Tracks on the middle frame")
    img = np.asarray(P["img"], float)
    hgt = width * img.shape[0] / img.shape[1]
    x0 = (pg.W - width) / 2.0
    a = pg.ax(x0, y0, width, hgt)
    a.imshow(img, cmap="gray", vmin=0, vmax=max(float(np.percentile(img, 99.8)), 1e-6),
             interpolation="nearest")
    mov_c, st_c = fx.color("preview_moving"), fx.color("preview_stuck")
    for xy, stuck in P.get("tracks", []):
        if len(xy) > 1:
            a.plot(xy[:, 0], xy[:, 1], "-", lw=0.6, alpha=0.55, color=st_c if stuck else mov_c)
    for e in P.get("moving", []):
        c = np.asarray(e[0])
        a.plot(c[:, 1], c[:, 0], "-", color=mov_c, lw=0.9)
    for e in P.get("static", []):
        c = np.asarray(e[0])
        a.plot(c[:, 1], c[:, 0], "-", color=st_c, lw=1.0)
    for key in ("floppy", "pinned"):
        for track in P.get(key, []):
            for c in track:
                a.plot(c[:, 1], c[:, 0], "-", color=fx.color(key), lw=0.6, alpha=0.75)
    dx = float(P.get("dx_nm") or r["pixel_size_nm"] * r["bin"])
    bar_um = 10.0
    bar_px = bar_um * 1000.0 / dx
    H, W = img.shape
    a.plot([0.04 * W, 0.04 * W + bar_px], [0.95 * H, 0.95 * H], "-", color="white", lw=3,
           solid_capstyle="butt")
    a.text(0.04 * W + bar_px / 2, 0.95 * H - 0.015 * H, "%g µm" % bar_um, color="white",
           ha="center", va="bottom", fontsize=fx.fs(8.5))
    a.set_xlim(-0.5, W - 0.5)
    a.set_ylim(H - 0.5, -0.5)
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    # legend under the image
    items = [("moving", mov_c), ("stuck", st_c), ("floppy track", fx.color("floppy")),
             ("pinned track", fx.color("pinned"))]
    ly = y0 + hgt + 0.08
    lx = x0
    for lab, c in items:
        la = pg.ax(lx, ly + 0.05, 0.22, 0.04)
        la.axis("off")
        la.plot([0, 1], [0.5, 0.5], "-", color=c, lw=2.2, transform=la.transAxes, clip_on=False)
        t = pg.text(lx + 0.28, ly, lab, fontsize=fx.fs(8.5), color=fx.color("report_ink"))
        bb = t.get_window_extent(renderer=pg.fig.canvas.get_renderer())
        lx = bb.x1 / pg.fig.dpi + 0.25
    pg.text(x0, ly + 0.24, "Contours: filaments in the middle frame.  Thin lines: their "
            "tracks over the %d frames." % r["n_frames"],
            fontsize=fx.fs(8), color=fx.color("report_muted"))
    pg.y = ly + 0.62


def _examples(pg, r, P, n=6):
    """Pinned / floppy example tracks, contours coloured by time."""
    fx = pg.fx
    y0 = pg.section("Attachment: floppy and pinned tracks")
    a_ = r["attachment"]
    m = r["metrics"]
    pg.text(pg.m, y0 - 0.02,
            "Pinned: %d of %d tracks (%s %%)    Floppy: %d of %d moving tracks (%s %%)    "
            "contours early → late" % (
                a_["pinned_tracks"], a_["tracks_for_pinned"], _num(m.get("pinned_pct"), 1),
                a_["floppy_tracks"], a_["tracks_for_floppy"], _num(m.get("floppy_pct"), 1)),
            fontsize=fx.fs(8.5), color=fx.color("report_muted"))
    pinned = P.get("pinned", [])
    seen = {id(t) for t in pinned}
    ex = ([("pinned", t) for t in pinned]
          + [("floppy", t) for t in P.get("floppy", []) if id(t) not in seen])[:n]
    y = y0 + 0.22
    size = (pg.text_w - (n - 1) * 0.12) / n
    if not ex:
        pg.text(pg.m, y, "No floppy or pinned tracks found.", fontsize=fx.fs(9.5),
                color=fx.color("report_ink"))
        pg.y = y + 0.45
        return
    img = np.asarray(P["img"], float)
    vmax = max(float(np.percentile(img, 99.8)), 1e-6)
    cmap = fx.scheme(fx.opt.get("time_colormap", "viridis"))["cmap"]
    for i, (kind, track) in enumerate(ex):
        a = pg.ax(pg.m + i * (size + 0.12), y + 0.18, size, size)
        allc = np.vstack(track)
        r0, c0 = allc.min(0).astype(int) - 6
        r1, c1 = allc.max(0).astype(int) + 7
        side = max(r1 - r0, c1 - c0)                           # square crop
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        r0, c0 = max(rc - side // 2, 0), max(cc - side // 2, 0)
        r1, c1 = min(r0 + side, img.shape[0]), min(c0 + side, img.shape[1])
        a.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        cols = cmap(np.linspace(0, 1, len(track)))
        for c, col in zip(track, cols):
            a.plot(c[:, 1] - c0, c[:, 0] - r0, "-", color=col, lw=1.0)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(fx.color(kind))
            sp.set_linewidth(1.5)
        pg.text(pg.m + i * (size + 0.12), y, kind, fontsize=fx.fs(8.5), fontweight="bold",
                color=fx.color(kind))
    pg.y = y + 0.18 + size + 0.40


def _tracking_rows(r):
    tr = r.get("tracking") or {}
    v = r["velocity_nm_s"]
    a_ = r["attachment"]
    secs = r.get("seconds", {})
    return [
        ("Frames analysed", "%d (frames %d\u2013%d)" % (r["n_frames"], min(r["frames"]),
                                                       max(r["frames"]))),
        ("Frame interval", "%s s (%s)" % (_num(tr.get("frame_interval_s"), 2, False),
                                          r.get("timing_source", "-"))),
        ("Binning / pixel size", "%d\u00d7%d / %s nm" % (r["bin"], r["bin"],
                                                          _num(tr.get("pixel_nm"), 1, False))),
        ("Detector", r["detector"]),
        ("Filaments per frame (moving)", _num(r["moving_per_frame"], 0)),
        ("Static filaments", _num(r["static_filaments"], 0)),
        ("Frame-to-frame links", _num(tr.get("links"), 0)),
        ("Tracks", _num(tr.get("tracks"), 0)),
        ("Tracks analysed (\u2265 %s links)" % tr.get("min_track_links", "-"),
         "%s  (moving %s, stuck %s)" % (_num(tr.get("tracks_analysed"), 0),
                                       _num(tr.get("tracks_moving"), 0),
                                       _num(tr.get("tracks_stuck"), 0))),
        ("Track length (links)", "median %s, max %s" % (_num(tr.get("median_track_links"), 0),
                                                       _num(tr.get("max_track_links"), 0))),
        ("Velocity windows (5 steps)", _num(r.get("tracked_points"), 0)),
        ("Velocity: mean / median / top 5 %", "%s / %s / %s nm/s" % (
            _num(v.get("mean_moving"), 0), _num(v.get("median_moving"), 0), _num(v.get("top5"), 0))),
        ("Velocity variation within tracks (CV)", _num(r.get("velocity_cv_median"), 2)),
        ("Floppy / pinned tracks", "%d of %d / %d of %d" % (
            a_["floppy_tracks"], a_["tracks_for_floppy"], a_["pinned_tracks"],
            a_["tracks_for_pinned"])),
        ("Analysis time", "%s s (detection %s s, linking %s s)" % (
            _num(secs.get("total"), 1), _num(secs.get("detect"), 1), _num(secs.get("link"), 1))),
    ]


def _threshold_rows(r):
    """Every quality metric with its value and the good / poor anchors."""
    from ..pipelines.preview import QUALITY_SPECS
    rows = []
    for q in sorted(r["quality"], key=lambda x: list(QUALITY_SPECS).index(x["metric"])):
        _, unit, good, poor, _ = QUALITY_SPECS[q["metric"]]
        nd = _DECIMALS.get(q["metric"], 2)
        better = "\u2265" if good > poor else "\u2264"
        worse = "\u2264" if good > poor else "\u2265"
        rows.append((_LABELS.get(q["metric"], q["label"]), metric_text(q),
                     "%s %s" % (better, _num(good, nd, False)),
                     "%s %s" % (worse, _num(poor, nd, False)), q["status"]))
    return rows


def _table(pg, title, rows, header=None, cols=(0.0, 3.0)):
    """Banded table; ``rows`` are tuples of strings (a 5th item = status colour)."""
    from matplotlib.patches import Rectangle
    fx = pg.fx
    ink, muted = fx.color("report_ink"), fx.color("report_muted")
    y0 = pg.section(title)
    if header:
        for c, h in zip(cols, header):
            pg.text(pg.m + 0.08 + c, y0, h, fontsize=fx.fs(8.5), fontweight="bold", color=muted)
        y0 += _ROW_H
    for i, row in enumerate(rows):
        yy = y0 + i * _ROW_H
        if i % 2 == 0:
            band = pg.ax(pg.m, yy - 0.035, pg.text_w, _ROW_H)
            band.axis("off")
            band.add_patch(Rectangle((0, 0), 1, 1, transform=band.transAxes,
                                     facecolor=fx.color("report_track"), alpha=0.55,
                                     edgecolor="none"))
        cells = row[:len(cols)]
        for j, (c, val) in enumerate(zip(cols, cells)):
            color = muted if j == 0 and not header else ink
            if j == 1 and len(row) > len(cols):                 # value coloured by status
                color = fx.color(row[-1]) if row[-1] != "good" else ink
            pg.text(pg.m + 0.08 + c, yy, val, fontsize=fx.fs(9.5), color=color,
                    fontweight="bold" if (j == 1 and len(row) > len(cols)
                                          and row[-1] != "good") else "normal")
    pg.y = y0 + _ROW_H * len(rows) + 0.25


def _footer(pg, i, n):
    fx = pg.fx
    pg.text(pg.m, pg.H - 0.42, "FASTrack fast-preview", fontsize=fx.fs(7.5),
            color=fx.color("report_muted"))
    pg.text(pg.W - pg.m, pg.H - 0.42, "page %d of %d" % (i, n), fontsize=fx.fs(7.5),
            color=fx.color("report_muted"), ha="right")
