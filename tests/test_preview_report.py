"""The fast-preview PDF/PNG report (viz/preview_report.py) on a synthetic result."""
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from fastrack.pipelines.preview import score_quality, verdict  # noqa: E402
from fastrack.viz.preview_report import PAGE_SIZES, metric_text  # noqa: E402
from fastrack.viz.theme import load_theme  # noqa: E402


def _result(floppy=48.0, n_img=200, seed=0):
    rng = np.random.default_rng(seed)
    metrics = dict(snr=20.0, focus_width_nm=270.0, flatness=0.9, saturation_pct=0.0,
                   background_to_signal=3.0, density=0.5, stuck_pct=24.0, floppy_pct=floppy,
                   pinned_pct=3.0, bleach_pct_per_frame=1.1, step_over_length=0.2)
    q = score_quality(metrics)
    track = [np.column_stack([np.arange(10) + 50 + k, np.full(10, 60 + k)]) for k in range(5)]
    trails = [(np.column_stack([np.linspace(10, 80, 8), np.linspace(20, 30, 8)]), False),
              (np.column_stack([np.full(6, 100.0), np.full(6, 100.0)]), True)]
    lv = np.column_stack([rng.uniform(800, 4000, 300), rng.normal(1500, 300, 300)])
    return dict(
        n_frames=20, frames=list(range(20)), bin=2, detector="ridge-fast",
        timing_source="metadata", pixel_size_nm=80.65, moving_per_frame=35.0,
        static_filaments=11, stuck_pct=24.0, tracked_points=300,
        velocity_nm_s=dict(mean_moving=1500.0, median_moving=1520.0, top5=2200.0),
        length_um=dict(median=2.9, mean=3.0, p10=1.0, p90=5.1, static_median=1.2),
        metrics=metrics, quality=q, verdict=verdict(q), velocity_cv_median=0.4,
        attachment=dict(floppy_tracks=3, pinned_tracks=1, tracks_for_floppy=10,
                        tracks_for_pinned=12),
        tracking=dict(frame_interval_s=0.5, links=400, tracks=120, tracks_analysed=33,
                      tracks_moving=27, tracks_stuck=6, median_track_links=2.0,
                      max_track_links=17, min_track_links=5, pixel_nm=161.3),
        seconds=dict(total=5.7, detect=3.0, link=0.4),
        _plot=dict(img=rng.random((n_img, n_img)), moving=[(track[0],)], static=[(track[1],)],
                   lengths_um=rng.uniform(0.5, 6, 400), static_lengths_um=rng.uniform(0.5, 2, 20),
                   velocities=lv[:, 1], len_vel=lv, max_len_vel=lv[::10], tracks=trails,
                   dx_nm=161.3, floppy=[track], pinned=[track[::-1]]),
    )


def _render(tmp_path, r, **opts):
    from fastrack.pipelines.preview import plot_dashboard
    cli = ["preview_report.%s=%s" % (k, v) for k, v in opts.items()]
    return plot_dashboard("slide_01/movie", r, str(tmp_path / "movie_preview"),
                          theme={"theme": "default", "cli": cli})


def test_report_writes_one_pdf_and_a_png_per_page(tmp_path):
    files = _render(tmp_path, _result())
    names = sorted(os.path.basename(f) for f in files)
    assert names[0] == "movie_preview.pdf"
    pngs = [n for n in names if n.endswith(".png")]
    assert pngs[0] == "movie_preview_p1.png" and len(pngs) >= 2
    from PIL import Image
    w, h = Image.open(tmp_path / "movie_preview_p1.png").size
    assert h > w                                                     # portrait
    assert abs(w / h - 8.5 / 11.0) < 0.01                            # US letter by default
    with open(tmp_path / "movie_preview.pdf", "rb") as f:
        assert f.read(5) == b"%PDF-"
    import json
    man = json.load(open(tmp_path / "figures.json"))
    assert man[0]["figure"] == "preview_report" and "movie_preview.pdf" in man[0]["files"]


@pytest.mark.parametrize("opts", [dict(page_size='"a4"'), dict(slider='"zones"'),
                                  dict(overlay_page='"1"'), dict(overlay_page='"2"')])
def test_report_options_render(tmp_path, opts):
    files = _render(tmp_path, _result(), **opts)
    assert any(f.endswith(".pdf") for f in files)
    if "page_size" in opts:
        from PIL import Image
        w, h = Image.open(tmp_path / "movie_preview_p1.png").size
        assert abs(w / h - PAGE_SIZES["a4"][0] / PAGE_SIZES["a4"][1]) < 0.01


def test_overlay_goes_to_page_two_when_it_does_not_fit_page_one(tmp_path):
    """12 quality rows leave ~1.5 in on page 1: the overlay can't get 2/3 of the
    page width there, so 'auto' puts it on page 2 (3 pages); a small image
    with a short page 1 goes on page 1 (2 pages)."""
    big = [f for f in _render(tmp_path, _result()) if f.endswith(".png")]
    assert len(big) == 3
    small = tmp_path / "small"
    small.mkdir()
    r = _result()
    r["quality"] = [q for q in r["quality"] if q["status"] == "good"][:2]   # short page 1
    lo = [f for f in _render(small, r, overlay_min_width=0.15) if f.endswith(".png")]
    assert len(lo) == 2


def test_report_all_good_and_no_examples(tmp_path):
    r = _result(floppy=0.0)
    r["metrics"].update(density=3.0, stuck_pct=5.0, pinned_pct=0.0)
    r["quality"] = score_quality(r["metrics"])
    r["verdict"] = verdict(r["quality"])
    r["_plot"].update(floppy=[], pinned=[], static=[], static_lengths_um=np.zeros(0))
    assert r["verdict"] == "GOOD"
    assert _render(tmp_path, r)


def test_report_can_be_switched_off_and_old_dashboard_on(tmp_path):
    from fastrack.pipelines.preview import plot_dashboard
    files = plot_dashboard("m", _result(), str(tmp_path / "m_preview"),
                           theme={"theme": "default", "skip": ["preview_report"],
                                  "cli": ["preview_dashboard.enabled=true"]})
    assert [os.path.basename(f) for f in files] == ["m_dashboard.png"]


def test_metric_text_units():
    assert metric_text(dict(metric="floppy_pct", value=48.123, unit="%")) == "48.1 %"
    assert metric_text(dict(metric="focus_width_nm", value=264.4, unit="nm")) == "264 nm"
    assert metric_text(dict(metric="density_low", value=0.5432, unit="")).endswith("µm²")
    assert metric_text(dict(metric="snr", value=None, unit="")) == "–"
    assert load_theme("default").formats("preview_report") == ["pdf", "png"]
