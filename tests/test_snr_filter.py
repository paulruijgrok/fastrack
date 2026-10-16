"""Signal-to-noise filter for the ridge detector, and the noise estimate behind it.

Covers: the preprocessing model's pixel-noise estimate (accuracy, persistence,
backward-compatible loading, refit of old models), the frame-source /
``Frame`` plumbing, the per-contour SNR and the ``min_snr`` post-filter (with
stub contours, no optional dependency), settings/CLI wiring, the detection
cache key, and an end-to-end ``fast --detector ridge --preprocess`` run on a
synthetic low-count movie (only when ``ridge-detector`` is installed).
"""
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
tifffile = pytest.importorskip("tifffile")

from fastrack import preprocess as pp  # noqa: E402
from fastrack.core.input import PreprocessedSource, open_source  # noqa: E402

RECIPE = {"background": "median", "model_frames": 0, "flat_sigma": 8.0,
          "denoise_sigma": 0.8, "clip_hi_pct": 99.9}


class StubLine:
    """Minimal stand-in for an upstream ridge ``Line`` (row/col centerline)."""

    def __init__(self, rows, cols):
        self.row = np.asarray(rows, dtype=float)
        self.col = np.asarray(cols, dtype=float)
        self.num = len(self.row)
        self.width_l = self.width_r = None


def _noisy_stack(T=30, H=160, W=160, sigma=4.0, seed=0, line=60.0):
    """~300 counts background + Gaussian noise (sigma) + one bright moving line."""
    rng = np.random.default_rng(seed)
    stack = 300.0 + rng.normal(0.0, sigma, (T, H, W))
    for t in range(T):
        c = (2 * t) % 40
        stack[t, 31:34, c:c + 20] += line
    return np.clip(stack, 0, 65535).astype(np.uint16)


def _write_mm_movie(root, stack, tail="561 nm"):
    d = os.path.join(root, "Pos0")
    os.makedirs(d)
    for i, fr in enumerate(stack):
        tifffile.imwrite(os.path.join(d, "img_%09d_%s_000.tif" % (i, tail)), fr)
    return d


# --------------------------------------------------------------------------- #
# noise estimate in the preprocessing model
# --------------------------------------------------------------------------- #
def test_estimate_noise_sd_recovers_known_sigma():
    """Without denoising, the estimate matches the injected noise; the moving
    line, the background level and the clipping of the correction don't bias it."""
    stack = _noisy_stack(sigma=4.0).astype(np.float32)
    bg, gain = pp.estimate_background_gain(stack, background="median", flat_sigma=0)
    sd = pp.estimate_noise_sd(stack, bg, gain, denoise_sigma=0)
    # frame-pair differences: static background cancels; the line is ~2% of pixels
    assert sd == pytest.approx(4.0, rel=0.1)


def test_estimate_noise_sd_ignores_static_structure_and_crowding():
    """A strong static pattern and many filaments don't inflate the estimate."""
    rng = np.random.default_rng(5)
    T, H, W = 10, 160, 160
    yy, xx = np.mgrid[0:H, 0:W]
    static = 200.0 * np.sin(xx / 7.0) * np.cos(yy / 11.0)      # junk / gradient
    stack = 300.0 + static + rng.normal(0.0, 3.0, (T, H, W))
    for t in range(T):                                          # ~8% of pixels lit
        for r in range(8, H, 20):
            c = (9 * t + r) % (W - 40)
            stack[t, r - 1:r + 2, c:c + 40] += 80.0
    stack = stack.astype(np.float32)
    bg, gain = pp.estimate_background_gain(stack, flat_sigma=0)
    assert pp.estimate_noise_sd(stack, bg, gain, denoise_sigma=0) == pytest.approx(3.0, rel=0.1)


def test_estimate_noise_sd_scales_with_noise():
    s2 = _noisy_stack(sigma=2.0, seed=1).astype(np.float32)
    s8 = _noisy_stack(sigma=8.0, seed=1).astype(np.float32)
    est = []
    for s in (s2, s8):
        bg, gain = pp.estimate_background_gain(s, flat_sigma=0)
        est.append(pp.estimate_noise_sd(s, bg, gain, denoise_sigma=0.8))
    assert est[1] / est[0] == pytest.approx(4.0, rel=0.08)


def test_fit_model_sets_noise_and_output_units():
    stack = _noisy_stack()
    m = pp.fit_model(stack, **RECIPE)
    assert m.noise_sd is not None and m.noise_sd > 0
    assert m.output_noise_sd(8) == pytest.approx(m.noise_sd / m.hi * 255.0)
    assert m.output_noise_sd(16) == pytest.approx(m.noise_sd / m.hi * 65535.0)


def test_model_noise_roundtrip_and_old_format(tmp_path):
    m = pp.fit_model(_noisy_stack(), **RECIPE)
    p = str(tmp_path / "m.npz")
    m.save(p)
    assert pp.PreprocessModel.load(p).noise_sd == pytest.approx(m.noise_sd)

    # a model written before noise_sd existed loads with noise_sd=None
    old = str(tmp_path / "old.npz")
    np.savez(old, background=m.background, gain=m.gain, hi=np.float64(m.hi),
             denoise_sigma=np.float64(m.denoise_sigma))
    assert pp.PreprocessModel.load(old).noise_sd is None


def test_preprocessed_source_refits_model_without_noise(tmp_path):
    """An old cached model (no noise_sd) is refitted once and overwritten; the
    pixels it produces are unchanged."""
    mm = _write_mm_movie(str(tmp_path / "movie"), _noisy_stack())
    inner = {"kind": "mm_dir", "directory": mm}
    model_path = str(tmp_path / "model.npz")
    src = PreprocessedSource(inner, RECIPE, model_path=model_path)
    fresh = src.ensure_model()
    np.savez(model_path, background=fresh.background, gain=fresh.gain,
             hi=np.float64(fresh.hi), denoise_sigma=np.float64(fresh.denoise_sigma))

    src2 = PreprocessedSource(inner, RECIPE, model_path=model_path)
    m2 = src2.ensure_model()
    assert m2.noise_sd == pytest.approx(fresh.noise_sd)
    assert pp.PreprocessModel.load(model_path).noise_sd == pytest.approx(fresh.noise_sd)
    np.testing.assert_array_equal(src2.read(0), src.read(0))


def test_source_noise_in_pipeline_units(tmp_path):
    mm = _write_mm_movie(str(tmp_path / "movie"), _noisy_stack())
    inner = {"kind": "mm_dir", "directory": mm}
    src = PreprocessedSource(inner, RECIPE)
    assert src.noise_sd() == pytest.approx(src.model.output_noise_sd(8) * 257.0)
    assert open_source(inner).noise_sd() is None          # raw sources: unknown


# --------------------------------------------------------------------------- #
# contour SNR and the min_snr post-filter (no optional dependency needed)
# --------------------------------------------------------------------------- #
def _line_image(seed=0, noise=3.0, line=40.0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.0, noise, (60, 60)), 0, None)   # clipped bg ~ 0
    img[29:32, 10:50] += line
    return img


def test_contour_snr_separates_line_from_noise():
    from fastrack.core.detection.ridge import contour_snr
    img = _line_image()
    on_line = StubLine([30] * 30, range(15, 45))
    in_noise = StubLine([10] * 30, range(15, 45))
    snr = contour_snr([on_line, in_noise], img, noise_sd=3.0)
    assert snr[0] > 8 and snr[1] < 3


def test_estimate_frame_noise_on_unclipped_image():
    from fastrack.core.detection.ridge import estimate_frame_noise
    rng = np.random.default_rng(3)
    img = 1000.0 + rng.normal(0.0, 5.0, (128, 128))
    assert estimate_frame_noise(img) == pytest.approx(5.0, rel=0.2)


class _FakeUpstream:
    """Upstream detector stand-in returning fixed contours."""
    contours = []

    def __init__(self, **kw):
        self.kw = kw

    def detect_lines(self, img):
        pass


def _adapter(min_snr, contours):
    from fastrack.core.detection.ridge import _RidgeAdapterBase

    class Fake(_FakeUpstream):
        pass
    Fake.contours = contours

    class Adapter(_RidgeAdapterBase):
        @staticmethod
        def _import_detector_cls():
            return Fake
    return Adapter(min_snr=min_snr)


def test_min_snr_filter_drops_noise_contours_and_keeps_params_clean():
    from fastrack.core.frame import Frame
    contours = [StubLine([30] * 30, range(15, 45)), StubLine([10] * 30, range(15, 45))]

    def run(min_snr, noise_sd):
        fr = Frame()
        fr.img = _line_image().astype(np.float64)
        fr.width, fr.height = fr.img.shape
        fr.noise_sd = noise_sd
        det = _adapter(min_snr, contours)
        assert "min_snr" not in det.params          # never passed upstream
        det.detect(fr)
        return len(fr.filXYs)

    assert run(0.0, 3.0) == 2                       # off: unchanged behaviour
    assert run(5.0, 3.0) == 1                       # noise contour dropped
    assert run(5.0, None) == 1                      # falls back to an estimate
    assert run(1e6, 3.0) == 0


# --------------------------------------------------------------------------- #
# settings, CLI and cache key
# --------------------------------------------------------------------------- #
def test_min_snr_setting_and_cli(monkeypatch, tmp_path):
    from fastrack.cli import fast_main
    from fastrack.config import Settings
    from fastrack.pipelines import gliding

    cfg = tmp_path / "c.toml"
    cfg.write_text('[analysis]\ndetection_algorithm = "ridge"\n[ridge]\nmin_snr = 5.0\n')
    kw = Settings.from_toml(str(cfg)).to_run_kwargs()
    assert kw["detection_params"]["min_snr"] == 5.0
    assert Settings().ridge.min_snr == 0.0

    seen = {}
    monkeypatch.setattr(gliding, "run", lambda **k: seen.update(k))
    fast_main(["-d", str(tmp_path), "--detector", "ridge", "--ridge-min-snr", "4"])
    assert seen["detection_params"]["min_snr"] == 4.0


def test_detection_tags_key_on_detector_params():
    from fastrack.pipelines.gliding import detection_tags
    assert detection_tags() == ("", "")                     # entropy unchanged
    a = detection_tags("ridge", {"low_contrast": 18, "min_snr": 0.0})
    b = detection_tags("ridge", {"low_contrast": 60, "min_snr": 0.0})
    c = detection_tags("ridge", {"low_contrast": 18, "min_snr": 5.0})
    assert a[0] == b[0] == c[0] == "__det_ridge"            # same output tree
    assert len({a[1], b[1], c[1]}) == 3                     # distinct caches
    assert a[1].startswith("_ridge_d")
    d = detection_tags("ridge", {"low_contrast": 18, "min_snr": 0.0}, RECIPE)
    assert d[0] == "__det_ridge__pp_" + pp.params_hash(RECIPE)
    assert d[1] == a[1] + "_pp" + pp.params_hash(RECIPE)


# --------------------------------------------------------------------------- #
# end to end with the real ridge detector
# --------------------------------------------------------------------------- #
def test_gliding_ridge_min_snr_end_to_end(tmp_path, monkeypatch):
    """On a noisy low-count movie, min_snr removes the noise ridges while the
    real moving lines survive; the frame noise reaches the detector from the
    preprocessing model."""
    pytest.importorskip("skimage")
    pytest.importorskip("ridge_detector")
    from fastrack.pipelines import gliding
    from fastrack.io.stores import STORES

    rng = np.random.default_rng(7)
    # lines move 15 px/frame, so no pixel is covered in more than 2 of 8
    # frames and the temporal-median background does not absorb them
    T, H, W = 8, 96, 160
    stack = 300.0 + rng.normal(0, 6.0, (T, H, W))
    for t in range(T):
        for r in (20, 48, 76):
            stack[t, r - 1:r + 2, 5 + 15 * t:35 + 15 * t] += 60.0
    stack = np.clip(stack, 0, 65535).astype(np.uint16)
    data = tmp_path / "data"
    mm = _write_mm_movie(str(data / "movie_1"), stack)
    monkeypatch.chdir(tmp_path)

    def detections(min_snr):
        params = dict(line_widths=[3], low_contrast=10, high_contrast=30,
                      min_len=8, min_snr=min_snr)
        gliding.run(main_dir=str(data), force_analysis=True, detection_algorithm="ridge",
                    detection_params=params, cache_layout="per-movie", nprocs=1,
                    min_path_length=2, preprocess=RECIPE)
        _, tag = gliding.detection_tags("ridge", params, RECIPE)
        store = STORES.create("per-movie")
        return [len(store.read(mm, tag, f)) for f in range(T)]

    loose = detections(0.0)
    strict = detections(6.0)
    assert sum(strict) < sum(loose)                 # noise ridges removed
    assert all(n >= 3 for n in strict), strict      # the three lines survive
