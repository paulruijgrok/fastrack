"""In-pipeline preprocessing: shared module, frame-source wrapper, settings/CLI,
cache keying, and an end-to-end ``fast --preprocess`` run on a synthetic movie."""
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
tifffile = pytest.importorskip("tifffile")

from fastrack import preprocess as pp  # noqa: E402
from fastrack.core.input import PreprocessedSource, open_source, to_pipeline_image  # noqa: E402

RECIPE = {"background": "median", "model_frames": 0, "flat_sigma": 8.0,
          "denoise_sigma": 0.8, "clip_hi_pct": 99.9}


def _low_count_stack(T=8, H=48, W=64, seed=0):
    """Taka-like data: ~300 counts background, a strong top->bottom gradient,
    faint (+120) moving lines, all in the bottom of the 16-bit range."""
    rng = np.random.default_rng(seed)
    grad = np.linspace(1.4, 0.6, H)[:, None]
    stack = 300.0 + rng.random((T, H, W)) * 40.0 * grad
    for t in range(T):
        for r in (10, 24, 38):
            stack[t, r - 1:r + 2, 4 + 3 * t:40 + 3 * t] += 120.0 * grad[r, 0]
    return stack.astype(np.uint16)


def _write_mm_movie(root, stack, tail="561 nm"):
    """A MicroManager-style frame folder ``root/Pos0/img_000000NNN_<tail>_000.tif``."""
    d = os.path.join(root, "Pos0")
    os.makedirs(d)
    for i, fr in enumerate(stack):
        tifffile.imwrite(os.path.join(d, "img_%09d_%s_000.tif" % (i, tail)), fr)
    return d


# --------------------------------------------------------------------------- #
# shared maths
# --------------------------------------------------------------------------- #
def test_fit_model_all_frames_equals_preprocess_stack():
    stack = _low_count_stack()
    model = pp.fit_model(stack, **RECIPE)
    ref = pp.preprocess_stack(stack, background="median", flat_sigma=8.0,
                              denoise_sigma=0.8, clip_hi_pct=99.9)
    assert np.array_equal(model.apply_stack(stack), ref)
    assert ref.dtype == np.uint8 and ref.max() == 255


def test_model_frames_subsamples_evenly_and_stays_close():
    stack = _low_count_stack(T=20)
    assert pp.sample_indices(20, 5) == [0, 5, 10, 14, 19]
    assert pp.sample_indices(20, 0) == list(range(20))
    assert pp.sample_indices(20, 50) == list(range(20))
    full = pp.fit_model(stack, **RECIPE)
    sub = pp.fit_model(stack, **dict(RECIPE, model_frames=5))
    # A sampled background is a noisier estimate of the same thing, not a
    # different thing: gain fields agree closely, stretch ceilings within ~10%.
    assert np.allclose(full.gain, sub.gain, rtol=0.02)
    assert abs(full.hi - sub.hi) / full.hi < 0.10


def test_model_save_load_roundtrip(tmp_path):
    stack = _low_count_stack()
    model = pp.fit_model(stack, **RECIPE)
    path = str(tmp_path / "m.npz")
    model.save(path)
    back = pp.PreprocessModel.load(path)
    assert np.array_equal(back.apply_stack(stack), model.apply_stack(stack))
    # flat_sigma=0 -> gain None survives the round trip
    m2 = pp.fit_model(stack, **dict(RECIPE, flat_sigma=0))
    m2.save(path)
    assert pp.PreprocessModel.load(path).gain is None


def test_correction_equalises_gradient_and_lifts_contrast():
    stack = _low_count_stack()
    out = pp.preprocess_stack(stack, flat_sigma=8.0)
    # raw: the legacy read leaves ~2 grey levels; corrected: full 8-bit use
    raw8 = (stack[0] >> 8)
    assert len(np.unique(raw8)) <= 3
    assert len(np.unique(out[0])) > 50
    # top and bottom filament rows end up at comparable brightness
    top, bottom = out[4, 9:12].max(), out[4, 37:40].max()
    assert bottom > 0.7 * top


def test_params_hash_ignores_key_order_and_extras():
    a = pp.params_hash(RECIPE)
    b = pp.params_hash(dict(reversed(list(RECIPE.items()))))
    assert a == b
    assert pp.params_hash(dict(RECIPE, flat_sigma=9.0)) != a


# --------------------------------------------------------------------------- #
# frame-source wrapper
# --------------------------------------------------------------------------- #
def test_preprocessed_source_matches_script_route(tmp_path):
    """Reading through PreprocessedSource == running the standalone tool and
    reading its 8-bit output with the legacy read (uint8 * 257)."""
    stack = _low_count_stack()
    mm = _write_mm_movie(str(tmp_path / "raw"), stack)
    inner = {"kind": "mm_dir", "directory": mm, "header": "img_000000",
             "tail": "561 nm", "frame_rate": 1.0}
    model_path = str(tmp_path / "model.npz")
    src = PreprocessedSource(inner, RECIPE, model_path=model_path)
    assert src.count == len(stack) and src.frame_numbers() == list(range(len(stack)))
    got = np.stack([src.read(i) for i in range(len(stack))])
    script = pp.preprocess_stack(stack, background="median", flat_sigma=8.0,
                                 denoise_sigma=0.8, clip_hi_pct=99.9)
    assert np.array_equal(got, np.stack([to_pipeline_image(f) for f in script]))
    assert got.dtype == np.uint16
    # the model was persisted; a rebuilt source (as a pool worker does) loads it
    assert os.path.isfile(model_path)
    again = open_source(src.descriptor())
    assert isinstance(again, PreprocessedSource)
    assert again.model_path == model_path
    assert np.array_equal(again.read(3), src.read(3))


def test_read_raw_keeps_native_depth(tmp_path):
    stack = _low_count_stack()
    mm = _write_mm_movie(str(tmp_path / "raw"), stack)
    tifffile.imwrite(str(tmp_path / "stack.tif"), stack)
    for src in (open_source({"kind": "mm_dir", "directory": mm, "tail": "561 nm"}),
                open_source({"kind": "tiff_stack", "path": str(tmp_path / "stack.tif")})):
        raw = src.read_raw(2)
        assert raw.dtype == np.uint16 and np.array_equal(raw, stack[2])
        assert np.array_equal(src.read(2), to_pipeline_image(stack[2]))


def test_entropy_quality_gate_passes_after_preprocessing(tmp_path):
    """The gate's absolute-contrast threshold (>1000 on the 16-bit scale) is
    what rejects low-count data raw; the full-range stretch makes it pass."""
    from fastrack.core.frame import Frame

    stack = _low_count_stack(H=96, W=96)
    inner = {"kind": "tiff_stack", "path": str(tmp_path / "s.tif")}
    tifffile.imwrite(inner["path"], stack)

    def gate(img):
        f = Frame()
        f.img = img
        f.width, f.height = img.shape
        f.fast_rank, f.morph_contrast = True, False
        return f.check_picture_quality()

    assert gate(open_source(inner).read(0)) == "bad"
    assert gate(PreprocessedSource(inner, RECIPE).read(0)) == "good"


# --------------------------------------------------------------------------- #
# settings / CLI
# --------------------------------------------------------------------------- #
def test_settings_default_off_and_recipe_when_enabled():
    from fastrack.config import Settings

    assert Settings().to_run_kwargs()["preprocess"] is None
    assert Settings().to_directional_kwargs()["preprocess"] is None
    s = Settings.from_sources({"preprocess": {"enabled": True, "flat_sigma": 30}})
    for kw in (s.to_run_kwargs(), s.to_directional_kwargs()):
        assert kw["preprocess"] == {"background": "median", "model_frames": 0,
                                    "flat_sigma": 30, "denoise_sigma": 0.8,
                                    "clip_hi_pct": 99.9}


def test_cli_flags_map_to_preprocess_settings(monkeypatch, tmp_path):
    from fastrack.cli import fast_main, fastplus_main
    from fastrack.pipelines import directional, gliding

    seen = {}
    monkeypatch.setattr(gliding, "run", lambda **kw: seen.setdefault("fast", kw))
    monkeypatch.setattr(directional, "run", lambda **kw: seen.setdefault("plus", kw))
    fast_main(["-d", str(tmp_path), "--preprocess", "--pp-flat-sigma", "25",
               "--pp-model-frames", "10"])
    fastplus_main(["-d", str(tmp_path), "--preprocess", "--pp-background", "mean"])
    assert seen["fast"]["preprocess"]["flat_sigma"] == 25.0
    assert seen["fast"]["preprocess"]["model_frames"] == 10
    assert seen["plus"]["preprocess"]["background"] == "mean"
    seen.clear()
    fast_main(["-d", str(tmp_path)])
    assert seen["fast"]["preprocess"] is None


# --------------------------------------------------------------------------- #
# end to end: fast --preprocess on a synthetic low-count movie
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cache_layout", ["per-movie", "per-frame"])
def test_gliding_run_with_preprocessing(tmp_path, monkeypatch, cache_layout):
    pytest.importorskip("skimage")
    from fastrack.pipelines import gliding

    stack = _low_count_stack(T=8, H=96, W=96)
    data = tmp_path / "data"
    mm = _write_mm_movie(str(data / "movie_1"), stack)
    monkeypatch.chdir(tmp_path)
    common = dict(main_dir=str(data), force_analysis=True, detection_algorithm="entropy",
                  cache_layout=cache_layout, nprocs=1, min_path_length=2)

    # raw: the entropy gate rejects the low-count frames -> no output tree with
    # detections; preprocessed: the run goes through and writes its own tree.
    gliding.run(**common)
    gliding.run(**common, preprocess=RECIPE)

    outs = sorted(os.listdir(tmp_path / "outputs"))
    pp_tag = "__pp_" + pp.params_hash(RECIPE)
    assert any(o.endswith(pp_tag) for o in outs), outs
    assert any(not o.endswith(pp_tag) for o in outs), outs
    # model persisted in the work dir (the movie folder for mm input), cache
    # tagged with the recipe so raw / corrected filXYs never collide
    assert os.path.isfile(os.path.join(mm, "preprocess_model_%s.npz" % pp.params_hash(RECIPE)))
    names = os.listdir(mm)
    assert any(n.startswith("filXYs_pp") for n in names), names
    assert not any(n.startswith("filXYs0") or n == "filXYs.npz" for n in names), names
