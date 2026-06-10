"""Tests for the layered settings (no heavy deps required)."""
from fastrack.config import Settings


def test_layering_later_wins():
    s = Settings.from_sources(
        {"hardware": {"pixel_size_nm": 64.5}},
        {"hardware": {"pixel_size_nm": 100.0}, "analysis": {"num_frames_ave": 7}},
    )
    assert s.hardware.pixel_size_nm == 100.0
    assert s.analysis.num_frames_ave == 7
    # untouched fields keep their defaults
    assert s.analysis.fast_rank is True


def test_flat_overrides_and_none_ignored():
    s = Settings().with_overrides(fast_rank=False, make_movie=True, nprocs=None)
    assert s.analysis.fast_rank is False
    assert s.runtime.make_movie is True
    # None must not clobber the default
    assert s.runtime.nprocs is None


def test_to_run_kwargs_maps_sections():
    k = Settings().to_run_kwargs()
    assert k["pixel_size"] == 80.65
    assert k["max_velocity"] == 2016.25
    assert k["detection_algorithm"] == "entropy"
    assert k["tracking_algorithm"] == "greedy"


def test_unknown_section_and_key_raise():
    for bad in ({"bogus": {}}, {"analysis": {"nope": 1}}):
        try:
            Settings.from_sources(bad)
            assert False, "expected KeyError for %r" % bad
        except KeyError:
            pass


def test_overlay_defaults_and_to_run_kwargs():
    k = Settings().to_run_kwargs()
    assert k["overlay_fps"] == 10.0
    assert k["overlay_frame_label"] is True
    assert k["overlay_time_label"] is True
    assert k["overlay_frame_interval_s"] == 1.0
    assert k["overlay_font_scale"] == 0.6


def test_overlay_overrides_flat():
    s = Settings().with_overrides(fps=24.0, frame_label=False, time_label=False)
    assert s.overlay.fps == 24.0
    assert s.overlay.frame_label is False
    assert s.overlay.time_label is False


def test_cli_field_map_targets_valid_fields():
    # Every CLI dest must map to a real Settings field, and applying them all
    # must not raise (catches typos / renamed fields in the override layer).
    from fastrack.cli import _CLI_TO_FIELD
    sample = {}
    for field_name in _CLI_TO_FIELD.values():
        sample[field_name] = False if field_name in ("dark_line", "frame_label") else 1
    s = Settings().with_overrides(**sample)  # raises KeyError on any bad field
    assert s is not None


def test_cli_overrides_layer_on_config():
    # Simulate: base config sets fps=5; CLI passes fps=30 -> CLI wins.
    base = Settings.from_sources({"overlay": {"fps": 5.0, "frame_label": False}})
    cli = {"fps": 30.0}  # only --overlay-fps explicitly passed
    merged = base.with_overrides(**cli)
    assert merged.overlay.fps == 30.0          # CLI override applied
    assert merged.overlay.frame_label is False  # config value preserved


def test_from_toml_roundtrip(tmp_path):
    import importlib.util
    if importlib.util.find_spec("tomllib") is None:
        import pytest
        pytest.skip("tomllib requires Python 3.11+")
    p = tmp_path / "cfg.toml"
    p.write_text(
        "[overlay]\nfps = 12.0\ntime_label = false\n"
        "[analysis]\ndetection_algorithm = 'ridge'\n"
    )
    s = Settings.from_toml(str(p))
    assert s.overlay.fps == 12.0
    assert s.overlay.time_label is False
    assert s.analysis.detection_algorithm == "ridge"
    # ridge detector params get emitted when detector == ridge
    assert s.to_run_kwargs()["detection_params"]["line_widths"] == [3]
