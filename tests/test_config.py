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
