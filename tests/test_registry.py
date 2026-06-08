"""Tests for the strategy registry (no heavy deps required)."""
from fastrack.registry import Registry


def test_register_create_available():
    reg = Registry("widget")

    @reg.register("a")
    class A:
        def __init__(self, v=1):
            self.v = v

    reg.register("b", lambda **kw: kw)

    assert "a" in reg and "b" in reg
    assert reg.available() == ["a", "b"]
    assert reg.create("a", v=5).v == 5
    assert reg.create("b", x=2) == {"x": 2}


def test_case_insensitive():
    reg = Registry("widget")
    reg.register("Greedy", dict)
    assert "greedy" in reg
    assert reg.create("GREEDY") == {}


def test_duplicate_and_unknown_raise():
    reg = Registry("widget")
    reg.register("a", dict)
    try:
        reg.register("a", list)
        assert False, "expected ValueError on duplicate"
    except ValueError:
        pass
    try:
        reg.create("missing")
        assert False, "expected KeyError on unknown"
    except KeyError:
        pass


def test_builtin_registries_populated():
    from fastrack.core.detection import DETECTORS
    from fastrack.core.tracking import LINKERS
    from fastrack.io.movie import MOVIE_WRITERS
    from fastrack.io.stores import STORES

    assert "entropy" in DETECTORS
    assert "greedy" in LINKERS
    assert "ffmpeg_h264" in MOVIE_WRITERS
    assert "npy" in STORES
