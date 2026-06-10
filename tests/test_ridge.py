"""Tests for the optional ridge-detection filament detector.

The registry and the Line->filXYs mapping are tested without the optional
``ridge-detector`` dependency (they don't need it).  The end-to-end detection
test ``importorskip``s the dependency and the image stack, so it runs only when
``pip install 'fastrack[ridge]'`` (and scipy/scikit-image/opencv) are present.
"""
import numpy as np
import pytest


def test_ridge_is_registered():
    # Registered even without the optional dependency installed (lazy import).
    from fastrack.core.detection import DETECTORS
    assert "ridge" in DETECTORS


def test_ridge_requires_dependency_or_constructs():
    """Without the extra, constructing raises a clear ImportError; with it, it builds."""
    from fastrack.core.detection import DETECTORS
    try:
        import ridge_detector  # noqa: F401
        have_dep = True
    except ImportError:
        have_dep = False

    if have_dep:
        det = DETECTORS.create("ridge", line_widths=[3])
        assert det.params["line_widths"] == [3]
    else:
        with pytest.raises(ImportError, match="fastrack\\[ridge\\]"):
            DETECTORS.create("ridge")


def test_contours_to_filxys_mapping():
    """The Line -> filXYs mapping is dependency-free and exercised with stubs."""
    from fastrack.core.detection.ridge import _contours_to_filxys

    class StubLine:
        def __init__(self, row, col, wl, wr):
            self.num = len(row)
            self.row = np.asarray(row, dtype=float)
            self.col = np.asarray(col, dtype=float)
            self.width_l = None if wl is None else np.asarray(wl, dtype=float)
            self.width_r = None if wr is None else np.asarray(wr, dtype=float)

    img = np.arange(100, dtype=np.uint16).reshape(10, 10)
    lines = [
        StubLine([1.4, 2.6, 3.0], [1, 1, 1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]),
        StubLine([0], [0], None, None),          # too short -> skipped
    ]
    filxys = _contours_to_filxys(lines, img)
    assert len(filxys) == 1
    contour, width, density, midpoint = filxys[0]
    assert contour.shape == (3, 2)
    assert contour.dtype.kind == "i"            # integer pixel coords
    # rows rounded: 1.4->1, 2.6->3, 3.0->3 ; cols all 1
    assert contour[:, 0].tolist() == [1, 3, 3]
    assert width == pytest.approx(1.2)          # mean(0.5+0.7)
    assert midpoint.tolist() == contour[3 // 2 - 1].tolist()


def test_contours_to_filxys_clips_to_image():
    from fastrack.core.detection.ridge import _contours_to_filxys

    class StubLine:
        num = 2
        row = np.array([-5.0, 100.0])           # out of bounds
        col = np.array([-1.0, 100.0])
        width_l = None
        width_r = None

    img = np.zeros((10, 10), dtype=np.uint16)
    (contour, width, density, midpoint), = _contours_to_filxys([StubLine()], img)
    assert contour[:, 0].min() >= 0 and contour[:, 0].max() <= 9
    assert contour[:, 1].min() >= 0 and contour[:, 1].max() <= 9
    assert width == 0.0                          # no width info


def test_detector_cache_isolation(tmp_path):
    """A tagged filXYs cache must not be shadowed by an untagged one in the same
    folder -- guards the bug where ridge linking re-used entropy's cached
    filaments because load_frame1/2 ignored the cache tag."""
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    pytest.importorskip("cv2")

    from fastrack.core.frame import Frame
    from fastrack.core.motility import Motility

    d = str(tmp_path)

    # A "ridge" (tagged) cache with a 3-point contour.
    fr = Frame()
    fr.directory, fr.frame_no, fr.cache_tag = d, 0, "_ridge"
    ridge_contour = np.array([[1, 1], [2, 2], [3, 3]])
    fr.filXYs = [[ridge_contour, 4.0, 900.0, ridge_contour[1]]]
    fr.save_filXYs()   # -> filXYs_ridge000.npy

    # An untagged (entropy) cache with a DIFFERENT 2-point contour.
    fr0 = Frame()
    fr0.directory, fr0.frame_no, fr0.cache_tag = d, 0, ""
    fr0.filXYs = [[np.array([[5, 5], [6, 6]]), 1.0, 1.0, np.array([6, 6])]]
    fr0.save_filXYs()  # -> filXYs000.npy

    # Loading with the ridge tag must pick up the ridge file (3 points), not the
    # untagged one (2 points).
    m = Motility()
    m.directory, m.cache_tag = d, "_ridge"
    m.load_frame1(0)
    assert len(m.frame1.filaments) == 1
    assert len(m.frame1.filaments[0].contour) == 3


def test_ridge_detect_end_to_end():
    """Full detection on a synthetic image (skips unless all deps are present)."""
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    pytest.importorskip("cv2")
    pytest.importorskip("ridge_detector")

    from fastrack.core.frame import Frame
    from fastrack.core.detection import DETECTORS

    # Synthetic frame: a few bright horizontal lines on a dark background.
    img = np.zeros((64, 64), dtype=np.uint16)
    for r in (16, 32, 48):
        img[r - 1:r + 2, 8:56] = 4000
    frame = Frame()
    frame.frame_no = 0
    frame.img = img
    frame.width, frame.height = img.shape

    detector = DETECTORS.create(
        "ridge", line_widths=[3], low_contrast=20, high_contrast=80,
        min_len=10, dark_line=False, estimate_width=True,
    )
    detector.detect(frame)

    assert len(frame.filaments) >= 1
    fil = frame.filaments[0]
    assert np.asarray(fil.contour).ndim == 2 and np.asarray(fil.contour).shape[1] == 2
    assert fil.fil_length > 0
    assert len(np.atleast_1d(fil.cm)) == 2
