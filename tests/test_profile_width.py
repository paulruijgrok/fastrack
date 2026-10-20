"""Filament cross-section width measured from the image (focus metric)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from fastrack.analysis.profile_width import (  # noqa: E402
    FWHM_PER_SIGMA, filament_profile_widths)

H = W = 256


def _line(sigma, angle, offset=0.0, amp=200.0, noise=6.0, seed=0, neighbour=None):
    """Straight Gaussian-profile line through the centre (+ optional parallel
    neighbour over half its length); returns image and integer centerlines."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = 128 + offset, 128 + 0.7 * offset
    d = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    s = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    img = 300 + amp * np.exp(-0.5 * (d / sigma) ** 2) * (np.abs(s) < 40)
    t = np.arange(-35, 36)
    conts = [np.column_stack([np.rint(cy + t * np.sin(angle)),
                              np.rint(cx + t * np.cos(angle))]).astype(int)]
    if neighbour is not None:
        img += amp * np.exp(-0.5 * ((d - neighbour) / sigma) ** 2) * (s > 0) * (s < 40)
        t2 = np.arange(1, 39)
        conts.append(np.column_stack(
            [np.rint(cy + t2 * np.sin(angle) + neighbour * np.cos(angle)),
             np.rint(cx + t2 * np.cos(angle) - neighbour * np.sin(angle))]).astype(int))
    return img + rng.normal(0, noise, img.shape), conts


@pytest.mark.parametrize("sigma", [1.2, 1.6, 2.2])
def test_recovers_known_width_at_any_angle_and_subpixel_offset(sigma):
    got = []
    for angle in np.linspace(0, np.pi, 5)[:-1]:
        for offset in (0.0, 0.3, 0.5):
            img, conts = _line(sigma, angle, offset)
            got.append(filament_profile_widths(img, conts)[0])
    assert np.allclose(np.array(got) / FWHM_PER_SIGMA, sigma, rtol=0.03)


def test_wide_blur_uses_a_wider_window():
    """sigma 3.5 px needs > the initial +-6 px window; must not under-read."""
    img, conts = _line(3.5, 0.3, amp=400)
    assert filament_profile_widths(img, conts)[0] / FWHM_PER_SIGMA == pytest.approx(3.5, rel=0.05)


def test_neighbour_profiles_are_rejected():
    img, conts = _line(1.6, 0.4, 0.3, neighbour=7)
    w = filament_profile_widths(img, conts, measure=[0])
    assert len(w) == 1
    assert w[0] / FWHM_PER_SIGMA == pytest.approx(1.6, rel=0.03)


def test_short_or_empty_contours_give_nan():
    img, _ = _line(1.5, 0.0)
    w = filament_profile_widths(img, [np.array([[128, 128], [128, 129]]), np.zeros((0, 2))])
    assert np.isnan(w).all()
