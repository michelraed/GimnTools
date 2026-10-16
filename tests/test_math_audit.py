"""Regression tests for the mathematical corrections introduced in 0.4.0.

Each test pins down one of the defects found during the pre-publication audit
(see ``docs/AUDIT.md``) so that it cannot silently come back. Every expected
value here is either exact (adjointness, analytic line integrals, coordinate
round-trips) or an analytic ground truth reconstructed from exact projections.
"""
import os
import sys

import numpy as np
import pytest
import scipy.fft as fft

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Gimntools.GimnRec.filters import (                       # noqa: E402
    ramLak, ramLak_legacy, cosine_filter, hann_filter, hamming_filter,
    shepp_logan_filter, apply_filter_to_sinogram,
    interpolate_distance, smooth_angular, convolve_sino_1d,
)
from Gimntools.GimnRec.projectors import (                    # noqa: E402
    siddon_forward_2d, siddon_backward_2d,
)
from Gimntools.GimnRec.reconstructor import siddon_reconstructor   # noqa: E402
from Gimntools.GimnRec.sinogramer import Sinogramer            # noqa: E402


SEG_CONFIG = {
    "crystal_type": "segmented",
    "sipm_pixel_n": np.array([8, 8]),
    "sipm_pitch": np.array([3.36, 3.36]),
    "crystal_size": np.array([20.0, 26.88, 26.88]),
    "rotation_angles": 12,
    "sino_range_mm": 26.88,
    "arround": 180,
}


# --------------------------------------------------------------------------
#  Projector: adjointness and absolute accuracy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N,Nr,Nth,K", [(32, 24, 17, 1), (32, 24, 17, 3),
                                        (48, 32, 19, 5), (31, 25, 13, 4)])
def test_projector_is_exactly_self_adjoint(N, Nr, Nth, K):
    rng = np.random.default_rng(0)
    px, bw = 1.0, 1.3
    grid_min = -N * px / 2
    ang = np.sort(rng.uniform(0, np.pi, Nth))
    dist = (np.arange(Nr) + 0.5) * bw - Nr * bw / 2
    x = rng.random((N, N))
    y = rng.random((Nr, Nth))
    lhs = float((siddon_forward_2d(x, ang, dist, N, px, grid_min, bw, K) * y).sum())
    rhs = float((x * siddon_backward_2d(y, ang, dist, N, px, grid_min, bw, K, 1)).sum())
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_projector_reproduces_analytic_chord_length():
    """Line integrals through a uniform disk must match 2*sqrt(R^2 - r^2)."""
    N, px, R = 401, 0.1, 10.0
    grid_min = -N * px / 2
    c = (np.arange(N) + 0.5) * px + grid_min
    yy, xx = np.meshgrid(c, c, indexing="ij")
    disk = ((xx ** 2 + yy ** 2) <= R ** 2).astype(float)
    dist = np.array([0.0, 2.0, 5.0, 8.0])
    ang = np.array([0.0, 0.7, 1.9])
    s = siddon_forward_2d(disk, ang, dist, N, px, grid_min, 0.0, 1)
    for i, d in enumerate(dist):
        exact = 2 * np.sqrt(R ** 2 - d ** 2)
        assert abs(s[i].mean() - exact) < 0.005 * exact


# --------------------------------------------------------------------------
#  Ramp filter: the DC term must be zero
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [8, 16, 32, 127, 128, 256])
def test_ramlak_is_the_ideal_ramp(n):
    assert np.allclose(ramLak(n), np.abs(fft.fftfreq(n)))
    assert ramLak(n)[0] == 0.0


@pytest.mark.parametrize("n", [8, 16, 32, 128])
def test_legacy_ramlak_carries_the_documented_dc_offset(n):
    """The old filter was |f| + 1/n; keep that reproducible but separate."""
    assert np.allclose(ramLak_legacy(n), np.abs(fft.fftfreq(n)) + 1.0 / n)


def test_ramp_filter_kills_a_constant_projection():
    """Zero DC gain: an unpadded constant projection must filter to exactly 0.

    Only meaningful without zero-padding: once the projection is padded it is
    a rectangle, not a constant, and a ramp legitimately responds to its
    edges.
    """
    out = apply_filter_to_sinogram(ramLak, np.ones((128, 4)), pad_factor=1)
    assert np.max(np.abs(out)) < 1e-9


def test_cosine_filter_rolls_off_to_zero_at_nyquist():
    n = 128
    c = cosine_filter(n)
    assert c[0] == 0.0
    assert abs(c[n // 2]) < 1e-12
    assert np.all(c >= -1e-12)


def test_apodized_fbp_filters_are_valid_ramps():
    n = 128
    ramp = ramLak(n)
    for factory in (hann_filter, hamming_filter, shepp_logan_filter):
        filt = factory(n)
        assert filt.shape == (n,)
        assert np.all(np.isfinite(filt))
        assert np.all(filt >= 0.0)
        assert filt[0] == 0.0
        assert np.all(filt <= ramp + 1e-15)
    assert abs(hann_filter(n)[n // 2]) < 1e-12


# --------------------------------------------------------------------------
#  FBP: absolute normalization against an analytic phantom
# --------------------------------------------------------------------------

def _gaussian_blob_case(N=256, Nr=256, Nth=180, fov=40.0, sigma=2.0):
    px, bw = fov / N, fov / Nr
    c = (np.arange(N) + 0.5) * px - fov / 2
    X, Y = np.meshgrid(c, c, indexing="xy")
    truth = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    r = (np.arange(Nr) + 0.5) * bw - fov / 2
    proj = np.sqrt(2 * np.pi) * sigma * np.exp(-r ** 2 / (2 * sigma ** 2))
    sino = np.repeat(proj[:, None], Nth, axis=1)[None, :, :]
    angles = (np.arange(Nth) + 0.5) * (180.0 / Nth)
    rec = siddon_reconstructor(sino, fov_mm=fov, sino_range_mm=fov,
                               n_img=N, verbose=False)
    return rec, angles, truth, X, Y, sigma


def test_fbp_is_quantitative():
    """FBP of exact projections must return the object, not an arbitrary scale."""
    rec, angles, truth, X, Y, sigma = _gaussian_blob_case()
    img = rec.fbp(angles, pad_factor=4)[0]
    core = (X ** 2 + Y ** 2) < sigma ** 2
    ratio = img[core].mean() / truth[core].mean()
    assert 0.99 < ratio < 1.01, f"FBP scale off: ratio={ratio:.4f}"
    far = (X ** 2 + Y ** 2) > (6 * sigma) ** 2
    assert abs(img[far].mean()) < 0.005


def test_fbp_padding_reduces_the_dc_bias():
    rec, angles, truth, X, Y, sigma = _gaussian_blob_case()
    far = (X ** 2 + Y ** 2) > (6 * sigma) ** 2
    bias = [abs(rec.fbp(angles, pad_factor=pf)[0][far].mean())
            for pf in (1, 2, 4)]
    assert bias[0] > bias[1] > bias[2]


def test_fbp_normalize_false_restores_the_legacy_scale():
    rec, angles, *_ = _gaussian_blob_case(N=64, Nr=64, Nth=45)
    a = rec.fbp(angles, normalize=False)
    b = rec.fbp(angles, normalize=True)
    k = (np.pi / 45) / rec.img_pixel ** 2
    assert np.allclose(b, a * k)
    # On this grid voxel == bin, so the pre-0.4.1 constant is numerically
    # identical; that coincidence is why the scale error went unnoticed.
    assert rec.img_pixel == pytest.approx(rec.sino_pixel)


def test_fbp_scale_is_independent_of_the_image_grid():
    """The reconstructed value must not depend on how the grid was chosen.

    Before 0.4.1 the constant carried a spurious factor voxel_mm/sino_pixel,
    so the same object reconstructed at 64 vs 512 pixels differed by 8x.
    """
    ratios = []
    for N in (64, 128, 256, 512):
        rec, angles, truth, X, Y, sigma = _gaussian_blob_case(N=N, Nr=256)
        img = rec.fbp(angles, pad_factor=4)[0]
        core = (X ** 2 + Y ** 2) < sigma ** 2
        ratios.append(img[core].mean() / truth[core].mean())
    assert max(ratios) / min(ratios) < 1.15, (
        f"FBP scale still depends on the grid: {ratios}")


def test_fbp_legacy_scale_reproduces_the_old_constant():
    rec, angles, *_ = _gaussian_blob_case(N=128, Nr=256, Nth=45)
    new = rec.fbp(angles, legacy_scale=False)
    old = rec.fbp(angles, legacy_scale=True)
    assert np.allclose(old, new * (rec.img_pixel / rec.sino_pixel))


# --------------------------------------------------------------------------
#  Radial resampling: the physical span must be preserved
# --------------------------------------------------------------------------

def test_interpolate_distance_preserves_the_radial_scale():
    """A delta at the outermost measured bin must stay at its true radius."""
    R, Nr, n_out = 26.88, 15, 128
    s = np.zeros((1, Nr, 1))
    s[0, -1, 0] = 1.0
    out = interpolate_distance(s, n_out=n_out, kind="linear")[0, :, 0]
    j = int(np.argmax(out))
    r_reported = (j + 0.5) * (R / n_out) - R / 2
    r_true = (Nr - 1 + 0.5) * (R / Nr) - R / 2
    # residual is bounded by half an output bin
    assert abs(r_reported - r_true) <= 0.5 * (R / n_out) + 1e-9


def test_legacy_endpoint_alignment_still_available_and_biased():
    R, Nr, n_out = 26.88, 15, 128
    s = np.zeros((1, Nr, 1))
    s[0, -1, 0] = 1.0
    out = interpolate_distance(s, n_out=n_out, kind="linear",
                               align="endpoints")[0, :, 0]
    assert int(np.argmax(out)) == n_out - 1     # the documented +6.31 % stretch


def test_interpolation_conserves_a_smooth_profile():
    """Away from the edges the resampled profile must sit on the true curve.

    The outermost output bins fall beyond the first/last input bin CENTRE, so
    interp1d extrapolates them with fill_value=0; those bins are excluded
    here and the behaviour is documented rather than silently tolerated.
    """
    R, Nr, n_out = 26.88, 64, 256
    r = (np.arange(Nr) + 0.5) * (R / Nr) - R / 2
    prof = np.exp(-r ** 2 / (2 * 4.0 ** 2))
    out = interpolate_distance(
        prof[None, :, None], n_out=n_out, kind="cubic")[0, :, 0]
    ro = (np.arange(n_out) + 0.5) * (R / n_out) - R / 2
    ref = np.exp(-ro ** 2 / (2 * 4.0 ** 2))
    edge = int(np.ceil(0.5 * n_out / Nr)) + 1
    assert np.max(np.abs(out[edge:-edge] - ref[edge:-edge])) < 1e-5


def test_conservative_count_rebin_has_no_ring_lobes_and_preserves_counts():
    """A central count bin must not acquire spline sidelobes.

    Cubic interpolation of this exact input produces detached positive
    lobes several millimetres from the source; their backprojection is a set
    of artificial concentric circles.  Histogram-overlap rebinning must keep
    the support contiguous, non-negative, and conserve every angular sum.
    """
    n_in, n_out, n_ang = 15, 128, 7
    sino = np.zeros((1, n_in, n_ang), dtype=np.float64)
    sino[0, n_in // 2, :] = np.arange(1, n_ang + 1)
    out = interpolate_distance(sino, n_out=n_out, kind="conservative")

    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), sino.sum(axis=1), rtol=0, atol=1e-12)
    support = np.flatnonzero(out[0, :, -1] > 0)
    assert np.all(np.diff(support) == 1)


# --------------------------------------------------------------------------
#  Angular smoothing: the 180-degree wrap must mirror the radial axis
# --------------------------------------------------------------------------

def test_smooth_angular_uses_the_parallel_beam_extension():
    """The angular axis continues as s(r, theta + 180) = s(-r, theta).

    A radially SYMMETRIC pattern that is constant in theta is invariant under
    that extension, so smoothing must leave it untouched, edges included.
    An ASYMMETRIC one must not: its continuation is the mirrored row, so the
    edge columns are pulled towards the value at -r. The old 'wrap' mode
    continued the row with itself and left both cases unchanged, which is
    only correct for the symmetric one.
    """
    Nr, Nth = 33, 40
    sym = np.zeros((1, Nr, Nth))
    sym[0, 5, :] = 1.0
    sym[0, Nr - 1 - 5, :] = 1.0                  # mirror partner at -r
    out = smooth_angular(sym, sigma_bins=2.0, mode="parallel180")
    assert np.allclose(out[0, 5, :], 1.0, atol=1e-9)

    asym = np.zeros((1, Nr, Nth))
    asym[0, 5, :] = 1.0                          # no mirror partner
    out = smooth_angular(asym, sigma_bins=2.0, mode="parallel180")
    assert out[0, 5, 0] < 0.7                    # edge pulled towards s(-r)=0
    assert np.isclose(out[0, 5, Nth // 2], 1.0, atol=1e-9)
    legacy = smooth_angular(asym, sigma_bins=2.0, mode="wrap")
    assert np.allclose(legacy[0, 5, :], 1.0, atol=1e-9)


def test_smooth_angular_preserves_total_counts():
    rng = np.random.default_rng(1)
    s = rng.random((2, 17, 48))
    out = smooth_angular(s, sigma_bins=1.5)
    assert abs(out.sum() - s.sum()) < 1e-6 * s.sum()


def test_smooth_angular_zero_sigma_is_a_no_op():
    s = np.random.default_rng(2).random((1, 9, 12))
    assert np.array_equal(smooth_angular(s, 0.0), s)


# --------------------------------------------------------------------------
#  Sinogram coordinates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 89.999, 90.0, 120.0, 179.5])
def test_global_to_sinogram_round_trip(phi_deg):
    s = Sinogramer(SEG_CONFIG)
    a = np.deg2rad(phi_deg)
    n = np.array([np.cos(a), np.sin(a)])
    d = np.array([-n[1], n[0]])
    r_true = 3.7
    p = r_true * n
    p1 = np.array([*(p - 5 * d), 0.0])
    p2 = np.array([*(p + 5 * d), 0.0])
    r, phi = s.global_to_sinogram(p1, p2, "z")
    assert abs(r - r_true) < 1e-9
    assert abs(phi - phi_deg) < 1e-9


def test_vertical_lor_is_not_dropped():
    """dx == 0 used to divide by zero and yield phi=180, silently discarded."""
    s = Sinogramer(SEG_CONFIG)
    r, phi = s.global_to_sinogram([0.0, -30.0, 0.0], [0.0, 30.0, 0.0], "z")
    assert np.isfinite(r) and np.isfinite(phi)
    assert 0.0 <= phi < 180.0
    assert s.get_pixel_position(r, phi) is not None


def test_degenerate_lor_returns_nan_and_is_rejected():
    s = Sinogramer(SEG_CONFIG)
    r, phi = s.global_to_sinogram([1.0, 2.0, 0.0], [1.0, 2.0, 0.0], "z")
    assert np.isnan(r) and np.isnan(phi)
    assert s.get_pixel_position(r, phi) is None


def test_radial_binning_rejects_out_of_field_lors_symmetrically():
    s = Sinogramer(SEG_CONFIG)
    half = SEG_CONFIG["sino_range_mm"] / 2
    assert s.get_pixel_position(-half, 90.0) == (0, s.sinogram.shape[2] // 2)
    for d in (-half - 1e-6, -half - 0.5, -half - 2.0,
              half, half + 0.5, half + 2.0):
        assert s.get_pixel_position(d, 90.0) is None


def test_radial_bin_centres_match_the_reconstructor_grid():
    """The binning in Sinogramer and the geometry in the projector must agree."""
    s = Sinogramer(SEG_CONFIG)
    R = SEG_CONFIG["sino_range_mm"]
    n_dist = s.sinogram.shape[1]
    bw = R / n_dist
    for i in range(n_dist):
        centre = (i + 0.5) * bw - R / 2
        assert s.get_pixel_position(centre, 0.5)[0] == i


def test_slice_index_never_goes_out_of_range():
    cfg = dict(SEG_CONFIG, crystal_type="monolithic")
    s = Sinogramer(cfg, matrix=np.array([15, 128, 128]))
    half = float(cfg["crystal_size"][1]) / 2
    assert s.get_slice_mono((-half, -half)) == 0
    assert s.get_slice_mono((0.0, 0.0)) == 7
    for pair in [(half, half), (half + 0.1, half), (-half - 0.1, -half)]:
        assert s.get_slice_mono(pair) == -1        # rejected, not out of range
    # fill_sino must survive a rejected slice
    s.fill_sino(0.0, 90.0, -1)
    assert s.sinogram.sum() == 0


# --------------------------------------------------------------------------
#  EM system model consistency
# --------------------------------------------------------------------------

def test_psf_convolution_adjoint_is_exact_for_any_kernel():
    rng = np.random.default_rng(3)
    for L in (4, 5, 8, 9):
        k = rng.random(L)                     # deliberately asymmetric
        x = rng.random((32, 7))
        y = rng.random((32, 7))
        lhs = float((convolve_sino_1d(x, k, adjoint=False) * y).sum())
        rhs = float((x * convolve_sino_1d(y, k, adjoint=True)).sum())
        assert abs(lhs - rhs) <= 1e-10 * abs(lhs)


def test_mlem_fixed_point_reproduces_a_uniform_disk():
    """With a consistent model, EM must converge to the object, not a scale of it."""
    N = Nr = 96
    Nth = 60
    fov = 30.0
    px, bw = fov / N, fov / Nr
    c = (np.arange(N) + 0.5) * px - fov / 2
    X, Y = np.meshgrid(c, c, indexing="xy")
    truth = ((X ** 2 + Y ** 2) < 6.0 ** 2).astype(float)
    angles = (np.arange(Nth) + 0.5) * (180.0 / Nth)
    rec0 = siddon_reconstructor(np.zeros((1, Nr, Nth)), fov_mm=fov,
                                sino_range_mm=fov, n_img=N, verbose=False)
    sino = rec0._project(truth, np.deg2rad(angles))[None, :, :]
    rec = siddon_reconstructor(sino, fov_mm=fov, sino_range_mm=fov,
                               n_img=N, verbose=False)
    img = rec.mlem(angles, iterations=60, verbose=False,
                   regularization_every=0)[0]
    core = (X ** 2 + Y ** 2) < 3.0 ** 2
    assert 0.95 < img[core].mean() < 1.05
