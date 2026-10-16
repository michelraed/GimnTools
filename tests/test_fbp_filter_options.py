"""Verify the parametric FBP filter/padding options.

The point of these tests is that the new options are a strict
GENERALISATION: at cutoff = 1.0 the parametric factory must return, bit for
bit, the arrays the shipped filter functions already returned, so no
previously produced reconstruction changes.
"""
import numpy as np
import pytest

from Gimntools.GimnRec.filters import (
    ramLak, cosine_filter, hann_filter, hamming_filter, shepp_logan_filter,
    butterworth_filter, make_fbp_filter, resolve_pad_length,
    apply_filter_to_sinogram,
)

SIZES = [64, 128, 129, 256, 512]

LEGACY = {
    "ramp": ramLak,
    "cosine": cosine_filter,
    "hann": hann_filter,
    "hamming": hamming_filter,
    "shepp-logan": shepp_logan_filter,
}


@pytest.mark.parametrize("window", sorted(LEGACY))
@pytest.mark.parametrize("size", SIZES)
def test_cutoff_one_reproduces_legacy_filter(window, size):
    """cutoff=1.0 must equal the legacy filter to machine precision."""
    new = make_fbp_filter(window, cutoff=1.0)(size)
    old = LEGACY[window](size)
    assert new.shape == old.shape
    assert np.max(np.abs(new - old)) < 1e-15


@pytest.mark.parametrize("window", sorted(LEGACY))
def test_cutoff_zeroes_everything_above_fc(window):
    size, cutoff = 256, 0.4
    f = np.abs(np.fft.fftfreq(size))
    flt = make_fbp_filter(window, cutoff=cutoff)(size)
    assert np.all(flt[f > 0.5 * cutoff] == 0.0)
    assert np.any(flt[f <= 0.5 * cutoff] > 0.0)


@pytest.mark.parametrize("window", sorted(LEGACY))
def test_hard_cutoff_window_touches_zero_at_fc(window):
    """With hard_cutoff the window is rescaled onto f_c, so the ramp value
    at f_c is the window's own Nyquist value (0 for cosine/hann, 0.08 for
    hamming, sinc(0.5) for shepp-logan, 1 for the bare ramp)."""
    size, cutoff = 4096, 0.5
    f = np.abs(np.fft.fftfreq(size))
    flt = make_fbp_filter(window, cutoff=cutoff, hard_cutoff=True)(size)
    fc = 0.5 * cutoff
    i = np.argmin(np.abs(f - fc))                  # closest bin below/at f_c
    expected_w = {"ramp": 1.0, "cosine": 0.0, "hann": 0.0,
                  "hamming": 0.08, "shepp-logan": float(np.sinc(0.5))}[window]
    assert flt[i] / f[i] == pytest.approx(expected_w, abs=2e-3)


def test_soft_cutoff_keeps_nyquist_shaping():
    """hard_cutoff=False shapes the window over the full band and only
    truncates; at f_c the window value is therefore NOT zero for hann."""
    size, cutoff = 4096, 0.5
    f = np.abs(np.fft.fftfreq(size))
    flt = make_fbp_filter("hann", cutoff=cutoff, hard_cutoff=False)(size)
    i = np.argmin(np.abs(f - 0.5 * cutoff))
    # hann at half of Nyquist: 0.5*(1+cos(pi/2)) = 0.5
    assert flt[i] / f[i] == pytest.approx(0.5, abs=2e-3)


def test_ramp_ignores_hard_cutoff_flag():
    a = make_fbp_filter("ramp", cutoff=0.6, hard_cutoff=True)(256)
    b = make_fbp_filter("ramp", cutoff=0.6, hard_cutoff=False)(256)
    assert np.array_equal(a, b)


def test_butterworth_is_minus_3db_at_cutoff():
    size, cutoff, order = 8192, 0.5, 5
    f = np.abs(np.fft.fftfreq(size))
    flt = butterworth_filter(size, cutoff=cutoff, order=order)
    i = np.argmin(np.abs(f - 0.5 * cutoff))
    assert flt[i] / f[i] == pytest.approx(1 / np.sqrt(2), abs=1e-3)


def test_butterworth_never_exactly_zero():
    flt = make_fbp_filter("butterworth", cutoff=0.3, order=8)(256)
    f = np.abs(np.fft.fftfreq(256))
    assert np.all(flt[f > 0] > 0.0)


def test_butterworth_order_steepens_rolloff():
    size, cutoff = 4096, 0.4
    f = np.abs(np.fft.fftfreq(size))
    i = np.argmin(np.abs(f - 0.45))          # well above f_c = 0.2
    v = [butterworth_filter(size, cutoff, o)[i] / f[i] for o in (2, 5, 10)]
    assert v[0] > v[1] > v[2]


@pytest.mark.parametrize("window", ["hann", "butterworth"])
def test_lower_cutoff_attenuates_more(window):
    size = 1024
    f = np.abs(np.fft.fftfreq(size))
    i = np.argmin(np.abs(f - 0.3))
    hi = make_fbp_filter(window, cutoff=0.9)(size)[i]
    lo = make_fbp_filter(window, cutoff=0.5)(size)[i]
    assert hi > lo


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_fbp_filter("gaussian")
    with pytest.raises(ValueError):
        make_fbp_filter("hann", cutoff=0.0)
    with pytest.raises(ValueError):
        make_fbp_filter("hann", cutoff=1.5)
    with pytest.raises(ValueError):
        resolve_pad_length(100, 2, "quadruple")


# --------------------------------------------------------------------------
#  zero-padding procedure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,factor,expected", [
    (100, 1, 100), (100, 2, 256), (100, 4, 512),
    (128, 2, 256), (128, 4, 512), (129, 2, 512),
])
def test_pow2_padding_is_unchanged(n, factor, expected):
    """'pow2' must reproduce the previous hard-coded formula."""
    assert resolve_pad_length(n, factor, "pow2") == expected


@pytest.mark.parametrize("n,factor", [(100, 1), (100, 2), (129, 4)])
def test_none_padding(n, factor):
    assert resolve_pad_length(n, factor, "none") == n


@pytest.mark.parametrize("n,factor,expected", [
    (100, 2, 200), (100, 3, 300), (129, 2, 258), (128, 4, 512),
])
def test_exact_padding(n, factor, expected):
    assert resolve_pad_length(n, factor, "exact") == expected


def test_pad_mode_default_matches_old_behaviour():
    """apply_filter_to_sinogram with defaults must be byte-identical to the
    pre-existing implementation (pow2 rounding of 2*n)."""
    rng = np.random.default_rng(0)
    sino = rng.random((100, 17))
    n_pad = int(2 ** np.ceil(np.log2(2 * 100)))
    flt = ramLak(n_pad)
    ref = np.zeros_like(sino)
    for a in range(sino.shape[1]):
        buf = np.zeros(n_pad)
        buf[:100] = sino[:, a]
        ref[:, a] = np.real(np.fft.ifft(np.fft.fft(buf) * flt))[:100]
    got = apply_filter_to_sinogram(ramLak, sino)
    assert np.max(np.abs(got - ref)) < 1e-12
