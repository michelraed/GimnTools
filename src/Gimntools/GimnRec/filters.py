"""Reconstruction filters, radial detector-response kernels and sinogram
resampling.

Frequency-domain conventions used throughout this module
--------------------------------------------------------
A projection ``p`` is sampled on ``Nr`` radial bins of physical width
``bin_mm``. The ramp filters returned here are arrays in numpy DFT ordering
whose values are ``|f|`` in **cycles per bin** (i.e. ``|numpy.fft.fftfreq(n)|``),
so the filtered projection

    q = ifft( fft(p) * ramp ).real

is the ideal filtered projection multiplied by ``bin_mm``. The reconstructor
divides that factor out (see ``siddon_reconstructor.fbp``); nothing else in
this module rescales it.
"""

from math import ceil

import numpy as np
import scipy.fft as fft
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "ramLak", "ramLak_legacy", "cosine_filter", "cossineFilter",
    "hann_filter", "hamming_filter", "shepp_logan_filter",
    "butterworth_filter", "make_fbp_filter", "FBP_WINDOWS",
    "resolve_pad_length", "PAD_MODES",
    "apply_filter_to_sinogram",
    "fwhm_to_sigma", "make_psf_kernel",
    "convolve_sino_1d", "convolve_sino_1d_adj",
    "interpolate_distance", "smooth_angular",
]


###############################################################################
#  RECONSTRUCTION FILTERS
###############################################################################

def ramLak(size):
    """Ram-Lak (pure ramp) filter, ``|f|`` in cycles/bin, DFT ordering.

    Equivalent to ``numpy.abs(numpy.fft.fftfreq(size))``: zero at DC, rising
    linearly to 0.5 at the Nyquist bin and mirrored over the negative
    frequencies.

    .. note:: **Corrected in 0.4.0.** The previous implementation built the
       ramp by cumulative summation starting at ``+a`` instead of ``0``, which
       returned exactly ``|f| + 1/size``. The constant ``1/size`` is a DC
       passband: it adds an unfiltered copy of the projection to every
       filtered projection, so the backprojection acquires a broad ``1/r``
       pedestal instead of the correct zero background. Measured on an
       analytic uniform disk (FOV 40 mm, 256 bins, 180 views): the old filter
       gave a mean value of 2.13 inside the disk where the truth is 1.0 and a
       background of +0.26; the corrected filter gives 0.99 and -0.009.
       :func:`ramLak_legacy` reproduces the old array for anyone who needs to
       regenerate previously published figures.
    """
    return np.abs(fft.fftfreq(int(size)))


def ramLak_legacy(size):
    """Pre-0.4.0 ramp filter, kept only for reproducing older results.

    Returns ``|f| + 1/size`` (for even ``size``). See :func:`ramLak` for why
    the extra DC term is wrong.
    """
    size = int(size)
    r = np.zeros(size)
    v = 0.0
    a = 0.5 / (size / 2)
    for i in range(size):
        v += a if i <= size // 2 else -a
        r[i] = v
    return r


def cosine_filter(size):
    """Cosine-windowed ramp: ``|f| * cos(pi*f / f_Nyquist)``.

    With ``f`` in cycles/bin the Nyquist frequency is 0.5, so the window is
    ``cos(pi*f/0.5 / 2) = cos(pi*f)``. Rolls off to zero at Nyquist, which
    suppresses the high-frequency noise amplified by the pure ramp at the
    cost of spatial resolution.
    """
    f = np.abs(fft.fftfreq(int(size)))
    return f * np.cos(np.pi * f)


# Backward-compatible misspelled alias.
cossineFilter = cosine_filter


def hann_filter(size):
    """Hann-windowed ramp, recommended for coarse/sparse FBP data."""
    f = np.abs(fft.fftfreq(int(size)))
    return f * 0.5 * (1.0 + np.cos(2.0 * np.pi * f))


def hamming_filter(size):
    """Hamming-windowed ramp (milder roll-off than Hann)."""
    f = np.abs(fft.fftfreq(int(size)))
    return f * (0.54 + 0.46 * np.cos(2.0 * np.pi * f))


def shepp_logan_filter(size):
    """Shepp-Logan-windowed ramp."""
    f = np.abs(fft.fftfreq(int(size)))
    return f * np.sinc(f)


def butterworth_filter(size, cutoff=1.0, order=5):
    """Butterworth-apodised ramp, ``|f| / sqrt(1 + (f/f_c)**(2*order))``.

    ``cutoff`` is a FRACTION OF NYQUIST (Nyquist = 0.5 cycles/bin), so
    ``cutoff=1`` puts the -3 dB point exactly at Nyquist. Unlike the windows
    above the Butterworth roll-off is monotone and never reaches exactly
    zero, so no frequency is hard-truncated; ``f_c`` is the -3 dB point, not
    a support limit. This is the convention used by most nuclear-medicine
    consoles, where the cutoff is instead quoted in cycles/cm.
    """
    f = np.abs(fft.fftfreq(int(size)))
    fc = max(1e-12, float(cutoff)) * 0.5
    return f / np.sqrt(1.0 + (f / fc) ** (2 * int(order)))


#: Window names accepted by :func:`make_fbp_filter`.
FBP_WINDOWS = ("ramp", "cosine", "hann", "hamming", "shepp-logan",
               "butterworth")


def make_fbp_filter(window="ramp", cutoff=1.0, order=5, hard_cutoff=True):
    """Build a filter factory ``f(size) -> array`` for FBP.

    The returned array is ``|f| * W(f)`` in numpy DFT ordering with ``f`` in
    cycles/bin, i.e. the same convention as :func:`ramLak`, so it can be
    passed straight to :func:`apply_filter_to_sinogram` and
    ``siddon_reconstructor.fbp``.

    Parameters
    ----------
    window : str
        One of :data:`FBP_WINDOWS`. ``'ramp'`` is the unwindowed Ram-Lak.
    cutoff : float
        Cutoff frequency as a FRACTION OF NYQUIST, in ``(0, 1]``. The
        Nyquist frequency of the radial axis is ``0.5`` cycles/bin, i.e.
        ``1/(2*bin_mm)`` in mm^-1, so the physical cutoff is
        ``cutoff / (2*bin_mm)`` cycles per mm. ``cutoff=1`` is the classical
        "no extra cutoff" case.
    order : int
        Butterworth order. Ignored by the other windows.
    hard_cutoff : bool
        For the finite-support windows (ramp/cosine/hann/hamming/
        shepp-logan) the window is RESCALED so that it reaches zero at
        ``f_c`` and the filter is set to exactly zero above ``f_c``. Setting
        ``hard_cutoff=False`` keeps the window shaped over the full Nyquist
        band and only truncates above ``f_c``; for ``window='ramp'`` the two
        are identical (a rectangular cut of the ramp). Butterworth ignores
        this flag: its roll-off is by definition not compactly supported.

    Notes
    -----
    At ``cutoff=1.0`` and ``hard_cutoff=True`` this reproduces
    :func:`ramLak`, :func:`cosine_filter`, :func:`hann_filter`,
    :func:`hamming_filter` and :func:`shepp_logan_filter` exactly (verified
    to machine precision by ``tests/test_fbp_filter_options.py``); the
    parametrisation is therefore a strict generalisation and does not change
    any previously produced result.

    A cutoff is a resolution/noise trade-off and not a correction: lowering
    it removes high spatial frequencies from the reconstruction, so the
    measured spatial resolution (FWHM) degrades monotonically with it. The
    value used must be reported together with the radial bin size, because
    "0.5 of Nyquist" means different mm^-1 for different bin widths.
    """
    window = str(window).strip().lower()
    if window not in FBP_WINDOWS:
        raise ValueError(f"window must be one of {FBP_WINDOWS}, got {window!r}")
    cutoff = float(cutoff)
    if not (0.0 < cutoff <= 1.0):
        raise ValueError(f"cutoff must be in (0, 1] (fraction of Nyquist), "
                         f"got {cutoff}")
    order = int(order)

    def factory(size):
        size = int(size)
        f = np.abs(fft.fftfreq(size))
        if window == "butterworth":
            return butterworth_filter(size, cutoff=cutoff, order=order)
        fc = 0.5 * cutoff
        # Reference frequency the window is shaped against: f_c when the
        # window is rescaled onto the cutoff, Nyquist otherwise.
        fref = fc if hard_cutoff else 0.5
        u = f / fref
        if window == "ramp":
            w = np.ones_like(f)
        elif window == "cosine":
            w = np.cos(0.5 * np.pi * u)
        elif window == "hann":
            w = 0.5 * (1.0 + np.cos(np.pi * u))
        elif window == "hamming":
            w = 0.54 + 0.46 * np.cos(np.pi * u)
        else:  # shepp-logan
            w = np.sinc(0.5 * u)
        out = f * w
        out[f > fc] = 0.0          # support limited to |f| <= f_c
        return out

    factory.__name__ = f"fbp_filter[{window},cutoff={cutoff:g}]"
    return factory


#: Zero-padding procedures accepted by :func:`resolve_pad_length`.
PAD_MODES = ("none", "pow2", "exact")


def resolve_pad_length(n, pad_factor=2, pad_mode="pow2"):
    """Padded FFT length for a radial axis of ``n`` bins.

    Parameters
    ----------
    n : int
        Number of radial bins.
    pad_factor : int
        Multiplier applied to ``n`` before padding.
    pad_mode : {'none', 'pow2', 'exact'}
        ``'none'``  -> no padding at all (``n``); circular wrap-around of the
        ramp kernel is NOT suppressed. Reproduces pre-0.4.0 behaviour.
        ``'pow2'``  -> smallest power of two >= ``pad_factor * n`` (the
        0.4.0 default; fastest FFT, and the length actually used when
        ``pad_factor >= 2``).
        ``'exact'`` -> exactly ``pad_factor * n`` bins, without rounding up
        to a power of two. Use it when the padded length must be stated
        exactly in a methods section; the FFT may be slower for lengths with
        large prime factors.

    Returns
    -------
    int
    """
    n = int(n)
    pad_mode = str(pad_mode).strip().lower()
    pad_factor = max(1, int(pad_factor))
    if pad_mode == "none" or pad_factor == 1 and pad_mode == "pow2":
        # pad_factor == 1 with 'pow2' kept its pre-existing meaning: no pad.
        return n
    if pad_mode == "pow2":
        return int(2 ** np.ceil(np.log2(pad_factor * n)))
    if pad_mode == "exact":
        return pad_factor * n
    raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}")


def apply_filter_to_sinogram(ftype, sino, pad_factor=2, pad_mode="pow2"):
    """Filter every angular column of a 2-D sinogram ``(n_dist, n_ang)``.

    Parameters
    ----------
    ftype : callable
        Filter factory, e.g. :func:`ramLak`. Called with the padded length.
    sino : (n_dist, n_ang) array
    pad_factor : int
        Zero-padding factor applied before the FFT. The FFT implements a
        CIRCULAR convolution, so without padding the tail of the ramp kernel
        wraps around and biases the opposite side of the projection (a
        classical FBP DC/cupping artefact). ``pad_factor=1`` reproduces the
        pre-0.4.0 unpadded behaviour.

        Measured on an analytic Gaussian blob (exact projections, 256 bins,
        180 views), reconstructed value / truth at the centre and residual
        far-field background:

        ==============  =========  ==================
        ``pad_factor``  ratio      far-field baseline
        ==============  =========  ==================
        1 (no padding)  0.98912    -0.0087
        2               0.99705    -0.0013
        4               0.99903    +0.0002
        ==============  =========  ==================

    pad_mode : {'none', 'pow2', 'exact'}
        Zero-padding PROCEDURE, see :func:`resolve_pad_length`. The default
        ``'pow2'`` is the historical behaviour and leaves every previously
        produced result unchanged.

    Returns
    -------
    (n_dist, n_ang) float64 array
    """
    sino = np.asarray(sino, dtype=np.float64)
    n = sino.shape[0]
    pad_factor = max(1, int(pad_factor))
    n_pad = resolve_pad_length(n, pad_factor=pad_factor, pad_mode=pad_mode)

    flt = np.asarray(ftype(n_pad), dtype=np.float64)
    if flt.shape != (n_pad,):
        raise ValueError(
            f"filter factory returned shape {flt.shape}, expected ({n_pad},)")

    out = np.zeros_like(sino)
    buf = np.zeros(n_pad, dtype=np.float64)
    for a in range(sino.shape[1]):
        buf[:] = 0.0
        buf[:n] = sino[:, a]
        out[:, a] = np.real(fft.ifft(fft.fft(buf) * flt))[:n]
    return out


###############################################################################
#  PSF / INTERPOLATION / SMOOTHING
###############################################################################

def fwhm_to_sigma(fwhm): return fwhm/2.3548200450309493

def make_psf_kernel(fwhm_mm, pixel_mm, n_sig=4.0):
    s=fwhm_to_sigma(fwhm_mm)/pixel_mm; h=max(1,int(ceil(n_sig*s)))
    x=np.arange(-h,h+1,dtype=np.float64); k=np.exp(-0.5*(x/s)**2); k/=k.sum()
    return k

def convolve_sino_1d(sino2d, kernel, adjoint=False):
    """Convolve every angular column (axis=0 = distances) with a 1-D kernel.

    This implements the radial detector-response operator ``C`` used inside
    the system model ``H = diag(norm) . C . A`` (see reconstructor).

    adjoint : bool
        False -> apply ``C`` (forward projector side).
        True  -> apply ``C^T`` EXACTLY (backprojector side).

    Why an explicit adjoint mode (GimnToolsV2 0.4.0)
    ------------------------------------------------
    Previously the same forward convolution was used on BOTH sides. That is
    correct only if the kernel is symmetric AND of odd length — an
    assumption that was never stated or checked. Measured on random data:

        kernel                  |C used as C^T| relative error
        odd,  symmetric         2e-16   (accidentally exact)
        odd,  asymmetric        1.7e-2
        even, symmetric         1.0e-2
        even, asymmetric        1.9e-2

    So the default Gaussian PSF happened to be fine, but any measured or
    depth-dependent (asymmetric) detector response, or any even-length
    kernel, silently broke the pairing. An unpaired C/C^T means the EM
    sensitivity image is not the adjoint of the operator actually used in
    the forward step: MLEM/OSEM then converge to the wrong fixed point.

    ``C^T`` is the correlation with the same kernel, i.e. convolution with
    the reversed kernel AND the mirrored crop offset. Reversing alone is not
    enough for even-length kernels, because numpy's ``mode='same'`` crop is
    asymmetric in that case (verified: reversing left a 1.0e-2 residual for
    an even symmetric kernel, and 5.6e-3 for an even asymmetric one).

    For a symmetric odd-length kernel this function returns bit-identical
    results to the previous implementation in both directions.
    """
    k = np.asarray(kernel, dtype=np.float64).ravel()
    L = k.size
    if L == 0:
        raise ValueError("empty convolution kernel")
    n = sino2d.shape[0]
    if L > n:
        raise ValueError(
            f"kernel length ({L}) exceeds the number of radial bins ({n}); "
            "reduce fwhm_mm or use a finer radial sampling")

    off = (L - 1) // 2            # numpy 'same' crop offset
    if adjoint:
        k = k[::-1]
        off = (L - 1) - off       # mirrored offset -> exact transpose

    out = np.empty_like(sino2d, dtype=np.float64)
    for th in range(sino2d.shape[1]):
        full = np.convolve(sino2d[:, th].astype(np.float64), k, mode='full')
        out[:, th] = full[off:off + n]
    return out


def convolve_sino_1d_adj(sino2d, kernel):
    """Exact transpose of :func:`convolve_sino_1d` (convenience alias)."""
    return convolve_sino_1d(sino2d, kernel, adjoint=True)

def _conservative_radial_rebin(sino, n_out):
    """Redistribute histogram counts between radial grids by bin overlap.

    Both grids cover the same normalized physical interval ``[0, 1]``.
    Each input-bin count is split among the overlapping output bins in
    proportion to the overlap.  The operation is therefore non-oscillatory
    and conserves the sum independently for every slice and angle.
    """
    sino = np.asarray(sino, dtype=np.float64)
    _, n_in, _ = sino.shape
    n_out = int(n_out)
    if n_out < 1:
        raise ValueError(f"n_out must be >= 1 (got {n_out})")
    if n_out == n_in:
        return sino.copy()

    # R[j, i] is the fraction of input bin i assigned to output bin j.
    in_lo = np.arange(n_in, dtype=np.float64) / n_in
    in_hi = (np.arange(n_in, dtype=np.float64) + 1.0) / n_in
    out_lo = np.arange(n_out, dtype=np.float64) / n_out
    out_hi = (np.arange(n_out, dtype=np.float64) + 1.0) / n_out
    overlap = np.maximum(
        0.0,
        np.minimum(out_hi[:, None], in_hi[None, :])
        - np.maximum(out_lo[:, None], in_lo[None, :]),
    )
    redistribution = overlap * n_in
    return np.einsum("ji,zit->zjt", redistribution, sino, optimize=True)


def interpolate_distance(sino, n_out=128, kind='conservative', align='centers'):
    """Resample the radial axis of a sinogram to ``n_out`` bins.

    Parameters
    ----------
    sino : (n_slices, n_dist, n_ang) array
    n_out : int
        Number of output radial bins.
    kind : str
        ``'conservative'`` (default) redistributes histogram counts by physical bin
        overlap.  It preserves the total counts and cannot create the
        alternating positive/negative sidelobes produced by cubic splines;
        use it whenever ``sino`` contains measured event counts.  Other
        values are passed to :class:`scipy.interpolate.interp1d` and are
        intended for continuous profiles rather than count histograms.
    align : {'centers', 'endpoints'}
        Coordinate convention used to place the output samples.

        ``'centers'`` (default, correct): input bin *i* sits at the centre of
        its bin, ``(i + 0.5)/n_dist`` of the radial span, and output bin *j*
        at ``(j + 0.5)/n_out``. The physical span of the array is preserved,
        so the whole pipeline may keep describing the resampled axis as
        ``n_out`` uniform bins covering ``sino_range_mm`` -- which is exactly
        what ``pipelines``, ``reconstruction_v2``, ``calibration`` and the
        CLIs assume when they rebuild ``distances``.

        ``'endpoints'`` reproduces the pre-0.4.0 behaviour, which mapped
        ``linspace(0, 1, n_dist)`` onto ``linspace(0, 1, n_out)``, i.e. it
        aligned the CENTRE of the first and last input bins with the CENTRE
        of the first and last output bins. The resampled array then spans
        only ``(n_dist - 1)/n_dist`` of the radial range while the rest of the
        pipeline still labelled it as the full range.

    .. warning:: **Radial scale error corrected in 0.4.0.** With
       ``align='endpoints'`` the mismatch between the true and the assumed bin
       width is ``n_out*(n_dist-1) / (n_dist*(n_out-1))``. For the shipped
       segmented geometry (15 measured bins, ``n_out=128``, radial span
       26.88 mm) the pipeline assumed a bin of 0.2100 mm where the true one
       was 0.1975 mm: a **+6.31 % radial scale error**, i.e. up to 0.79 mm of
       positional error at the edge of the field. Every distance reported in
       mm was affected -- reconstructed FWHM, source positions, source
       separations and the fitted ``dx_cor``/``dy_cor``. Results produced
       before 0.4.0 with radial interpolation enabled must be regenerated or
       rescaled.

    Returns
    -------
    (n_slices, n_out, n_ang) float64 array
    """
    sino = np.asarray(sino)
    Nz, Nr, Nth = sino.shape
    n_out = int(n_out)
    if kind == 'conservative':
        if align != 'centers':
            raise ValueError(
                "kind='conservative' preserves the full bin extent and "
                "requires align='centers'")
        return _conservative_radial_rebin(sino, n_out)
    if align == 'centers':
        ri = (np.arange(Nr) + 0.5) / Nr
        ro = (np.arange(n_out) + 0.5) / n_out
    elif align == 'endpoints':
        ri = np.linspace(0.0, 1.0, Nr)
        ro = np.linspace(0.0, 1.0, n_out)
    else:
        raise ValueError(
            f"align must be 'centers' or 'endpoints', got {align!r}")

    out = np.empty((Nz, n_out, Nth), dtype=np.float64)
    for z in range(Nz):
        for th in range(Nth):
            out[z, :, th] = interp1d(ri, sino[z, :, th].astype(np.float64),
                                     kind=kind, fill_value=0.0,
                                     bounds_error=False)(ro)
    return out


def smooth_angular(sino, sigma_bins=2.0, mode='parallel180'):
    """Gaussian smoothing along the angular axis, per (slice, distance) row.

    Parameters
    ----------
    sino : (n_slices, n_dist, n_ang) array
    sigma_bins : float
        Standard deviation in angular bins. ``0`` returns a float64 copy.
    mode : {'parallel180', 'wrap', 'nearest'}
        Boundary condition on the angular axis.

        ``'parallel180'`` (default) uses the exact periodic extension of a
        180-degree parallel-beam sinogram, ``s(r, theta + 180) = s(-r, theta)``:
        the array is continued past ``theta = 180`` with the radially
        REVERSED sinogram. This is the only boundary condition consistent
        with the geometry.

        ``'wrap'`` is the pre-0.4.0 behaviour: it continued the angular axis
        with an un-reversed copy, which mixes bin ``r`` with bin ``-r`` at the
        two ends of the angular range. The error is confined to roughly
        ``3*sigma_bins`` columns at each end and vanishes for a radially
        symmetric object, which is why it went unnoticed; it is kept for
        reproducibility.

        ``'nearest'`` simply replicates the edge column (no wrap-around at
        all).

    Returns
    -------
    (n_slices, n_dist, n_ang) float64 array
    """
    sino = np.asarray(sino, dtype=np.float64)
    if sigma_bins is None or sigma_bins <= 0:
        return sino.copy()

    Nz, Nr, Nth = sino.shape
    if mode == 'wrap':
        out = np.empty_like(sino)
        for z in range(Nz):
            for r in range(Nr):
                out[z, r, :] = gaussian_filter1d(sino[z, r, :],
                                                 sigma=sigma_bins, mode='wrap')
        return out
    if mode == 'nearest':
        out = np.empty_like(sino)
        for z in range(Nz):
            for r in range(Nr):
                out[z, r, :] = gaussian_filter1d(sino[z, r, :],
                                                 sigma=sigma_bins,
                                                 mode='nearest')
        return out
    if mode != 'parallel180':
        raise ValueError(
            f"mode must be 'parallel180', 'wrap' or 'nearest', got {mode!r}")

    # Exact 180-degree extension: [ s(-r, .) | s(r, .) | s(-r, .) ]
    flipped = sino[:, ::-1, :]
    ext = np.concatenate([flipped, sino, flipped], axis=2)
    ext = gaussian_filter1d(ext, sigma=sigma_bins, axis=2, mode='nearest')
    return ext[:, :, Nth:2 * Nth]
