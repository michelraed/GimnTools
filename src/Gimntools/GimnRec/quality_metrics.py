"""Quantitative metrics for point-, dual- and triple-source (capillary)
reconstructions.

Includes:
  - FWHM by Gaussian fit (more robust than bin-crossing FWHM)
  - FWHM along two directions
  - Source separation in dual/triple phantoms
  - Peak-to-valley contrast between adjacent sources
  - Position-error statistics

All fit-based metrics come with an uncertainty estimated from the fit
covariance and are reported in physical units (mm), assuming that
``fov_mm`` and the image shape are consistent.

Acknowledged limitations:
  - All functions assume a SQUARE image (``shape[0] == shape[1]``); a
    ``ValueError`` is raised otherwise.
  - The Gaussian-fit FWHM assumes a locally Gaussian PSF. For systems with
    a non-Gaussian PSF (long tails, etc.) the bin-crossing FWHM is
    reported alongside.
  - ``measure_triple_source`` reports the position error twice (component-wise
    and as a Euclidean distance); they differ by sqrt(2), so always state
    which one is quoted.
  - The peak-to-valley contrast in ``measure_dual_source`` is a heuristic:
    "peak" is the mean of the first/last 10% of the profile that joins the
    two fitted peaks, "valley" is the minimum of its central 20%.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
#  Basic utilities
# ---------------------------------------------------------------------------

def _gaussian_1d(x, amp, mu, sigma, bg):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg


def _gaussian_2d(xy, amp, mu_x, mu_y, sigma_x, sigma_y, theta, bg):
    x, y = xy
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    return bg + amp * np.exp(-(a*(x-mu_x)**2 + 2*b*(x-mu_x)*(y-mu_y)
                                + c*(y-mu_y)**2))


def _sigma_to_fwhm(s): return 2.3548200450309493 * s


def image_axes(image_2d, fov_mm):
    """Bin-centre coordinates (mm) of a square image with the given FOV."""
    if image_2d.shape[0] != image_2d.shape[1]:
        raise ValueError(
            f"quality_metrics assumes a square image; got {image_2d.shape}"
        )
    N = image_2d.shape[0]
    mm = np.linspace(-fov_mm/2 + (fov_mm/N)/2,
                      fov_mm/2 - (fov_mm/N)/2, N)
    return mm


def fwhm_by_bins(values, axis_mm, baseline="min"):
    """FWHM by linear interpolation at the half-maximum crossings.

    This is the single FWHM estimator of the package; :mod:`Gimntools.GimnRec.plots`
    calls it too, so a profile measured in a figure and the same profile
    measured in a report give identical numbers.

    Parameters
    ----------
    values : 1-D array
        Profile samples.
    axis_mm : 1-D array
        Coordinate of each sample, in mm. Must be BIN CENTRES (see
        :func:`image_axes`); using bin edges stretches the result by
        ``N/(N-1)``.
    baseline : {'min', 'zero'}
        Level from which the half maximum is taken.

        ``'min'`` (default) uses ``(max + min)/2``, i.e. it treats the
        profile minimum as the background. This is the usual choice for a
        reconstructed profile sitting on a non-zero pedestal.

        ``'zero'`` uses ``max/2``, the textbook definition, valid only when
        the background is genuinely zero. It reproduces the pre-0.4.0
        behaviour of ``plots._compute_fwhm``; the two differ by roughly
        ``background/peak`` and must never be mixed inside one table.

    Returns
    -------
    (fwhm_mm, (left_mm, right_mm))
        ``(nan, (nan, nan))`` when no sample reaches the half maximum.
    """
    v = np.asarray(values, np.float64)
    axis_mm = np.asarray(axis_mm, np.float64)
    if v.size == 0 or not np.isfinite(v).any():
        return float('nan'), (float('nan'), float('nan'))
    if baseline == "min":
        hm = (v.max() + v.min()) / 2.0
    elif baseline == "zero":
        hm = v.max() / 2.0
    else:
        raise ValueError(f"baseline must be 'min' or 'zero', got {baseline!r}")

    above = v >= hm
    if not above.any():
        return float('nan'), (float('nan'), float('nan'))
    idx = np.where(above)[0]
    li, ri = idx[0], idx[-1]
    if li > 0:
        f = (hm - v[li-1]) / (v[li] - v[li-1] + 1e-12)
        lm = axis_mm[li-1] + f*(axis_mm[li] - axis_mm[li-1])
    else:
        lm = axis_mm[li]
    if ri < len(v)-1:
        f = (hm - v[ri]) / (v[ri+1] - v[ri] + 1e-12)
        rm = axis_mm[ri] + f*(axis_mm[ri+1] - axis_mm[ri])
    else:
        rm = axis_mm[ri]
    return abs(rm - lm), (lm, rm)


# ---------------------------------------------------------------------------
#  Single point source
# ---------------------------------------------------------------------------

def measure_point_source(image_2d, fov_mm, expected_xy=None, roi_mm=8.0):
    """
    Measure a single point source in the image.

    Returns
    -------
    dict with:
      'peak_xy_mm'       : (x, y) of the fitted maximum
      'fwhm_x_mm'        : FWHM along x (Gaussian fit)
      'fwhm_y_mm'        : FWHM along y (Gaussian fit)
      'fwhm_x_bins_mm'   : bin-crossing FWHM (no fit)
      'fwhm_y_bins_mm'   : bin-crossing FWHM (no fit)
      'sigma_fit_xy_mm'  : fit uncertainties of the peak position
      'fit_amplitude'    : fitted amplitude
      'fit_background'   : fitted background
      'fit_residual_rms' : RMS of the 2-D fit residual over the ROI
    """
    mm = image_axes(image_2d, fov_mm)
    N = image_2d.shape[0]
    pixel_mm = fov_mm / N

    # Locate the approximate maximum
    if expected_xy is None:
        iy, ix = np.unravel_index(np.argmax(image_2d), image_2d.shape)
    else:
        ix = int(np.argmin(np.abs(mm - expected_xy[0])))
        iy = int(np.argmin(np.abs(mm - expected_xy[1])))
    cx0, cy0 = mm[ix], mm[iy]

    # ROI around it
    r_pix = int(np.ceil(roi_mm / pixel_mm))
    ymin, ymax = max(0, iy - r_pix), min(N, iy + r_pix + 1)
    xmin, xmax = max(0, ix - r_pix), min(N, ix + r_pix + 1)
    roi = image_2d[ymin:ymax, xmin:xmax].astype(np.float64)
    mm_x = mm[xmin:xmax]; mm_y = mm[ymin:ymax]
    X, Y = np.meshgrid(mm_x, mm_y)

    # 2-D Gaussian fit
    amp0 = roi.max() - roi.min()
    bg0 = roi.min()
    p0 = [amp0, cx0, cy0, 1.5, 1.5, 0.0, bg0]
    try:
        popt, pcov = curve_fit(_gaussian_2d, (X.ravel(), Y.ravel()),
                                 roi.ravel(), p0=p0, maxfev=20000)
        amp, mu_x, mu_y, sx, sy, theta, bg = popt
        sx = abs(sx); sy = abs(sy)
        sigma_xy = np.sqrt(np.diag(pcov))
        # sx/sy parameterize the principal axes of a rotated Gaussian;
        # the X/Y line-profile widths use the diagonal of its PRECISION
        # matrix, not the marginal variances.
        ct, st = np.cos(theta), np.sin(theta)
        fwhm_x_fit = _sigma_to_fwhm(1.0 / np.sqrt(ct**2 / sx**2 + st**2 / sy**2))
        fwhm_y_fit = _sigma_to_fwhm(1.0 / np.sqrt(st**2 / sx**2 + ct**2 / sy**2))
        residual = roi.ravel() - _gaussian_2d((X.ravel(), Y.ravel()), *popt)
        rms = float(np.sqrt(np.mean(residual ** 2)))
    except Exception as e:
        amp = bg = float('nan'); mu_x = cx0; mu_y = cy0
        sx = sy = theta = float('nan')
        fwhm_x_fit = fwhm_y_fit = float('nan')
        sigma_xy = [float('nan')] * 7
        rms = float('nan')

    # Bin-crossing FWHM (profiles through the fitted peak)
    iy_peak = int(np.argmin(np.abs(mm - mu_y)))
    ix_peak = int(np.argmin(np.abs(mm - mu_x)))
    fwhm_x_bins, _ = fwhm_by_bins(image_2d[iy_peak, :], mm)
    fwhm_y_bins, _ = fwhm_by_bins(image_2d[:, ix_peak], mm)

    return {
        'peak_xy_mm':       (float(mu_x), float(mu_y)),
        'fwhm_x_mm':        float(fwhm_x_fit),
        'fwhm_y_mm':        float(fwhm_y_fit),
        'fwhm_principal_1_mm': float(_sigma_to_fwhm(sx)),
        'fwhm_principal_2_mm': float(_sigma_to_fwhm(sy)),
        'principal_angle_deg': float(np.rad2deg(theta) % 180),
        'fwhm_x_bins_mm':   float(fwhm_x_bins),
        'fwhm_y_bins_mm':   float(fwhm_y_bins),
        'sigma_fit_xy_mm':  (float(sigma_xy[1]), float(sigma_xy[2])),
        'fit_amplitude':    float(amp),
        'fit_background':   float(bg),
        'fit_residual_rms': float(rms),
        'expected_xy':      expected_xy,
    }


# ---------------------------------------------------------------------------
#  Two sources (e.g. dual-source phantom at +/-7.5 mm)
# ---------------------------------------------------------------------------

def measure_dual_source(image_2d, fov_mm, expected_pair_xy,
                         search_radius_mm=3.0):
    """
    For two sources expected at positions = [(x1,y1), (x2,y2)], measure:
      - effective position of each (local Gaussian fit)
      - FWHM of each source
      - measured vs expected separation
      - peak-to-valley contrast along the line joining the two peaks
        (heuristic; see module docstring)
    """
    mm = image_axes(image_2d, fov_mm)
    per_source = []
    for xy in expected_pair_xy:
        per_source.append(measure_point_source(
            image_2d, fov_mm, expected_xy=xy,
            roi_mm=max(search_radius_mm * 2, 4.0),
        ))

    # Separation
    (x1, y1) = per_source[0]['peak_xy_mm']
    (x2, y2) = per_source[1]['peak_xy_mm']
    sep_measured = float(np.hypot(x2 - x1, y2 - y1))
    sep_expected = float(np.hypot(expected_pair_xy[1][0] - expected_pair_xy[0][0],
                                    expected_pair_xy[1][1] - expected_pair_xy[0][1]))

    # Peak-to-valley contrast: profile along the line joining the sources
    n_samples = 200
    ts = np.linspace(0, 1, n_samples)
    xs = x1 + ts*(x2 - x1); ys = y1 + ts*(y2 - y1)
    # simple bilinear interpolation
    N = image_2d.shape[0]
    pixel_mm = fov_mm / N
    ix_f = (xs + fov_mm/2) / pixel_mm - 0.5
    iy_f = (ys + fov_mm/2) / pixel_mm - 0.5
    ix0 = np.clip(np.floor(ix_f).astype(int), 0, N-2)
    iy0 = np.clip(np.floor(iy_f).astype(int), 0, N-2)
    fx = ix_f - ix0; fy = iy_f - iy0
    prof = ((1-fx)*(1-fy) * image_2d[iy0, ix0] +
            (1-fx)*  fy  * image_2d[iy0+1, ix0] +
              fx *(1-fy) * image_2d[iy0, ix0+1] +
              fx *  fy  * image_2d[iy0+1, ix0+1])
    # peaks = first/last 10% of the profile; valley = central 20%
    peak_val = (prof[:n_samples//10].mean() + prof[-n_samples//10:].mean()) / 2
    valley_val = prof[2*n_samples//5 : 3*n_samples//5].min()
    if peak_val > 1e-12:
        contrast = float((peak_val - valley_val) / peak_val)
    else:
        contrast = float('nan')

    return {
        'sources': per_source,
        'separation_measured_mm': sep_measured,
        'separation_expected_mm': sep_expected,
        'separation_error_mm':    sep_measured - sep_expected,
        'profile_along_line':     prof,
        'profile_t':              ts,
        'peak_valley_contrast':   contrast,
    }


# ---------------------------------------------------------------------------
#  Three sources (capillaries)
# ---------------------------------------------------------------------------

def measure_triple_source(image_2d, fov_mm, expected_triplet_xy,
                            roi_mm=4.0):
    """
    For 3 capillaries at expected positions, measure each one locally.

    Returns two RMS position-error summaries, which differ by a factor
    sqrt(2) for isotropic errors and must not be confused:

    ``rms_position_error_component_mm``
        RMS over the 2*N individual x and y error components. This is what
        the historical key ``rms_position_error_mm`` has always contained.
    ``rms_position_error_distance_mm``
        RMS of the per-source Euclidean displacement
        ``sqrt(dx**2 + dy**2)``, which is what "RMS position error" usually
        means in a report.
    """
    per_source = [measure_point_source(image_2d, fov_mm, expected_xy=xy,
                                         roi_mm=roi_mm)
                  for xy in expected_triplet_xy]
    # Position errors
    pos_errors = []
    for ps, exp in zip(per_source, expected_triplet_xy):
        m = ps['peak_xy_mm']
        pos_errors.append((m[0] - exp[0], m[1] - exp[1]))
    pos_err_arr = np.asarray(pos_errors, dtype=np.float64)

    # Two different summaries, reported side by side because the name "RMS
    # position error" is ambiguous and they differ by sqrt(2) for isotropic
    # errors. State which one you quote.
    #   component : RMS over all 2*N individual x and y errors
    #   distance  : RMS of the per-source EUCLIDEAN displacement
    rms_component = float(np.sqrt(np.mean(pos_err_arr ** 2)))
    rms_distance = float(np.sqrt(np.mean((pos_err_arr ** 2).sum(axis=1))))

    return {
        'sources': per_source,
        'expected_xy': expected_triplet_xy,
        'position_errors_mm': pos_errors,
        # Kept under its historical key; it is the COMPONENT-wise RMS.
        'rms_position_error_mm': rms_component,
        'rms_position_error_component_mm': rms_component,
        'rms_position_error_distance_mm': rms_distance,
    }


# ---------------------------------------------------------------------------
#  Consolidated report
# ---------------------------------------------------------------------------

def print_metrics_report(metrics, label=""):
    """Print a human-readable report of the computed metrics."""
    print("\n" + "="*70)
    print(f"  QUALITY METRICS — {label}")
    print("="*70)

    if 'fwhm_x_mm' in metrics:  # single source
        m = metrics
        print(f"  Peak position    : ({m['peak_xy_mm'][0]:+.2f}, "
              f"{m['peak_xy_mm'][1]:+.2f}) mm")
        if m.get('expected_xy'):
            err = (m['peak_xy_mm'][0] - m['expected_xy'][0],
                   m['peak_xy_mm'][1] - m['expected_xy'][1])
            print(f"  Position error   : ({err[0]:+.2f}, {err[1]:+.2f}) mm")
        print(f"  FWHM_x (gauss)   : {m['fwhm_x_mm']:.2f} mm  "
              f"(bins: {m['fwhm_x_bins_mm']:.2f} mm)")
        print(f"  FWHM_y (gauss)   : {m['fwhm_y_mm']:.2f} mm  "
              f"(bins: {m['fwhm_y_bins_mm']:.2f} mm)")
        print(f"  Amplitude / BG   : {m['fit_amplitude']:.2f} / "
              f"{m['fit_background']:.2f}")
        print(f"  Fit residual RMS : {m['fit_residual_rms']:.3f}")

    elif 'sources' in metrics and len(metrics.get('sources', [])) == 2:
        for i, s in enumerate(metrics['sources']):
            print(f"\n  --- Source {i+1} (expected at "
                  f"{s['expected_xy']}) ---")
            print(f"    peak = ({s['peak_xy_mm'][0]:+.2f}, "
                  f"{s['peak_xy_mm'][1]:+.2f}) mm  "
                  f"FWHM=({s['fwhm_x_mm']:.2f},{s['fwhm_y_mm']:.2f}) mm")
        print(f"\n  Measured separation : {metrics['separation_measured_mm']:.2f} mm")
        print(f"  Expected separation : {metrics['separation_expected_mm']:.2f} mm")
        print(f"  Separation error    : {metrics['separation_error_mm']:+.2f} mm")
        print(f"  Peak-valley contrast: {metrics['peak_valley_contrast']:.3f} "
              "(1.0 = perfectly resolved sources)")

    elif 'sources' in metrics and len(metrics.get('sources', [])) == 3:
        for i, (s, e, err) in enumerate(zip(metrics['sources'],
                                              metrics['expected_xy'],
                                              metrics['position_errors_mm'])):
            print(f"\n  --- Capillary {i+1} (expected at ({e[0]:+.1f},{e[1]:+.1f})) ---")
            print(f"    peak = ({s['peak_xy_mm'][0]:+.2f}, "
                  f"{s['peak_xy_mm'][1]:+.2f}) mm")
            print(f"    error = ({err[0]:+.2f}, {err[1]:+.2f}) mm")
            print(f"    FWHM = ({s['fwhm_x_mm']:.2f}, {s['fwhm_y_mm']:.2f}) mm")
        print(f"\n  RMS position error (per component) : "
              f"{metrics['rms_position_error_component_mm']:.2f} mm")
        print(f"  RMS position error (distance)      : "
              f"{metrics['rms_position_error_distance_mm']:.2f} mm")


def plot_metric_overview(image_2d, fov_mm, metrics, title="",
                           save_path=None):
    """Image + profiles + overlay of measured/expected positions."""
    mm = image_axes(image_2d, fov_mm)
    ext = [-fov_mm/2, fov_mm/2, -fov_mm/2, fov_mm/2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].imshow(image_2d, cmap='gray', origin='lower', extent=ext)
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    axes[0].set_title(title)

    # mark positions
    if 'sources' in metrics:
        for s, exp in zip(metrics['sources'], metrics.get('expected_xy',
                          [s.get('expected_xy') for s in metrics['sources']])):
            if exp:
                axes[0].plot(exp[0], exp[1], 'g+', ms=14, mew=1.5,
                              label='expected' if s is metrics['sources'][0] else None)
            m = s['peak_xy_mm']
            axes[0].plot(m[0], m[1], 'rx', ms=10, mew=1.5,
                          label='measured' if s is metrics['sources'][0] else None)
        axes[0].legend(fontsize=9)
    elif 'peak_xy_mm' in metrics:
        if metrics.get('expected_xy'):
            e = metrics['expected_xy']
            axes[0].plot(e[0], e[1], 'g+', ms=14, mew=1.5, label='expected')
        m = metrics['peak_xy_mm']
        axes[0].plot(m[0], m[1], 'rx', ms=10, mew=1.5, label='measured')
        axes[0].legend(fontsize=9)

    # Profiles through the first peak
    if 'sources' in metrics:
        m0 = metrics['sources'][0]['peak_xy_mm']
    else:
        m0 = metrics['peak_xy_mm']
    ix = int(np.argmin(np.abs(mm - m0[0])))
    iy = int(np.argmin(np.abs(mm - m0[1])))
    axes[1].plot(mm, image_2d[iy, :], 'b-', lw=1.2)
    axes[1].axvline(m0[0], c='r', ls='--', lw=0.8)
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("Intensity")
    axes[1].set_title(f"Horizontal profile at y={m0[1]:.1f} mm")
    axes[1].grid(alpha=0.3)

    axes[2].plot(mm, image_2d[:, ix], 'g-', lw=1.2)
    axes[2].axvline(m0[1], c='r', ls='--', lw=0.8)
    axes[2].set_xlabel("y (mm)"); axes[2].set_ylabel("Intensity")
    axes[2].set_title(f"Vertical profile at x={m0[0]:.1f} mm")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
