"""Centre-of-rotation (COR) analysis and sinogram correction.

Model: for a point source at (Sx, Sy) the radial centroid of the sinogram
at each angle theta is a sinusoid::

    c(theta) = Sx*cos(theta) + Sy*sin(theta)

A displacement (dx, dy) of the true rotation axis (COR) relative to the
geometric centre adds an *identical* sinusoid::

    c(theta) = (Sx+dx)*cos(theta) + (Sy+dy)*sin(theta) + b

COR and source position are therefore **degenerate**: the COR can only be
isolated if the expected source position is known. The ideal case is a
CENTRED source (or an activity distribution symmetric about the origin),
where Sx = Sy = 0 and any sinusoid left in the centroids is COR.

.. note::
   This module was previously shipped **without its imports** (it was split
   out of a monolithic script) and raised ``NameError`` at call time. The
   imports below fix that; the numerical behaviour is unchanged.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from Gimntools.GimnRec.utils import coincidence_to_lor
from Gimntools.GimnRec.filters import interpolate_distance, smooth_angular

def cor_centroid_per_angle(sino_2d, distances):
    """Radial centroid (mm) of each angular column of a 2-D sinogram.

    sino_2d:   (n_dist, n_ang) sinogram of one slice
    distances: (n_dist,) r coordinates (bin centres, mm)
    Returns:   (n_ang,) array, NaN where the column is empty.
    """
    n_dist, n_ang = sino_2d.shape
    out = np.full(n_ang, np.nan, dtype=np.float64)
    for th in range(n_ang):
        col = sino_2d[:, th].astype(np.float64)
        s = col.sum()
        if s > 1e-9:
            out[th] = (distances * col).sum() / s
    return out


def estimate_cor(sino_3d, distances, angles_deg,
                 expected_source_xy=(0.0, 0.0),
                 slice_idx=None, n_slices_avg=3,
                 plot=True, name=""):
    """Estimate the centre-of-rotation offset by fitting a sinusoid to
    the radial centroids of the sinogram.

    expected_source_xy : (Sx, Sy) expected source position (mm). Use (0, 0)
                         for a centred source or a distribution symmetric
                         about the origin.
    slice_idx          : slice used (None -> middle of the volume).
    n_slices_avg       : number of slices summed around slice_idx.
    plot               : if True, show a diagnostic figure.

    Returns: dx_mm, dy_mm, diagnostics_dict
    """
    n_slices, n_dist, n_ang = sino_3d.shape
    angles_rad = np.deg2rad(angles_deg)
    Sx, Sy = expected_source_xy

    if slice_idx is None:
        slice_idx = n_slices // 2
    z_lo = max(0, slice_idx - n_slices_avg // 2)
    z_hi = min(n_slices, z_lo + n_slices_avg)
    sino_2d = sino_3d[z_lo:z_hi].sum(axis=0)

    cents = cor_centroid_per_angle(sino_2d, distances)
    valid = np.isfinite(cents)
    if valid.sum() < 3:
        print("[COR] insufficient data — returning (0, 0)")
        return 0.0, 0.0, {'centroids': cents, 'fit': None}

    expected = Sx * np.cos(angles_rad) + Sy * np.sin(angles_rad)
    cents_sub = cents - expected

    M = np.column_stack([
        np.cos(angles_rad[valid]),
        np.sin(angles_rad[valid]),
        np.ones(valid.sum()),
    ])
    sol, *_ = np.linalg.lstsq(M, cents_sub[valid], rcond=None)
    dx, dy, b = float(sol[0]), float(sol[1]), float(sol[2])

    fit_curve = dx * np.cos(angles_rad) + dy * np.sin(angles_rad) + b
    residual = cents_sub - fit_curve
    rms = float(np.sqrt(np.nanmean(residual ** 2)))

    print(f"[COR] {name or 'sinogram'}: dx={dx:+.3f} mm, dy={dy:+.3f} mm, "
          f"b={b:+.3f} mm  (residual RMS={rms:.3f} mm, slices {z_lo}..{z_hi-1})")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(angles_deg, cents, 'b.', ms=3, label='Measured centroid')
        axes[0].plot(angles_deg, expected, 'g--', lw=1.0,
                     label=f'Expected source ({Sx:.2f}, {Sy:.2f}) mm')
        axes[0].plot(angles_deg, expected + fit_curve, 'r-', lw=1.4,
                     label=f'Total fit (dx={dx:+.2f}, dy={dy:+.2f})')
        axes[0].set_xlabel('Angle (deg)'); axes[0].set_ylabel('Radial centroid (mm)')
        axes[0].set_title('Radial centroid vs angle')
        axes[0].legend(loc='best', fontsize=9); axes[0].grid(True, alpha=0.3)

        axes[1].plot(angles_deg, residual, 'k.', ms=3)
        axes[1].axhline(0, c='r', ls='--', lw=0.8)
        axes[1].set_xlabel('Angle (deg)'); axes[1].set_ylabel('Residual (mm)')
        axes[1].set_title(f'Fit residual (RMS={rms:.3f} mm)')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'COR analysis — {name}', fontsize=12)
        plt.tight_layout(); plt.show()

    return dx, dy, {
        'centroids': cents, 'expected': expected, 'fit_curve': fit_curve,
        'residual': residual, 'dx': dx, 'dy': dy, 'b': b, 'rms': rms,
        'angles_deg': np.asarray(angles_deg),
        'slice_range': (z_lo, z_hi - 1),
    }


def estimate_cor_offset(sino_3d, distances, angles_deg=None,
                        slice_idx=None, n_slices_avg=3, min_fraction=0.0):
    """Estimate the CONSTANT radial offset of the rotation axis (mm).

    Model
    -----
    A misaligned rotation axis shifts every angular column of a
    parallel-beam sinogram by the SAME amount ``d``. The radial centroid of
    column theta is then::

        c(theta) = d + Sx*cos(theta) + Sy*sin(theta)

    so ``d`` is the mean of the centroids over a full 180-degree set, and the
    remaining sinusoid is the object's own off-centre position (Sx, Sy).

    This is the quantity consumed by ``siddon_reconstructor(cor_mm=...)``.
    It is NOT the same as ``estimate_cor``, which fits (dx, dy) — an
    angle-dependent sinusoidal shift — and is corrected by resampling the
    sinogram with ``apply_cor``.

    Limitations (must be respected before trusting the number)
    ----------------------------------------------------------
    * The estimate is unbiased only if the activity distribution is
      symmetric about the rotation axis, or if the angles cover 180 degrees
      uniformly so the sinusoidal term averages to zero. With few or
      unevenly spaced angles the sinusoid does NOT cancel and leaks into d.
      The returned diagnostics report the fitted sinusoid amplitude; if it
      is comparable to d, a constant offset is not the right model.
    * Background, scatter and truncation bias any centroid estimator. Use
      ``min_fraction`` to ignore columns with too few counts.
    * The estimate is degenerate with a true lateral displacement of the
      object: a centred phantom is required to separate them.

    Parameters
    ----------
    sino_3d : (n_slices, n_dist, n_ang)
    distances : (n_dist,) radial bin centres in mm, WITHOUT any COR applied.
    angles_deg : (n_ang,) used only for the sinusoid diagnostic.
    min_fraction : columns whose counts are below this fraction of the
        maximum column are ignored (0 = keep all non-empty columns).

    Returns
    -------
    d_mm : float
    info : dict with 'centroids', 'n_used', 'std_mm', 'sinusoid_amplitude_mm'
    """
    n_slices, n_dist, n_ang = sino_3d.shape
    if slice_idx is None:
        slice_idx = n_slices // 2
    z_lo = max(0, slice_idx - n_slices_avg // 2)
    z_hi = min(n_slices, z_lo + n_slices_avg)
    sino_2d = np.asarray(sino_3d[z_lo:z_hi], dtype=np.float64).sum(axis=0)

    col_sums = sino_2d.sum(axis=0)
    thr = max(1e-9, float(min_fraction) * float(col_sums.max(initial=0.0)))
    cents = cor_centroid_per_angle(sino_2d, distances)
    valid = np.isfinite(cents) & (col_sums > thr)

    if valid.sum() < 2:
        print("[COR] insufficient data — returning 0.0 mm")
        return 0.0, {'centroids': cents, 'n_used': int(valid.sum()),
                     'std_mm': float('nan'),
                     'sinusoid_amplitude_mm': float('nan')}

    d = float(np.mean(cents[valid]))
    std = float(np.std(cents[valid]))

    amp = float('nan')
    if angles_deg is not None:
        th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))[valid]
        M = np.column_stack([np.cos(th), np.sin(th), np.ones(th.size)])
        sol, *_ = np.linalg.lstsq(M, cents[valid], rcond=None)
        amp = float(np.hypot(sol[0], sol[1]))
        d = float(sol[2])   # constant term of the fit: cleaner than the mean
                            # when the angular sampling is not uniform

    print(f"[COR] constant offset d = {d:+.4f} mm "
          f"({valid.sum()}/{n_ang} columns, centroid sd = {std:.4f} mm, "
          f"object sinusoid amplitude = {amp:.4f} mm, "
          f"slices {z_lo}..{z_hi-1})")
    if np.isfinite(amp) and amp > 2.0 * abs(d) and amp > 1e-3:
        print("[COR] WARNING: the sinusoidal (object off-centre) term "
              "dominates the constant term. A constant cor_mm cannot "
              "correct that; see estimate_cor/apply_cor for the (dx, dy) "
              "model, or re-centre the phantom.")

    return d, {'centroids': cents, 'n_used': int(valid.sum()),
               'std_mm': std, 'sinusoid_amplitude_mm': amp,
               'slice_range': (z_lo, z_hi - 1)}


def apply_cor(sino_3d, dx_mm, dy_mm, distances, angles_deg):
    """Apply the COR correction by shifting each angular column radially
    by -(dx*cos(theta) + dy*sin(theta)) mm.

    Uses scipy.ndimage.shift with order=3 for fractional shifts.
    """
    from scipy.ndimage import shift as nd_shift
    n_slices, n_dist, n_ang = sino_3d.shape
    angles_rad = np.deg2rad(angles_deg)

    sino_pixel = (distances[-1] - distances[0]) / (n_dist - 1)
    shifts_mm  = -(dx_mm * np.cos(angles_rad) + dy_mm * np.sin(angles_rad))
    shifts_pix = shifts_mm / sino_pixel

    out = np.empty_like(sino_3d, dtype=np.float64)
    for z in range(n_slices):
        for th in range(n_ang):
            out[z, :, th] = nd_shift(
                sino_3d[z, :, th].astype(np.float64),
                shifts_pix[th],
                order=1, prefilter=False, mode='constant', cval=0.0,
            )
    return out


def analyze_cor_from_file(cal_file, bg_file, detector_config, detectors_angles,
                          parameters, rsectors,
                          n_dist_out=128, angular_sigma=0.5,
                          expected_source_xy=(0.0, 0.0),
                          show_plots=True):
    """Full COR calibration pipeline from a .root file.

    Reads the file (subtracting background if provided), builds the
    sinogram through the SAME path as ``process_file`` (interpolation +
    smoothing) and estimates (dx, dy).

    Returns: (dx_mm, dy_mm)
    """
    print("=" * 60)
    print("  COR CALIBRATION")
    print(f"  File: {os.path.basename(cal_file)}")
    print("=" * 60)

    sino_src, _ = coincidence_to_lor(cal_file, detector_config, detectors_angles,
                                      parameters, rsectors=rsectors)
    if bg_file is not None and os.path.exists(bg_file):
        sino_bg, _ = coincidence_to_lor(bg_file, detector_config, detectors_angles,
                                         parameters, rsectors=rsectors)
        sino_raw = sino_src - sino_bg
        print(f"  Background subtracted: {os.path.basename(bg_file)}")
    else:
        sino_raw = sino_src

    sino_interp = interpolate_distance(
        sino_raw, n_out=n_dist_out, kind='conservative')
    sino_smooth = smooth_angular(sino_interp, sigma_bins=angular_sigma)

    sino_range_mm = detector_config.get("sino_range_mm",
                                         detector_config["crystal_size"][1])
    sino_pixel = sino_range_mm / n_dist_out
    sino_distances = np.linspace(-sino_range_mm/2 + sino_pixel/2,
                                  sino_range_mm/2 - sino_pixel/2,
                                  n_dist_out)
    n_ang = sino_smooth.shape[2]
    angles_deg = (np.arange(n_ang) + 0.5) * (180.0 / n_ang)

    dx, dy, _ = estimate_cor(
        sino_smooth, sino_distances, angles_deg,
        expected_source_xy=expected_source_xy,
        plot=show_plots,
        name=os.path.basename(cal_file),
    )

    print(f"\n  >>> Calibrated COR: dx={dx:+.3f} mm, dy={dy:+.3f} mm <<<\n")
    return dx, dy
