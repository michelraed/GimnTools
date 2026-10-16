"""
calibration.py
==============

Joint geometric calibration for a 2-detector preclinical PET system.

Estimates the system's geometric parameters from acquisitions with sources
at known laboratory positions.

Parameters estimated jointly:
    dx_cor, dy_cor : offset of the rotation axis with respect to the
                     assumed geometric centre (mm)
    phi_offset     : angular offset of the gantry encoder (degrees)
    delta_s_i      : small corrections to the nominal source positions
                     (Gaussian prior of magnitude ``sigma_pos_prior``)

Radial-centroid model in the sinogram::

    c(theta) =  Sum_i I_i * [
                  (Sx_i + dSx_i) * cos(theta + phi_off)
                + (Sy_i + dSy_i) * sin(theta + phi_off)
                ] / Sum_i I_i
              + dx_cor * cos(theta + phi_off)
              + dy_cor * sin(theta + phi_off)
              + b_const

Identifiability (read this before quoting any number from this module)
---------------------------------------------------------------------
Expanding the model above gives

    c(theta) = (Sx_w + dx_cor) * cos(theta + phi_off)
             + (Sy_w + dy_cor) * sin(theta + phi_off) + b_const

so ``dx_cor`` enters only as a SUM with the source-position corrections
``dSx_i``, and likewise for y. For a single file the two are exactly
degenerate: nothing in the data separates "the axis moved by 1 mm" from "the
source was 1 mm off its nominal position". What breaks the degeneracy is

  * the SHARING structure -- ``dx_cor``/``dy_cor``/``phi_off`` are common to
    every file while each source keeps its own ``dSx_i``, so at least two
    files with DIFFERENT nominal source positions are required, and
  * the Gaussian priors (``sigma_cor_prior``, ``sigma_pos_prior``,
    ``sigma_phi_prior_deg``), which are what pins the solution down when the
    data are uninformative.

``phi_offset`` is the weakest parameter of the set: with the source
corrections free, a rotation of the fitted ``(A, B)`` amplitudes can absorb
most of it. The practical test is to compare the reported sigma against the
prior. In the calibration shipped in ``examples/CONFIG_PETSYS``,
``phi_offset = -0.212 +/- 2.166 deg`` against a prior of 2.0 deg: the
posterior is essentially the prior, the encoder offset is NOT constrained by
those acquisitions, and the value is consistent with zero. Report it that
way rather than as a measurement.

The uncertainties returned in ``sigma_*`` come from ``(J^T J)^-1`` scaled by
the reduced chi-square of the FULL residual vector, prior rows included. They
are therefore posterior (MAP) uncertainties, not likelihood-only ones.

Acknowledged limitations (to be stated in any publication):
  - Out-of-plane detector tilt (rotation about axes parallel to the face)
    is not modelled. It would require additional acquisitions.
  - ``dx_cor``/``dy_cor`` and the source corrections are degenerate up to the
    priors, as described above.
  - Differential radial miscalibration between the two detectors
    (radius_1 != radius_2) is not separated. It would require eccentric
    sources along multiple orthogonal directions.
  - The centroid model weights sources by intensity; when intensities are
    unknown (capillaries), equal intensities are assumed.

Typical usage:
    >>> specs = [
    ...   {'path': '...Source_Centre.root', 'sources': [(0,0)], 'I': [1.0],
    ...    'label': 'C', 'n_gantry_angles': 12},
    ...   {'path': '...Source_OFF.root', 'sources': [(7.5,0)], 'I': [1.0],
    ...    'label': '+x', 'n_gantry_angles': 12},
    ...   {'path': '...Source_OFF_2.root', 'sources': [(-7.5,0)], 'I': [1.0],
    ...    'label': '-x', 'n_gantry_angles': 24},
    ...   {'path': '...TwoSources_OFF.root', 'sources': [(7.5,0),(-7.5,0)],
    ...    'I': [1.0, 1.0], 'label': '+/-x', 'n_gantry_angles': 12},
    ... ]
    >>> params, report = joint_calibration(specs, bg_file, detector_config,
    ...                                    detectors_angles, parameters, rsec)
    >>> save_calibration(params, 'calibration.json')
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
#  Model: predicted centroid as a function of the parameters
# ---------------------------------------------------------------------------

def predict_centroid(angles_rad, sources_xy, intensities,
                     dx_cor, dy_cor, phi_offset_rad, b_const,
                     delta_sources_xy=None):
    """
    Radial centroid predicted by the geometric model, for a set of sources
    with given intensities.

    Parameters
    ----------
    angles_rad : (N,) array
        Gantry angles (radians), as reported by the encoder.
    sources_xy : list of (Sx, Sy)
        Nominal source positions in mm (LAB frame, origin = nominal COR).
    intensities : list of float
        Relative source intensities (centroid weighting).
    dx_cor, dy_cor : float
        Offset of the true rotation axis relative to the nominal centre.
    phi_offset_rad : float
        Encoder reference offset (radians).
    b_const : float
        Constant term (radial sinogram offset, usually small).
    delta_sources_xy : list of (dSx, dSy) or None
        Source-position corrections (mm). If None, zeros are assumed.

    Returns
    -------
    c_pred : (N,) array
        Predicted radial centroid (mm) at each angle.
    """
    if delta_sources_xy is None:
        delta_sources_xy = [(0.0, 0.0)] * len(sources_xy)

    I_total = sum(intensities) + 1e-12
    eff_angles = angles_rad + phi_offset_rad
    cos_a = np.cos(eff_angles); sin_a = np.sin(eff_angles)

    # Intensity-weighted source centroid
    Sx_w = sum(I * (s[0] + d[0]) for I, s, d in
               zip(intensities, sources_xy, delta_sources_xy)) / I_total
    Sy_w = sum(I * (s[1] + d[1]) for I, s, d in
               zip(intensities, sources_xy, delta_sources_xy)) / I_total

    return Sx_w * cos_a + Sy_w * sin_a + dx_cor * cos_a + dy_cor * sin_a + b_const


# ---------------------------------------------------------------------------
#  Extraction of the measured centroid per angle
# ---------------------------------------------------------------------------

def centroid_per_angle(sino_2d, distances):
    """Radial centroid (mm) of each angular column of a 2-D sinogram."""
    n_dist, n_ang = sino_2d.shape
    out = np.full(n_ang, np.nan, dtype=np.float64)
    for th in range(n_ang):
        col = sino_2d[:, th].astype(np.float64)
        s = col.sum()
        if s > 1e-9:
            out[th] = (distances * col).sum() / s
    return out


# ---------------------------------------------------------------------------
#  Building the calibration sinograms
#  (depends on pipeline functions — passed in as callables)
# ---------------------------------------------------------------------------

def build_calibration_sinogram(file_path, bg_file, detector_config,
                                detectors_angles, parameters, rsectors,
                                coincidence_to_lor_fn,
                                interpolate_distance_fn, smooth_angular_fn,
                                n_dist_out=128, angular_sigma=0.5,
                                n_slices_avg=3):
    """
    Read a calibration file, subtract background, interpolate and smooth
    through the SAME pipeline as the measurement data. Returns the 2-D
    sinogram summed over the central slices, plus its axes.
    """
    sino_src, _ = coincidence_to_lor_fn(file_path, detector_config,
                                         detectors_angles, parameters,
                                         rsectors=rsectors)
    if bg_file and os.path.exists(bg_file):
        sino_bg, _ = coincidence_to_lor_fn(bg_file, detector_config,
                                            detectors_angles, parameters,
                                            rsectors=rsectors)
        sino_raw = sino_src - sino_bg
    else:
        sino_raw = sino_src

    sino_i = interpolate_distance_fn(
        sino_raw, n_out=n_dist_out, kind='conservative')
    sino_s = smooth_angular_fn(sino_i, sigma_bins=angular_sigma)

    n_sl = sino_s.shape[0]
    z0 = max(0, n_sl // 2 - n_slices_avg // 2)
    z1 = min(n_sl, z0 + n_slices_avg)
    sino_2d = sino_s[z0:z1].sum(axis=0)

    sino_range_mm = detector_config.get("sino_range_mm",
                                         detector_config["crystal_size"][1])
    sino_pixel = sino_range_mm / n_dist_out
    distances = np.linspace(-sino_range_mm/2 + sino_pixel/2,
                             sino_range_mm/2 - sino_pixel/2,
                             n_dist_out)

    n_ang = sino_s.shape[2]
    angles_deg = (np.arange(n_ang) + 0.5) * (180.0 / n_ang)

    return sino_2d, distances, angles_deg, (z0, z1 - 1)


# ---------------------------------------------------------------------------
#  Joint fit over multiple files
# ---------------------------------------------------------------------------

def joint_calibration(file_specs, bg_file, detector_config,
                       detectors_angles, parameters, rsectors,
                       coincidence_to_lor_fn,
                       interpolate_distance_fn, smooth_angular_fn,
                       n_dist_out=128, angular_sigma=0.5,
                       sigma_pos_prior=1.0,        # mm
                       fit_source_corrections=True,
                       sigma_phi_prior_deg=2.0,    # degrees
                       sigma_cor_prior=2.0,        # mm
                       verbose=True):
    """
    Joint fit of the geometric parameters over multiple calibration files.

    Parameters
    ----------
    file_specs : list of dict
        Each dict must contain:
            'path'           : path to the .root file
            'sources'        : list of (Sx, Sy) in mm (nominal positions)
            'I'              : list of relative intensities
            'label'          : identification string
            'n_gantry_angles': 12 or 24 (informative; the sinogram already
                               holds all the angular columns)
    sigma_pos_prior : float
        Magnitude of the Gaussian prior on the source-position corrections
        (mm). Reflects the mechanical positioning uncertainty.
    sigma_phi_prior_deg : float
        Magnitude of the prior on the encoder offset (degrees).
    sigma_cor_prior : float
        Magnitude of the prior on the COR offset (mm).
    fit_source_corrections : bool
        If True, also fit small corrections (dSx_i, dSy_i) for each source
        within the prior. If False, the nominal positions are fixed.

    Returns
    -------
    params : dict
        Estimated parameters + uncertainties + metadata.
    report : dict
        Observed/predicted/residual data per file, for plotting and audit.
    """
    # 1) Build all the sinograms
    obs_per_file = []
    for spec in file_specs:
        if not os.path.exists(spec['path']):
            print(f"[SKIP] {spec['label']}: file not found")
            continue
        if verbose:
            print(f"  Reading {spec['label']}: {os.path.basename(spec['path'])}")
        sino_2d, dist, ang_deg, sl_range = build_calibration_sinogram(
            spec['path'], bg_file, detector_config, detectors_angles,
            parameters, rsectors,
            coincidence_to_lor_fn, interpolate_distance_fn,
            smooth_angular_fn,
            n_dist_out=n_dist_out, angular_sigma=angular_sigma,
        )
        cents = centroid_per_angle(sino_2d, dist)
        valid = np.isfinite(cents)
        obs_per_file.append({
            'spec': spec,
            'angles_deg': ang_deg,
            'angles_rad': np.deg2rad(ang_deg),
            'centroids': cents,
            'valid': valid,
            'slice_range': sl_range,
            'sino_2d_shape': sino_2d.shape,
            'n_valid': int(valid.sum()),
        })
        if verbose:
            print(f"     {valid.sum()}/{len(cents)} valid angles")

    if not obs_per_file:
        raise RuntimeError("No readable calibration file.")

    # Diagnostic: total number of valid centroids. If ~0, the sinograms
    # are empty and the fit would consist only of the priors (a meaningless
    # result). The typical PETSYS cause is the rsectors map (chipID ->
    # detector) not matching the real chipIDs in the file (the "0..N-1"
    # generator is GATE-style, not PETSYS), the [e_min, e_max] energy window
    # rejecting everything, or a geometry wrong for the hardware. We raise an
    # actionable message instead of fitting silently and misleadingly.
    total_valid = sum(o['n_valid'] for o in obs_per_file)
    if total_valid == 0:
        raise RuntimeError(
            "Calibration aborted: 0 angles with a valid centroid across all "
            "files — the sinograms came out empty. Check, in this order:\n"
            " 1) rsectors: the KEYS must be the real chipIDs of the PETSYS "
            "file (not 0,1,...). The 'Generate (0-360)' button is for GATE; "
            "for PETSYS load your own chipID->detector map.\n"
            " 2) the e_min/e_max energy window in detector_config (units!).\n"
            " 3) a geometry (radius, crystal_size, Half_FOV) that matches "
            "your PETSYS hardware, not the GATE defaults."
        )

    # 2) Parameterization of the optimization vector
    # x = [dx_cor, dy_cor, phi_off_deg, b_const, dSx_0, dSy_0, dSx_1, dSy_1, ...]
    n_sources_total = sum(len(o['spec']['sources']) for o in obs_per_file)

    x0 = [0.0, 0.0, 0.0, 0.0]
    if fit_source_corrections:
        x0 += [0.0] * (2 * n_sources_total)

    # Mapping: for each file, the indices of its deltas in the x vector
    src_idx_start = []
    cursor = 4
    for o in obs_per_file:
        n_src = len(o['spec']['sources'])
        if fit_source_corrections:
            src_idx_start.append(cursor)
            cursor += 2 * n_src
        else:
            src_idx_start.append(None)

    # 3) Residual function (priors act as regularization terms)
    def residuals(x):
        dx_cor, dy_cor = x[0], x[1]
        phi_off_rad = np.deg2rad(x[2])
        b_const = x[3]

        res = []
        for o, src_start in zip(obs_per_file, src_idx_start):
            spec = o['spec']
            sources = spec['sources']
            I = spec['I']

            if src_start is not None:
                n_src = len(sources)
                deltas = [(x[src_start + 2*i], x[src_start + 2*i + 1])
                          for i in range(n_src)]
            else:
                deltas = None

            c_pred = predict_centroid(
                o['angles_rad'], sources, I,
                dx_cor, dy_cor, phi_off_rad, b_const,
                delta_sources_xy=deltas,
            )
            r = (o['centroids'] - c_pred)
            res.extend(r[o['valid']].tolist())

        # Priors (Tikhonov-style)
        # dx_cor, dy_cor
        res.append(x[0] / sigma_cor_prior)
        res.append(x[1] / sigma_cor_prior)
        # phi_off
        res.append(x[2] / sigma_phi_prior_deg)
        # b_const (no strong prior; left free)
        # source deltas
        if fit_source_corrections:
            for i in range(4, len(x)):
                res.append(x[i] / sigma_pos_prior)

        return np.asarray(res)

    # 4) Optimization
    # GimnToolsV2 FIX: use 'trf' (trust-region) instead of 'lm'. MINPACK's
    # Levenberg-Marquardt REQUIRES n_residuals >= n_variables and crashes
    # otherwise. Since b_const has no prior, with sparse data (few angles
    # with a valid centroid) the residual count can drop below the number of
    # variables. 'trf' solves the regularized (Tikhonov-prior) problem even
    # when under-determined; for well-determined problems it reaches
    # essentially the same minimum.
    sol = least_squares(residuals, x0, method='trf', max_nfev=10000)
    x_hat = sol.x

    # 5) Uncertainty estimate via the Gaussian approximation (J^T J)^-1
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * (sol.cost * 2.0 /
                                          max(len(sol.fun) - len(x_hat), 1))
        sigmas = np.sqrt(np.diag(cov))
    except Exception:
        sigmas = np.full_like(x_hat, np.nan)

    # 6) Decode the estimated parameters
    params = {
        'dx_cor_mm':     float(x_hat[0]),
        'dy_cor_mm':     float(x_hat[1]),
        'phi_offset_deg': float(x_hat[2]),
        'b_const_mm':    float(x_hat[3]),
        'sigma_dx_cor':  float(sigmas[0]),
        'sigma_dy_cor':  float(sigmas[1]),
        'sigma_phi':     float(sigmas[2]),
        'sigma_b':       float(sigmas[3]),
        'source_corrections': [],
        'priors': {
            'sigma_pos_mm':  sigma_pos_prior,
            'sigma_cor_mm':  sigma_cor_prior,
            'sigma_phi_deg': sigma_phi_prior_deg,
        },
        'fit_meta': {
            'cost': float(sol.cost),
            'status': int(sol.status),
            'n_obs_total': int(sum(o['n_valid'] for o in obs_per_file)),
            'n_params': int(len(x_hat)),
            'message': sol.message,
        },
    }

    if fit_source_corrections:
        cursor = 4
        for spec_idx, o in enumerate(obs_per_file):
            spec = o['spec']
            n_src = len(spec['sources'])
            corr_list = []
            for i in range(n_src):
                idx_x = cursor + 2*i
                idx_y = cursor + 2*i + 1
                corr_list.append({
                    'source_idx': i,
                    'nominal': spec['sources'][i],
                    'delta_xy_mm': (float(x_hat[idx_x]), float(x_hat[idx_y])),
                    'sigma_xy_mm': (float(sigmas[idx_x]), float(sigmas[idx_y])),
                    'effective_xy': (
                        float(spec['sources'][i][0] + x_hat[idx_x]),
                        float(spec['sources'][i][1] + x_hat[idx_y]),
                    ),
                })
            cursor += 2 * n_src
            params['source_corrections'].append({
                'label': spec['label'],
                'corrections': corr_list,
            })

    # 7) Build the report with predictions and residuals
    report = {'files': []}
    for o, src_start in zip(obs_per_file, src_idx_start):
        if src_start is not None:
            n_src = len(o['spec']['sources'])
            deltas = [(x_hat[src_start + 2*i], x_hat[src_start + 2*i + 1])
                      for i in range(n_src)]
        else:
            deltas = None
        c_pred = predict_centroid(
            o['angles_rad'], o['spec']['sources'], o['spec']['I'],
            x_hat[0], x_hat[1], np.deg2rad(x_hat[2]), x_hat[3],
            delta_sources_xy=deltas,
        )
        residual = o['centroids'] - c_pred
        rms = float(np.sqrt(np.nanmean(residual[o['valid']] ** 2)))
        report['files'].append({
            'label': o['spec']['label'],
            'angles_deg': o['angles_deg'],
            'centroids_obs': o['centroids'],
            'centroids_pred': c_pred,
            'residual': residual,
            'valid': o['valid'],
            'rms_residual_mm': rms,
        })

    if verbose:
        print("\n" + "="*70)
        print("  JOINT CALIBRATION RESULT")
        print("="*70)
        print(f"  dx_cor      = {params['dx_cor_mm']:+.3f} ± "
              f"{params['sigma_dx_cor']:.3f} mm")
        print(f"  dy_cor      = {params['dy_cor_mm']:+.3f} ± "
              f"{params['sigma_dy_cor']:.3f} mm")
        print(f"  phi_offset  = {params['phi_offset_deg']:+.3f} ± "
              f"{params['sigma_phi']:.3f} deg")
        print(f"  b_const     = {params['b_const_mm']:+.3f} ± "
              f"{params['sigma_b']:.3f} mm")
        print(f"  N obs       = {params['fit_meta']['n_obs_total']}, "
              f"N params = {params['fit_meta']['n_params']}")
        print()
        for fi in report['files']:
            print(f"  [{fi['label']:12s}] residual RMS = "
                  f"{fi['rms_residual_mm']:.3f} mm")
        if fit_source_corrections:
            print("\n  Source-position corrections (deltas):")
            for sc in params['source_corrections']:
                print(f"  --- {sc['label']} ---")
                for c in sc['corrections']:
                    nom = c['nominal']; d = c['delta_xy_mm']
                    eff = c['effective_xy']; sg = c['sigma_xy_mm']
                    print(f"    nominal=({nom[0]:+.2f},{nom[1]:+.2f}) -> "
                          f"delta=({d[0]:+.2f},{d[1]:+.2f}) +/- "
                          f"({sg[0]:.2f},{sg[1]:.2f})  "
                          f"effective=({eff[0]:+.2f},{eff[1]:+.2f})")

    return params, report


# ---------------------------------------------------------------------------
#  Report visualization
# ---------------------------------------------------------------------------

def plot_calibration_report(report, save_path=None):
    """Plot observed/predicted centroids and residuals per file."""
    n = len(report['files'])
    fig, axes = plt.subplots(2, n, figsize=(5*n, 7), squeeze=False)
    for j, fi in enumerate(report['files']):
        ax_top = axes[0, j]; ax_bot = axes[1, j]
        ang = fi['angles_deg']
        ax_top.plot(ang, fi['centroids_obs'], 'b.', ms=3, label='Observed')
        ax_top.plot(ang, fi['centroids_pred'], 'r-', lw=1.5, label='Predicted')
        ax_top.set_xlabel("Angle (deg)"); ax_top.set_ylabel("Radial centroid (mm)")
        ax_top.set_title(f"{fi['label']}")
        ax_top.grid(alpha=0.3); ax_top.legend(fontsize=8)

        ax_bot.plot(ang, fi['residual'], 'k.', ms=3)
        ax_bot.axhline(0, c='r', ls='--', lw=0.8)
        ax_bot.set_xlabel("Angle (deg)"); ax_bot.set_ylabel("Residual (mm)")
        ax_bot.set_title(f"RMS = {fi['rms_residual_mm']:.3f} mm")
        ax_bot.grid(alpha=0.3)
    fig.suptitle("Joint calibration — observed vs model",
                  fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

def save_calibration(params, path):
    """Save parameters as JSON. Can be loaded later to apply the
    corrections in the reconstruction pipeline."""
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, default=lambda o: float(o)
                   if hasattr(o, 'item') else str(o))
    print(f"[calibration] saved: {path}")


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  Characterization of the flood phantom (NOT used for normalization)
# ---------------------------------------------------------------------------

def characterize_flood_phantom(flood_file, detector_config, detectors_angles,
                                 parameters, rsectors,
                                 coincidence_to_lor_fn,
                                 interpolate_distance_fn, smooth_angular_fn,
                                 n_dist_out=128, angular_sigma=0.5,
                                 save_path=None):
    """
    Characterize the phantom used as flood, WITHOUT using it as a
    normalization source.

    Reports:
      - Effective centre-of-mass position of the phantom
      - Tilt indication (angular variation of the centroid vs an ideal
        sinusoid)
      - Effective-size estimate (FWHM of the radial distribution)

    It is documented here that this phantom is not appropriate for
    detector-to-detector normalization (asymmetry + off-centring + coupled
    rotation).
    """
    print("="*70)
    print("  FLOOD-PHANTOM CHARACTERIZATION (diagnostic, NOT normalization)")
    print("="*70)

    sino_2d, dist, ang_deg, sl_range = build_calibration_sinogram(
        flood_file, None, detector_config, detectors_angles,
        parameters, rsectors,
        coincidence_to_lor_fn, interpolate_distance_fn, smooth_angular_fn,
        n_dist_out=n_dist_out, angular_sigma=angular_sigma,
    )

    cents = centroid_per_angle(sino_2d, dist)
    valid = np.isfinite(cents)
    ang_rad = np.deg2rad(ang_deg)

    # Simple sinusoid fit to estimate (cx, cy) and b
    M = np.column_stack([
        np.cos(ang_rad[valid]),
        np.sin(ang_rad[valid]),
        np.ones(valid.sum()),
    ])
    sol, *_ = np.linalg.lstsq(M, cents[valid], rcond=None)
    cx_eff, cy_eff, b = float(sol[0]), float(sol[1]), float(sol[2])

    fit = cx_eff*np.cos(ang_rad) + cy_eff*np.sin(ang_rad) + b
    residual = cents - fit
    rms = float(np.sqrt(np.nanmean(residual[valid] ** 2)))

    # Effective-size estimate: typical width of the radial distribution
    # at each angle, from the second moment
    widths = []
    for th in range(sino_2d.shape[1]):
        col = sino_2d[:, th].astype(np.float64)
        s = col.sum()
        if s > 1e-9:
            mean_r = (dist * col).sum() / s
            var_r  = (col * (dist - mean_r)**2).sum() / s
            if var_r > 0:
                widths.append(2.355 * np.sqrt(var_r))  # Gaussian-equivalent FWHM
    fwhm_typ = float(np.median(widths)) if widths else float('nan')

    print(f"\n  Effective centre of mass: ({cx_eff:+.2f}, {cy_eff:+.2f}) mm")
    print(f"  Constant offset b:        {b:+.3f} mm")
    print(f"  Sinusoid residual RMS:    {rms:.3f} mm")
    print(f"  Typical radial width:     {fwhm_typ:.2f} mm (eq. FWHM)")
    print()
    print("  CONCLUSIONS:")
    print("  - If (cx, cy) != (0, 0), the phantom was off-centred with respect to the COR.")
    print("  - If the residual RMS is large, the phantom shape leaves a")
    print("    non-sinusoidal geometric signature in the sinogram (expected for a square/cube).")
    print("  - This phantom is NOT appropriate for per-bin (r, theta) normalization;")
    print("    that requires an acquisition with a uniform cylindrical source,")
    print("    centred and STATIC with respect to the laboratory.")

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sino_range_mm = detector_config.get("sino_range_mm",
                                              detector_config["crystal_size"][1])
        axes[0].imshow(sino_2d, aspect='auto', origin='lower', cmap='hot',
                        extent=[0, 180, -sino_range_mm/2, sino_range_mm/2])
        axes[0].set_xlabel("Angle (deg)"); axes[0].set_ylabel("r (mm)")
        axes[0].set_title("Phantom sinogram (central slice)")
        axes[1].plot(ang_deg, cents, 'b.', ms=3, label='Measured centroid')
        axes[1].plot(ang_deg, fit, 'r-', lw=1.5,
                      label=f"Sinusoid: ({cx_eff:+.2f},{cy_eff:+.2f})")
        axes[1].set_xlabel("Angle (deg)"); axes[1].set_ylabel("Radial centroid (mm)")
        axes[1].set_title("Centroid vs angle"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(ang_deg, residual, 'k.', ms=3)
        axes[2].axhline(0, c='r', ls='--', lw=0.8)
        axes[2].set_xlabel("Angle (deg)"); axes[2].set_ylabel("Residual (mm)")
        axes[2].set_title(f"Residual (RMS={rms:.2f} mm)"); axes[2].grid(alpha=0.3)
        fig.suptitle("Characterization of the phantom used as flood", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return {
        'cx_effective_mm': cx_eff,
        'cy_effective_mm': cy_eff,
        'b_const_mm': b,
        'rms_sinusoidal_residual_mm': rms,
        'fwhm_typical_radial_mm': fwhm_typ,
    }
