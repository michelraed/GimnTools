"""
reconstruction_v2.py
====================

Refined reconstruction pipeline that incorporates the parameters estimated
by the joint calibration (calibration.joint_calibration).

Corrections/model:
  1) Centre of rotation (COR): compensated in the projector GEOMETRY.
  2) Encoder offset (phi_offset): modifies the projection angles, not counts.
  3) Raw prompt data and exposure-scaled additive background remain separate
     in Poisson MLEM/OSEM; linear FBP can use the background difference.
  4) Supports arbitrary positive gantry-angle count consistent with data.

Deliberately NOT done (documented limitations):
  - Out-of-plane detector tilt.
  - Per-crystal-pair efficiency correction (requires an adequate flood).
  - Attenuation or scatter correction.

Usage:
    >>> from Gimntools.GimnRec.calibration import load_calibration
    >>> cal = load_calibration('calibration.json')
    >>> res = process_file_v2(fname, bgname, cfg, det_angles, params, rsec,
    ...                        calibration=cal,
    ...                        coincidence_to_lor_fn=coincidence_to_lor,
    ...                        ...)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift


# ---------------------------------------------------------------------------
#  Applying the corrections to the sinogram
# ---------------------------------------------------------------------------

def apply_cor_shift(sino_3d, dx_mm, dy_mm, distances, angles_deg):
    """
    Apply the centre-of-rotation correction: an angle-dependent radial
    shift of -(dx*cos(theta) + dy*sin(theta)) mm.
    """
    n_slices, n_dist, n_ang = sino_3d.shape
    angles_rad = np.deg2rad(angles_deg)
    sino_pixel = (distances[-1] - distances[0]) / (n_dist - 1)
    shifts_mm = -(dx_mm * np.cos(angles_rad) + dy_mm * np.sin(angles_rad))
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


def apply_angular_offset(sino_3d, angles_deg, phi_offset_deg):
    """Resample a parallel-beam sinogram angularly with Radon parity.

    Across a 180-degree boundary p(r,theta+pi)=p(-r,theta). This function
    is a display/compatibility resampler; for Poisson MLEM/OSEM prefer
    modifying the physical angle array (angles_deg + phi_offset_deg), with
    no bin interpolation or artificial count correlations.
    """
    arr = np.asarray(sino_3d, dtype=np.float64)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError("sinogram must be finite (slice, radial, angle)")
    theta = np.asarray(angles_deg, dtype=float)
    if theta.shape != (arr.shape[2],) or not np.isfinite(theta).all():
        raise ValueError("angles_deg does not match sinogram")
    if not np.isfinite(phi_offset_deg):
        raise ValueError("phi_offset_deg must be finite")
    if abs(phi_offset_deg) < 1e-12:
        return arr.copy()
    # Uniform angular bins are required for this interpolation.
    n_ang = arr.shape[2]
    step = 180.0 / n_ang
    if not np.allclose(np.diff(theta), step, atol=1e-6, rtol=1e-6):
        raise ValueError("angular resampling requires uniformly spaced angles")
    shift_bins = phi_offset_deg / step
    out = np.empty_like(arr)
    for j in range(n_ang):
        u = j - shift_bins
        i = int(np.floor(u))
        f = u - i
        left = arr[:, ::-1, i % n_ang] if (i // n_ang) % 2 else arr[:, :, i % n_ang]
        right = (arr[:, ::-1, (i+1) % n_ang] if ((i+1) // n_ang) % 2
                 else arr[:, :, (i+1) % n_ang])
        out[:, :, j] = (1.0-f)*left + f*right
    return out


# ---------------------------------------------------------------------------
#  Main pipeline (replacement for the legacy process_file)
# ---------------------------------------------------------------------------

def process_file_v2(fname, bgname, cfg, det_angles, params, rsec,
                     calibration,
                     coincidence_to_lor_fn,
                     interpolate_distance_fn, smooth_angular_fn,
                     siddon_reconstructor_cls,
                     ramLak_fn,
                     plot_stages_fn=None,
                     plot_comparison_fn=None,
                     plot_profiles_fn=None,
                     fwhm_mm=3.0, n_dist_out=128, angular_sigma=0.5,
                     recon_iters=10, recon_subsets=5, mlem_iters=100,
                     gauss_every=5, gauss_sigma=0.4, use_psf=False,   # -> regularization_*
                     rotation_angles_override=None,
                     save_intermediates=True,
                     output_prefix="",
                     n_img=None, voxel_mm=None, rays_per_bin="auto",
                     backproject_chunks=1, interpolate_radial=False,
                     cor_mm=0.0, background_scale=1.0,
                     methods=("osem", "mlem", "fbp")):
    """
    Refined reconstruction pipeline.

    Applies:
      - Poisson MLEM/OSEM on measured prompt count bins with additive background
      - Background subtraction/rebin/smoothing for linear FBP ONLY
      - Calibrated COR and encoder offsets in the system geometry, not counts
      - Configurable OSEM / MLEM / FBP selection
      - No flood sensitivity correction (norm_sino=None).

    Supports acquisitions with 12 or 24 gantry positions, controlled by
    `rotation_angles_override`. If None, uses cfg['rotation_angles'].

    Parameters
    ----------
    calibration : dict
        Output of calibration.joint_calibration or load_calibration().
    rotation_angles_override : int or None
        For 24-angle acquisitions, pass 24. For regular 12-angle files,
        leave None.

    Image-grid control (volume-driven reconstruction)
    ------------------------------------------------
    n_img, voxel_mm : int / float, optional
        Reconstruction matrix (n_img x n_img) and in-plane voxel size (mm).
        Give either of them, or both. When either is given, the image grid
        is defined EXPLICITLY instead of being inherited from the number of
        radial bins, and ``interpolate_radial`` should normally be set to
        False so the measured bins are used as acquired.
          * n_img only     -> voxel = fov_mm / n_img
          * voxel_mm only  -> n_img  = round(fov_mm / voxel_mm)
          * both           -> fov is redefined as n_img * voxel_mm
        Leaving both as None uses the original measured number of radial bins
        for EM; FBP may use a different bin count if it is resampled.
    interpolate_radial : bool
        Whether to radially rebin the sinogram to ``n_dist_out``.  The
        default is False: reconstruct directly from the measured bins.
        When enabled, the FBP-only difference is rebinned conservatively.
        MLEM/OSEM ALWAYS receive the original count bins. Rebinning creates
        correlated bin noise, incompatible with a naive independent-Poisson
        EM likelihood.
    rays_per_bin : int or 'auto'
        Sub-rays traced per radial bin (see siddon_reconstructor). 'auto'
        keeps the sub-ray spacing below one voxel, which is required once
        the voxel is smaller than the radial bin.
    backproject_chunks : int
        >1 parallelizes the backprojection over angle chunks.
    cor_mm : float
        CONSTANT radial offset of the rotation axis (mm), applied to the
        projector geometry (no sinogram resampling). This is independent of
        the calibrated (dx, dy) sinusoidal COR correction applied in step 5;
        use `corrections.estimate_cor_offset` to obtain it. Both can be used
        together, but if the calibration already contains a non-zero
        (dx, dy) check that you are not correcting the same misalignment
        twice.
    """
    name = os.path.basename(fname).replace(".root", "")
    fov_mm = 2.0 * cfg["Half_FOV"]
    sino_range_mm = cfg.get("sino_range_mm", cfg["crystal_size"][1])

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Image FOV: {fov_mm:.1f} mm | sino range: {sino_range_mm:.2f} mm")

    # 1) Optional override of the number of rotation angles
    cfg_local = dict(cfg)
    if rotation_angles_override is not None:
        cfg_local['rotation_angles'] = int(rotation_angles_override)
        print(f"  rotation_angles override: {rotation_angles_override}")

    # 2) The measured prompt counts Y and the background estimate B must
    # remain SEPARATE for Poisson EM: Y~Poisson(Ax+B). Direct subtraction gives
    # Skellam-like noise, not nonnegative Poisson measurements.
    if not np.isfinite(background_scale) or background_scale < 0:
        raise ValueError("background_scale must be nonnegative and finite")
    methods = tuple(dict.fromkeys(m.lower().strip() for m in methods))
    if not methods or any(m not in ("osem", "mlem", "fbp") for m in methods):
        raise ValueError("methods must be a nonempty selection of osem,mlem,fbp")
    sino_src, _ = coincidence_to_lor_fn(fname, cfg_local, det_angles,
                                         params, rsectors=rsec)
    if bgname is None:
        sino_bg = np.zeros_like(sino_src)
    else:
        sino_bg, _ = coincidence_to_lor_fn(bgname, cfg_local, det_angles,
                                           params, rsectors=rsec)
    sino_src = np.asarray(sino_src, dtype=np.float64)
    sino_bg = np.asarray(sino_bg, dtype=np.float64)
    if (sino_src.shape != sino_bg.shape or sino_src.ndim != 3 or
            not np.isfinite(sino_src).all() or not np.isfinite(sino_bg).all() or
            np.any(sino_src < 0) or np.any(sino_bg < 0)):
        raise ValueError("measurement/background must be matched finite nonnegative 3-D count histograms")
    background = background_scale * sino_bg
    if not np.isfinite(background).all():
        raise ValueError("scaled background overflows float64")
    n_ang = sino_src.shape[2]
    angles = (np.arange(n_ang) + 0.5) * (180.0 / n_ang)
    dx = float(calibration.get('dx_cor_mm', 0.0))
    dy = float(calibration.get('dy_cor_mm', 0.0))
    phi_off = float(calibration.get('phi_offset_deg', 0.0))
    if not np.isfinite([dx, dy, phi_off]).all():
        raise ValueError("calibration offsets must be finite")
    # The physical angles can span beyond 180 degrees: do NOT reduce modulo
    # 180 unless radial parity is applied. Siddon works with full angles.
    angles = angles + phi_off
    print(f"  [CAL] geometric COR dx={dx:+.3f}, dy={dy:+.3f} mm; "
          f"angle offset {phi_off:+.3f} deg (NO count resampling)")
    print(f"  Counts: prompts={sino_src.sum():.0f}  "
          f"background (scaled)={background.sum():.1f}")
    # FBP is a LINEAR transform of a background-subtracted estimate; it does
    # not apply the Poisson likelihood, so negative bins are permitted here.
    sino_fbp = sino_src - background
    if interpolate_radial and 'fbp' in methods:
        sino_fbp = interpolate_distance_fn(sino_fbp, n_out=n_dist_out,
                                            kind='conservative')
        print(f"  FBP radial rebin only: {sino_src.shape[1]} -> {n_dist_out}")
    if angular_sigma > 0 and 'fbp' in methods:
        sino_fbp = smooth_angular_fn(sino_fbp, sigma_bins=angular_sigma)
        print("  FBP angular smoothing only (EM raw count histograms unchanged)")
    sino_cal = sino_src  # legacy result key; now explicitly unresampled prompts
    sino_raw = sino_src
    sino_s = sino_fbp
    n_dist_eff = sino_src.shape[1]
    sino_pixel = sino_range_mm / n_dist_eff
    distances = np.linspace(-sino_range_mm/2 + sino_pixel/2,
                            sino_range_mm/2 - sino_pixel/2, n_dist_eff)

    # 6) Stage plots
    if plot_stages_fn is not None and save_intermediates:
        plot_stages_fn([
            ("Measured prompts (Poisson)", sino_src),
            ("Scaled background", background),
            ("FBP preprocessed difference", sino_fbp),
        ], fov=sino_range_mm, title=name, sl=min(7, sino_cal.shape[0]-1))

    # 8) Reconstruction: the same measured counts are used for OSEM and
    # MLEM; the forward model includes the estimated additive background.
    grid_kw = dict(fov_mm=fov_mm, sino_range_mm=sino_range_mm,
                   rays_per_bin=rays_per_bin,
                   backproject_chunks=backproject_chunks,
                   cor_mm=cor_mm, dx_cor_mm=dx, dy_cor_mm=dy)
    if n_img is not None or voxel_mm is not None:
        grid_kw["n_img"] = n_img
        grid_kw["voxel_mm"] = voxel_mm
        if n_img is not None and voxel_mm is not None:
            grid_kw["fov_mm"] = None
    psf_k = None
    if use_psf and any(m in methods for m in ("osem", "mlem")):
        from Gimntools.GimnRec.filters import make_psf_kernel
        psf_k = make_psf_kernel(fwhm_mm, sino_pixel)
    outputs = {}
    geometry = None
    for method in methods:
        if method == "fbp":
            rec = siddon_reconstructor_cls(sino_fbp, **grid_kw)
            out = rec.fbp(angles, filter_type=ramLak_fn)
        elif method == "osem":
            rec = siddon_reconstructor_cls(sino_src, **grid_kw)
            out = rec.osem(angles, iterations=recon_iters,
                           subsets_n=recon_subsets, psf_kernel=psf_k,
                           background_sino=background, norm_sino=None,
                           regularization_every=gauss_every,
                           regularization_sigma=gauss_sigma)
        else:
            rec = siddon_reconstructor_cls(sino_src, **grid_kw)
            out = rec.mlem(angles, iterations=mlem_iters, psf_kernel=psf_k,
                           background_sino=background, norm_sino=None,
                           regularization_every=gauss_every,
                           regularization_sigma=gauss_sigma)
        outputs[method] = out
        geometry = rec.geometry()
        print(f"  {method.upper()}: {out.shape}")

    if plot_comparison_fn is not None and all(m in outputs for m in ("osem", "mlem", "fbp")):
        plot_comparison_fn(outputs['osem'], outputs['mlem'], outputs['fbp'],
                           sl=min(7, outputs['osem'].shape[0]-1),
                           fov=geometry['fov_mm'], title=name)

    # Store JSON as Unicode arrays rather than object dictionaries. The
    # resulting NPZ can be loaded securely with allow_pickle=False.
    import json
    def serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): serializable(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [serializable(v) for v in obj]
        return obj
    out_name = f"{output_prefix}recon_{name}.npz"
    np.savez_compressed(out_name, sino_raw=sino_src,
                        sino_background=background,
                        sino_fbp=sino_fbp, sino_calibrated=sino_cal,
                        distances=distances, angles_deg=angles,
                        calibration_used_json=json.dumps(serializable(calibration)),
                        geometry_json=json.dumps(serializable(geometry)),
                        background_scale=float(background_scale),
                        **outputs)
    print(f"  Saved: {out_name}")

    return {
        'name': name, **outputs,
        'sino_raw': sino_src, 'sino_background': background,
        'sino_fbp': sino_fbp, 'sino_calibrated': sino_cal,
        'distances': distances, 'angles_deg': angles,
        'fov_mm': geometry['fov_mm'], 'voxel_mm': geometry['voxel_mm'],
        'geometry': geometry,
    }
