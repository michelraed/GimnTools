"""Legacy single-file reconstruction pipeline (``process_file``).

This module keeps the original "all-in-one" per-file pipeline: build the
sinogram from a ROOT file, subtract background, interpolate/smooth, optionally
estimate/apply a centre-of-rotation (COR) correction, and reconstruct with
OSEM / MLEM / FBP.

For new work prefer :func:`Gimntools.GimnRec.reconstruction_v2.process_file_v2`,
which uses calibrated corrections and dependency injection.

.. note::
   This module was previously shipped **without its imports** (it was split
   out of a monolithic script) and raised ``NameError`` at call time. The
   imports below fix that; the numerical behaviour of ``process_file`` is
   unchanged.
"""

import os

import numpy as np

from Gimntools.GimnRec.utils import coincidence_to_lor
from Gimntools.GimnRec.filters import (
    interpolate_distance,
    smooth_angular,
    make_psf_kernel,
)
from Gimntools.GimnRec.corrections import estimate_cor, apply_cor
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.plots import (
    plot_stages_mm,
    plot_comparison_mm,
    plot_profiles_fwhm,
)

def process_file(fname, bgname, cfg, det_angles, params, rsec,
                  fwhm_mm=6.0, n_dist_out=128, angular_sigma=2.0,
                  recon_iters=10, recon_subsets=8, mlem_iters=80,
                  gauss_every=5, gauss_sigma=0.5, use_psf=True,
                  norm_3d=None,
                  cor_dxdy=None,
                  estimate_cor_self=False,
                  cor_expected_source_xy=(0.0, 0.0),
                  n_img=None, voxel_mm=None, rays_per_bin="auto",
                  backproject_chunks=1, interpolate_radial=False,
                  cor_mm=0.0):
    """Per-file reconstruction pipeline (legacy).

    COR parameters
    --------------
    cor_dxdy : (dx, dy) in mm
        Pre-calibrated centre-of-rotation offset applied to the sinogram.
        ``None`` disables the correction.
    estimate_cor_self : bool
        If True, estimate the COR from THIS file (valid only for a centred
        source, or an activity distribution symmetric about the origin).
        Overrides ``cor_dxdy``.
    cor_expected_source_xy : (Sx, Sy)
        Expected source position for the self-estimation. The default
        ``(0, 0)`` covers a centred / symmetric source.

    Image-grid parameters (volume-driven reconstruction)
    ---------------------------------------------------
    n_img, voxel_mm : reconstruction matrix and in-plane voxel size (mm).
        Give either or both; when given, the image grid stops being a side
        effect of ``n_dist_out``. Both None = legacy behaviour.
    interpolate_radial : set False to reconstruct from the measured radial
        bins instead of the cubic-upsampled ones.
    rays_per_bin : sub-rays per radial bin ('auto' recommended).
    cor_mm : constant radial offset of the rotation axis (mm), applied to the
        projector geometry. Independent of `cor_dxdy`, which resamples the
        sinogram with an angle-dependent shift.
    """
    name = os.path.basename(fname).replace(".root", "")
    fov_mm = 2.0 * cfg["Half_FOV"]
    sino_range_mm = cfg.get("sino_range_mm", cfg["crystal_size"][1])
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  Image FOV: {fov_mm:.1f}mm | Sino range: {sino_range_mm:.1f}mm")

    # 1. Sinograms
    sino_src, _ = coincidence_to_lor(fname, cfg, det_angles, params, rsectors=rsec)
    sino_bg, _  = coincidence_to_lor(bgname, cfg, det_angles, params, rsectors=rsec)
    # MLEM/OSEM receive the measured prompts, with background additive.
    # This legacy function assumes identical measurement/background exposures;
    # use the calibrated v2 pipeline if the acquisition times differ.
    import warnings
    warnings.warn("Legacy process_file assumes equal background acquisition exposure; "
                  "use process_file_v2(background_scale=...) for quantitative work.",
                  DeprecationWarning, stacklevel=2)
    sino_raw = np.asarray(sino_src, dtype=float)
    bg_model = np.asarray(sino_bg, dtype=float)
    if (bg_model.shape != sino_raw.shape or not np.isfinite(bg_model).all() or
            not np.isfinite(sino_raw).all() or
            np.any(bg_model < 0) or np.any(sino_raw < 0)):
        raise ValueError("source/background sinograms must match, be finite and nonnegative")

    # Optional display/FBP interpolation; NEVER sent to Poisson MLEM/OSEM.
    sino_diff = sino_raw - bg_model
    if interpolate_radial:
        sino_interp = interpolate_distance(
            sino_diff, n_out=n_dist_out, kind='conservative')
    else:
        sino_interp = sino_diff.copy()
        print(f"  Radial interpolation OFF ({sino_interp.shape[1]} measured bins)")
    # Smoothing changes the measurement likelihood. EM always uses raw bins.
    sino_smooth = (smooth_angular(sino_interp, sigma_bins=angular_sigma)
                   if angular_sigma > 0 else sino_interp)
    print(f"  Sinogram: {sino_smooth.shape}")

    plot_stages_mm([("Raw", sino_raw), ("Interpolated", sino_interp),
                     ("Smooth", sino_smooth)],
                    fov=sino_range_mm, title=name, sl=min(7,sino_raw.shape[0]-1))

    # 2.5 Axes of the processed FBP sinogram (bin centres, mm and degrees).
    #     Derived from the ACTUAL bin count, which equals n_dist_out only
    #     when the radial interpolation is enabled.
    n_dist_eff = sino_smooth.shape[1]
    sino_pixel = sino_range_mm / n_dist_eff
    sino_distances = np.linspace(-sino_range_mm/2 + sino_pixel/2,
                                  sino_range_mm/2 - sino_pixel/2,
                                  n_dist_eff)
    n_ang = sino_smooth.shape[2]
    angles = (np.arange(n_ang) + 0.5) * (180.0 / n_ang)

    # 2.6 COR — optional self-estimation + correction
    if estimate_cor_self:
        dx_self, dy_self, _ = estimate_cor(
            sino_raw,
            np.linspace(-sino_range_mm/2+sino_range_mm/(2*sino_raw.shape[1]),
                        sino_range_mm/2-sino_range_mm/(2*sino_raw.shape[1]),
                        sino_raw.shape[1]), angles,
            expected_source_xy=cor_expected_source_xy,
            plot=True, name=f"{name} (self-cal)",
        )
        cor_dxdy = (dx_self, dy_self)

    if cor_dxdy is not None:
        dx, dy = float(cor_dxdy[0]), float(cor_dxdy[1])
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            print(f"  [COR] applying dx={dx:+.3f} mm, dy={dy:+.3f} mm")
            sino_pre = sino_smooth.copy()
            # COR is applied in the projector geometry, never to count bins.

    # 3. PSF kernel (radial: uses the real bin width)
    psf_pixel = sino_range_mm / sino_raw.shape[1]
    psf_k = make_psf_kernel(fwhm_mm, psf_pixel) if use_psf else None
    if psf_k is not None:
        print(f"  PSF: FWHM={fwhm_mm}mm, kernel={len(psf_k)}px")

    # 4. Flood normalization (goes INSIDE the EM update, not into the sinogram)
    #    norm_3d[z] is passed as norm_sino for each slice
    #    The reconstructor uses: sens = A^T . norm, forward = norm . A . x

    # 5. Reconstruction
    sino_final = sino_raw
    # FBP uses the (optionally smoothed/rebinned) linear difference.
    sino_fbp = sino_smooth

    # Take the central slice of the normalization volume, if available
    norm_ref = None
    if norm_3d is not None:
        sl_ref = norm_3d.shape[0] // 2
        norm_ref = norm_3d[sl_ref]  # central slice used as reference
        print(f"  Flood norm: active (range [{norm_ref.min():.2f}, {norm_ref.max():.2f}])")

    grid_kw = dict(fov_mm=fov_mm, sino_range_mm=sino_range_mm,
                   rays_per_bin=rays_per_bin,
                   backproject_chunks=backproject_chunks,
                   cor_mm=cor_mm,
                   dx_cor_mm=(cor_dxdy[0] if cor_dxdy is not None else 0.0),
                   dy_cor_mm=(cor_dxdy[1] if cor_dxdy is not None else 0.0))
    if n_img is not None or voxel_mm is not None:
        grid_kw["n_img"] = n_img
        grid_kw["voxel_mm"] = voxel_mm
        if n_img is not None and voxel_mm is not None:
            grid_kw["fov_mm"] = None

    # Both EM and FBP use the same image grid even if FBP radial samples differ.
    if n_img is None and voxel_mm is None:
        grid_kw['n_img'] = sino_raw.shape[1]
    rec1 = siddon_reconstructor(sino_final, **grid_kw)
    print(f"  OSEM ({recon_iters}it × {recon_subsets}sub)...")
    osem = rec1.osem(angles, iterations=recon_iters, subsets_n=recon_subsets,
                      psf_kernel=psf_k, norm_sino=norm_ref,
                      background_sino=bg_model,
                      regularization_every=gauss_every, regularization_sigma=gauss_sigma)

    rec2 = siddon_reconstructor(sino_final, **grid_kw)
    print(f"  MLEM ({mlem_iters}it)...")
    mlem = rec2.mlem(angles, iterations=mlem_iters, psf_kernel=psf_k,
                      norm_sino=norm_ref, background_sino=bg_model,
                      regularization_every=gauss_every, regularization_sigma=gauss_sigma)

    rec3 = siddon_reconstructor(sino_fbp, **grid_kw)
    print(f"  FBP...")
    fbp = rec3.fbp(angles)

    # 6. Visualization (uses the resolved FOV, which may differ from
    #    2*Half_FOV when n_img and voxel_mm were both specified)
    fov_mm = rec1.fov_mm
    plot_comparison_mm(osem, mlem, fbp, sl=min(7,osem.shape[0]-1), fov=fov_mm, title=name)

    sl = min(7, osem.shape[0]-1)
    fwhm_data = {}
    for lab, r in [("OSEM",osem),("MLEM",mlem),("FBP",fbp)]:
        fwhm_data[lab] = plot_profiles_fwhm(r[sl], fov_mm, f"{name} — {lab} s{sl}")

    # 7. Save
    sp = f"recon_{name}.npz"
    np.savez_compressed(sp, sino_raw=sino_raw, sino_smooth=sino_smooth,
                         osem=osem, mlem=mlem, fbp=fbp)
    print(f"  Saved: {sp}")

    print(f"\n  {'Method':<8} {'FWHM_H':>8} {'FWHM_V':>8} {'Pico (mm)':>16}")
    for lab in ["OSEM","MLEM","FBP"]:
        d = fwhm_data[lab]
        print(f"  {lab:<8} {d['fwhm_h']:>7.2f}  {d['fwhm_v']:>7.2f}  "
              f"({d['peak_mm'][0]:>5.1f},{d['peak_mm'][1]:>5.1f})")

    return {'name':name, 'osem':osem, 'mlem':mlem, 'fbp':fbp,
            'sino_smooth':sino_smooth, 'fwhm':fwhm_data,
            'fov_mm': rec1.fov_mm, 'voxel_mm': rec1.voxel_mm,
            'geometry': rec1.geometry()}
