"""
reconstruct_gate — reconstruction CLI for GATE simulations.

More flexible than reconstruct_petsys: no geometric corrections (the
simulator geometry is exact) and no background subtraction. Exposes
`process_as` and `rotate_along`, and allows toggling the radial
interpolation (useful for crystalID data, which is otherwise very
pixelated).

The rotation/projection/slice axis is derived from
detector_config['rotate_along'] (see Gimntools.GimnRec.utils._resolve_axes)
— it is no longer hard-coded.

Example:
    reconstruct_gate \\
        --measurement sim.root \\
        --detector-config cfg/detector_config_gate.json \\
        --detectors-angles cfg/detectors_angles.json \\
        --rsectors cfg/rsectors_gate.json \\
        --n-gantry-angles 10 --process-as GeneralGlobalPosition \\
        --rotate-along z --interpolate \\
        --output-dir ./out_gate --save-format nrrd
"""

import argparse
import os
import sys
import json

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(
        prog="reconstruct_gate",
        description="PET reconstruction for GATE simulations (flexible, no corrections).",
    )
    p.add_argument("--measurement", required=True, help="Simulation ROOT file.")
    p.add_argument("--detector-config", required=True,
                   help="detector_config JSON (GATE format).")
    p.add_argument("--detectors-angles", required=True,
                   help="JSON {detector_name: angle}.")
    p.add_argument("--rsectors", required=True,
                   help="JSON {rsectorID: detector_name}.")
    p.add_argument("--parameters", default=None,
                   help="Optional JSON with the branch list; GATE defaults if omitted.")

    # Geometry / reading
    p.add_argument("--n-gantry-angles", type=int, default=None,
                   help="Override rotation_angles. Default: use the detector_config value.")
    p.add_argument("--process-as", default=None,
                   help="Override detector_config['process_as'] "
                        "(CrystalID | GateGlobalPositions | GeneralGlobalPosition).")
    p.add_argument("--rotate-along", default=None, choices=["x", "y", "z"],
                   help="Override detector_config['rotate_along']. "
                        "Defines the rotation axis, projection plane and slice axis.")
    p.add_argument("--first-rotation", default=None, choices=["x", "y", "z"],
                   help="Axis used to place the detectors initially. "
                        "None = same as --rotate-along.")
    p.add_argument("--matrix", default="128,128,128",
                   help="Sinogram dimensions for the global process_as modes (Z,Nr,Ntheta "
                        "are recomputed internally for segmented).")

    # Sinogram processing
    p.add_argument("--interpolate", action="store_true",
                   help="Radially interpolate to --n-dist-out bins "
                        "(FBP only; iterative reconstruction uses original measured bins).")
    p.add_argument("--n-dist-out", type=int, default=128)
    p.add_argument("--angular-sigma", type=float, default=0.5,
                   help="FBP-only angular smoothing sigma; never changes EM counts. 0 disables it.")

    # Image grid (volume-driven reconstruction)
    p.add_argument("--n-img", type=int, default=None,
                   help="Reconstruction matrix (n-img x n-img). Defines the "
                        "image grid EXPLICITLY instead of inheriting it from "
                        "the number of radial bins. Combine with --voxel-mm "
                        "to also fix the physical voxel size (the FOV then "
                        "becomes n-img*voxel-mm and overrides 2*Half_FOV).")
    p.add_argument("--voxel-mm", type=float, default=None,
                   help="In-plane voxel size (mm). With --n-img it fixes the "
                        "FOV; alone it is combined with 2*Half_FOV to give "
                        "n-img = round(FOV/voxel).")
    p.add_argument("--rays-per-bin", default="auto",
                   help="Sub-rays traced per radial bin ('auto' or an int). "
                        "'auto' keeps the sub-ray spacing <= one voxel, "
                        "which is required when the voxel is smaller than "
                        "the radial bin. 1 = previous behaviour.")
    p.add_argument("--backproject-chunks", type=int, default=1,
                   help="Parallel accumulation buffers in the backprojector "
                        "(>1 is faster; changes only float summation order).")
    p.add_argument("--cor-mm", type=float, default=0.0,
                   help="Radial position (mm) of the rotation axis in the "
                        "sinogram: a CONSTANT offset of every angular column. "
                        "Applied to the projector geometry, so the counts are "
                        "never resampled and sub-bin values are exact.")
    p.add_argument("--estimate-cor", action="store_true",
                   help="Estimate the constant COR offset from the data "
                        "(valid only for an object centred on the axis) and "
                        "use it. Overrides --cor-mm.")

    # Reconstruction
    p.add_argument("--methods", default="osem,mlem,fbp")
    p.add_argument("--osem-iters", type=int, default=10)
    p.add_argument("--osem-subsets", type=int, default=4)
    p.add_argument("--mlem-iters", type=int, default=50)
    p.add_argument("--gauss-every", type=int, default=5)
    p.add_argument("--gauss-sigma", type=float, default=0.4)
    p.add_argument("--use-psf", action="store_true",
                   help="Include a 1-D Gaussian radial detector-response "
                        "kernel in the system model of MLEM/OSEM (it enters "
                        "both the forward step and the sensitivity image).")
    p.add_argument("--fwhm-mm", type=float, default=3.0,
                   help="FWHM (mm) of the --use-psf kernel.")
    p.add_argument("--geom-norm", action="store_true",
                   help="Apply the geometric sensitivity normalization "
                        "(pair-density norm for rotating flat detectors). "
                        "Corrects the centre-heavy intensity profile; see "
                        "README.md. EM methods receive it as norm_sino; "
                        "FBP pre-divides the sinogram.")
    p.add_argument("--legacy-em", action="store_true",
                   help="Use the old MLEM/OSEM behaviour (for comparison).")

    # Output
    p.add_argument("--output-dir", default="./outputs_reconstrucao_gate")
    p.add_argument("--save-format", choices=["mhd", "nrrd", "none"], default="mhd")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.no_plots:
        import matplotlib
        matplotlib.use("Agg")

    from Gimntools.GimnRec.utils import coincidence_to_lor, save_reconstructions
    from Gimntools.GimnRec.filters import (interpolate_distance, smooth_angular,
                                            ramLak, make_psf_kernel)
    from Gimntools.GimnRec.reconstructor import siddon_reconstructor
    from Gimntools.GimnRec.data_loaders import (
        get_default_parameters, load_detector_config, validate_detector_config,
        load_detectors_angles, load_rsectors,
    )

    if not os.path.exists(args.measurement):
        sys.exit(f"[ERROR] measurement not found: {args.measurement}")
    for label, path in [("detector-config", args.detector_config),
                        ("detectors-angles", args.detectors_angles),
                        ("rsectors", args.rsectors)]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    detector_config = load_detector_config(args.detector_config)
    for w in validate_detector_config(detector_config, "gate"):
        print(f"[WARNING] {w}")
    if args.process_as:
        detector_config["process_as"] = args.process_as
    if args.rotate_along:
        detector_config["rotate_along"] = args.rotate_along
    if args.n_gantry_angles is not None:
        detector_config["rotation_angles"] = args.n_gantry_angles

    # Ensure NumPy arrays in the fields that need them
    for k in ("crystal_size", "sipm_pitch", "sipm_pixel_n"):
        if isinstance(detector_config.get(k), list):
            detector_config[k] = np.asarray(detector_config[k])

    detectors_angles = load_detectors_angles(args.detectors_angles)
    rsectors = load_rsectors(args.rsectors)

    if args.parameters:
        with open(args.parameters) as f:
            parameters = json.load(f)
    else:
        parameters = get_default_parameters("gate")

    matrix = [int(x) for x in args.matrix.split(",")]
    if len(matrix) != 3 or any(n < 1 for n in matrix):
        build_parser().error("--matrix must be three positive integers Z,Nr,Ntheta")
    methods = list(dict.fromkeys(m.strip().lower() for m in args.methods.split(",") if m.strip()))
    if not methods or any(m not in ("osem", "mlem", "fbp") for m in methods):
        build_parser().error("--methods must specify a nonempty selection of osem,mlem,fbp")
    if not np.isfinite(args.angular_sigma) or args.angular_sigma < 0:
        build_parser().error("--angular-sigma must be finite and nonnegative")
    if args.n_dist_out < 1:
        build_parser().error("--n-dist-out must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    name = os.path.basename(args.measurement).replace(".root", "")
    fov_mm = 2.0 * detector_config["Half_FOV"]
    sino_range_mm = detector_config.get("sino_range_mm",
                                        float(detector_config["crystal_size"][1]))

    print(f"Building sinogram: {os.path.basename(args.measurement)}")
    sino_raw, _ = coincidence_to_lor(
        args.measurement, detector_config, detectors_angles, parameters,
        matrix=matrix, rsectors=rsectors,
        first_rotation=args.first_rotation, verbose=True,
    )
    print(f"  Raw sinogram: {sino_raw.shape}, counts={sino_raw.sum():.0f}")

    # Poisson EM requires ORIGINAL, independent and nonnegative count bins:
    # neither rebinning nor smoothing is applied to iterative methods.
    sino = np.asarray(sino_raw, dtype=np.float64)
    if sino.ndim != 3 or min(sino.shape) < 1 or not np.isfinite(sino).all() or np.any(sino < 0):
        raise ValueError("GATE sinogram must be a finite nonnegative 3-D count histogram")
    sino_fbp = sino.copy()
    if args.interpolate and "fbp" in methods:
        sino_fbp = interpolate_distance(sino_fbp, n_out=args.n_dist_out, kind="conservative")
        print(f"  FBP-only radial rebin: {sino.shape[1]} -> {sino_fbp.shape[1]}")
    if args.angular_sigma > 0 and "fbp" in methods:
        sino_fbp = smooth_angular(sino_fbp, sigma_bins=args.angular_sigma)
        print(f"  FBP-only angular smoothing sigma={args.angular_sigma}")

    n_dist = sino.shape[1]
    n_ang = sino.shape[2]
    angles = (np.arange(n_ang) + 0.5) * (180.0 / n_ang)

    norm = None
    norm_fbp = None
    if args.geom_norm:
        from Gimntools.GimnRec.normalization import geometric_norm_sinogram
        print("Computing geometric normalization (pair density)...")
        norm = geometric_norm_sinogram(detector_config, detectors_angles, n_dist, n_ang)
        if "fbp" in methods:
            norm_fbp = geometric_norm_sinogram(detector_config, detectors_angles,
                                                sino_fbp.shape[1], n_ang)
            if args.angular_sigma > 0:
                norm_fbp = smooth_angular(norm_fbp[None],
                                           sigma_bins=args.angular_sigma)[0]

    if args.rays_per_bin == "auto":
        rays_per_bin = "auto"
    else:
        try:
            rays_per_bin = int(args.rays_per_bin)
        except (TypeError, ValueError):
            build_parser().error("--rays-per-bin must be 'auto' or a positive integer")
        if rays_per_bin < 1 or str(rays_per_bin) != str(args.rays_per_bin):
            build_parser().error("--rays-per-bin must be 'auto' or a positive integer")
    cor_mm = args.cor_mm
    if args.estimate_cor:
        from Gimntools.GimnRec.corrections import estimate_cor_offset
        bin_w = sino_range_mm / n_dist
        bins = np.linspace(-sino_range_mm/2 + bin_w/2,
                            sino_range_mm/2 - bin_w/2, n_dist)
        cor_mm, _ = estimate_cor_offset(sino, bins, angles)
    if abs(cor_mm) > 0:
        print(f"  COR offset: {cor_mm:+.4f} mm "
              f"({cor_mm/(sino_range_mm/n_dist):+.3f} bins)")

    # Radial detector-response kernel (part of the system model H).
    # NOTE (0.4.0): --use-psf and --fwhm-mm were previously parsed and then
    # silently ignored in this CLI; psf_kernel was never passed to mlem/osem.
    psf_k = None
    if args.use_psf:
        psf_k = make_psf_kernel(args.fwhm_mm, sino_range_mm / n_dist)
        print(f"  PSF: FWHM={args.fwhm_mm} mm, kernel={len(psf_k)} bins")

    grid_kw = dict(fov_mm=fov_mm, sino_range_mm=sino_range_mm,
                   n_img=n_dist, rays_per_bin=rays_per_bin,
                   backproject_chunks=args.backproject_chunks,
                   cor_mm=cor_mm)
    if args.n_img is not None or args.voxel_mm is not None:
        grid_kw["n_img"] = args.n_img
        grid_kw["voxel_mm"] = args.voxel_mm
        if args.n_img is not None and args.voxel_mm is not None:
            grid_kw["fov_mm"] = None

    recs = {}
    geom = None
    if "osem" in methods:
        r = siddon_reconstructor(sino, **grid_kw)
        geom = r.geometry()
        print(f"OSEM ({args.osem_iters}it × {args.osem_subsets}sub)...")
        recs["osem"] = r.osem(angles, iterations=args.osem_iters,
                              subsets_n=args.osem_subsets, norm_sino=norm,
                              psf_kernel=psf_k,
                              regularization_every=args.gauss_every,
                              regularization_sigma=args.gauss_sigma,
                              legacy=args.legacy_em)
    if "mlem" in methods:
        r = siddon_reconstructor(sino, **grid_kw)
        geom = geom or r.geometry()
        print(f"MLEM ({args.mlem_iters}it)...")
        recs["mlem"] = r.mlem(angles, iterations=args.mlem_iters, norm_sino=norm,
                              psf_kernel=psf_k,
                              regularization_every=args.gauss_every,
                              regularization_sigma=args.gauss_sigma,
                              legacy=args.legacy_em)
    if "fbp" in methods:
        if norm_fbp is not None:
            from Gimntools.GimnRec.normalization import apply_norm_to_sinogram
            sino_fbp = apply_norm_to_sinogram(sino_fbp, norm_fbp)
        r = siddon_reconstructor(sino_fbp, **grid_kw)
        geom = geom or r.geometry()
        print("FBP...")
        recs["fbp"] = r.fbp(angles, filter_type=ramLak)

    out_npz = os.path.join(args.output_dir, f"recon_{name}.npz")
    # JSON rather than object-dtype NPZ: opening never requires pickle.
    np.savez_compressed(out_npz, sino_raw=sino_raw, sino_final=sino,
                        sino_fbp=sino_fbp, angles_deg=angles,
                        geometry_json=json.dumps(geom or {}), **recs)
    print(f"NPZ saved: {out_npz}")

    if args.save_format != "none":
        res_for_save = {"name": name, **recs}
        # Spacing comes from the RESOLVED image geometry, not from the number
        # of radial bins (they are independent since the volume-driven update).
        save_reconstructions(res_for_save, detector_config, n_dist,
                             os.path.join(args.output_dir, args.save_format),
                             fmt=args.save_format,
                             fov_mm=(geom or {}).get("fov_mm"),
                             voxel_mm=(geom or {}).get("voxel_mm"))

    print(f"\n[OK] GATE reconstructed: {name}  | methods: {list(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
