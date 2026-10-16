"""
reconstruct_petsys — reconstruction CLI for PETSYS (hardware) data.

Includes the geometric-correction path (COR + encoder offset) via a
calibration loaded from JSON, delegating to Gimntools.GimnRec.reconstruction_v2.

Example:
    reconstruct_petsys \\
        --measurement measurement.root --background bg.root \\
        --detector-config cfg/detector_config_petsys.json \\
        --detectors-angles cfg/detectors_angles.json \\
        --rsectors cfg/rsectors_petsys.json \\
        --calibration cfg/calibration.json \\
        --n-gantry-angles 12 \\
        --output-dir ./out --save-format mhd
"""

import argparse
import os
import sys
import json

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(
        prog="reconstruct_petsys",
        description="PET reconstruction for PETSYS data (with calibrated corrections).",
    )
    # Required inputs
    p.add_argument("--measurement", required=True, help="Measurement ROOT file.")
    p.add_argument("--background", required=True,
                   help="Background ROOT file; retained as an additive count model.")
    p.add_argument("--background-scale", type=float, required=True,
                   help="Prompt exposure time / background exposure time; required "
                        "to scale the additive background counts (1 for equal exposures).")
    p.add_argument("--detector-config", required=True,
                   help="JSON do detector_config (formato PETSYS).")
    p.add_argument("--detectors-angles", required=True,
                   help="JSON {detector_name: angle}.")
    p.add_argument("--rsectors", required=True,
                   help="JSON {chipID: nome_detector}.")
    p.add_argument("--calibration", required=True,
                   help="Calibration JSON (dx_cor_mm, dy_cor_mm, phi_offset_deg).")
    p.add_argument("--parameters", default=None,
                   help="Optional JSON with the branch list; "
                        "PETSYS defaults if omitted.")

    # Reconstruction
    p.add_argument("--n-gantry-angles", type=int, default=None,
                   help="Override rotation_angles (12 or 24). "
                        "Default: use the detector_config value.")
    p.add_argument("--n-dist-out", type=int, default=128)
    interp = p.add_mutually_exclusive_group()
    interp.add_argument("--interpolate", dest="interpolate", action="store_true",
                        help="Conservatively rebin radial counts to --n-dist-out. "
                             "Off by default; raw measured bins are preferred "
                             "for MLEM/OSEM.")
    interp.add_argument("--no-interpolate", dest="interpolate",
                        action="store_false", help="Use measured radial bins "
                        "(default; retained for command-line compatibility).")
    p.set_defaults(interpolate=False)
    p.add_argument("--angular-sigma", type=float, default=0.5)

    # Image grid (volume-driven reconstruction)
    p.add_argument("--n-img", type=int, default=None,
                   help="Reconstruction matrix (n-img x n-img), set "
                        "explicitly instead of following the radial bins.")
    p.add_argument("--voxel-mm", type=float, default=None,
                   help="In-plane voxel size (mm). With --n-img it fixes the "
                        "FOV (n-img*voxel-mm, overriding 2*Half_FOV).")
    p.add_argument("--rays-per-bin", default="auto",
                   help="Sub-rays per radial bin ('auto' or int; 1 = old "
                        "behaviour).")
    p.add_argument("--backproject-chunks", type=int, default=1,
                   help="Parallel accumulation buffers in the backprojector.")
    p.add_argument("--cor-mm", type=float, default=0.0,
                   help="Radial position (mm) of the rotation axis in the "
                        "sinogram (constant offset of every angular column), "
                        "applied to the projector geometry without resampling "
                        "the counts.")
    p.add_argument("--methods", default="osem,mlem,fbp",
                   help="Comma-separated list: osem,mlem,fbp.")
    p.add_argument("--osem-iters", type=int, default=10)
    p.add_argument("--osem-subsets", type=int, default=5)
    p.add_argument("--mlem-iters", type=int, default=50)
    p.add_argument("--gauss-every", type=int, default=5)
    p.add_argument("--gauss-sigma", type=float, default=0.4)
    p.add_argument("--use-psf", action="store_true")
    p.add_argument("--fwhm-mm", type=float, default=3.0)
    p.add_argument("--legacy-em", action="store_true",
                   help="Use the old MLEM/OSEM behaviour (for comparison).")

    # Output
    p.add_argument("--output-dir", default="./outputs_reconstrucao")
    p.add_argument("--save-format", choices=["mhd", "nrrd", "none"], default="mhd",
                   help="Volume export format (or 'none').")
    p.add_argument("--no-plots", action="store_true",
                   help="Desliga as figuras matplotlib (modo batch).")
    return p


def _require(path, label):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {label} not found: {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Deferred imports (avoids import cost when only --help is wanted)
    from Gimntools.GimnRec.utils import coincidence_to_lor, save_reconstructions
    from Gimntools.GimnRec.filters import interpolate_distance, smooth_angular, ramLak
    from Gimntools.GimnRec.reconstructor import siddon_reconstructor
    from Gimntools.GimnRec.reconstruction_v2 import process_file_v2
    from Gimntools.GimnRec.data_loaders import (
        get_default_parameters, load_detector_config, validate_detector_config,
        load_detectors_angles, load_rsectors,
    )
    from Gimntools.GimnRec.calibration import load_calibration

    if args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
    from Gimntools.GimnRec.plots import plot_stages_mm, plot_comparison_mm

    for label, path in [
        ("measurement", args.measurement), ("background", args.background),
        ("detector-config", args.detector_config),
        ("detectors-angles", args.detectors_angles),
        ("rsectors", args.rsectors), ("calibration", args.calibration),
    ]:
        _require(path, label)

    detector_config = load_detector_config(args.detector_config)
    for w in validate_detector_config(detector_config, "petsys"):
        print(f"[WARNING] {w}")
    detectors_angles = load_detectors_angles(args.detectors_angles)
    rsectors = load_rsectors(args.rsectors)
    calibration = load_calibration(args.calibration)

    if args.parameters:
        with open(args.parameters) as f:
            parameters = json.load(f)
    else:
        parameters = get_default_parameters("petsys")

    methods = list(dict.fromkeys(m.strip().lower() for m in args.methods.split(",")
                                      if m.strip()))
    if not methods or any(m not in ("osem", "mlem", "fbp") for m in methods):
        build_parser().error("--methods must specify osem,mlem,fbp (nonempty)")
    if not np.isfinite(args.background_scale) or args.background_scale < 0:
        build_parser().error("--background-scale must be finite and nonnegative")
    os.makedirs(args.output_dir, exist_ok=True)

    plot_stages_fn = None if args.no_plots else plot_stages_mm
    plot_comp_fn = None if args.no_plots else plot_comparison_mm

    try:
        rays_per_bin = int(args.rays_per_bin)
    except (TypeError, ValueError):
        rays_per_bin = "auto"

    res = process_file_v2(
        fname=args.measurement, bgname=args.background,
        cfg=detector_config, det_angles=detectors_angles,
        params=parameters, rsec=rsectors,
        calibration=calibration,
        coincidence_to_lor_fn=coincidence_to_lor,
        interpolate_distance_fn=interpolate_distance,
        smooth_angular_fn=smooth_angular,
        siddon_reconstructor_cls=siddon_reconstructor,
        ramLak_fn=ramLak,
        plot_stages_fn=plot_stages_fn,
        plot_comparison_fn=plot_comp_fn,
        fwhm_mm=args.fwhm_mm, n_dist_out=args.n_dist_out,
        angular_sigma=args.angular_sigma,
        recon_iters=args.osem_iters, recon_subsets=args.osem_subsets,
        mlem_iters=args.mlem_iters,
        gauss_every=args.gauss_every, gauss_sigma=args.gauss_sigma,
        use_psf=args.use_psf,
        rotation_angles_override=args.n_gantry_angles,
        output_prefix=os.path.join(args.output_dir, ""),
        n_img=args.n_img, voxel_mm=args.voxel_mm,
        rays_per_bin=rays_per_bin,
        backproject_chunks=args.backproject_chunks,
        interpolate_radial=args.interpolate,
        cor_mm=args.cor_mm,
        background_scale=args.background_scale, methods=methods,
    )

    if args.save_format != "none":
        res_for_save = {"name": res["name"]}
        for m in methods:
            if m in res:
                res_for_save[m] = res[m]
        save_reconstructions(res_for_save, detector_config, args.n_dist_out,
                             os.path.join(args.output_dir, args.save_format),
                             fmt=args.save_format,
                             fov_mm=res.get("fov_mm"),
                             voxel_mm=res.get("voxel_mm"))

    print(f"\n[OK] Reconstruction finished: {res['name']}")
    print(f"     Shapes: " + ", ".join(f"{m}={res[m].shape}" for m in methods))
    if res.get("geometry"):
        g = res["geometry"]
        print(f"     Grid: {g['n_img']}x{g['n_img']} px @ "
              f"{g['voxel_mm']:.4f} mm | radial bin {g['sino_bin_mm']:.4f} mm "
              f"| rays/bin {g['rays_per_bin']}")
    print(f"     Outputs in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
