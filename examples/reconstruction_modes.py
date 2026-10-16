"""
reconstruction_modes.py — executable catalogue of ALL the reconstruction
modes supported by GimnToolsV2.

Each `mode_*` function is self-contained and demonstrates a combination of:
  - data format:        PETSYS (hardware) or GATE (simulation)
  - process_as:         CrystalID | GateGlobalPositions | GeneralGlobalPosition
  - rotation axis:      rotate_along (x|y|z) — derived from the config (no longer hard-coded)
  - pre-processing:     radial interpolation and/or angular smoothing (on/off)
  - corrections:        COR + encoder offset (PETSYS only, via calibration)
  - algoritmo:          OSEM | MLEM | FBP
  - EM options:         PSF (on/off), Gaussian regularization, legacy on/off

IMPORTANT (scientific honesty): this file does NOT invent data. The
ROOT/JSON paths are placeholders you must point to your real files. Without
data, the functions raise FileNotFoundError on purpose, instead of
fabricating a result.

To run a mode:
    python reconstruction_modes.py gate_general_osem
or import a function and call it with your own paths.
"""

import os
import sys
import numpy as np

# --- imports do pacote ---
from Gimntools.GimnRec.utils import coincidence_to_lor, salvar_reconstrucoes
from Gimntools.GimnRec.filters import interpolate_distance, smooth_angular, ramLak, make_psf_kernel
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.reconstruction_v2 import process_file_v2
from Gimntools.GimnRec.data_loaders import (
    get_default_parameters, load_detector_config, validate_detector_config,
    load_detectors_angles, load_rsectors,
)
from Gimntools.GimnRec.calibration import load_calibration


# ===========================================================================
#  Helpers
# ===========================================================================

def _check(path, label):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"[{label}] not found: {path!r}. Edit the paths at the top of the "
            "function to point to your real data."
        )


def _load_gate_config(cfg_json, ang_json, rsec_json,
                      process_as=None, rotate_along=None, n_gantry=None):
    cfg = load_detector_config(cfg_json)
    for w in validate_detector_config(cfg, "gate"):
        print(f"[WARNING] {w}")
    if process_as:
        cfg["process_as"] = process_as
    if rotate_along:
        cfg["rotate_along"] = rotate_along
    if n_gantry is not None:
        cfg["rotation_angles"] = n_gantry
    for k in ("crystal_size", "sipm_pitch", "sipm_pixel_n"):
        if isinstance(cfg.get(k), list):
            cfg[k] = np.asarray(cfg[k])
    ang = load_detectors_angles(ang_json)
    rsec = load_rsectors(rsec_json)
    return cfg, ang, rsec


def _reconstruct(sino, fov_mm, sino_range_mm, method,
                 iters, subsets, gauss_every, gauss_sigma,
                 psf_kernel=None, legacy=False,
                 n_img=None, voxel_mm=None, rays_per_bin="auto",
                 backproject_chunks=1):
    angles = (np.arange(sino.shape[2]) + 0.5) * (180.0 / sino.shape[2])
    grid_kw = dict(fov_mm=fov_mm, sino_range_mm=sino_range_mm,
                   rays_per_bin=rays_per_bin,
                   backproject_chunks=backproject_chunks)
    if n_img is not None or voxel_mm is not None:
        grid_kw["n_img"] = n_img
        grid_kw["voxel_mm"] = voxel_mm
        if n_img is not None and voxel_mm is not None:
            grid_kw["fov_mm"] = None
    r = siddon_reconstructor(sino, **grid_kw)
    if method == "osem":
        return r.osem(angles, iterations=iters, subsets_n=subsets,
                      psf_kernel=psf_kernel, norm_sino=None,
                      regularization_every=gauss_every, regularization_sigma=gauss_sigma,
                      legacy=legacy), angles
    if method == "mlem":
        return r.mlem(angles, iterations=iters, psf_kernel=psf_kernel,
                      norm_sino=None, regularization_every=gauss_every,
                      regularization_sigma=gauss_sigma, legacy=legacy), angles
    if method == "fbp":
        return r.fbp(angles, filter_type=ramLak), angles
    raise ValueError(f"unknown method: {method}")


# ===========================================================================
#  GATE MODES — no corrections (exact simulator geometry)
# ===========================================================================

def _gate_pipeline(root, cfg_json, ang_json, rsec_json, *,
                   process_as, rotate_along, n_gantry,
                   interpolate=False, n_dist_out=128, angular_sigma=0.5,
                   method="osem", iters=10, subsets=4,
                   gauss_every=5, gauss_sigma=0.4,
                   use_psf=False, fwhm_mm=3.0, legacy=False,
                   n_img=None, voxel_mm=None, rays_per_bin="auto",
                   backproject_chunks=1,
                   matrix=(128, 128, 128), save_fmt=None, out_dir="./out_modes"):
    """Generic, parameterizable GATE pipeline (base of all GATE modes)."""
    for p, l in [(root, "ROOT"), (cfg_json, "config"),
                 (ang_json, "angles"), (rsec_json, "rsectors")]:
        _check(p, l)
    cfg, ang, rsec = _load_gate_config(cfg_json, ang_json, rsec_json,
                                       process_as, rotate_along, n_gantry)
    params = get_default_parameters("gate")

    name = os.path.basename(root).replace(".root", "")
    fov_mm = 2.0 * cfg["Half_FOV"]
    sino_range_mm = cfg.get("sino_range_mm", float(cfg["crystal_size"][1]))

    sino, _ = coincidence_to_lor(root, cfg, ang, params, matrix=list(matrix),
                                 rsectors=rsec, verbose=True)
    print(f"  raw sinogram {sino.shape}, counts={sino.sum():.0f}")

    if interpolate:
        sino = interpolate_distance(
            sino, n_out=n_dist_out, kind="conservative")
    if angular_sigma and angular_sigma > 0:
        sino = smooth_angular(sino, sigma_bins=angular_sigma)

    psf = None
    if use_psf:
        psf = make_psf_kernel(fwhm_mm, sino_range_mm / sino.shape[1])

    vol, _ = _reconstruct(sino, fov_mm, sino_range_mm, method, iters, subsets,
                          gauss_every, gauss_sigma, psf_kernel=psf, legacy=legacy,
                          n_img=n_img, voxel_mm=voxel_mm,
                          rays_per_bin=rays_per_bin,
                          backproject_chunks=backproject_chunks)
    print(f"  volume {vol.shape}  method={method} legacy={legacy}")

    if save_fmt:
        salvar_reconstrucoes({"name": name, method: vol}, cfg, sino.shape[1],
                             os.path.join(out_dir, save_fmt), fmt=save_fmt)
    return vol


# ---- Concrete GATE modes (edit the paths) ----

GATE_ROOT = "/path/to/simulation.root"              # <-- edit
GATE_CFG = "./CONFIG_GATE/detector_config_gate.json"
GATE_ANG = "./CONFIG_GATE/detectors_angles.json"
GATE_RSEC = "./CONFIG_GATE/rsectors_gate.json"


def gate_general_osem():
    """GATE - GeneralGlobalPosition - rotate z - OSEM (no interpolation)."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="osem", iters=10, subsets=4)


def gate_general_mlem():
    """GATE · GeneralGlobalPosition · rotate z · MLEM."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="mlem", iters=50)


def gate_general_fbp():
    """GATE · GeneralGlobalPosition · rotate z · FBP (RamLak)."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="fbp")


def gate_globalpos_osem():
    """GATE · GateGlobalPositions · rotate y · OSEM."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GateGlobalPositions", rotate_along="y",
                          n_gantry=10, method="osem", iters=10, subsets=4)


def gate_crystalid_osem_interp():
    """GATE - CrystalID - OSEM WITH interpolation (avoids coarse pixelation).

    CrystalID data have few native radial bins; interpolating to
    n_dist_out=128 is recommended — hence interpolate=True here.
    """
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="CrystalID", rotate_along="z",
                          n_gantry=10, interpolate=True, n_dist_out=128,
                          method="osem", iters=10, subsets=4)


def gate_general_osem_psf():
    """GATE - GeneralGlobalPosition - OSEM with PSF model (FWHM 3 mm)."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="osem", iters=10, subsets=4,
                          use_psf=True, fwhm_mm=3.0)


def gate_general_osem_legacy():
    """GATE - OSEM with legacy EM (old behaviour) — for comparison."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="osem", iters=10, subsets=4,
                          legacy=True)


def gate_general_osem_save_nrrd():
    """GATE · OSEM · exportar em NRRD."""
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, method="osem", save_fmt="nrrd")


def gate_general_osem_volume_driven():
    """GATE - OSEM on an EXPLICIT 256x256 grid, no sinogram upsampling.

    The image matrix is stated directly instead of being inherited from the
    number of radial bins. `interpolate=False` keeps the measured radial
    bins, and the projector automatically traces enough sub-rays per bin to
    cover the finer voxels.
    """
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, interpolate=False,
                          method="osem", iters=10, subsets=4,
                          n_img=256, backproject_chunks=8)


def gate_general_osem_fixed_voxel():
    """GATE - OSEM with a FIXED physical voxel of 0.25 mm.

    Useful when several acquisitions/scanners must be reconstructed onto a
    common voxel size (e.g. for NEMA metrics or for comparison against a
    GATE ground truth). The matrix is derived as round(FOV / voxel).
    """
    return _gate_pipeline(GATE_ROOT, GATE_CFG, GATE_ANG, GATE_RSEC,
                          process_as="GeneralGlobalPosition", rotate_along="z",
                          n_gantry=10, interpolate=False,
                          method="osem", iters=10, subsets=4,
                          voxel_mm=0.25, backproject_chunks=8)


# ===========================================================================
#  PETSYS MODES — with calibrated corrections (COR + encoder offset)
# ===========================================================================

PETSYS_ROOT = "/path/to/measurement.root"           # <-- edit
PETSYS_BG = "/path/to/background.root"               # <-- edit
PETSYS_CFG = "./CONFIG/detector_config_petsys.json"
PETSYS_ANG = "./CONFIG/detectors_angles.json"
PETSYS_RSEC = "./CONFIG/rsectors_petsys.json"
PETSYS_CAL = "./CONFIG/calibration.json"


def _petsys_pipeline(method_focus, *, n_gantry=12, use_psf=False, legacy=False,
                     save_fmt=None, out_dir="./out_modes_petsys"):
    """Pipeline PETSYS via process_file_v2 (aplica COR + phi_offset calibrados).

    process_file_v2 already runs OSEM+MLEM+FBP in one go; method_focus only
    selects which one to return/save. Corrections come from the calibration JSON.
    """
    for p, l in [(PETSYS_ROOT, "ROOT"), (PETSYS_BG, "BG"), (PETSYS_CFG, "config"),
                 (PETSYS_ANG, "angles"), (PETSYS_RSEC, "rsectors"),
                 (PETSYS_CAL, "calibration")]:
        _check(p, l)
    cfg = load_detector_config(PETSYS_CFG)
    for w in validate_detector_config(cfg, "petsys"):
        print(f"[WARNING] {w}")
    ang = load_detectors_angles(PETSYS_ANG)
    rsec = load_rsectors(PETSYS_RSEC)
    cal = load_calibration(PETSYS_CAL)
    params = get_default_parameters("petsys")

    res = process_file_v2(
        fname=PETSYS_ROOT, bgname=PETSYS_BG, cfg=cfg, det_angles=ang,
        params=params, rsec=rsec, calibration=cal,
        coincidence_to_lor_fn=coincidence_to_lor,
        interpolate_distance_fn=interpolate_distance,
        smooth_angular_fn=smooth_angular,
        siddon_reconstructor_cls=siddon_reconstructor, ramLak_fn=ramLak,
        plot_stages_fn=None, plot_comparison_fn=None,
        use_psf=use_psf, rotation_angles_override=n_gantry,
        output_prefix=os.path.join(out_dir, ""),
    )
    os.makedirs(out_dir, exist_ok=True)
    if save_fmt:
        n_dist = res["distances"].shape[0] if "distances" in res else 128
        salvar_reconstrucoes({"name": res["name"], method_focus: res[method_focus]},
                             cfg, n_dist,
                             os.path.join(out_dir, save_fmt), fmt=save_fmt)
    return res[method_focus]


def petsys_osem_12():
    """PETSYS - 12 angles - OSEM (with calibrated COR + phi_offset)."""
    return _petsys_pipeline("osem", n_gantry=12)


def petsys_mlem_12():
    """PETSYS - 12 angles - MLEM (with corrections)."""
    return _petsys_pipeline("mlem", n_gantry=12)


def petsys_fbp_12():
    """PETSYS - 12 angles - FBP (with corrections)."""
    return _petsys_pipeline("fbp", n_gantry=12)


def petsys_osem_24():
    """PETSYS - 24 angles - OSEM (e.g. 24-position acquisitions)."""
    return _petsys_pipeline("osem", n_gantry=24)


def petsys_osem_psf():
    """PETSYS - OSEM with PSF + calibrated corrections."""
    return _petsys_pipeline("osem", n_gantry=12, use_psf=True)


def petsys_osem_save_mhd():
    """PETSYS · OSEM · exportar em MHD."""
    return _petsys_pipeline("osem", n_gantry=12, save_fmt="mhd")


# ===========================================================================
#  Mode registry + minimal CLI
# ===========================================================================

MODES = {
    # GATE
    "gate_general_osem":          gate_general_osem,
    "gate_general_mlem":          gate_general_mlem,
    "gate_general_fbp":           gate_general_fbp,
    "gate_globalpos_osem":        gate_globalpos_osem,
    "gate_crystalid_osem_interp": gate_crystalid_osem_interp,
    "gate_general_osem_psf":      gate_general_osem_psf,
    "gate_general_osem_legacy":   gate_general_osem_legacy,
    "gate_general_osem_save_nrrd": gate_general_osem_save_nrrd,
    "gate_general_osem_volume_driven": gate_general_osem_volume_driven,
    "gate_general_osem_fixed_voxel":   gate_general_osem_fixed_voxel,
    # PETSYS
    "petsys_osem_12":  petsys_osem_12,
    "petsys_mlem_12":  petsys_mlem_12,
    "petsys_fbp_12":   petsys_fbp_12,
    "petsys_osem_24":  petsys_osem_24,
    "petsys_osem_psf": petsys_osem_psf,
    "petsys_osem_save_mhd": petsys_osem_save_mhd,
}


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "list"):
        print("Available modes:\n")
        for k, fn in MODES.items():
            print(f"  {k:30s} — {(fn.__doc__ or '').strip().splitlines()[0]}")
        print("\nUso: python reconstruction_modes.py <modo>")
        print("(Edit the GATE_*/PETSYS_* paths at the top to point to your data.)")
        return 0
    mode = argv[0]
    if mode not in MODES:
        print(f"Unknown mode: {mode}. Use 'list' to see the available modes.")
        return 2
    MODES[mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
