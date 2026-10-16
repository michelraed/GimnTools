"""
data_loaders.py
===============

Abstraction layer for loading coincidence data in the two supported
formats:

    - PETSYS: branch "tomographicCoincidences", events with
      SiPMPosX_*, SiPMPosY_*, chipID_*, energyCorrected_*, slice_1,
      angleGantry_1. Each event carries the gantry angle.

    - GATE:   branch "Coincidences", events with globalPos*1/2,
      crystalID1/2, rsectorID1/2, runID. The gantry angle is inferred
      from runID (each run = one angle, with step astep).

This layer does NOT replace the package's `coincidence_to_lor` — it just
invokes it with the appropriate configuration, exposes sensible per-format
defaults, and provides detectors_angles/rsectors generators.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
#  Angle + rsectors generator (same as used in the notebooks)
# ---------------------------------------------------------------------------

def generate_rsector_angle(n_detectors):
    """Generate equally spaced detectors over [0, 360).

    Exactly reproduces the notebooks' `generateRsectorAngle`:
    detector-(i+1) -> angle i*360/N, and rsectorID i -> detector-(i+1).

    Returns
    -------
    detectors_angles : dict {str: float}   ex.: {'detector-1': 0.0, ...}
    rsectors         : dict {int: str}      ex.: {0: 'detector-1', ...}
    """
    n = int(n_detectors)
    if n < 1:
        raise ValueError("n_detectors must be >= 1")
    angles = np.linspace(0, 360, n, endpoint=False)
    detectors_angles = {}
    rsectors = {}
    for i, angle in enumerate(angles):
        name = f"detector-{i + 1}"
        detectors_angles[name] = float(angle)
        rsectors[i] = name
    return detectors_angles, rsectors


# ---------------------------------------------------------------------------
#  Default parameters per format
# ---------------------------------------------------------------------------

PETSYS_DEFAULT_PARAMETERS = [
    "SiPMPosX_1", "SiPMPosY_1", "SiPMPosX_2", "SiPMPosY_2",
    "chipID_1", "chipID_2",
    "energyCorrected_1", "energyCorrected_2",
    "slice_1", "angleGantry_1",
]

GATE_DEFAULT_PARAMETERS = [
    "globalPosX1", "globalPosY1", "globalPosZ1",
    "globalPosX2", "globalPosY2", "globalPosZ2",
    "crystalID1", "crystalID2",
    "rsectorID1", "rsectorID2",
    "runID",
]


def get_default_parameters(data_format):
    """Return the default parameter list for the given format."""
    fmt = data_format.upper()
    if fmt == "PETSYS":
        return list(PETSYS_DEFAULT_PARAMETERS)
    elif fmt == "GATE":
        return list(GATE_DEFAULT_PARAMETERS)
    else:
        raise ValueError(f"unknown data_format: {data_format!r}")


# ---------------------------------------------------------------------------
#  Unified loader (delegates to the package's coincidence_to_lor)
# ---------------------------------------------------------------------------

def get_coincidence_loader():
    """
    Return the package's `coincidence_to_lor` function.

    The function supports both formats via `detector_config['data_from']`
    (the string "PETSYS" or "GATE"): pass the matching config and the
    function branches internally.
    """
    from Gimntools.GimnRec.utils import coincidence_to_lor
    return coincidence_to_lor


def get_pipeline_functions():
    """
    Return a dict with the package functions used by the calibration and
    reconstruction modules.

    NOTE (GimnToolsV2 fix): the old version imported from a legacy
    monolithic module `petsys_pipeline_completo`, which no longer exists in
    this package layout — that made this function fail unconditionally. It
    now imports directly from the `Gimntools.GimnRec` submodules.
    """
    from Gimntools.GimnRec.utils import (
        coincidence_to_lor, salvar_reconstrucoes_mhd,
    )
    from Gimntools.GimnRec.filters import (
        interpolate_distance, smooth_angular, ramLak,
    )
    from Gimntools.GimnRec.reconstructor import siddon_reconstructor
    from Gimntools.GimnRec.plots import (
        plot_stages_mm, plot_comparison_mm, plot_profiles_fwhm,
    )
    return {
        'coincidence_to_lor':       coincidence_to_lor,
        'interpolate_distance':     interpolate_distance,
        'smooth_angular':           smooth_angular,
        'siddon_reconstructor':     siddon_reconstructor,
        'ramLak':                   ramLak,
        'plot_stages_mm':           plot_stages_mm,
        'plot_comparison_mm':       plot_comparison_mm,
        'plot_profiles_fwhm':       plot_profiles_fwhm,
        'salvar_reconstrucoes_mhd': salvar_reconstrucoes_mhd,
    }


# ---------------------------------------------------------------------------
#  Detector config — load / save / validate
# ---------------------------------------------------------------------------

# Fields that must be numpy arrays in detector_config (JSON gives lists)
_DETECTOR_ARRAY_FIELDS = ['crystal_size', 'sipm_pitch', 'sipm_pixel_n']

# Required fields per format
_REQUIRED_PETSYS = {
    'data_from', 'pla_thickness', 'radius', 'crystal_size',
    'sipm_pitch', 'sipm_pixel_n', 'crystal_type', 'rotation_angles',
    'arround', 'detNum', 'Half_FOV', 'rotate_along',
    'e_min', 'e_max', 'det_z_rotation',
}
_REQUIRED_GATE = {
    'data_from', 'pla_thickness', 'radius', 'crystal_size',
    'sipm_pitch', 'sipm_pixel_n', 'crystal_type', 'rotation_angles',
    'arround', 'detNum', 'Half_FOV', 'process_as',
}


def _strip_comments(d):
    """Drop keys starting with '_' (used for annotations inside the JSONs)."""
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items()
                if not (isinstance(k, str) and k.startswith('_'))}
    if isinstance(d, list):
        return [_strip_comments(x) for x in d]
    return d


def load_detector_config(json_path):
    """Read detector_config from JSON, converting array fields to np.ndarray."""
    with open(json_path) as f:
        cfg = _strip_comments(json.load(f))
    for k in _DETECTOR_ARRAY_FIELDS:
        if k in cfg and isinstance(cfg[k], list):
            cfg[k] = np.asarray(cfg[k])
    return cfg


def save_detector_config(cfg, json_path):
    """Save detector_config as JSON (arrays converted to lists)."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)


def validate_detector_config(cfg, data_format):
    """
    Validate that detector_config has the fields expected for the format.
    Returns a list of warnings (empty = OK).
    """
    fmt = data_format.upper()
    cfg_fmt = cfg.get('data_from', '').upper()
    warnings = []

    if cfg_fmt != fmt:
        warnings.append(
            f"detector_config.data_from='{cfg_fmt}' does not match "
            f"--data-format={fmt}. Using {fmt}."
        )
        cfg['data_from'] = fmt

    required = _REQUIRED_PETSYS if fmt == "PETSYS" else _REQUIRED_GATE
    missing = required - set(cfg.keys())
    if missing:
        warnings.append(f"detector_config is missing fields: {sorted(missing)}")
    return warnings


def require_valid_detector_config(cfg, data_format):
    """Reject physically invalid or incomplete detector configuration.

    Unlike :func:`validate_detector_config` (UI-friendly warnings), this
    strict gate is used before processing real ROOT events. It deliberately
    does not infer missing physical dimensions or acquisition settings.
    """
    fmt = str(data_format).upper()
    if fmt not in ("PETSYS", "GATE"):
        raise ValueError("data_from must be 'PETSYS' or 'GATE'")
    if not isinstance(cfg, dict):
        raise ValueError("detector_config must be a dictionary")
    required = _REQUIRED_PETSYS if fmt == "PETSYS" else _REQUIRED_GATE
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"detector_config missing required fields: {missing}")
    if str(cfg['data_from']).upper() != fmt:
        raise ValueError("detector_config.data_from does not match data format")

    def positive(key, zero_allowed=False):
        try:
            a = np.asarray(cfg[key], dtype=float)
            if a.size != 1:
                raise ValueError()
            x = float(a.item())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"detector_config.{key} must be a finite scalar") from exc
        if not np.isfinite(x) or (x < 0 if zero_allowed else x <= 0):
            raise ValueError(f"detector_config.{key} must be finite and " +
                             ("nonnegative" if zero_allowed else "positive"))
        return x

    for key in ("radius", "Half_FOV", "pla_thickness"):
        positive(key, zero_allowed=(key == "pla_thickness"))
    if "sino_range_mm" in cfg:
        positive("sino_range_mm")
    if "slice_thickness_mm" in cfg:
        positive("slice_thickness_mm")
    for key, n in (("crystal_size", 3), ("sipm_pitch", 2)):
        try:
            a = np.asarray(cfg[key], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"detector_config.{key} must contain {n} positive entries") from exc
        if a.shape != (n,) or not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f"detector_config.{key} must contain {n} positive finite entries")
    n = np.asarray(cfg['sipm_pixel_n'], dtype=float)
    if n.shape != (2,) or not np.all(np.isfinite(n)) or np.any(n < 1) or np.any(n != np.floor(n)):
        raise ValueError("detector_config.sipm_pixel_n must contain two positive integers")
    for key in ('rotation_angles', 'detNum'):
        value = positive(key)
        if value != int(value):
            raise ValueError(f"detector_config.{key} must be a positive integer")
    if "arround" in cfg:
        positive("arround")
    if str(cfg.get('rotate_along', 'y')).lower() not in ("x", "y", "z"):
        raise ValueError("detector_config.rotate_along must be x, y or z")
    if fmt == "GATE" and cfg['process_as'] not in (
        "CrystalID", "GateGlobalPositions", "GeneralGlobalPosition",
        "PaulGlobalPosition"
    ):
        raise ValueError("detector_config.process_as is unknown")
    if fmt == "PETSYS":
        emin = positive("e_min", zero_allowed=True)
        emax = positive("e_max")
        if emin >= emax:
            raise ValueError("detector_config requires e_min < e_max")
        angle = np.asarray(cfg['det_z_rotation'], dtype=float)
        if angle.size != 1 or not np.isfinite(angle.item()):
            raise ValueError("detector_config.det_z_rotation must be finite")
    return cfg


# ---------------------------------------------------------------------------
#  Detectors angles + rsectors — load
# ---------------------------------------------------------------------------

def load_detectors_angles(json_path):
    """
    Read a {detector_name: angle_degrees} dict from JSON.
    Ex.: {"detector-1": 0, "detector-2": -180}
    """
    with open(json_path) as f:
        return _strip_comments(json.load(f))


def load_rsectors(json_path):
    """
    Read a {chipID_int: detector_name} dict.
    JSON forces keys to strings, so they are converted back to int here.
    Ex.: {"0": "detector-1", "15": "detector-2"}
    """
    with open(json_path) as f:
        raw = _strip_comments(json.load(f))
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except ValueError:
            # in case GATE really uses string keys
            out[k] = v
    return out


# ---------------------------------------------------------------------------
#  Specs (calibration / measurement) — load
# ---------------------------------------------------------------------------

def load_calibration_specs(json_path):
    """
    Read a list of dicts with {path, sources, I, label, n_gantry_angles}.

    Resolves `path` relative to the JSON's directory (when relative).
    """
    with open(json_path) as f:
        specs = _strip_comments(json.load(f))

    base_dir = os.path.dirname(os.path.abspath(json_path))
    for s in specs:
        if not os.path.isabs(s['path']):
            s['path'] = os.path.normpath(os.path.join(base_dir, s['path']))
        # Convert sources [[x,y],...] to [(x,y),...] (tuples)
        s['sources'] = [tuple(p) for p in s['sources']]
    return specs


def load_measurement_specs(json_path):
    """
    Read a list of dicts with:
        {path, n_gantry_angles, kind: 'point'|'dual'|'triple',
         expected_xy: [[x,y], ...]}
    """
    with open(json_path) as f:
        specs = _strip_comments(json.load(f))

    base_dir = os.path.dirname(os.path.abspath(json_path))
    for s in specs:
        if not os.path.isabs(s['path']):
            s['path'] = os.path.normpath(os.path.join(base_dir, s['path']))
        if 'expected_xy' in s and s['expected_xy']:
            s['expected_xy'] = [tuple(p) for p in s['expected_xy']]
    return specs
