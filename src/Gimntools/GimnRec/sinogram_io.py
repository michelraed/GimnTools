"""Portable sinogram archives with the measured acquisition metadata."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np

from Gimntools.GimnRec.storage import dumps


_FIELDS = ("name", "source", "fov_mm", "sino_range_mm", "detector_config",
           "cor_mm", "dx_cor_mm", "dy_cor_mm", "acquisition", "rebinned")
_VERSION = "gimntools-sinogram-v1"


def save_sinogram(path, entry):
    metadata = {key: entry[key] for key in _FIELDS}
    acquisition = dict(metadata["acquisition"])
    for key in ("measurement", "background"):
        if acquisition.get(key):
            acquisition[key] = str(acquisition[key]).replace("\\", "/").rsplit("/", 1)[-1]
    metadata["acquisition"] = acquisition
    arrays = {
        "format": _VERSION,
        "metadata": dumps(metadata),
        "counts": entry["sino"],
        "angles_deg": entry["angles"],
    }
    arrays.update({key: entry[key] for key in ("norm", "background") if entry[key] is not None})
    path = Path(path)
    output = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(output.name)
    try:
        with output:
            np.savez_compressed(output, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_sinogram(path):
    try:
        with np.load(path, allow_pickle=False) as file:
            if file["format"].item() != _VERSION:
                raise ValueError("Unsupported sinogram format")
            entry = json.loads(file["metadata"].item())
            if not isinstance(entry["name"], str) or not isinstance(entry["detector_config"], dict):
                raise ValueError("Sinogram archive must contain a name and detector setup")
            measured = entry["acquisition"]
            count = measured.get("observed_count") if isinstance(measured, dict) else None
            if type(count) is not int or count < 0:
                raise ValueError("Sinogram archive lacks a measured position or run count")
            if entry["source"] == "GATE":
                if measured.get("count_kind") != "runs" or count != len(measured["gate_run_ids"]):
                    raise ValueError("GATE archive has inconsistent run metadata")
            elif entry["source"] == "PETSYS":
                if measured.get("count_kind") != "positions" or count != len(measured["object_angles_deg"]):
                    raise ValueError("PETSYS archive has inconsistent rotation metadata")
            else:
                raise ValueError("Sinogram archive must identify PETSYS or GATE data")
            result = {key: entry[key] for key in _FIELDS}
            result.update(sino=file["counts"], angles_override=file["angles_deg"],
                          norm=file["norm"] if "norm" in file else None,
                          background=file["background"] if "background" in file else None)
    except (KeyError, TypeError, AttributeError, BadZipFile) as exc:
        raise ValueError("Select a sinogram NPZ saved by GimnTools with acquisition metadata") from exc
    return result
