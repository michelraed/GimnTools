"""Recover list mode from the coincidence trees used for sinogram creation."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from Gimntools.GimnRec.data_loaders import require_valid_detector_config
from Gimntools.GimnRec.detector_response import PETSYS_ID_BRANCHES, _gimnroot_offsets
from Gimntools.GimnRec.listmode_io import event_rows, object_axes, write_listmode
from Gimntools.GimnRec.listmode_response import READOUT_TOLERANCE_MM
from Gimntools.GimnRec.systemSpace import DetectorReference, systemSpace
from Gimntools.GimnRec.utils import _gate_centers, _gate_points, _gate_run_step, _petsys_points


def export_root_listmode(root_path, output, config, detector_angles, mapping, *,
                         gate_global_hits=False, gate_energy_unit="MeV",
                         progress=None, chunk_size="16 MB"):
    """Export actual coincidences in reconstruction XYZ (mm) and keV."""
    import uproot

    fmt = config.get("data_from")
    require_valid_detector_config(config, fmt)
    if Path(root_path).resolve() == Path(output).resolve():
        raise ValueError("Choose a different output path from the source ROOT")
    if gate_energy_unit not in ("MeV", "keV"):
        raise ValueError("Choose the energy unit stored in the GATE ROOT")
    config = dict(config, crystal_size=np.asarray(config["crystal_size"], dtype=float))
    axis, axes = object_axes(config)
    space = systemSpace(detector_angles, axis, config, axis)
    mapping = {int(key): name for key, name in mapping.items() if name is not None}
    metadata = {
        "source": "root coincidences", "format": fmt, "input_file": Path(root_path).name,
        "detector_config": {key: np.asarray(value).tolist() for key, value in config.items()},
        "detector_angles_deg": {name: float(angle) for name, angle in detector_angles.items()},
        "detector_mapping": mapping, "rotation_axis": axis, "object_axes": ["xyz"[a] for a in axes],
    }
    tree_name = "tomographicCoincidences" if fmt == "PETSYS" else "Coincidences"
    with uproot.open(root_path) as root:
        if tree_name not in root:
            raise ValueError(f"ROOT needs the {tree_name} event tree. A histogram cannot supply the two hits")
        tree = root[tree_name]
        available = set(tree.keys())
        verify_channels = False
        if fmt == "PETSYS":
            rotation = Rotation.from_euler("z", config["det_z_rotation"], degrees=True)
            origin = DetectorReference(np.zeros(3), config["crystal_size"],
                                       config["pla_thickness"], config["radius"])
            energy_fields = ["energyCorrected_1", "energyCorrected_2"]
            branches = ["angleGantry_1", *energy_fields]
            for hit in (1, 2):
                branches.extend(f"{name}_{hit}" for name in ("chipID", "SiPMPosX", "SiPMPosY"))
            if not mapping or set(mapping.values()) - detector_angles.keys():
                raise ValueError("Map the PETSYS chips to detector heads before exporting")
            verify_channels = (config["crystal_type"] == "segmented"
                               and list(config["sipm_pixel_n"]) == [8, 8]
                               and set(PETSYS_ID_BRANCHES[:2]) <= available)
            if verify_channels:
                branches.extend(key for key in PETSYS_ID_BRANCHES if key in available)
            metadata["hit_positions"] = "recorded SiPM positions"
        else:
            energy_fields = next(([f"{key}1", f"{key}2"] for key in ("energy", "edep")
                                  if {f"{key}1", f"{key}2"} <= available), None)
            if energy_fields is None:
                raise ValueError("GATE ROOT needs energy1/2 or edep1/2 branches for list-mode export")
            step = _gate_run_step(config)
            centers = (_gate_centers(space, config, mapping)
                       if config["process_as"] == "CrystalID" and not gate_global_hits else None)
            branches = [*energy_fields]
            if step or centers is not None or "runID" in available:
                branches.append("runID")
            if centers is None:
                branches.extend(f"globalPos{coord}{hit}" for hit in (1, 2) for coord in "XYZ")
            else:
                branches.extend(f"{name}{hit}" for hit in (1, 2) for name in ("crystalID", "rsectorID"))
            metadata.update(input_energy_unit=gate_energy_unit,
                            hit_positions="recorded global positions" if centers is None else "crystal centres")
        if not set(branches) <= available:
            raise ValueError(f"ROOT is missing required coincidence fields: {sorted(set(branches) - available)}")

        def batches():
            visited = 0
            by_step, pose_indices = {}, {}
            alignment_bounds = {}
            invalid_alignment = 0
            pitch = np.asarray(config["sipm_pitch"], dtype=float)
            for data in tree.iterate(branches, step_size=chunk_size, library="np"):
                size = len(data[energy_fields[0]])
                visited += size
                if fmt == "PETSYS":
                    energy_window = (config["e_min"], config["e_max"])
                    points, accepted = _petsys_points(data, slice(None), space, mapping, rotation, origin, energy_window)
                    if not points:
                        if progress:
                            progress(visited, tree.num_entries)
                        continue
                    poses = data["angleGantry_1"][accepted].astype(float)
                    indices = np.full(len(poses), -1, dtype=np.int64)
                    energies = [data[key][accepted] for key in energy_fields]
                else:
                    points, _ = _gate_points(data, slice(None), space, centers, step)
                    indices = (data["runID"].astype(np.int64) if "runID" in branches
                               else np.full(size, -1, dtype=np.int64))
                    poses = indices * step
                    scale = 1000. if gate_energy_unit == "MeV" else 1.
                    energies = [data[key].astype(float) * scale for key in energy_fields]
                rows = event_rows(*(hit[:, axes] for hit in points), energies)
                valid = np.isfinite(poses) & np.isfinite(rows).all(axis=1)
                valid &= np.any(rows[:, :3] != rows[:, 4:7], axis=1)
                if verify_channels and valid.any():
                    retained = accepted.copy()
                    retained[accepted] = valid
                    for hit in (1, 2):
                        offsets, good, _, _ = _gimnroot_offsets(data, slice(None), hit, retained, pitch)
                        invalid_alignment += np.count_nonzero(~good)
                        chips = data[f"chipID_{hit}"][retained][good]
                        for chip in np.unique(chips):
                            points = offsets[good][chips == chip]
                            head = mapping[int(chip)]
                            low, high = alignment_bounds.get(head, (np.inf, -np.inf))
                            alignment_bounds[head] = (np.minimum(low, points.min(axis=0)),
                                                      np.maximum(high, points.max(axis=0)))
                if fmt == "PETSYS":
                    pose_values, first_appearance, inverse = np.unique(
                        poses[valid], return_index=True, return_inverse=True,
                    )
                    ids = np.empty(len(pose_values), dtype=np.int64)
                    for i in np.argsort(first_appearance):
                        ids[i] = pose_indices.setdefault(float(pose_values[i]), len(pose_indices))
                    indices[valid] = ids[inverse]
                pairs = np.column_stack((indices[valid], poses[valid]))
                for (index, angle), count in zip(*np.unique(pairs, axis=0, return_counts=True)):
                    key = (int(index), float(angle))
                    by_step[key] = by_step.get(key, 0) + int(count)
                yield rows[valid], poses[valid], indices[valid]
                if progress:
                    progress(visited, tree.num_entries)
            metadata["input_coincidences"] = visited
            metadata["poses"] = [{"step": index, "angle_deg": angle, "coincidences": count}
                                 for (index, angle), count in sorted(by_step.items())]
            if fmt == "PETSYS" and by_step:
                metadata["detector_config"]["rotation_angles"] = len(by_step)
            if verify_channels:
                unstable = sorted(head for head, (low, high) in alignment_bounds.items()
                                  if np.any(high - low > 2 * READOUT_TOLERANCE_MM))
                if invalid_alignment or unstable:
                    metadata["petsys_alignment_error"] = (
                        f"{invalid_alignment} invalid channel/index hits, inconsistent translations on {unstable}"
                    )
                elif set(mapping.values()) <= alignment_bounds.keys():
                    metadata["petsys_bin_offsets_mm"] = {
                        head: ((low + high) / 2).tolist() for head, (low, high) in alignment_bounds.items()
                    }

        return write_listmode(output, batches(), metadata)
