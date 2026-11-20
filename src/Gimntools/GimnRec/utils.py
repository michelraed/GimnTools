"""Sinogram building from coincidence data (ROOT files) and volume export.

Main entry points:
  - :func:`coincidence_to_lor` — build a (slice, distance, angle) sinogram
    from a GATE or PETSYS ROOT file.
  - :func:`save_reconstructions` — save reconstructed volumes as MHD/NRRD
    (Portuguese alias kept: :func:`salvar_reconstrucoes`).
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import index

from scipy.spatial.transform import Rotation as ROT
from Gimntools.GimnRec.systemSpace import DetectorReference, systemSpace
import numpy as np
import pandas as pd
import os
import warnings


def _resolve_axes(detector_config):
    """Derive the system axes from detector_config['rotate_along'].

    In tomography the rotation axis is PARALLEL to the slice axis; the
    projection happens in the plane perpendicular to it. global_to_sinogram
    already follows this convention:
        rotate_along='z' -> project onto (x, y), slice = z (index 2)
        rotate_along='y' -> project onto (x, z), slice = y (index 1)
        rotate_along='x' -> project onto (y, z), slice = x (index 0)

    Returns (rotate_along, slice_axis_index).

    METHODOLOGY NOTE (GimnToolsV2): in the original code the gantry rotation
    axis was hard-coded to 'y' (systemSpace.rotateGantry/rotateBy and the
    systemSpace construction inside coincidence_to_lor), while the PROJECTION
    axis varied per `process_as` branch ('y' for GateGlobalPositions, 'z' for
    GeneralGlobalPosition). That is internally inconsistent whenever
    rotate_along != 'y'. Everything now derives from rotate_along.
    For rotate_along='y' the result is IDENTICAL to the old code; it only
    changes when a different axis is configured.
    """
    ra = str(detector_config.get("rotate_along", "y")).lower()
    if ra not in ("x", "y", "z"):
        raise ValueError(f"invalid rotate_along: {ra!r} (expected 'x', 'y' or 'z')")
    slice_axis_index = {"x": 0, "y": 1, "z": 2}[ra]
    return ra, slice_axis_index



_LOR_CHUNK_SIZE = 32_768


def _lor_coordinates(p1, p2):
    du, dv = (p2 - p1).T
    midpoint = 0.5 * (p1 + p2)
    phi = np.arctan2(du, -dv) % np.pi
    distance = midpoint[:, 0] * np.cos(phi) + midpoint[:, 1] * np.sin(phi)
    angle = np.rad2deg(phi)
    degenerate = (du == 0) & (dv == 0)
    distance[degenerate] = angle[degenerate] = np.nan
    return distance, angle


def _lor_bins(distance, angle, axial, shape, radial_half):
    nz, nr, na = shape
    radial = np.floor(((distance + radial_half) / (2.0 * radial_half)) * nr)
    angular = np.floor((angle / 180.0) * na)
    valid = ((axial >= 0) & (axial < nz) & (radial >= 0) & (radial < nr)
             & (angular >= 0) & (angular < na))
    return ((axial[valid].astype(np.intp) * nr + radial[valid].astype(np.intp)) * na
            + angular[valid].astype(np.intp))


def _petsys_points(data, block, space, rsectors, local_rotation, origin, energy_range):
    """Transform energy-selected GimnROOT hits into the shared object frame."""
    chips = [data[f"chipID_{hit}"][block] for hit in (1, 2)]
    energies = [data[f"energyCorrected_{hit}"][block] for hit in (1, 2)]
    e_min, e_max = energy_range
    reject = ((energies[0] < e_min) | (energies[0] > e_max)
              | (energies[1] < e_min) | (energies[1] > e_max))
    mapped_chips = list(rsectors)
    accepted = ~reject & np.isin(chips[0], mapped_chips) & np.isin(chips[1], mapped_chips)

    gantry_angle = data["angleGantry_1"][block][accepted]
    points = []
    if gantry_angle.size:
        gantry = ROT.from_euler(space.rotate_along, -gantry_angle[:, None], degrees=True)
        for hit, ids in enumerate(chips, 1):
            ids = ids[accepted]
            coords = np.column_stack((data[f"SiPMPosX_{hit}"][block][accepted],
                                      data[f"SiPMPosY_{hit}"][block][accepted],
                                      np.zeros(ids.size)))
            coords = local_rotation.apply(coords)[:, ::-1] + origin
            for chip in np.unique(ids):
                group = ids == chip
                coords[group] = space.detectors[rsectors[chip]].apply(coords[group])
            points.append(gantry.apply(coords))
    return points, accepted


def _petsys_chunk(start, *, data, space, rsectors, local_rotation, origin,
                  plane, energy_range, radial_half):
    block = slice(start, start + _LOR_CHUNK_SIZE)
    points, accepted = _petsys_points(data, block, space, rsectors, local_rotation, origin, energy_range)
    if not points:
        return np.empty(0), np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)
    distance, angle = _lor_coordinates(*(coords[:, plane] for coords in points))
    axial = data["slice_1"][block][accepted]
    if not np.issubdtype(axial.dtype, np.integer):
        # Preserve truncation toward zero and the scalar reader's error on non-finite slices.
        if np.all(np.isfinite(axial) & (axial >= -2.0**63) & (axial < 2.0**63)):
            axial = axial.astype(np.int64)
        else:
            axial = np.asarray([int(value) for value in axial])

    bins = _lor_bins(distance, angle, axial, space.sinogram.shape, radial_half)
    return distance, angle, axial, bins


def _collect_lor_chunks(processor, sino, total, workers):
    workers = index(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    starts = range(0, total, _LOR_CHUNK_SIZE)
    columns = {"distance": [], "angle": [], "slice": []}

    def collect(results):
        for distance, angle, axial, bins in results:
            # Workers compute geometry. One ordered reduction owns the histogram.
            np.add.at(sino.ravel(), bins, 1)
            if distance.size:
                for key, values in zip(columns, (distance, angle, axial)):
                    columns[key].append(values)

    workers = min(workers, len(starts))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Bound outstanding chunks so large files do not queue every future at once.
            for first in range(0, len(starts), workers * 2):
                collect(executor.map(processor, starts[first:first + workers * 2]))
    else:
        collect(map(processor, starts))
    events = pd.DataFrame({key: np.concatenate(parts) if parts else []
                           for key, parts in columns.items()})
    return sino, events


def _petsys_sinogram(data, space, config, rsectors, workers):
    total = len(data["SiPMPosX_1"])
    if workers is None:
        workers = min(2, os.cpu_count() or 1) if total >= 1_000_000 else 1
    processor = partial(
        _petsys_chunk, data=data, space=space,
        rsectors={chip: name for chip, name in rsectors.items() if name is not None},
        local_rotation=ROT.from_euler("z", config["det_z_rotation"], degrees=True),
        origin=DetectorReference(np.zeros(3), config["crystal_size"],
                                 config["pla_thickness"], config["radius"]),
        plane=space._PLANE_AXES[space.rotate_along],
        energy_range=(config["e_min"], config["e_max"]),
        radial_half=float(config.get("sino_range_mm", config["crystal_size"][1])) / 2,
    )
    return _collect_lor_chunks(processor, space.sinogram, total, workers)


def _gate_centers(space, cfg, rsectors):
    columns, rows = map(int, cfg["sipm_pixel_n"])
    row, column = np.divmod(np.arange(columns * rows), columns)
    local = np.column_stack((np.zeros(len(row)), -np.asarray(space.positionsY)[row],
                             -np.asarray(space.positionsX)[columns - 1 - column]))
    origin = DetectorReference(np.zeros(3), cfg["crystal_size"], cfg["pla_thickness"], cfg["radius"])
    return {sector: space.detectors[name].apply(local + origin)
            for sector, name in rsectors.items() if name in space.detectors}


def _gate_points(data, block, space, centers, step_deg, *, pose_degrees=None):
    points = []
    axial = None
    if centers is not None:
        axial = np.zeros(len(data["crystalID1"][block]), dtype=np.intp)
        total_crystals = space.cols * space.rows
        for hit in (1, 2):
            crystal = data[f"crystalID{hit}"][block]
            if np.any((crystal < 0) | (crystal >= total_crystals) | (crystal != np.floor(crystal))):
                raise ValueError("crystal_id outside segmented detector grid")
            crystal = crystal.astype(np.intp, copy=False)
            axial += crystal // space.cols
            sector = data[f"rsectorID{hit}"][block]
            coords = np.empty((len(crystal), 3))
            for item in np.unique(sector):
                group = sector == item
                if item not in centers:
                    raise ValueError(f"GATE rsectorID {item} is missing from the detector mapping")
                coords[group] = centers[item][crystal[group]]
            points.append(coords)
    else:
        points = [np.column_stack([data[f"globalPos{axis}{hit}"][block] for axis in "XYZ"])
                  .astype(float, copy=False) for hit in (1, 2)]
    if pose_degrees is None and step_deg:
        pose_degrees = step_deg * data["runID"][block].astype(np.int64)
    if pose_degrees is not None:
        rotation = ROT.from_euler(space.rotate_along, -pose_degrees[:, None], degrees=True)
        points = [rotation.apply(coords) for coords in points]
    return points, axial


def _gate_chunk(start, *, data, space, centers, step_deg, slice_axis, radial_half, axial_width):
    block = slice(start, start + _LOR_CHUNK_SIZE)
    points, axial = _gate_points(data, block, space, centers, step_deg)
    if axial is None:
        # Direct/cross-plane axial bins use the mean of the two positions.
        nz = space.sinogram.shape[0]
        summed = points[0][:, slice_axis] + points[1][:, slice_axis]
        bins = np.floor((summed + axial_width) / (2 * axial_width / nz))
        valid = (bins >= 0) & (bins < nz)
        axial = np.where(valid, bins, -1).astype(np.intp)
    plane = space._PLANE_AXES[space.rotate_along]
    distance, angle = _lor_coordinates(*(coords[:, plane] for coords in points))
    return distance, angle, axial, _lor_bins(distance, angle, axial, space.sinogram.shape, radial_half)


def _gate_sinogram(data, space, cfg, rsectors, step_deg, slice_axis, workers):
    total = len(data["runID"])
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    centers = _gate_centers(space, cfg, rsectors) if cfg["process_as"] == "CrystalID" else None
    processor = partial(_gate_chunk, data=data, space=space, centers=centers,
                        step_deg=step_deg, slice_axis=slice_axis,
                        radial_half=float(cfg.get("sino_range_mm", cfg["crystal_size"][1])) / 2,
                        axial_width=float(cfg["crystal_size"][1]))
    return _collect_lor_chunks(processor, space.sinogram, total, workers)


def _gate_run_step(config):
    # GeneralGlobalPosition already stores hits in the object frame.
    if config["process_as"] == "GeneralGlobalPosition":
        return 0.
    return (360 / config["rotation_angles"]) / config["detNum"]


def coincidence_to_lor(file_name, detector_config, detectors_angles, parameters,
                       matrix=[128,128,128], rsectors={0:"detector-1",1:"detector-2"},
                       first_rotation=None, verbose=False, *, geometry_out=None, workers=None,
                       infer_rotation_angles=False):
    """Build the (slice, distance, angle) sinogram from a ROOT file.

    Parameters
    ----------
    workers : int or None
        Number of geometry workers. None selects up to four for GATE and
        up to two for PETSYS acquisitions with at least one million events.
        One worker processes vectorized batches on the caller thread.
    geometry_out : dict or None
        Receives the geometry, observed PETSYS poses and GATE run IDs.
    infer_rotation_angles : bool
        Size a PETSYS sinogram from the number of distinct recorded poses.
        GATE uses the configured rotation plan to transform each run ID.
    first_rotation : str or None
        Axis used to place the detectors initially. If None, uses the same
        axis as rotate_along (the geometrically consistent choice: detector
        ring perpendicular to the rotation axis). It can be overridden when
        the physical setup places the detectors on one axis and rotates the
        gantry about another.
    """
    # Fail early on invalid hardware configuration rather than processing
    # thousands of events with corrupt geometric coefficients.
    from Gimntools.GimnRec.data_loaders import require_valid_detector_config
    require_valid_detector_config(detector_config, detector_config.get("data_from", ""))
    detector_config = dict(detector_config,
                           crystal_size=np.asarray(detector_config["crystal_size"], dtype=float))
    import uproot as up   # ROOT dependency only when reading actual ROOT files
    rotate_along, slice_axis = _resolve_axes(detector_config)
    first_rot = (rotate_along if first_rotation is None
                 else str(first_rotation).lower())
    if first_rot not in ("x", "y", "z"):
        raise ValueError("first_rotation must be 'x', 'y' or 'z'")
    if verbose:
        print(f"[coincidence_to_lor] rotate_along={rotate_along!r}, "
              f"first_rotation={first_rot!r}, slice_axis_index={slice_axis}")

    if detector_config["data_from"] == "GATE":
        PETSYS = systemSpace(detectors_angles, rotate_along, detector_config, first_rot)
        with up.open(file_name) as root:
            c = root["Coincidences"].arrays(parameters, library="np")
        if detector_config["process_as"] in ("GateGlobalPositions","PaulGlobalPosition","GeneralGlobalPosition"):
            PETSYS.set_matrix(matrix)
        astep = _gate_run_step(detector_config)
        if geometry_out is not None:
            from Gimntools.GimnRec.detector_response import acquisition_geometry
            runs = np.unique(c["runID"])
            geometry_out.update(acquisition_geometry(
                file_name, detector_config, detectors_angles, rsectors,
                first_rot, -astep * runs.astype(float),
            ))
            geometry_out["gate_run_ids"] = runs.astype(int).tolist()
            geometry_out["count_kind"] = "runs"
            geometry_out["observed_count"] = len(runs)
        return _gate_sinogram(c, PETSYS, detector_config, rsectors, astep, slice_axis, workers)

    elif detector_config["data_from"] == "PETSYS":
        with up.open(file_name) as root:
            tree = root["tomographicCoincidences"]
            branches = parameters
            if geometry_out is not None:
                from Gimntools.GimnRec.detector_response import PETSYS_ID_BRANCHES
                available = set(tree.keys())
                branches = list(dict.fromkeys([*parameters, *(
                    name for name in PETSYS_ID_BRANCHES if name in available
                )]))
            co = tree.arrays(branches, library="np")
        poses = None
        if infer_rotation_angles or geometry_out is not None:
            poses = np.unique(co["angleGantry_1"].astype(float))
        if infer_rotation_angles:
            if not poses.size or not np.isfinite(poses).all():
                raise ValueError("PETSYS rotation detection requires finite positions in angleGantry_1")
            detector_config["rotation_angles"] = len(poses)
        PETSYS = systemSpace(detectors_angles, rotate_along, detector_config, first_rot)
        if geometry_out is not None:
            from Gimntools.GimnRec.detector_response import (
                acquisition_geometry, check_petsys_lattice,
            )
            geometry_out.update(acquisition_geometry(
                file_name, detector_config, detectors_angles, rsectors,
                first_rot, -poses,
            ))
            geometry_out["count_kind"] = "positions"
            geometry_out["observed_count"] = int(np.count_nonzero(np.isfinite(poses)))
            geometry_out.update(check_petsys_lattice(co, detector_config, rsectors))
        ev_n = co['SiPMPosX_1'].shape[0]
        print(f"PETSYS: {ev_n} events — {os.path.basename(file_name)}")
        sino, df = _petsys_sinogram(co, PETSYS, detector_config, rsectors, workers)
        print(f"  Sino: {sino.shape}, max={sino.max():.0f}, LORs={len(df)}")
        return sino, df
    else:
        print("Unrecognized data_from value."); return None, None


def resolve_slice_thickness(detector_config, n_slices):
    """Axial spacing (mm) between reconstructed slices.

    The slice index of a coincidence is the SUM of the two axial coordinates
    (``Sinogramer.get_slice_seg`` for segmented crystals,
    ``Sinogramer.get_slice_mono`` for monolithic ones). As in any
    direct/cross-plane PET rebinning, the mid-plane of the LOR is HALF that
    sum, so consecutive slices are separated by half a detector element, not
    a full one.

    Resolution order:

    1. ``detector_config['slice_thickness_mm']`` if present -- always wins,
       and is the recommended way to pin the value down.
    2. Segmented crystals reached through ``process_as='CrystalID'``:
       ``sipm_pitch[1] / 2``, since the index is the sum of two crystal ROW
       indices spaced by the pitch.
    3. Monolithic rebinning (the ``*GlobalPosition*`` branches): the axial
       field of view in mid-plane coordinates is ``crystal_size[1]`` spread
       over ``n_slices`` bins, hence ``crystal_size[1] / n_slices``.

    Otherwise (in particular PETSYS data, where the slice index comes
    pre-computed in the ``slice_1`` branch and its convention is a property of
    the acquisition software, not of this package) the function falls back to
    ``sipm_pitch[1] / 2`` and warns. Set ``slice_thickness_mm`` explicitly to
    silence the warning.

    .. note:: **Corrected in 0.4.0.** The exporter previously wrote
       ``sipm_pitch[1]`` as the axial spacing, i.e. a FULL detector pitch.
       For the shipped 8x8 / 3.36 mm geometry that is 3.36 mm where the
       segmented rebinning gives 1.68 mm -- a factor of 2 in the axial
       direction of every exported MHD/NRRD header. Volumes exported before
       0.4.0 are axially stretched by that factor; any axial distance
       measured on them must be halved (or the header corrected).

    Parameters
    ----------
    detector_config : dict
    n_slices : int
        Number of slices in the volume actually being exported.

    Returns
    -------
    float
        Axial spacing in mm.
    """
    explicit = detector_config.get("slice_thickness_mm")
    if explicit is not None:
        return float(explicit)

    pitch_axial = float(np.asarray(detector_config["sipm_pitch"],
                                   dtype=float)[1])
    process_as = str(detector_config.get("process_as", "")).strip()
    data_from = str(detector_config.get("data_from", "")).upper()

    if data_from == "GATE" and process_as == "CrystalID":
        return pitch_axial / 2.0
    if data_from == "GATE" and process_as.endswith("GlobalPosition") \
            or process_as == "GateGlobalPositions":
        axial_fov = float(np.asarray(detector_config["crystal_size"],
                                     dtype=float)[1])
        return axial_fov / max(int(n_slices), 1)

    warnings.warn(
        "[export] the axial slice spacing could not be derived from "
        f"detector_config (data_from={data_from!r}, process_as={process_as!r}); "
        f"falling back to sipm_pitch[1]/2 = {pitch_axial / 2.0:.4f} mm. Set "
        "detector_config['slice_thickness_mm'] to the value your acquisition "
        "actually uses -- the exported header is only as trustworthy as this "
        "number.", stacklevel=3)
    return pitch_axial / 2.0


def _build_sitk_image(img_array, detector_config, n_dist_out=None,
                      fov_mm=None, voxel_mm=None, slice_thickness_mm=None):
    """Convert a ``(Z, Y, X)`` volume into a centred SimpleITK image.

    Shared by the MHD and NRRD exporters.

    The in-plane spacing is derived from the ACTUAL in-plane size of the
    volume (``fov_mm / img_array.shape[-1]``), not from ``n_dist_out``: the
    two are independent since the image grid became an explicit
    reconstruction parameter. ``n_dist_out`` is kept for call compatibility
    and only used to warn about a mismatch. ``voxel_mm`` overrides the
    in-plane spacing; ``slice_thickness_mm`` overrides the axial one
    (otherwise :func:`resolve_slice_thickness` decides).
    """
    img_array = np.asarray(img_array).astype(np.float32)
    n_xy = int(img_array.shape[-1])
    if fov_mm is None:
        fov_mm = detector_config["Half_FOV"] * 2.0
    fov_mm = float(fov_mm)

    if voxel_mm is not None:
        spacing_xy = float(voxel_mm)
        fov_mm = spacing_xy * n_xy
    else:
        spacing_xy = fov_mm / n_xy

    if n_dist_out is not None and int(n_dist_out) != n_xy:
        print(f"[NOTE] volume is {n_xy} px in-plane but n_dist_out="
              f"{int(n_dist_out)}; spacing derived from the volume "
              f"({spacing_xy:.4f} mm/px).")

    if slice_thickness_mm is not None:
        spacing_z = float(slice_thickness_mm)
    else:
        spacing_z = resolve_slice_thickness(detector_config,
                                            img_array.shape[0])

    import SimpleITK as sitk  # optional until volume export
    sitk_img = sitk.GetImageFromArray(img_array)  # SimpleITK reads [Z, Y, X]
    sitk_img.SetSpacing([spacing_xy, spacing_xy, spacing_z])  # order [X, Y, Z]
    origin_xy = -(fov_mm / 2.0) + (spacing_xy / 2.0)
    origin_z = -((img_array.shape[0] * spacing_z) / 2.0) + (spacing_z / 2.0)
    sitk_img.SetOrigin([origin_xy, origin_xy, origin_z])
    return sitk_img, (spacing_xy, spacing_xy, spacing_z)


def save_reconstructions(res_dict, detector_config, n_dist_out=None, output_dir=None,
                         fmt="mhd", keys=('osem', 'mlem', 'fbp'),
                         fov_mm=None, voxel_mm=None, slice_thickness_mm=None,
                         chaves_recon=None):
    """Save the reconstructions in the dict as MHD or NRRD via SimpleITK.

    fmt: 'mhd' (MetaImage, .mhd+.raw) or 'nrrd'. Both use SimpleITK, which
         is already a project dependency; NRRD adds no new dependency.

    fov_mm / voxel_mm: optional overrides for the in-plane geometry. Pass
         ``voxel_mm=rec.voxel_mm`` (from the reconstructor) when the image
         grid was chosen independently of the sinogram, so the exported
         header carries the true voxel size.

    slice_thickness_mm: optional override for the axial spacing. When None,
         :func:`resolve_slice_thickness` derives it from detector_config.

    chaves_recon: deprecated Portuguese alias of ``keys``.

    The Portuguese alias :func:`salvar_reconstrucoes` points to this same
    function and is kept for backward compatibility.
    """
    if chaves_recon is not None:
        keys = chaves_recon
    if output_dir is None:
        raise ValueError("output_dir is required")
    fmt = fmt.lower().lstrip('.')
    if fmt not in ("mhd", "nrrd"):
        raise ValueError(f"Unsupported format: {fmt!r} (use 'mhd' or 'nrrd')")
    os.makedirs(output_dir, exist_ok=True)
    base_name = res_dict.get('name', 'unnamed_reconstruction')

    saved = []
    for key in keys:
        img_array = res_dict.get(key)
        if img_array is None:
            continue
        sitk_img, sp = _build_sitk_image(
            img_array, detector_config, n_dist_out, fov_mm=fov_mm,
            voxel_mm=voxel_mm, slice_thickness_mm=slice_thickness_mm)
        response = (res_dict.get("geometry") or {}).get("detector_response")
        if key in ("mlem", "osem") and response is not None:
            import json
            sitk_img.SetMetaData("GimnToolsDetectorResponse", json.dumps(response))
        out_filename = os.path.join(output_dir, f"{base_name}_{key}.{fmt}")
        import SimpleITK as sitk
        sitk.WriteImage(sitk_img, out_filename)
        saved.append(out_filename)
        print(f"[{fmt.upper()} saved] {out_filename} | "
              f"Spacing: {sp[0]:.3f}x{sp[1]:.3f}x{sp[2]:.3f} mm")
    return saved


# Portuguese aliases kept for backward compatibility with existing scripts.
salvar_reconstrucoes = save_reconstructions


def salvar_reconstrucoes_mhd(res_dict, detector_config, n_dist_out, output_dir):
    """Backward compatibility: save as MHD (delegates to save_reconstructions)."""
    return save_reconstructions(res_dict, detector_config, n_dist_out,
                                output_dir, fmt="mhd")
