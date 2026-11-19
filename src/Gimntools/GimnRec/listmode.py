"""Three-dimensional list-mode EM from coincidence endpoints in millimetres."""

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation


EVENT_COLUMNS = ("x1", "y1", "z1", "E1", "x2", "y2", "z2", "E2")
CALIBRATION_FIELDS = ("sensitivity", "normalization_lors", "normalization_weights", "detector_model")


def _simulator_model(read, accepted):
    description = read("metadata_json")
    try:
        info = json.loads(description.item()) if description is not None else None
    except (ValueError, TypeError):
        return None
    if not isinstance(info, dict) or info.get("source") != "pseudo-simulator":
        return None
    step_index = read("step_index")
    required = {"detector_config", "detector_angles_deg", "detector_mapping", "poses", "hit_positions"}
    if (not required <= info.keys() or step_index is None or step_index.shape != accepted.shape
            or not np.issubdtype(step_index.dtype, np.integer)):
        raise ValueError("The pseudo-simulator NPZ lacks its detector geometry or per-event acquisition steps")
    from Gimntools.GimnRec.listmode_response import SimulatedListmode

    return SimulatedListmode(info, step_index[accepted])


def load_listmode(path, energy_window=None):
    """Load an N×8 'events' array or eight named columns. Energies use keV.

    Endpoints are in a common object XYZ frame unless a per-event
    `gantry_deg` vector is supplied. In that case rotate both endpoints
    by -gantry_deg around `rotation_axis` (default z) into the object frame.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            keys = {key.casefold(): key for key in archive.files}

            def read(key, default=None):
                return archive[keys[key.casefold()]] if key.casefold() in keys else default

            data = read("events")
            if data is None:
                data = np.column_stack([archive[keys[name.casefold()]] for name in EVENT_COLUMNS])
            data = np.asarray(data, dtype=float)
            if data.ndim != 2 or data.shape[1] != 8 or not len(data):
                raise ValueError("List-mode events must have shape (N, 8)")
            good = np.isfinite(data).all(axis=1)
            good &= np.any(data[:, :3] != data[:, 4:7], axis=1)
            if energy_window is not None:
                low, high = energy_window
                if not np.isfinite([low, high]).all() or not 0 <= low < high:
                    raise ValueError("Enter a valid energy window in keV")
                good &= ((data[:, [3, 7]] >= low) & (data[:, [3, 7]] <= high)).all(axis=1)
            angles = read("gantry_deg")
            if angles is not None:
                if angles.shape != (len(data),):
                    raise ValueError("gantry_deg must have one angle per event")
                good &= np.isfinite(angles)
            rays = np.empty((np.count_nonzero(good), 6))
            for target, field in enumerate((0, 1, 2, 4, 5, 6)):
                rays[:, target] = data[good, field]
            if not len(rays):
                raise ValueError("No valid coincidences pass the energy window")
            if angles is not None:
                axis = str(read("rotation_axis", np.array("z")).item()).lower()
                if axis not in ("x", "y", "z"):
                    raise ValueError("rotation_axis must be x, y or z")
                rotation = Rotation.from_euler(axis, -angles[good], degrees=True)
                rays[:, :3] = rotation.apply(rays[:, :3])
                rays[:, 3:] = rotation.apply(rays[:, 3:])

            sensitivity = read("sensitivity")
            norm_lors = read("normalization_lors")
            norm_weights = read("normalization_weights")
            if sensitivity is not None and norm_lors is not None:
                raise ValueError("Specify either sensitivity or normalization_lors")
            if norm_weights is not None and norm_lors is None:
                raise ValueError("normalization_weights requires normalization_lors")
            simulator = _simulator_model(read, good)
            return rays, {"total_events": len(data), "accepted_events": len(rays),
                          "energy_window_kev": energy_window,
                          "sensitivity": sensitivity, "normalization_lors": norm_lors,
                          "normalization_weights": norm_weights, "detector_model": simulator}
    except (KeyError, AttributeError, BadZipFile) as exc:
        raise ValueError("Select a list-mode NPZ containing events or x1,y1,z1,E1,x2,y2,z2,E2") from exc


@njit(cache=True)
def _trace_3d(ray, shape, spacing, corner, columns, lengths):
    direction = ray[3:] - ray[:3]
    enter, exit_at = 0.0, 1.0
    distance = np.sqrt(np.sum(direction * direction))
    if not (0 < distance < np.inf):
        return 0
    for axis in range(3):
        lo = corner[axis]
        hi = lo + shape[axis] * spacing[axis]
        p, d = ray[axis], direction[axis]
        if d == 0.0:
            if p < lo or p >= hi:
                return 0
        else:
            a, b = (lo - p) / d, (hi - p) / d
            enter = max(enter, min(a, b))
            exit_at = min(exit_at, max(a, b))
    if enter >= exit_at:
        return 0

    index = np.empty(3, dtype=np.int64)
    step = np.zeros(3, dtype=np.int64)
    next_at = np.full(3, np.inf)
    stride = np.full(3, np.inf)
    for axis in range(3):
        d = direction[axis]
        pos = (ray[axis] + d * enter - corner[axis]) / spacing[axis]
        nearest = np.rint(pos)
        if abs(pos - nearest) < 1e-10:
            pos = nearest
        index[axis] = min(shape[axis] - 1, max(0, int(np.floor(pos)) - int(d < 0.0 and pos == nearest)))
        if d != 0.0:
            step[axis] = 1 if d > 0.0 else -1
            face = corner[axis] + (index[axis] + int(d > 0.0)) * spacing[axis]
            next_at[axis] = (face - ray[axis]) / d
            stride[axis] = spacing[axis] / abs(d)

    n = 0
    t = enter
    while t < exit_at:
        t1 = min(next_at[0], next_at[1], next_at[2], exit_at)
        if t1 > t:
            columns[n] = (index[2] * shape[1] + index[1]) * shape[0] + index[0]
            lengths[n] = (t1 - t) * distance
            n += 1
        if t1 >= exit_at - 1e-12:
            break
        for axis in range(3):
            if next_at[axis] <= t1 + 1e-12:
                index[axis] += step[axis]
                next_at[axis] += stride[axis]
                if index[axis] < 0 or index[axis] >= shape[axis]:
                    return n
        t = t1
    return n


@njit(cache=True, parallel=True)
def _trace_block(rays, shape, spacing, corner):
    width = np.sum(shape)
    indices = np.empty((len(rays), width), dtype=np.int32)
    values = np.empty((len(rays), width), dtype=np.float64)
    counts = np.empty(len(rays), dtype=np.int32)
    for row in prange(len(rays)):
        counts[row] = _trace_3d(rays[row], shape, spacing, corner, indices[row], values[row])
    ptr = np.empty(len(rays) + 1, dtype=np.int32)
    ptr[0] = 0
    for row in range(len(counts)):
        ptr[row + 1] = ptr[row] + counts[row]
    packed_indices = np.empty(ptr[-1], dtype=np.int32)
    packed_values = np.empty(ptr[-1], dtype=np.float64)
    for row in prange(len(rays)):
        packed_indices[ptr[row]:ptr[row + 1]] = indices[row, :counts[row]]
        packed_values[ptr[row]:ptr[row + 1]] = values[row, :counts[row]]
    return packed_values, packed_indices, ptr


def _ray_matrix(rays, shape_xyz, spacing, corner):
    maximum = get_num_threads()
    active = min(maximum, max(1, len(rays) // 512))
    set_num_threads(active)
    try:
        packed = _trace_block(rays, shape_xyz, spacing, corner)
    finally:
        set_num_threads(maximum)
    return csr_matrix(packed, shape=(len(rays), int(np.prod(shape_xyz))))


def _volume_grid(shape, spacing, center):
    shape = tuple(shape)
    spacing, center = np.asarray(spacing, dtype=float), np.asarray(center, dtype=float)
    if (len(shape) != 3 or any(int(n) != n or n < 1 for n in shape)
            or spacing.shape != (3,) or center.shape != (3,)
            or not np.isfinite([spacing, center]).all() or np.any(spacing <= 0)):
        raise ValueError("Enter a positive 3D matrix and voxel spacing and a finite FOV centre")
    shape_xyz = np.asarray(shape[::-1], dtype=np.int64)
    if np.prod(shape_xyz) >= np.iinfo(np.int32).max:
        raise ValueError("The volume exceeds the 3D projector index range")
    return shape_xyz, spacing, center, center - shape_xyz * spacing / 2


def _phantom_support(shape, spacing, corner, phantom):
    if not phantom or "diameter_mm" not in phantom or "height_mm" not in phantom:
        raise ValueError("Phantom support requires a pseudo-simulator NPZ with cylinder dimensions")
    diameter, height = float(phantom["diameter_mm"]), float(phantom["height_mm"])
    if not np.isfinite([diameter, height]).all() or min(diameter, height) <= 0:
        raise ValueError("Invalid phantom cylinder dimensions in the NPZ")
    positions = [origin + (np.arange(n) + .5) * step
                 for origin, n, step in zip(corner[::-1], shape, spacing[::-1])]
    z, y, x = np.ix_(*positions)
    # Include every voxel intersecting the cylinder, including partial volume at its edge.
    nearest_x = np.maximum(0., np.abs(x) - spacing[0] / 2)
    nearest_y = np.maximum(0., np.abs(y) - spacing[1] / 2)
    return ((nearest_x**2 + nearest_y**2 <= (diameter / 2)**2)
            & (np.abs(z) <= (height + spacing[2]) / 2)).ravel()


def reconstruct_listmode(rays, shape, spacing, *, center=(0, 0, 0),
                         iterations=5, subsets=1, relaxation=0.8, ratio_cap=1e6,
                         sensitivity=None, normalization_lors=None, normalization_weights=None,
                         detector_model=None, psf=None, sensitivity_floor=None, restrict_to_phantom=False,
                         model_progress=None, cache_bytes=128 * 1024**2,
                         progress=None, checkpoint=None, update_callback=None):
    """Reconstruct a (z, y, x) volume with matched 3D Siddon operators.

    A saved pseudo-simulator acquisition supplies the detector-pair response
    and scanner sensitivity for the selected grid. Other inputs can supply
    voxel sensitivity or independent scanner normalization LORs. Otherwise
    the approximation is uniform sensitivity on illuminated voxels.
    """
    rays = np.asarray(rays, dtype=float)
    shape_xyz, spacing, center, corner = _volume_grid(shape, spacing, center)
    if (rays.ndim != 2 or rays.shape[1] != 6 or not len(rays)
            or not np.isfinite(rays).all()):
        raise ValueError("List-mode LORs must be a finite nonempty N×6 array")
    if (not isinstance(iterations, (int, np.integer)) or iterations < 1
            or not isinstance(subsets, (int, np.integer)) or not 1 <= subsets <= len(rays)
            or not 0 < relaxation <= 1 or not np.isfinite(ratio_cap) or ratio_cap <= 0):
        raise ValueError("Check iterations, subsets, EM relaxation and ratio cap")

    shape = tuple(shape_xyz[::-1])
    num_voxels = int(np.prod(shape_xyz))
    block_size = min(8192, max(1, 2**20 // int(shape_xyz.sum())))
    order = np.random.default_rng(0).permutation(len(rays)) if subsets > 1 else None
    edges = np.linspace(0, len(rays), subsets + 1, dtype=int)

    def matrix(points):
        return _ray_matrix(points, shape_xyz, spacing, corner)

    response = None
    if detector_model is not None:
        from Gimntools.GimnRec.listmode_response import ListmodeResponse, prepare_listmode_response

        response = detector_model if isinstance(detector_model, ListmodeResponse) else prepare_listmode_response(
            rays, shape, spacing, center, detector_model, progress=model_progress,
        )
    if psf is not None:
        from Gimntools.GimnRec.listmode_psf import ListmodePSF, prepare_listmode_psf

        if not isinstance(psf, ListmodePSF):
            psf = prepare_listmode_psf(shape, spacing, center, psf, response.detector_config if response else None)
        if psf.shape != shape:
            raise ValueError("The SV-PSF must match the list-mode image grid")

    sensitivity_model = "uniform"
    if sensitivity is not None:
        sensitivity = np.asarray(sensitivity, dtype=float)
        if sensitivity.shape != shape or not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError("NPZ sensitivity must be finite, nonnegative and match the (z,y,x) matrix")
        sensitivity = sensitivity.ravel()
        sensitivity_model = "provided voxel sensitivity"
    elif normalization_lors is not None:
        reference = np.asarray(normalization_lors, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 6 or not np.isfinite(reference).all():
            raise ValueError("Normalization LORs must be finite N×6 endpoints")
        weights = np.ones(len(reference)) if normalization_weights is None else np.asarray(normalization_weights, dtype=float)
        if weights.shape != (len(reference),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Normalization weights must be finite and nonnegative, one per LOR")
        sensitivity = np.zeros(num_voxels)
        for start in range(0, len(reference), block_size):
            stop = start + block_size
            sensitivity += matrix(reference[start:stop]).T @ weights[start:stop]
        sensitivity_model = "independent normalization LORs"
    elif response is not None:
        sensitivity = response.sensitivity
        sensitivity_model = response.name

    support = np.zeros(num_voxels, dtype=bool)
    traced = np.zeros(subsets, dtype=np.int64)

    if response is not None and response.operators:
        operators = response.operators
        event_rows = response.event_rows if order is None else response.event_rows[order]
        crosses = np.concatenate([np.diff(a.indptr) > 0 for a in operators])
        offsets = np.r_[0, np.cumsum([a.shape[0] for a in operators])]
        for a in operators:
            support[a.indices] = True
        for sub, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            traced[sub] = np.count_nonzero(crosses[event_rows[first:last]])

        def blocks(sub, first, last):
            counts = np.bincount(event_rows[first:last], minlength=len(crosses))
            for a, lo, hi in zip(operators, offsets[:-1], offsets[1:]):
                yield a, counts[lo:hi]
    else:
        cache = {}

        def blocks(sub, first, last):
            for start in range(first, last, block_size):
                a = cache.get(start)
                if a is None:
                    stop = min(start + block_size, last)
                    a = matrix(rays[start:stop] if order is None else rays[order[start:stop]])
                yield a, 1.

        for sub, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            for start, (a, _) in zip(range(first, last, block_size), blocks(sub, first, last)):
                size = a.data.nbytes + a.indices.nbytes + a.indptr.nbytes
                if size <= cache_bytes:
                    cache[start] = a
                    cache_bytes -= size
                support[a.indices] = True
                traced[sub] += np.count_nonzero(np.diff(a.indptr))
    if not traced.all():
        raise ValueError("At least one subset has no LOR crossing the image. Check the FOV and coordinate frame")
    simulator_sensitivity = response is not None and sensitivity is response.sensitivity
    floor = (.01 if simulator_sensitivity else 0.) if sensitivity_floor is None else sensitivity_floor
    if not np.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("List-mode sensitivity floor must be a fraction from 0 to less than 1")
    if sensitivity is None:
        sensitivity = support.astype(float)
    elif not simulator_sensitivity and np.any(sensitivity[support] == 0):
        raise ValueError("Normalization has zero exposure in voxels crossed by measured LORs")
    if psf is not None:
        support = psf.transpose(support.astype(float)) > 0
        sensitivity = psf.transpose(sensitivity)
    support &= (sensitivity > 0) & (sensitivity >= floor * sensitivity.max())
    if restrict_to_phantom == "auto":
        restrict_to_phantom = bool(response and response.phantom)
    if restrict_to_phantom:
        support &= _phantom_support(shape, spacing, corner, response.phantom if response else None)
    if not support.any():
        raise ValueError("The data and detector response have no common image support")
    if floor or restrict_to_phantom:
        illuminated = support.astype(float)
        if psf is not None:
            illuminated = psf.forward(illuminated)
        for sub, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            traced[sub] = sum(int(np.sum(np.where(a @ illuminated > 0, observed, 0)))
                              for a, observed in blocks(sub, first, last))
        if not traced.all():
            raise ValueError("At least one subset misses the active reconstruction support")
    x = np.zeros(num_voxels)
    x[support] = traced.sum() / sensitivity[support].sum()

    for it in range(1, iterations + 1):
        for sub, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            correction = np.zeros(num_voxels)
            blurred = psf.forward(x) if psf is not None else x
            for a, observed in blocks(sub, first, last):
                predicted = a @ blurred
                ratio = np.divide(1., predicted, out=np.zeros_like(predicted), where=predicted > 0)
                np.minimum(ratio, ratio_cap, out=ratio)
                ratio *= observed
                correction += a.T @ ratio
            if psf is not None:
                correction = psf.transpose(correction)
            exposure = (last - first) / len(rays)
            update = correction[support] / (sensitivity[support] * exposure)
            x[support] *= 1 + relaxation * (update - 1)
            if update_callback is not None:
                update_callback(it, sub + 1, x.reshape(shape))
        if checkpoint is not None:
            checkpoint(it, x.reshape(shape).copy())
        if progress is not None:
            progress(it)
    geometry = {"origin": tuple(corner + spacing / 2),
                "spacing": tuple(spacing), "center": tuple(center),
                "traced_events": int(traced.sum()), "input_events": len(rays),
                "sensitivity": sensitivity_model, "sensitivity_floor": floor,
                "phantom_support": bool(restrict_to_phantom), "sv_psf": psf.name if psf else None,
                "iterations": iterations, "subsets": subsets}
    return x.reshape(shape), geometry
