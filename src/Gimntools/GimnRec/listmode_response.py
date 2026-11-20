"""Matched 3D response for list mode with a saved detector geometry."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from Gimntools.GimnRec.pseudo_simulator import _crystal_trace, detector_geometry


@dataclass(frozen=True)
class ListmodeAcquisition:
    """Scanner settings and per-event step from an acquisition archive."""

    metadata: dict
    event_steps: np.ndarray


@dataclass(frozen=True)
class ListmodeResponse:
    sensitivity: np.ndarray
    operators: tuple[csr_matrix, ...]
    event_rows: np.ndarray | None
    name: str
    detector_config: dict
    phantom: dict | None
    details: dict


READOUT_TOLERANCE_MM = 5e-5


@njit(cache=True, nogil=True)
def _optical_rays(first, second, first_heads, second_heads, heads, translations, lower, upper,
                  grid, pitch, pixels, face, mu, segmented):
    weights = np.zeros(len(first))
    for i in range(len(weights)):
        local_a, local_b = first[i], second[i]
        if not (np.all(local_a >= lower) and np.all(local_a <= upper)
                and np.all(local_b >= lower) and np.all(local_b <= upper)):
            continue
        left, right = first_heads[i], second_heads[i]
        a, b = heads[left], heads[right]
        direction = b @ local_b + translations[right] - a @ local_a - translations[left]
        length = np.sqrt(direction @ direction)
        if length == 0.:
            continue
        direction /= length
        _, _, path_a = _crystal_trace(local_a, a.T @ direction, lower, upper, grid,
                                      pitch, pixels, face, np.inf, False, segmented)
        _, _, path_b = _crystal_trace(local_b, b.T @ -direction, lower, upper, grid,
                                      pitch, pixels, face, np.inf, False, segmented)
        # For fixed interaction points the photon directions and pre-hit
        # attenuation are constant along the line inside the bore. The
        # change of variables from directions to hit pairs contributes 1/L².
        weights[i] = np.exp(-mu * (path_a + path_b)) / (length * length)
    return weights


def _crystal_centers(geometry):
    coords = np.indices(tuple(geometry.pixels)).reshape(2, -1).T
    return (coords - (geometry.pixels - 1) / 2) * geometry.pitch


def _detector_channels(geometry, crystals):
    left, right = np.triu_indices(len(geometry.heads), k=1)
    crystals_per_head = len(crystals)
    sub = np.arange(crystals_per_head)
    first = (left[:, None] * crystals_per_head + sub).ravel()
    second = (right[:, None] * crystals_per_head + sub).ravel()
    limit = len(geometry.heads) * crystals_per_head
    codes = (first.reshape(-1, crystals_per_head, 1) * limit
             + second.reshape(-1, 1, crystals_per_head)).ravel()
    return np.sort(codes), limit


def _head_readout(points, geometry, rotation):
    scanner = rotation.apply(points[:, np.argsort(geometry.axes)])
    heads = np.full(len(points), -1, dtype=np.int64)
    positions = np.empty((len(points), 2))
    for i, head in enumerate(geometry.heads):
        local = scanner @ head
        valid = np.abs(local[:, 0] - geometry.lower[0]) <= READOUT_TOLERANCE_MM
        if np.any(heads[valid] >= 0):
            raise ValueError("A list-mode readout matches more than one detector plane")
        heads[valid] = i
        positions[valid] = local[valid, 1:] @ geometry.grid.T
    if np.any(heads < 0):
        raise ValueError("List-mode hits do not match the detector planes in their saved geometry and poses")
    return heads, positions


def _readout_offsets(rays, steps, geometry, rotations, info):
    if info.get("petsys_alignment_error"):
        raise ValueError("PETSYS channel alignment could not be verified: " + info["petsys_alignment_error"])
    stored = info.get("petsys_bin_offsets_mm")
    if stored is not None:
        if not isinstance(stored, dict) or any(name not in stored for name in geometry.names):
            raise ValueError("PETSYS archive alignment must cover every detector head")
        values = np.asarray([stored[name] for name in geometry.names], dtype=float)
        if values.shape != (len(geometry.heads), 2) or not np.isfinite(values).all():
            raise ValueError("Invalid PETSYS archive alignment offsets")
        return values, "verified ROOT channels"
    low = np.full((len(geometry.heads), 2), np.inf)
    high = np.full_like(low, -np.inf)
    for i, rotation in enumerate(rotations):
        selected = steps == i
        if not selected.any():
            continue
        for start in (0, 3):
            heads, uv = _head_readout(rays[selected, start:start + 3], geometry, rotation)
            for h in np.unique(heads):
                values = uv[heads == h]
                low[h] = np.minimum(low[h], values.min(axis=0))
                high[h] = np.maximum(high[h], values.max(axis=0))
    # Seeing both extremal pixels resolves the whole-pitch ambiguity of a
    # translated lattice. Every hit is verified against that lattice below.
    span = (geometry.pixels - 1) * geometry.pitch
    if not np.allclose(high - low, span, rtol=0, atol=2 * READOUT_TOLERANCE_MM):
        raise ValueError("Cannot identify every PETSYS lattice origin from this NPZ. Export from ROOT with "
                         "channel IDs or select direct endpoint projection for these data")
    return (low + high) / 2, "verified complete coordinate lattice"


def _hit_channel(points, geometry, rotation, offsets):
    heads, positions = _head_readout(points, geometry, rotation)
    nominal = positions - offsets[heads]
    grid_pos = nominal / geometry.pitch + (geometry.pixels - 1) / 2
    pixels = np.rint(grid_pos).astype(np.int64)
    if (np.any((pixels < 0) | (pixels >= geometry.pixels))
            or np.any(np.abs(grid_pos - pixels) * geometry.pitch > READOUT_TOLERANCE_MM)):
        raise ValueError("List-mode hits do not match the verified crystal lattice")
    per_head = int(np.prod(geometry.pixels))
    return heads * per_head + pixels[:, 0] * geometry.pixels[1] + pixels[:, 1]


def _sample_pairs(codes, total_channels, crystals, geometry, translations, unit):
    """Integrate within each active crystal volume, including DOI and both gaps."""
    count = len(crystals)
    subdivisions = unit.shape[1]
    first, second = np.divmod(codes, total_channels)
    left, right = np.repeat(first // count, subdivisions), np.repeat(second // count, subdivisions)
    hits = []
    for columns, channel in ((unit[:, :, :3], first), (unit[:, :, 3:], second)):
        local = np.empty((len(codes), subdivisions, 3))
        local[:, :, 0] = geometry.lower[0] + columns[:, :, 0] * (geometry.upper[0] - geometry.lower[0])
        if geometry.segmented:
            uv = crystals[channel % count, None, :] + geometry.face * (columns[:, :, 1:] - .5)
            local[:, :, 1:] = uv @ geometry.grid
        else:
            local[:, :, 1:] = geometry.lower[1:] + columns[:, :, 1:] * (geometry.upper[1:] - geometry.lower[1:])
        hits.append(local.reshape(-1, 3))
    weights = _optical_rays(*hits, left, right, geometry.heads, translations, geometry.lower, geometry.upper,
                            geometry.grid, geometry.pitch, geometry.pixels, geometry.face,
                            geometry.mu, geometry.segmented)
    a = np.einsum("nij,nj->ni", geometry.heads[left], hits[0]) + translations[left]
    b = np.einsum("nij,nj->ni", geometry.heads[right], hits[1]) + translations[right]
    return a, b, weights / subdivisions


def prepare_listmode_response(rays, shape, spacing, center, acquisition, *, detector_radius_mm=None,
                              progress=None):
    """Prepare the finite-pair projector and sensitivity for the requested voxel grid."""
    from Gimntools.GimnRec.listmode import _ray_matrix, _volume_grid

    grid_shape, spacing, _, corner = _volume_grid(shape, spacing, center)
    info = acquisition.metadata
    from_root = info["source"] == "root coincidences"
    petsys = from_root and info["format"] == "PETSYS"
    config = dict(info["detector_config"])
    if "mu_511_mm_inv" in info:
        config["mu_511_mm_inv"] = info["mu_511_mm_inv"]
    encoded = detector_geometry(config, info["detector_angles_deg"], info["detector_mapping"])
    if detector_radius_mm is not None:
        if not petsys or not np.isfinite(detector_radius_mm) or detector_radius_mm <= 0:
            raise ValueError("Enter a positive radius override for segmented PETSYS data")
        config["radius"] = float(detector_radius_mm)
    geometry = encoded if detector_radius_mm is None else detector_geometry(
        config, info["detector_angles_deg"], info["detector_mapping"],
    )
    pose_info = info["poses"]
    pose_degrees = np.asarray([pose["angle_deg"] for pose in pose_info], dtype=float)
    if from_root:
        has_exposure = all("exposure" in pose for pose in pose_info)
        if not has_exposure and any("exposure" in pose for pose in pose_info):
            raise ValueError("Provide exposure for every list-mode acquisition pose")
        exposure = np.asarray([pose["exposure"] for pose in pose_info], dtype=float) if has_exposure else np.ones(len(pose_info))
    else:
        exposure = np.asarray([pose["primaries"] for pose in pose_info], dtype=float)
    # ROOT run IDs need not start at zero or be contiguous.
    step_ids = np.asarray([pose["step"] for pose in pose_info], dtype=np.int64)
    sorted_ids, order_ids = np.unique(step_ids, return_index=True)
    ix = np.searchsorted(sorted_ids, acquisition.event_steps)
    if (not len(pose_degrees) or not np.isfinite(pose_degrees).all()
            or len(sorted_ids) != len(step_ids)
            or not np.isfinite(exposure).all() or np.any(exposure <= 0)
            or ix.shape != (len(rays),) or np.any(ix >= len(sorted_ids))
            or not np.array_equal(sorted_ids[ix], acquisition.event_steps)):
        raise ValueError("Invalid acquisition steps in the list-mode NPZ")
    steps = order_ids[ix]
    exposure /= exposure.sum()
    rotations = [Rotation.from_euler(geometry.axis, angle, degrees=True) for angle in pose_degrees]
    crystals = _crystal_centers(geometry) if geometry.segmented else np.zeros((1, 2))
    pair_codes, channels = _detector_channels(geometry, crystals)
    readout = geometry.segmented and (from_root or info["hit_positions"] == "readout plane")
    details = {"detector_radius_mm": float(config["radius"]), "encoded_radius_mm": float(encoded.lower[0] - config["pla_thickness"]),
               "positions": len(rotations)}
    if from_root:
        details["exposure"] = "provided" if has_exposure else "assumed equal"
    if petsys:
        offsets, calibration = _readout_offsets(rays, steps, encoded, rotations, info)
        details.update(alignment=calibration, readout_offsets_mm=dict(zip(geometry.names, offsets.tolist())))
    else:
        offsets = np.zeros((len(geometry.heads), 2))
    # GimnROOT publishes corrected positions in the detector plane. Decode
    # labels in the translated readout lattice, and integrate within the
    # correspondingly translated crystal volumes in the scanner frame.
    local_shifts = np.column_stack((np.zeros(len(offsets)), offsets @ geometry.grid))
    translations = np.einsum("nij,nj->ni", geometry.heads, local_shifts)
    if readout:
        observed = np.empty(len(rays), dtype=np.int64)
        for step, rotation in enumerate(rotations):
            selected = steps == step
            if not selected.any():
                continue
            a = _hit_channel(rays[selected, :3], encoded, rotation, offsets)
            b = _hit_channel(rays[selected, 3:], encoded, rotation, offsets)
            pair = np.minimum(a, b) * channels + np.maximum(a, b)
            ix = np.searchsorted(pair_codes, pair)
            if np.any(ix >= len(pair_codes)) or not np.array_equal(pair_codes[ix], pair):
                raise ValueError("A crystal-pair coincidence must join two different mapped heads")
            observed[selected] = step * len(pair_codes) + ix
        observed, event_rows = np.unique(observed, return_inverse=True)
    collected = [[] for _ in rotations]
    sensitivity = np.zeros(int(grid_shape.prod()))
    # Independent, stratified detector-volume integration. The simulated
    # primary positions and the observed counts never enter the sensitivity.
    order = 15
    if geometry.segmented:
        requested = np.ceil(np.log2(max(1., geometry.face / spacing.min())))
        order = max(6 if readout else 7, min(8, 4 + int(requested)))
    sobol = qmc.Sobol(d=6, scramble=True, seed=175).random_base2(order)
    digital = (sobol * 2.**32).astype(np.uint32)
    shifts = np.random.default_rng(241)
    batch_pairs = max(1, 4096 // len(digital))
    for start in range(0, len(pair_codes), batch_pairs):
        stop = min(start + batch_pairs, len(pair_codes))
        # Independent digital shifts retain stratification within each pair
        # without repeating crystal offsets across the axial detector lattice.
        shift = shifts.integers(0, 2**32, (stop - start, 1, 6), dtype=np.uint32)
        unit = ((digital[None, :, :] ^ shift).astype(float) + .5) / 2.**32
        a, b, weight = _sample_pairs(pair_codes[start:stop], channels, crystals, geometry, translations, unit)
        # Before the forward/adjoint operations sum the sub-rays belonging
        # to each physical channel, including bins with zero observed counts.
        if readout:
            groups = np.arange(0, len(weight) + 1, len(digital), dtype=np.int32)
            reduce_rows = csr_matrix((weight, np.arange(len(weight), dtype=np.int32), groups),
                                     shape=(stop - start, len(weight)))
        for pose, rotation in enumerate(rotations):
            inverse = rotation.inv()
            object_rays = np.column_stack((inverse.apply(a)[:, geometry.axes],
                                           inverse.apply(b)[:, geometry.axes]))
            if readout:
                projection = reduce_rows @ _ray_matrix(object_rays, grid_shape, spacing, corner)
                sensitivity += exposure[pose] * np.asarray(projection.sum(axis=0)).ravel()
                base = pose * len(pair_codes)
                first, last = np.searchsorted(observed, (base + start, base + stop))
                if first < last:
                    collected[pose].append(projection[observed[first:last] - base - start].astype(np.float32))
            else:
                for offset in range(0, len(object_rays), 4096):
                    block = slice(offset, offset + 4096)
                    projection = _ray_matrix(object_rays[block], grid_shape, spacing, corner)
                    sensitivity += exposure[pose] * (projection.T @ weight[block])
        if progress:
            progress(stop, len(pair_codes))
    if not sensitivity.any():
        raise ValueError("No detector response crosses the image. Check the list-mode FOV")
    if readout:
        # Combine one pose at a time so the full sparse response never has
        # to be held twice in memory during construction.
        for i, blocks in enumerate(collected):
            collected[i] = vstack(blocks, format="csr") if blocks else None
        operators = tuple(a for a in collected if a is not None)
        name = ("measured PETSYS crystal-pair response" if petsys else "GATE crystal-pair response"
                if from_root else "finite crystal-pair response")
        return ListmodeResponse(sensitivity, operators, event_rows, name, config, info.get("phantom"), details)
    return ListmodeResponse(sensitivity, (), None, "simulated scanner sensitivity", config, info.get("phantom"), details)
