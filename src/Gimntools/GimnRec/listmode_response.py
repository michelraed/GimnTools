"""Matched 3D response for list mode from the PET pseudo-simulator."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from Gimntools.GimnRec.pseudo_simulator import _crystal_trace, detector_geometry


@dataclass(frozen=True)
class SimulatedListmode:
    """Scanner settings and per-event step from a simulator archive."""

    metadata: dict
    event_steps: np.ndarray


@dataclass(frozen=True)
class ListmodeResponse:
    sensitivity: np.ndarray
    operators: tuple[csr_matrix, ...]
    event_rows: np.ndarray | None
    name: str


@njit(cache=True, nogil=True)
def _optical_rays(first, second, first_heads, second_heads, heads, lower, upper,
                  grid, pitch, pixels, face, mu, segmented):
    weights = np.zeros(len(first))
    for i in range(len(weights)):
        local_a, local_b = first[i], second[i]
        if not (np.all(local_a >= lower) and np.all(local_a <= upper)
                and np.all(local_b >= lower) and np.all(local_b <= upper)):
            continue
        a, b = heads[first_heads[i]], heads[second_heads[i]]
        direction = b @ local_b - a @ local_a
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


def _hit_channel(points, geometry, rotation, crystals_per_head):
    scanner = rotation.apply(points[:, np.argsort(geometry.axes)])
    result = np.full(len(points), -1, dtype=np.int64)
    for i, head in enumerate(geometry.heads):
        local = scanner @ head
        uv = local[:, 1:] @ geometry.grid.T
        grid_pos = uv / geometry.pitch + (geometry.pixels - 1) / 2
        pixel = np.rint(grid_pos).astype(np.int64)
        # Simulator archives saved exact readout centres in mm. The check
        # verifies that the pose metadata and measurement identify the same bin.
        valid = ((np.abs(local[:, 0] - geometry.lower[0]) < 1e-5)
                 & (np.abs(grid_pos - pixel).max(axis=1) < 1e-5)
                 & np.all((pixel >= 0) & (pixel < geometry.pixels), axis=1))
        if np.any(result[valid] >= 0):
            raise ValueError("The pseudo-simulator readout matches more than one crystal head")
        result[valid] = i * crystals_per_head + pixel[valid, 0] * geometry.pixels[1] + pixel[valid, 1]
    if np.any(result < 0):
        raise ValueError("The pseudo-simulator readout does not match its saved detector geometry and poses")
    return result


def _sample_pairs(codes, total_channels, crystals, geometry, unit):
    """Integrate within each active crystal volume, including DOI and both gaps."""
    count = len(crystals)
    subdivisions = len(unit)
    first, second = np.divmod(codes, total_channels)
    left, right = np.repeat(first // count, subdivisions), np.repeat(second // count, subdivisions)
    hits = []
    for columns, channel in ((unit[:, :3], first), (unit[:, 3:], second)):
        local = np.empty((len(codes), subdivisions, 3))
        local[:, :, 0] = geometry.lower[0] + columns[:, 0] * (geometry.upper[0] - geometry.lower[0])
        if geometry.segmented:
            uv = crystals[channel % count, None, :] + geometry.face * (columns[:, 1:] - .5)
            local[:, :, 1:] = uv @ geometry.grid
        else:
            local[:, :, 1:] = geometry.lower[1:] + columns[:, 1:] * (geometry.upper[1:] - geometry.lower[1:])
        hits.append(local.reshape(-1, 3))
    weights = _optical_rays(*hits, left, right, geometry.heads, geometry.lower, geometry.upper,
                            geometry.grid, geometry.pitch, geometry.pixels, geometry.face,
                            geometry.mu, geometry.segmented)
    a = np.einsum("nij,nj->ni", geometry.heads[left], hits[0])
    b = np.einsum("nij,nj->ni", geometry.heads[right], hits[1])
    return a, b, weights / subdivisions


def prepare_listmode_response(rays, shape, spacing, center, simulator, *, progress=None):
    """Prepare the finite-pair projector and sensitivity for the requested voxel grid."""
    from Gimntools.GimnRec.listmode import _ray_matrix, _volume_grid

    grid_shape, spacing, _, corner = _volume_grid(shape, spacing, center)
    info = simulator.metadata
    config = dict(info["detector_config"])
    if "mu_511_mm_inv" in info:
        config["mu_511_mm_inv"] = info["mu_511_mm_inv"]
    geometry = detector_geometry(config, info["detector_angles_deg"], info["detector_mapping"])
    pose_info = info["poses"]
    pose_degrees = np.asarray([pose["angle_deg"] for pose in pose_info], dtype=float)
    exposure = np.asarray([pose["primaries"] for pose in pose_info], dtype=float)
    steps = simulator.event_steps
    if (not len(pose_degrees) or not np.isfinite(pose_degrees).all()
            or list(range(len(pose_degrees))) != [pose["step"] for pose in pose_info]
            or not np.isfinite(exposure).all() or np.any(exposure <= 0)
            or steps.shape != (len(rays),) or np.any((steps < 0) | (steps >= len(pose_degrees)))):
        raise ValueError("Invalid acquisition steps in the pseudo-simulator NPZ")
    exposure /= exposure.sum()
    rotations = [Rotation.from_euler(geometry.axis, angle, degrees=True) for angle in pose_degrees]
    crystals = _crystal_centers(geometry) if geometry.segmented else np.zeros((1, 2))
    pair_codes, channels = _detector_channels(geometry, crystals)
    readout = geometry.segmented and info["hit_positions"] == "readout plane"
    if readout:
        observed = np.empty(len(rays), dtype=np.int64)
        for step, rotation in enumerate(rotations):
            selected = steps == step
            if not selected.any():
                continue
            a = _hit_channel(rays[selected, :3], geometry, rotation, len(crystals))
            b = _hit_channel(rays[selected, 3:], geometry, rotation, len(crystals))
            pair = np.minimum(a, b) * channels + np.maximum(a, b)
            ix = np.searchsorted(pair_codes, pair)
            if np.any(ix >= len(pair_codes)) or not np.array_equal(pair_codes[ix], pair):
                raise ValueError("A pseudo-simulator coincidence must join two different mapped heads")
            observed[selected] = step * len(pair_codes) + ix
        observed, event_rows = np.unique(observed, return_inverse=True)
    collected = [[] for _ in rotations]
    sensitivity = np.zeros(int(grid_shape.prod()))
    # Independent, stratified detector-volume integration. The simulated
    # primary positions and the observed counts never enter the sensitivity.
    order = 15
    if geometry.segmented:
        requested = np.ceil(np.log2(max(1., geometry.face / spacing.min())))
        order = max(6, min(8, 4 + int(requested)))
    unit = qmc.Sobol(d=6, scramble=True, seed=175).random_base2(order)
    batch_pairs = max(1, 4096 // len(unit))
    for start in range(0, len(pair_codes), batch_pairs):
        stop = min(start + batch_pairs, len(pair_codes))
        a, b, weight = _sample_pairs(pair_codes[start:stop], channels, crystals, geometry, unit)
        # Before the forward/adjoint operations sum the sub-rays belonging
        # to each physical channel, including bins with zero observed counts.
        if readout:
            groups = np.arange(0, len(weight) + 1, len(unit), dtype=np.int32)
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
        return ListmodeResponse(sensitivity, operators, event_rows, "finite crystal-pair response")
    return ListmodeResponse(sensitivity, (), None, "simulated scanner sensitivity")
