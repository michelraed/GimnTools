"""Geometric PET coincidence simulation using the configured detector montage."""

from dataclasses import asdict
import math
from operator import index

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

from Gimntools.GimnRec.data_loaders import require_valid_detector_config
from Gimntools.GimnRec.detector_response import crystal_response_settings
from Gimntools.GimnRec.listmode_io import event_rows, object_axes, write_listmode
from Gimntools.GimnRec.phantoms import unit_directions


def acquisition_plan(config, steps=None, arc_deg=None, start_deg=0.):
    steps = index(config["rotation_angles"] if steps is None else steps)
    if arc_deg is None:
        arc_deg = (360 / config["detNum"] if config["data_from"] == "GATE"
                   else config["arround"])
    if steps < 1 or not np.isfinite([arc_deg, start_deg]).all() or not 0 < abs(arc_deg) <= 360:
        raise ValueError("Enter a positive step count, a nonzero rotation arc up to ±360° and a finite start angle")
    return start_deg + np.arange(steps) * (arc_deg / steps)


@njit(cache=True)
def _clip(lo, hi, position, direction, lower, upper):
    if abs(direction) < 1e-14:
        if lower <= position <= upper:
            return lo, hi
        return hi, lo
    a, b = (lower - position) / direction, (upper - position) / direction
    return max(lo, min(a, b)), min(hi, max(a, b))


@njit(cache=True)
def _interaction(p, d, lower, upper, grid, pitch, pixels, face, travel,
                 center_hits, segmented):
    """Sample the first interaction through consecutive active crystal intervals."""
    lo, hi = 0., np.inf
    for axis in range(3):
        lo, hi = _clip(lo, hi, p[axis], d[axis], lower[axis], upper[axis])
        if hi <= lo:
            return np.inf, np.zeros(3)
    if not segmented:
        if travel >= hi - lo:
            return np.inf, np.zeros(3)
        hit = p + (lo + travel) * d
        if center_hits:
            hit[0] = lower[0]
        return lo + travel, hit

    uv = grid @ p[1:]
    duv = grid @ d[1:]
    half = pixels * pitch / 2
    for j in range(2):
        lo, hi = _clip(lo, hi, uv[j], duv[j], -half[j], half[j])
        if hi <= lo:
            return np.inf, np.zeros(3)

    # Walk grid cells in ray order. Pitch defines the cells, while the
    # smaller crystal face leaves inactive gaps between neighbouring cells.
    current = lo
    while current < hi:
        centers = np.empty(2)
        next_at = hi
        for j in range(2):
            coord = (uv[j] + (current + 1e-10) * duv[j] + half[j]) / pitch[j]
            cell = min(max(int(math.floor(coord)), 0), pixels[j] - 1)
            centers[j] = (cell + .5) * pitch[j] - half[j]
            if abs(duv[j]) > 1e-14:
                boundary = centers[j] + math.copysign(pitch[j] / 2, duv[j])
                crossing = (boundary - uv[j]) / duv[j]
                if current < crossing < next_at:
                    next_at = crossing
        start, end = current, next_at
        for j in range(2):
            start, end = _clip(start, end, uv[j], duv[j],
                               centers[j] - face / 2, centers[j] + face / 2)
            if end <= start:
                break
        active_length = max(0., end - start)
        if travel < active_length:
            interaction = start + travel
            if center_hits:
                coords = np.empty(3)
                coords[0] = lower[0]
                coords[1:] = grid.T @ centers
                return interaction, coords
            return interaction, p + interaction * d
        travel -= active_length
        current = next_at

    return np.inf, np.zeros(3)


@njit(cache=True, nogil=True)
def _coincidences(points, directions, travel, heads, lower, upper, grid,
                  pitch, pixels, face, center_hits, segmented):
    endpoints = np.empty((len(points), 6))
    count = 0
    for event in range(len(points)):
        nearest = np.full(2, np.inf)
        head_ids = np.full(2, -1)
        hits = np.empty((2, 3))
        for h in range(len(heads)):
            transform = heads[h].T
            local = transform @ points[event]
            direction = transform @ directions[event]
            for photon in range(2):
                signed = direction if photon == 0 else -direction
                length, point = _interaction(local, signed, lower, upper, grid, pitch, pixels,
                                             face, travel[event, photon, h], center_hits, segmented)
                if length < nearest[photon]:
                    nearest[photon] = length
                    head_ids[photon] = h
                    hits[photon] = heads[h] @ point
        if head_ids[0] >= 0 and head_ids[1] >= 0 and head_ids[0] != head_ids[1]:
            endpoints[count, :3], endpoints[count, 3:] = hits[0], hits[1]
            count += 1
    return endpoints[:count]


def simulate_listmode(path, phantom, config, detector_angles, mapping, *,
                      primaries, steps=None, arc_deg=None, start_deg=0., seed=0,
                      hit_mode="Match acquisition", progress=None, chunk_size=16384):
    """Acquire the same number of annihilations per pose and save detected pairs."""
    fmt = config.get("data_from")
    require_valid_detector_config(config, fmt)
    primaries, seed, chunk_size = index(primaries), index(seed), index(chunk_size)
    if primaries < 1 or seed < 0 or chunk_size < 1:
        raise ValueError("Use positive primaries per step and a nonnegative random seed")
    poses = acquisition_plan(config, steps, arc_deg, start_deg)
    axis, axes = object_axes(config)
    response = crystal_response_settings(config)
    names = tuple(dict.fromkeys(name for name in mapping.values() if name is not None))
    if set(names) - detector_angles.keys():
        raise ValueError("The detector mapping refers to a head missing from the montage")
    angles = np.asarray([detector_angles[name] for name in names], dtype=float)
    if not np.isfinite(angles).all() or len(angles) < 2:
        raise ValueError("Map at least two detector heads with finite angles in the acquisition tab")
    heads = Rotation.from_euler(axis, angles[:, None], degrees=True).as_matrix()
    if len(np.unique(np.round(heads[:, :, 0], 9), axis=0)) != len(heads):
        raise ValueError("The configured rotation axis and detector angles place heads at the same position")

    front = config["radius"] + config["pla_thickness"]
    depth, size_y, size_z = config["crystal_size"]
    lower = np.array([front, -size_y / 2, -size_z / 2])
    upper = np.array([front + depth, size_y / 2, size_z / 2])
    if phantom.diameter_mm / 2 >= front:
        raise ValueError("The phantom radius must be smaller than the inner detector radius")
    segmented = config["crystal_type"] == "segmented"
    if not segmented and config["crystal_type"] != "monolithic":
        raise ValueError("Choose segmented or monolithic crystals in the acquisition tab")
    if fmt == "PETSYS" and not config["e_min"] <= 511 <= config["e_max"]:
        raise ValueError("The PETSYS energy window must include 511 keV for simulated events")
    pitch = np.asarray(config["sipm_pitch"], dtype=float)
    pixels = np.asarray(config["sipm_pixel_n"], dtype=np.int64)
    if segmented and response.face_width_mm > pitch.min():
        raise ValueError("The active face width cannot exceed the SiPM pitch")
    theta = np.deg2rad(config.get("det_z_rotation", 0.) if fmt == "PETSYS" else 0.)
    sine, cosine = math.sin(theta), math.cos(theta)
    grid = np.array([[sine, cosine], [cosine, -sine]])
    if hit_mode not in ("Match acquisition", "Crystal centres", "Interaction positions"):
        raise ValueError("Unknown detector position output mode")
    center_hits = hit_mode == "Crystal centres" or (
        hit_mode == "Match acquisition" and (fmt == "PETSYS" or config["process_as"] == "CrystalID")
    )
    rng = np.random.default_rng(seed)
    metadata = {
        "source": "pseudo-simulator", "format": fmt, "phantom": asdict(phantom),
        "detector_config": {key: np.asarray(value).tolist() for key, value in config.items()},
        "detector_angles_deg": dict(zip(names, angles.tolist())),
        "detector_mapping": {int(key): name for key, name in mapping.items() if name is not None},
        "object_axes": ["xyz"[a] for a in axes], "rotation_axis": axis,
        "hit_positions": "readout plane" if center_hits else "first interaction",
        "mu_511_mm_inv": response.mu_per_mm, "seed": seed, "primaries_per_step": primaries,
        "poses": [], "model": "isotropic back-to-back 511 keV photons, first crystal interaction",
    }
    scanner_axes = np.argsort(axes)

    def batches():
        for step, angle in enumerate(poses):
            rotation = Rotation.from_euler(axis, angle, degrees=True)
            inverse = rotation.inv()
            count = 0
            for offset in range(0, primaries, chunk_size):
                size = min(chunk_size, primaries - offset)
                points = rotation.apply(phantom.sample(rng, size)[:, scanner_axes])
                directions = unit_directions(rng, size)
                travel = rng.exponential(1 / response.mu_per_mm, (size, 2, len(heads)))
                hits = _coincidences(points, directions, travel, heads, lower, upper, grid,
                                     pitch, pixels, response.face_width_mm, center_hits, segmented)
                if len(hits):
                    first_hit = inverse.apply(hits[:, :3])[:, axes]
                    second_hit = inverse.apply(hits[:, 3:])[:, axes]
                    yield event_rows(first_hit, second_hit, (511., 511.)), angle, step
                    count += len(hits)
            pose = {"step": step, "angle_deg": float(angle), "primaries": primaries, "coincidences": count}
            metadata["poses"].append(pose)
            if progress:
                progress(step + 1, len(poses), count)

    return write_listmode(path, batches(), metadata)
