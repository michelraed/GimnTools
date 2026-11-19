"""Matched, slice-wise detector response operators for Poisson reconstruction."""

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.special import roots_legendre

from Gimntools.GimnRec.doi import first_interaction_quadrature
from Gimntools.GimnRec.projectors import _siddon_trace


@dataclass(frozen=True)
class ResponseSettings:
    """A positive attenuation coefficient enables DOI; a LUT enables SV-PSF."""

    mu_per_mm: float | None = None
    face_width_mm: float | None = None
    face_order: int = 3
    depth_order: int = 3
    lut_path: str | None = None
    analytical: bool = False
    material: str | None = None


# Mixture mass coefficients at 511 keV from xraylib 4.3.0 CS_Total
# and nominal densities of 7.10 g/cm³ (Lu1.8Y0.2SiO5) and 7.13 g/cm³ (Bi4Ge3O12).
CRYSTAL_MU_511 = {"LYSO": 0.08184, "BGO": 0.09629}
CRYSTAL_MATERIALS = (*CRYSTAL_MU_511, "Custom")
DEFAULT_FACE_WIDTH_MM = 3.0


def crystal_material(value):
    name = str(value or "LYSO").strip().upper()
    if name == "CUSTOM":
        return "Custom"
    if name not in CRYSTAL_MU_511:
        raise ValueError(f"Unknown crystal material {value!r}. Choose LYSO, BGO or Custom")
    return name


def crystal_response_settings(config, *, material=None, mu=None, face=None,
                              face_order=3, depth_order=3, lut_path=None, analytical=False):
    """Fill missing response settings from the selected sinogram's detector profile."""
    config = config or {}
    material = crystal_material(material or config.get("crystal_material"))
    if mu in (None, ""):
        if material == crystal_material(config.get("crystal_material")):
            mu = config.get("mu_511_mm_inv")
        if mu in (None, ""):
            mu = CRYSTAL_MU_511.get(material)
    if face in (None, ""):
        face = config.get("crystal_face_width_mm", DEFAULT_FACE_WIDTH_MM)

    def positive(value, label):
        try:
            number = float(str(value).replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"Enter a positive {label}") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"Enter a positive {label}")
        return number

    try:
        face_order, depth_order = int(str(face_order)), int(str(depth_order))
    except ValueError as exc:
        raise ValueError("Face and depth quadrature orders must be integers in [1, 12]") from exc
    if not (1 <= face_order <= 12 and 1 <= depth_order <= 12):
        raise ValueError("Face and depth quadrature orders must be integers in [1, 12]")
    return ResponseSettings(positive(mu, "511 keV attenuation coefficient in mm⁻¹"),
                            positive(face, "segment face width in mm"),
                            face_order, depth_order, lut_path, analytical, material)


def acquisition_geometry(path, config, detector_angles, rsectors, first_rotation,
                         object_angles):
    """Snapshot the geometry and acquisition poses used for raw histogramming."""
    return dict(
        measurement=str(path), detector_config={
            key: np.asarray(value).tolist() for key, value in config.items()
        },
        detector_angles={str(key): float(value) for key, value in detector_angles.items()},
        rsectors={int(key): str(value) for key, value in rsectors.items()},
        first_rotation=str(first_rotation),
        object_angles_deg=np.unique(np.asarray(object_angles, dtype=float)).tolist(),
    )


def _transverse_is_x(axis, rotation):
    return (axis == "y") != (round(rotation / 90) % 2 == 1)


# GimnROOT SiPM::CorrectChannelToPixelPs returns (row I, column J).
# Source: GimnROOT 31b862f, source/2-histograms/SiPM.C.
_GIMNROOT_PIXELS = np.array([
    (1, 7), (1, 6), (2, 4), (0, 6), (0, 5), (3, 7), (0, 4), (2, 6),
    (1, 5), (3, 3), (0, 7), (3, 4), (3, 6), (2, 5), (2, 7), (1, 4),
    (6, 6), (7, 7), (6, 7), (7, 6), (4, 7), (4, 6), (5, 6), (5, 7),
    (7, 4), (7, 5), (6, 4), (3, 5), (6, 5), (5, 5), (5, 4), (4, 4),
    (4, 5), (4, 3), (4, 2), (5, 2), (5, 3), (6, 2), (3, 2), (6, 3),
    (7, 2), (7, 3), (4, 0), (4, 1), (5, 0), (6, 1), (5, 1), (6, 0),
    (7, 0), (7, 1), (0, 3), (3, 0), (3, 1), (2, 1), (2, 0), (1, 0),
    (1, 1), (0, 2), (2, 2), (1, 3), (0, 0), (1, 2), (2, 3), (0, 1),
])
PETSYS_ID_BRANCHES = ("channelID_1", "channelID_2", "posI_1", "posI_2", "posJ_2")


def _gimnroot_offsets(data, block, hit, valid, pitches):
    # posJ_1 in older GimnROOT files is copied from the second hit.
    channels = data[f"channelID_{hit}"][block][valid]
    good = np.isfinite(channels) & (channels >= 0) & (channels < 64) & (channels == np.floor(channels))
    pixels = _GIMNROOT_PIXELS[np.where(good, channels, 0).astype(int)]
    for key, column in ((f"posI_{hit}", 0), ("posJ_2", 1)):
        if key in data and (column == 0 or hit == 2):
            good &= data[key][block][valid] == pixels[:, column]
    centers = (pixels[:, ::-1] - 3.5) * (pitches * [1, -1])
    measured = np.column_stack([data[f"SiPMPos{axis}_{hit}"][block][valid] for axis in "XY"])
    offsets = measured - centers
    good &= np.isfinite(offsets).all(axis=1)
    return offsets, good, pixels, measured


def check_petsys_lattice(data, config, rsectors):
    """Check the nominal transverse crystal encoding for accepted ROOT hits."""
    axis = config.get("rotate_along", "y").lower()
    rotation = float(config["det_z_rotation"])
    if (config["crystal_type"] != "segmented" or axis not in ("y", "z")
            or not np.isclose(rotation / 90, round(rotation / 90), rtol=0, atol=1e-8)):
        return {}
    is_x = _transverse_is_x(axis, rotation)
    coordinate = "SiPMPosX" if is_x else "SiPMPosY"
    n = config["sipm_pixel_n"][0 if is_x else 1]
    pitch = config["sipm_pitch"][0 if is_x else 1]
    chip_ids = [chip for chip, head in rsectors.items() if head is not None]
    heads = {rsectors[chip] for chip in chip_ids}
    pitches = np.asarray(config["sipm_pitch"], dtype=float)
    has_ids = (list(config["sipm_pixel_n"]) == [8, 8]
               and all(key in data for key in PETSYS_ID_BRANCHES[:2]))
    bounds = {}
    center_bounds = {}
    axis_index = 0 if is_x else 1
    off_lattice = same_head = invalid_ids = 0
    for first in range(0, len(data[f"{coordinate}_1"]), 100_000):
        block = slice(first, first + 100_000)
        chips = [data[f"chipID_{hit}"][block] for hit in (1, 2)]
        e1, e2 = (data[f"energyCorrected_{hit}"][block] for hit in (1, 2))
        valid = (~((e1 < config["e_min"]) | (e1 > config["e_max"])
                   | (e2 < config["e_min"]) | (e2 > config["e_max"]))
                 & np.isin(chips[0], chip_ids) & np.isin(chips[1], chip_ids))
        bad = np.zeros(valid.shape, dtype=bool)
        for hit, ids in enumerate(chips, 1):
            if has_ids:
                offsets, good, pixels, measured = _gimnroot_offsets(data, block, hit, valid, pitches)
                invalid_ids += int(np.count_nonzero(~good))
                used_chips = ids[valid][good]
                for chip in np.unique(used_chips):
                    group = used_chips == chip
                    points = offsets[good][group]
                    low, high = bounds.get(int(chip), (np.inf, -np.inf))
                    bounds[int(chip)] = (np.minimum(low, points.min(axis=0)),
                                         np.maximum(high, points.max(axis=0)))
                    minima, maxima = center_bounds.setdefault(
                        rsectors[chip], (np.full(n, np.inf), np.full(n, -np.inf)))
                    indices = pixels[good][group, 1 - axis_index]
                    coordinates = measured[good][group, axis_index]
                    np.minimum.at(minima, indices, coordinates)
                    np.maximum.at(maxima, indices, coordinates)
            else:
                index = data[f"{coordinate}_{hit}"][block] / pitch + (n - 1) / 2
                nearest = np.rint(index)
                bad |= (~np.isfinite(index) | (np.abs(index - nearest) > 0.01)
                        | (nearest < 0) | (nearest >= n))
        off_lattice += int(np.count_nonzero(valid & bad))
        for head in heads:
            ids = [chip for chip in chip_ids if rsectors[chip] == head]
            same_head += int(np.count_nonzero(valid & np.isin(chips[0], ids) & np.isin(chips[1], ids)))

    result = dict(petsys_off_lattice_events=off_lattice, petsys_same_head_events=same_head)
    if has_ids:
        offsets_by_head = {}
        ambiguous = [
            chip for chip, (low, high) in bounds.items()
            if np.any(high - low > 0.02 * pitches)
        ]
        missing = set(chip_ids) - bounds.keys()
        for chip, (low, high) in bounds.items():
            head = rsectors[chip]
            offset = (low + high) / 2
            previous = offsets_by_head.setdefault(head, offset)
            if not np.allclose(offset, previous, rtol=0, atol=0.01 * pitches):
                ambiguous.append(chip)
        # One encoded center per pixel is required. Float32 ROOT serialization
        # can round coordinates, but it cannot explain a varying calibration.
        variable = []
        raw_centers = (np.arange(n) - (n - 1) / 2) * pitch * (1 if is_x else -1)
        measured_centers = {}
        for head, (low, high) in center_bounds.items():
            observed = np.isfinite(low)
            if np.any(high[observed] - low[observed] >
                      8 * np.spacing(np.maximum(1, np.abs(low[observed])).astype(np.float32))):
                variable.append(head)
            centers = raw_centers + offsets_by_head[head][axis_index]
            centers[observed] = (low[observed] + high[observed]) / 2
            measured_centers[head] = centers.tolist()
        if invalid_ids or ambiguous or missing or variable:
            result["petsys_alignment_error"] = (
                f"{invalid_ids} invalid channel/index hits, inconsistent translations for chips "
                f"{sorted(set(ambiguous))}, no accepted hits for chips {sorted(missing)}, "
                f"variable encoded centers on heads {sorted(variable)}"
            )
        else:
            result["petsys_bin_offsets_mm"] = {
                head: offset.tolist() for head, offset in offsets_by_head.items()
            }
            result["petsys_bin_centers_mm"] = measured_centers
    return result


@njit
def _trace_weighted(rays, weights, n, pixel, grid_min, row):
    grid_max = grid_min + n * pixel
    for ray, weight in zip(rays, weights):
        if (max(ray[0], ray[2]) < grid_min or min(ray[0], ray[2]) >= grid_max
                or max(ray[1], ray[3]) < grid_min or min(ray[1], ray[3]) >= grid_max):
            continue
        iy, ix, length, count = _siddon_trace(ray[0], ray[1], ray[2], ray[3], n, pixel, grid_min)
        for j in range(count):
            row[iy[j] * n + ix[j]] += weight * length[j]


def _bins_at_coordinate(coordinate):
    nearest = round(coordinate)
    if abs(coordinate - nearest) < 1e-9:
        # An exact bin edge can round either way after frame transforms.
        yield nearest - 1, 0.5
        yield nearest, 0.5
    else:
        yield int(np.floor(coordinate)), 1.0


def _nominal_pairs(acquisition, sino_shape, sino_range, face_width, *, analytic=False):
    """Enumerate transverse crystal pairs, including zero-count measurements."""
    from Gimntools.GimnRec.sinogramer import Sinogramer
    from Gimntools.GimnRec.systemSpace import systemSpace

    cfg = acquisition["detector_config"]
    detector_angles = acquisition["detector_angles"]
    active = set(acquisition["rsectors"].values())
    axis = cfg.get("rotate_along", "y").lower()
    if (cfg["crystal_type"] != "segmented" or axis not in ("y", "z")
            or axis != acquisition["first_rotation"]
            or len(active) != 2 or not active.issubset(detector_angles)):
        raise ValueError("DOI requires two active opposing segmented detectors in the slice plane")
    heads = sorted(active)
    if not np.isfinite([detector_angles[name] for name in heads]).all() or not np.isclose(
        (detector_angles[heads[1]] - detector_angles[heads[0]]) % 360, 180.0,
        rtol=0, atol=1e-6,
    ):
        raise ValueError("DOI requires opposing detector orientations")
    if acquisition.get("petsys_alignment_error"):
        raise ValueError("Cannot verify GimnROOT channel alignment: " + acquisition["petsys_alignment_error"])
    if acquisition.get("petsys_off_lattice_events", 0) or acquisition.get("petsys_same_head_events", 0):
        raise ValueError(
            "PETSYS DOI requires opposite-head coincidences and transverse SiPM hit positions "
            "at nominal segment centers (within 1% of the configured pitch). ROOT contains "
            f"{acquisition.get('petsys_off_lattice_events', 0)} off-lattice and "
            f"{acquisition.get('petsys_same_head_events', 0)} same-head events"
        )
    if cfg["data_from"] == "GATE" and cfg["process_as"] != "CrystalID":
        raise ValueError("GATE DOI requires the CrystalID sinogram input mode")
    poses = np.asarray(acquisition["object_angles_deg"], dtype=float)
    if not poses.size or not np.isfinite(poses).all():
        raise ValueError("DOI requires finite acquisition poses from the ROOT data")
    poses = np.unique(poses)
    if poses.size != cfg["rotation_angles"]:
        raise ValueError(
            f"DOI needs all configured discrete poses. ROOT contains {poses.size}, "
            f"detector_config specifies {cfg['rotation_angles']}"
        )

    space = systemSpace(detector_angles, axis, cfg, axis)
    if cfg["data_from"] == "PETSYS":
        rotation = float(cfg["det_z_rotation"])
        if not np.isclose(rotation / 90, round(rotation / 90), rtol=0, atol=1e-8):
            raise ValueError("PETSYS DOI requires the SiPM lattice aligned to the slice axes")
    else:
        rotation = 0.0
    # After rotation and coordinate reversal the X input contributes to
    # the Z axis at 0 degrees and to the Y axis at 90 degrees.
    transverse_is_x = _transverse_is_x(axis, rotation)
    centers = space.positionsX if transverse_is_x else space.positionsY
    pitch = space.PichX if transverse_is_x else space.PichY
    depth, local_y, local_z = cfg["crystal_size"]
    block_width = local_z if axis == "y" else local_y
    if face_width > pitch or 2 * max(map(abs, centers)) + face_width > block_width + 1e-8:
        raise ValueError("DOI segment face width exceeds the configured pitch or block envelope")

    plane = list(Sinogramer._PLANE_AXES[axis])
    local_tangent = np.eye(3)[plane[1]]
    local_normal = np.array([1.0, 0.0, 0.0])
    front = cfg["radius"] + cfg["pla_thickness"]
    n_dist, n_ang = sino_shape
    exposure = np.asarray(acquisition.get("pose_exposure", np.ones(poses.size)), dtype=float)
    if analytic:
        if (exposure.shape != poses.shape or not np.isfinite(exposure).all()
                or np.any(exposure < 0) or not np.isfinite(exposure.sum()) or exposure.sum() <= 0):
            raise ValueError("Analytical response requires a nonnegative exposure weight per sorted pose")
        exposure = exposure / exposure.sum()
    rows = {}
    measured_centers = acquisition.get("petsys_bin_centers_mm")
    if measured_centers is not None:
        from scipy.spatial.transform import Rotation
        local_rotation = Rotation.from_euler("z", rotation, degrees=True)
        raw_axis = np.eye(3)[0 if transverse_is_x else 1]

        def local_centers(coordinates):
            return local_rotation.apply(np.asarray(coordinates)[:, None] * raw_axis)[:, ::-1]

        physical_local = local_centers(centers)
        measured_local = [local_centers(measured_centers[name]) for name in heads]
    else:
        physical_local = np.asarray(centers)[:, None] * local_tangent
        measured_local = [physical_local] * 2
    local_offsets = physical_local @ local_tangent
    for pose, dwell in zip(poses, exposure if analytic else np.ones(poses.size)):
        space.rotateBy(float(pose))
        rotations = [space.global_rot * space.detectors[name] for name in heads]
        normals = [rot.apply(local_normal)[plane] for rot in rotations]
        tangents = [rot.apply(local_tangent)[plane] for rot in rotations]
        endpoints = [
            rot.apply(front * local_normal + physical_local) for rot in rotations
        ]
        bin_endpoints = endpoints if measured_centers is None else [
            rot.apply(front * local_normal + local)
            for rot, local in zip(rotations, measured_local)
        ]
        for i, (p1, measured1) in enumerate(zip(endpoints[0], bin_endpoints[0])):
            for j, (p2, measured2) in enumerate(zip(endpoints[1], bin_endpoints[1])):
                r, phi = space.global_to_sinogram(measured1, measured2, axis)
                for angular, wa in _bins_at_coordinate(phi / 180 * n_ang):
                    # Across the angular seam the radial coordinate changes sign.
                    raw_r = -r if angular < 0 or angular >= n_ang else r
                    coord = (raw_r + sino_range / 2) / sino_range * n_dist
                    for radial, wr in _bins_at_coordinate(coord):
                        if 0 <= radial < n_dist:
                            row = radial * n_ang + (angular % n_ang)
                            segments = None
                            if analytic:
                                segments = (local_offsets - local_offsets[i],
                                            local_offsets - local_offsets[j])
                            rows.setdefault(row, []).append(
                                (p1[plane], p2[plane], normals, tangents, segments,
                                 phi, wa * wr * dwell))
    return rows, depth


def _scintillator_path(depth, offset, cosine, tangent, segments, width):
    """Length in this head from the slab entrance to the interaction point."""
    limit = depth / cosine
    path = np.zeros_like(limit)
    transverse = np.abs(tangent) > 1e-14
    denominator = np.where(transverse, tangent, 1.0)
    for center in segments:
        enter = (offset - center - width / 2) / denominator
        leave = (offset - center + width / 2) / denominator
        chord = np.maximum(0, np.minimum(limit, np.maximum(enter, leave))
                           - np.maximum(0, np.minimum(enter, leave)))
        within = np.abs(offset - center) <= width / 2
        path += np.where(transverse, chord, np.where(within, limit, 0.0))
    return path


def _pair_rays(pair, offsets, face_weights, depths, depth_weights, mu, width):
    """Integrate interaction points, testing front-face entry for both photons."""
    p1, p2, normals, tangents, segments, _phi, bin_weight = pair
    normal1, normal2 = normals
    tangent1, tangent2 = tangents
    points1 = p1 + offsets[:, None] * tangent1 + depths[:, None] * normal1
    points2 = p2 + offsets[:, None] * tangent2 + depths[:, None] * normal2
    delta = points1[:, None, :] - points2[None, :, :]
    length = np.linalg.norm(delta, axis=-1)
    direction = delta / length[..., None]
    cosine1 = direction @ normal1
    cosine2 = -direction @ normal2
    if segments is not None:
        path1 = _scintillator_path(depths[:, None], offsets[:, None], cosine1,
                                  direction @ tangent1, segments[0], width)
        path2 = _scintillator_path(depths[None, :], offsets[None, :], cosine2,
                                  -direction @ tangent2, segments[1], width)
        face = face_weights[:, None] * face_weights[None, :]
        depth = depth_weights[:, None] * depth_weights[None, :]
        weights = face * depth * np.exp(-mu * (path1 + path2)) * bin_weight / length
        # Emission is modeled in the gap between the detector fronts.
        # Keep the requested grid and integrate only this segment of each ray.
        front1 = points1[:, None, :] - (depths[:, None] / cosine1)[..., None] * direction
        front2 = points2[None, :, :] + (depths[None, :] / cosine2)[..., None] * direction
        rays = np.concatenate((front1, front2), axis=-1)
        return rays.reshape(-1, 4), weights.ravel()
    entrance1 = offsets[:, None] - depths[:, None] * (direction @ tangent1) / cosine1
    entrance2 = offsets[None, :] + depths[None, :] * (direction @ tangent2) / cosine2
    accepted = (np.abs(entrance1) <= width / 2) & (np.abs(entrance2) <= width / 2)
    i, j = np.nonzero(accepted)
    # Depth quadrature samples the normal-incidence first-interaction measure.
    # Correct its exponential to the actual incidence of each joined ray.
    extra_path = depths[i] * (1 / cosine1[i, j] - 1) + depths[j] * (1 / cosine2[i, j] - 1)
    weights = face_weights[i] * face_weights[j] * depth_weights[i] * depth_weights[j]
    weights *= bin_weight * np.exp(-mu * extra_path) / length[i, j]
    return np.column_stack((points1[i], points2[j])), weights


def _doi_matrix(recon, angles_deg, settings, acquisition, *, analytic=False):
    width, mu = settings.face_width_mm, settings.mu_per_mm
    if width is None or not np.isfinite([width, mu]).all() or min(width, mu) <= 0:
        raise ValueError("DOI requires positive finite segment face width and mu (mm^-1)")
    if any(not isinstance(order, (int, np.integer)) or not 1 <= order <= 12
           for order in (settings.face_order, settings.depth_order)):
        raise ValueError("DOI quadrature orders must be integers in [1, 12]")
    shape = (recon.n_dist, recon.n_ang)
    pairs, depth = _nominal_pairs(
        acquisition, shape, recon.sino_range, width,
        analytic=analytic,
    )
    ray_count = sum(map(len, pairs.values())) * (settings.face_order * settings.depth_order)**2
    if ray_count > 10_000_000:
        raise ValueError("DOI quadrature would trace more than 10 million rays. Reduce the orders")
    face_nodes, face_weights = roots_legendre(settings.face_order)
    if analytic:
        nodes, weights = roots_legendre(settings.depth_order)
        depths = (nodes + 1) * depth / 2
        depth_weights = mu * depth / 2 * weights
    else:
        depths, depth_weights, acceptance = first_interaction_quadrature(
            depth, mu, 1.0, settings.depth_order)
        depth_weights *= acceptance
    offsets = np.repeat(width / 2 * face_nodes, settings.depth_order)
    face_weights = np.repeat(width / 2 * face_weights, settings.depth_order)
    depths = np.tile(depths, settings.face_order)
    depth_weights = np.tile(depth_weights, settings.face_order)
    raw_angles = (np.arange(recon.n_ang) + 0.5) * 180 / recon.n_ang
    angle_shift = np.asarray(angles_deg) - raw_angles
    if not np.allclose(angle_shift, angle_shift[0], rtol=0, atol=1e-8):
        raise ValueError("DOI expects the raw angular grid plus a constant encoder offset")
    shift = np.deg2rad(angle_shift[0])
    rotation = np.array([[np.cos(shift), np.sin(shift)], [-np.sin(shift), np.cos(shift)]])
    n = recon.n_img
    data, indices = [], []
    indptr = [0]
    row_data = np.zeros(n * n)
    for k in range(recon.n_dist * recon.n_ang):
        row_data.fill(0)
        for pair in pairs.get(k, ()):
            rays, weights = _pair_rays(pair, offsets, face_weights, depths, depth_weights, mu, width)
            rays = rays.reshape((-1, 2, 2)) @ rotation
            theta = np.deg2rad(pair[-2]) + shift
            # A wrapped angular row uses the opposite radial sign.
            if abs((k % recon.n_ang + 0.5) * 180 / recon.n_ang - pair[-2]) > 90:
                theta += np.pi
            translation = np.array([recon.dx_cor_mm, recon.dy_cor_mm])
            translation += recon.cor_mm * np.array([np.cos(theta), np.sin(theta)])
            rays -= translation
            _trace_weighted(rays.reshape((-1, 4)), weights, n, recon.img_pixel,
                            recon.img_min, row_data)
        nonzero = np.flatnonzero(row_data)
        indptr.append(indptr[-1] + nonzero.size)
        if indptr[-1] > 10_000_000:
            raise ValueError("DOI matrix exceeds the sparse size limit. Reduce the image grid or quadrature order")
        if nonzero.size:
            indices.append(nonzero)
            data.append(row_data[nonzero])
    if not data:
        raise ValueError("DOI geometry has no sensitivity on this image and sinogram grid")
    coefficients = np.concatenate(data)
    if analytic:
        # In 2D the line measure is db dtheta with orientation modulo pi.
        # Divide by voxel area to average over uniformly distributed sources.
        coefficients /= np.pi * recon.img_pixel**2
    return csr_matrix((coefficients, np.concatenate(indices), indptr),
                      shape=(recon.n_dist * recon.n_ang, n * n))


def _axis(lut, name):
    axis = np.asarray(lut[name], dtype=float)
    if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)):
        raise ValueError(f"SV-PSF {name} must be a finite, strictly increasing 1D axis in mm")
    return axis


def lut_reference_grid(recon):
    return np.asarray([recon.sino_range, recon.fov_mm, recon.n_img, recon.rays_per_bin,
                       recon.cor_mm, recon.dx_cor_mm, recon.dy_cor_mm])


def _verify_lut_acquisition(provenance, current):
    if not current:
        return
    try:
        original = json.loads(provenance)["analytical_geometry"]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Analytical LUT is missing its acquisition provenance") from exc
    geometry_keys = ("data_from", "process_as", "crystal_type", "rotate_along",
                     "radius", "pla_thickness", "crystal_size", "sipm_pitch",
                     "sipm_pixel_n", "det_z_rotation")
    old, new = original["detector_config"], current["detector_config"]
    old_centers = original.get("petsys_bin_centers_mm", {})
    new_centers = current.get("petsys_bin_centers_mm", {})
    if (any(old.get(key) != new.get(key) for key in geometry_keys)
            or original.get("first_rotation") != current.get("first_rotation")
            or original.get("detector_angles") != current.get("detector_angles")
            or original.get("object_angles_deg") != current.get("object_angles_deg")
            or old_centers.keys() != new_centers.keys()
            or any(not np.allclose(old_centers[head], new_centers[head],
                                   rtol=0, atol=1e-5) for head in old_centers)):
        raise ValueError("Analytical LUT requires the original detector and pose geometry")


def _sv_matrix(recon, angles_deg, path, compose_doi, acquisition):
    content = Path(path).read_bytes()
    with np.load(io.BytesIO(content), allow_pickle=False) as lut:
        missing = {"x_mm", "y_mm", "dx_mm", "dy_mm", "kernels",
                   "provenance", "doi_included"} - set(lut.files)
        if missing:
            raise ValueError(f"SV-PSF LUT missing fields: {', '.join(sorted(missing))}")
        x, y, dx, dy = (_axis(lut, key) for key in ("x_mm", "y_mm", "dx_mm", "dy_mm"))
        kernels = np.asarray(lut["kernels"], dtype=float)
        provenance = lut["provenance"]
        contains_doi = lut["doi_included"]
        if provenance.ndim or provenance.dtype.kind != "U" or not provenance.item().strip():
            raise ValueError("SV-PSF requires a scalar Unicode provenance description")
        if contains_doi.ndim or contains_doi.dtype.kind != "b":
            raise ValueError("SV-PSF doi_included must be a scalar boolean")
        if compose_doi and contains_doi.item():
            raise ValueError("Cannot combine DOI with a LUT that already includes DOI")
        if (kernels.shape != (len(y), len(x), len(dy), len(dx))
                or not np.isfinite(kernels).all() or np.any(kernels < 0)):
            raise ValueError("SV-PSF kernels must be finite, nonnegative and shaped (y,x,dy,dx)")
        if not np.allclose(kernels.sum(axis=(2, 3)), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("Every SV-PSF kernel must sum to 1")
        factors = None
        if "reference_factors" in lut:
            factors = np.asarray(lut["reference_factors"], dtype=float)
            expected = lut_reference_grid(recon)
            if (not contains_doi.item() or
                    factors.shape != (recon.n_dist, recon.n_ang) or
                    not np.isfinite(factors).all() or np.any(factors < 0) or
                    "reference_grid" not in lut or "reference_angles_deg" not in lut or
                    not np.array_equal(lut["reference_grid"], expected) or
                    lut["reference_angles_deg"].shape != angles_deg.shape or
                    not np.allclose(lut["reference_angles_deg"], angles_deg, rtol=0, atol=1e-8)):
                raise ValueError("Analytical LUT reference must match the original image and sinogram geometry")
            _verify_lut_acquisition(provenance.item(), acquisition)
    n = recon.n_img
    if n * n * len(dx) * len(dy) > 5_000_000:
        raise ValueError("SV-PSF matrix exceeds the sparse size limit. Reduce the image grid or LUT support")
    bins = recon.img_min + (np.arange(n) + 0.5) * recon.img_pixel
    # Linear interpolation on the calibrated source grid. Singleton axes
    # are constant and out-of-range sources use the nearest calibrated edge.
    interp = RegularGridInterpolator((y, x), kernels) if x.size * y.size > 1 else None
    matrices = []
    for first in range(0, n * n, 4096):
        source = np.arange(first, min(first + 4096, n * n))
        iy, ix = np.divmod(source, n)
        points = np.column_stack((bins[iy].clip(y[0], y[-1]), bins[ix].clip(x[0], x[-1])))
        if interp is None:
            weights = np.broadcast_to(kernels[0, 0], (len(source), len(dy), len(dx)))
        else:
            weights = interp(points)
        rows, cols, data = [], [], []
        for j, shift_y in enumerate(dy / recon.img_pixel):
            if abs(shift_y) >= n:
                continue
            by = int(np.floor(shift_y))
            fy = shift_y - by
            for i, shift_x in enumerate(dx / recon.img_pixel):
                if abs(shift_x) >= n:
                    continue
                bx = int(np.floor(shift_x))
                fx = shift_x - bx
                for oy, wy in ((by, 1 - fy), (by + 1, fy)):
                    for ox, wx in ((bx, 1 - fx), (bx + 1, fx)):
                        coeff = weights[:, j, i] * wy * wx
                        ty, tx = iy + oy, ix + ox
                        valid = (coeff > 0) & (ty >= 0) & (ty < n) & (tx >= 0) & (tx < n)
                        rows.append((ty[valid] * n + tx[valid]))
                        cols.append(source[valid] - first)
                        data.append(coeff[valid])
        block_shape = (len(source), n * n)
        block = coo_matrix(
            (np.concatenate(data), (np.concatenate(cols), np.concatenate(rows))),
            shape=block_shape,
        ).tocsr() if data else csr_matrix(block_shape)
        matrices.append(block)
    # The rows accumulated above are source pixels. Forward maps into target pixels.
    matrix = vstack(matrices, format="csr").T
    if not matrix.nnz:
        raise ValueError("SV-PSF LUT has no support within the reconstructed image grid")
    metadata = dict(sha256=hashlib.sha256(content).hexdigest(),
                    provenance=provenance.item(), doi_included=contains_doi.item(),
                    uses_reference_factors=factors is not None)
    return matrix, factors, metadata


class DetectorResponse:
    """An EM system model built for a particular image and acquisition grid."""

    def __init__(self, recon, angles_deg, settings, acquisition=None):
        self.recon = recon
        self.angles = np.asarray(angles_deg, dtype=float).copy()
        if self.angles.shape != (recon.n_ang,) or not np.isfinite(self.angles).all():
            raise ValueError("Invalid acquisition angles for detector response")
        use_doi = settings.mu_per_mm is not None
        if not use_doi and not settings.lut_path:
            raise ValueError("Select DOI or supply an SV-PSF LUT")
        if use_doi and not acquisition:
            raise ValueError("DOI needs the acquisition geometry. Build the sinogram from ROOT first")
        self.metadata = asdict(settings)
        if settings.lut_path:
            self.metadata["lut_path"] = str(settings.lut_path)
        self.metadata["angles_deg"] = self.angles.tolist()
        if acquisition is not None:
            self.metadata["acquisition"] = acquisition
        self.sv = self.reference_factors = None
        if settings.lut_path:
            self.sv, self.reference_factors, self.metadata["lut"] = _sv_matrix(
                recon, self.angles, settings.lut_path, use_doi, acquisition)

            warnings.warn("SV-PSF is exploratory until the LUT is independently calibrated "
                          "for this scanner and reconstruction model", stacklevel=2)
        if settings.analytical:
            from Gimntools.GimnRec.analytical_response import analytical_projection_matrix
            self.doi = analytical_projection_matrix(recon, self.angles, settings, acquisition)
        else:
            self.doi = _doi_matrix(recon, self.angles, settings, acquisition) if use_doi else None
        if use_doi and not settings.analytical:
            warnings.warn("DOI uses a 2D nominal crystal-pair aggregation by raw sinogram bin "
                          "with equal exposure per pose", stacklevel=2)

    def operators(self, indices=None):
        """Return forward and transpose for the full data or an angular subset."""
        r = self.recon
        angles = np.deg2rad(self.angles if indices is None else self.angles[indices])
        matrix = self.doi
        if matrix is not None and indices is not None:
            rows = np.arange(r.n_dist * r.n_ang).reshape(r.n_dist, r.n_ang)[:, indices]
            matrix = matrix[rows.ravel()]
        image_shape = (r.n_img, r.n_img)
        sino_shape = (r.n_dist, angles.size)
        transpose = matrix.T if matrix is not None else None
        sv_transpose = self.sv.T if self.sv is not None else None
        factors = self.reference_factors
        if factors is not None and indices is not None:
            factors = factors[:, indices]

        def forward(image):
            if self.sv is not None:
                image = (self.sv @ image.ravel()).reshape(image_shape)
            if matrix is None:
                projection = r._project(image, angles)
                return projection if factors is None else projection * factors
            return (matrix @ image.ravel()).reshape(sino_shape)

        def backward(sino):
            if transpose is None:
                image = r._backproject(sino if factors is None else sino * factors, angles)
            else:
                image = (transpose @ sino.ravel()).reshape(image_shape)
            if sv_transpose is not None:
                image = (sv_transpose @ image.ravel()).reshape(image_shape)
            return image

        return forward, backward


def add_response_arguments(parser):
    group = parser.add_argument_group("Detector response (MLEM/OSEM)")
    group.add_argument("--crystal-material", choices=CRYSTAL_MATERIALS,
                       help="Enable DOI with a 511 keV material preset or Custom attenuation.")
    group.add_argument("--doi-mu-mm-inv", type=float,
                       help="511 keV linear attenuation coefficient in inverse mm. Enables DOI.")
    group.add_argument("--doi-face-mm", type=float,
                       help="Override the configured active transverse crystal segment width in mm.")
    group.add_argument("--doi-face-order", type=int, default=3)
    group.add_argument("--doi-depth-order", type=int, default=3)
    group.add_argument("--analytical-response", action="store_true",
                       help="Use the full first-interaction analytical projector with DOI.")
    group.add_argument("--sv-psf-lut", help="Spatially variant response NPZ (see detector response docs).")


def response_from_arguments(args, methods, config=None):
    use_doi = (args.analytical_response or args.crystal_material is not None
               or args.doi_mu_mm_inv is not None)
    if args.analytical_response and args.sv_psf_lut:
        raise ValueError("--analytical-response is a standalone projection model")
    if args.doi_face_mm is not None and not use_doi:
        raise ValueError("--doi-face-mm requires a DOI response")
    if not use_doi and not args.sv_psf_lut:
        return None
    if args.use_psf or args.legacy_em:
        raise ValueError("DOI/SV-PSF cannot be combined with --use-psf or --legacy-em")
    if "mlem" not in methods and "osem" not in methods:
        raise ValueError("DOI/SV-PSF requires MLEM or OSEM")
    if not use_doi:
        return ResponseSettings(lut_path=args.sv_psf_lut)
    return crystal_response_settings(
        config, material=args.crystal_material, mu=args.doi_mu_mm_inv, face=args.doi_face_mm,
        face_order=args.doi_face_order, depth_order=args.doi_depth_order,
        lut_path=args.sv_psf_lut, analytical=args.analytical_response,
    )
