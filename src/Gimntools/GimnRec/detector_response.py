"""Matched, slice-wise detector response operators for Poisson reconstruction."""

import hashlib
import io
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
    chip_ids = list(rsectors)
    off_lattice = same_head = 0
    for first in range(0, len(data[f"{coordinate}_1"]), 100_000):
        block = slice(first, first + 100_000)
        chip1, chip2 = data["chipID_1"][block], data["chipID_2"][block]
        e1, e2 = data["energyCorrected_1"][block], data["energyCorrected_2"][block]
        valid = ((e1 >= config["e_min"]) & (e1 <= config["e_max"])
                 & (e2 >= config["e_min"]) & (e2 <= config["e_max"])
                 & np.isin(chip1, chip_ids) & np.isin(chip2, chip_ids))
        bad = np.zeros(valid.shape, dtype=bool)
        for hit in (1, 2):
            index = data[f"{coordinate}_{hit}"][block] / pitch + (n - 1) / 2
            nearest = np.rint(index)
            bad |= (~np.isfinite(index) | (np.abs(index - nearest) > 0.01)
                    | (nearest < 0) | (nearest >= n))
        off_lattice += int(np.count_nonzero(valid & bad))
        for head in set(rsectors.values()):
            ids = [key for key in rsectors if rsectors[key] == head]
            same_head += int(np.count_nonzero(valid & np.isin(chip1, ids) & np.isin(chip2, ids)))
    return dict(petsys_off_lattice_events=off_lattice, petsys_same_head_events=same_head)


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


def _nominal_pairs(acquisition, sino_shape, sino_range, face_width):
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
    rows = {}
    for pose in poses:
        space.rotateBy(float(pose))
        rotations = [space.global_rot * space.detectors[name] for name in heads]
        normals = [rot.apply(local_normal)[plane] for rot in rotations]
        tangents = [rot.apply(local_tangent)[plane] for rot in rotations]
        endpoints = [
            rot.apply(front * local_normal + np.asarray(centers)[:, None] * local_tangent)
            for rot in rotations
        ]
        for p1 in endpoints[0]:
            for p2 in endpoints[1]:
                r, phi = space.global_to_sinogram(p1, p2, axis)
                for angular, wa in _bins_at_coordinate(phi / 180 * n_ang):
                    # Across the angular seam the radial coordinate changes sign.
                    raw_r = -r if angular < 0 or angular >= n_ang else r
                    coord = (raw_r + sino_range / 2) / sino_range * n_dist
                    for radial, wr in _bins_at_coordinate(coord):
                        if 0 <= radial < n_dist:
                            row = radial * n_ang + (angular % n_ang)
                            rows.setdefault(row, []).append(
                                (p1[plane], p2[plane], normals, tangents, phi, wa * wr))
    return rows, depth


def _pair_rays(pair, offsets, face_weights, depths, depth_weights, mu, width):
    """Integrate interaction points, testing front-face entry for both photons."""
    p1, p2, normals, tangents, _phi, bin_weight = pair
    normal1, normal2 = normals
    tangent1, tangent2 = tangents
    points1 = p1 + offsets[:, None] * tangent1 + depths[:, None] * normal1
    points2 = p2 + offsets[:, None] * tangent2 + depths[:, None] * normal2
    delta = points1[:, None, :] - points2[None, :, :]
    length = np.linalg.norm(delta, axis=-1)
    direction = delta / length[..., None]
    cosine1 = direction @ normal1
    cosine2 = -direction @ normal2
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


def _doi_matrix(recon, angles_deg, settings, acquisition):
    width, mu = settings.face_width_mm, settings.mu_per_mm
    if width is None or not np.isfinite([width, mu]).all() or min(width, mu) <= 0:
        raise ValueError("DOI requires positive finite segment face width and mu (mm^-1)")
    if any(not isinstance(order, (int, np.integer)) or not 1 <= order <= 12
           for order in (settings.face_order, settings.depth_order)):
        raise ValueError("DOI quadrature orders must be integers in [1, 12]")
    shape = (recon.n_dist, recon.n_ang)
    pairs, depth = _nominal_pairs(acquisition, shape, recon.sino_range, width)
    ray_count = sum(map(len, pairs.values())) * (settings.face_order * settings.depth_order)**2
    if ray_count > 10_000_000:
        raise ValueError("DOI quadrature would trace more than 10 million rays. Reduce the orders")
    face_nodes, face_weights = roots_legendre(settings.face_order)
    depths, depth_weights, acceptance = first_interaction_quadrature(
        depth, mu, 1.0, settings.depth_order)
    offsets = np.repeat(width / 2 * face_nodes, settings.depth_order)
    face_weights = np.repeat(width / 2 * face_weights, settings.depth_order)
    depths = np.tile(depths, settings.face_order)
    depth_weights = np.tile(acceptance * depth_weights, settings.face_order)
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
    return csr_matrix((np.concatenate(data), np.concatenate(indices), indptr),
                      shape=(recon.n_dist * recon.n_ang, n * n))


def _axis(lut, name):
    axis = np.asarray(lut[name], dtype=float)
    if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)):
        raise ValueError(f"SV-PSF {name} must be a finite, strictly increasing 1D axis in mm")
    return axis


def _sv_matrix(recon, path, compose_doi):
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
                    provenance=provenance.item(), doi_included=contains_doi.item())
    return matrix, metadata


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
        self.sv = None
        if settings.lut_path:
            self.sv, self.metadata["lut"] = _sv_matrix(recon, settings.lut_path, use_doi)
            warnings.warn("SV-PSF is exploratory until the LUT is independently calibrated "
                          "for this scanner and reconstruction model", stacklevel=2)
        self.doi = _doi_matrix(recon, self.angles, settings, acquisition) if use_doi else None
        if use_doi:
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

        def forward(image):
            if self.sv is not None:
                image = (self.sv @ image.ravel()).reshape(image_shape)
            if matrix is None:
                return r._project(image, angles)
            return (matrix @ image.ravel()).reshape(sino_shape)

        def backward(sino):
            if transpose is None:
                image = r._backproject(sino, angles)
            else:
                image = (transpose @ sino.ravel()).reshape(image_shape)
            if sv_transpose is not None:
                image = (sv_transpose @ image.ravel()).reshape(image_shape)
            return image

        return forward, backward


def add_response_arguments(parser):
    group = parser.add_argument_group("Detector response (MLEM/OSEM)")
    group.add_argument("--doi-mu-mm-inv", type=float,
                       help="511 keV linear attenuation coefficient in inverse mm. Enables DOI.")
    group.add_argument("--doi-face-mm", type=float,
                       help="Active transverse crystal segment width in mm. Required with DOI.")
    group.add_argument("--doi-face-order", type=int, default=3)
    group.add_argument("--doi-depth-order", type=int, default=3)
    group.add_argument("--sv-psf-lut", help="Spatially variant response NPZ (see detector response docs).")


def response_from_arguments(args, methods):
    if args.doi_mu_mm_inv is None and not args.sv_psf_lut:
        if args.doi_face_mm is not None:
            raise ValueError("--doi-face-mm requires --doi-mu-mm-inv")
        return None
    if args.use_psf or args.legacy_em:
        raise ValueError("DOI/SV-PSF cannot be combined with --use-psf or --legacy-em")
    if "mlem" not in methods and "osem" not in methods:
        raise ValueError("DOI/SV-PSF requires MLEM or OSEM")
    if (args.doi_mu_mm_inv is None) != (args.doi_face_mm is None):
        raise ValueError("Supply both --doi-mu-mm-inv and --doi-face-mm for DOI")
    return ResponseSettings(args.doi_mu_mm_inv, args.doi_face_mm,
                            args.doi_face_order, args.doi_depth_order, args.sv_psf_lut)
