"""Source-dependent blur and matched transpose for 3D list-mode EM."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, csr_matrix, vstack

from Gimntools.GimnRec.detector_response import _axis
from Gimntools.GimnRec.reconstruction_backend import sparse_product


FWHM_TO_SIGMA = 1 / np.sqrt(8 * np.log(2))


@dataclass(frozen=True)
class ListmodePSF:
    shape: tuple
    matrices: tuple
    name: str
    planar: bool = False

    def forward(self, values, accelerator=None):
        if self.planar:
            plane = values.reshape(self.shape[0], -1).T
            return sparse_product(self.matrices[0], plane, accelerator).T.ravel()
        for matrix in self.matrices:
            values = sparse_product(matrix, values, accelerator)
        return values

    def transpose(self, values, accelerator=None):
        if self.planar:
            plane = values.reshape(self.shape[0], -1).T
            return sparse_product(self.matrices[0], plane, accelerator, transpose=True).T.ravel()
        for matrix in reversed(self.matrices):
            values = sparse_product(matrix, values, accelerator, transpose=True)
        return values


def _normalize_columns(matrix):
    total = np.asarray(matrix.sum(axis=0)).ravel()
    if np.any(total <= 0):
        raise ValueError("The SV-PSF has no in-grid support at a source voxel. Check the grid and kernel offsets")
    return matrix.multiply(1 / total).tocsr()


def _axis_blur(shape, axis, sigma_pixels):
    count = int(np.prod(shape))
    source = np.arange(count, dtype=np.int32)
    stride = int(np.prod(shape[axis + 1:]))
    position = source // stride % shape[axis]
    sigma = np.broadcast_to(sigma_pixels, shape).ravel()
    radius = np.minimum(shape[axis] - 1, np.ceil(3. * sigma)).astype(np.int32)
    if 2 * radius.sum(dtype=np.int64) + count > 15_000_000:
        raise ValueError("The Gaussian SV-PSF exceeds the sparse size limit. Reduce the FWHM or the image grid")
    rows, columns, weights = [], [], []
    for shift in range(-radius.max(), radius.max() + 1):
        shifted = position + shift
        valid = (shifted >= 0) & (shifted < shape[axis]) & (abs(shift) <= radius)
        if not valid.any():
            continue
        indices = source[valid]
        rows.append(indices + shift * stride)
        columns.append(indices)
        if shift:
            weights.append(np.exp(-.5 * (shift / sigma[valid])**2))
        else:
            weights.append(np.ones(len(indices)))
    matrix = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
                        shape=(count, count))
    return _normalize_columns(matrix)


def _gaussian_field(shape, spacing, center, settings, config):
    fwhm = np.asarray(settings["fwhm_mm"], dtype=float)
    growth = float(settings["growth"])
    if fwhm.shape != (2,) or not np.isfinite(fwhm).all() or np.any(fwhm < 0) or not np.isfinite(growth) or growth < 0:
        raise ValueError("Enter nonnegative XY and Z FWHM and nonnegative SV-PSF growth")
    if not fwhm.any():
        raise ValueError("Enter a positive SV-PSF FWHM in at least one direction")
    sizes_xyz = np.asarray(shape[::-1]) * spacing
    config = config or {}
    face = np.asarray(config.get("crystal_size", []), dtype=float)
    axis = config.get("rotate_along", "y")
    xy_reference = float(config.get("sino_range_mm", min(face[1:]) if face.size == 3 else min(sizes_xyz[:2]))) / 2
    z_reference = (face[2 if axis == "z" else 1] if face.size == 3 else sizes_xyz[2]) / 2
    position = [origin + (np.arange(n) - (n - 1) / 2) * step
                for origin, n, step in zip(center[::-1], shape, spacing[::-1])]
    z, y, x = np.ix_(*position)
    transverse = np.minimum(1., (x*x + y*y) / xy_reference**2)
    axial = np.minimum(1., (z / z_reference)**2)
    xy_sigma = fwhm[0] * FWHM_TO_SIGMA * (1. + growth * transverse)
    z_sigma = fwhm[1] * FWHM_TO_SIGMA * (1. + growth * np.maximum(transverse, axial))
    matrices = tuple(_axis_blur(shape, axis, sigma / spacing[2 - axis])
                     for axis, sigma in ((2, xy_sigma), (1, xy_sigma), (0, z_sigma))
                     if np.any(sigma))
    return ListmodePSF(shape, matrices, "3D Gaussian SV-PSF")


def _lut_projection(shape, positions, spacing, fields, offsets, kernels):
    count = int(np.prod(shape))
    if count * int(np.prod(kernels.shape[len(shape):])) > 12_000_000:
        raise ValueError("The SV-PSF LUT exceeds the sparse size limit. Use a smaller kernel or image grid")
    interp = RegularGridInterpolator(fields, kernels)
    shifts = []
    for index in np.ndindex(*kernels.shape[len(shape):]):
        if not np.any(kernels[(...,) + index]):
            continue
        shift = np.array([values[i] / step for values, i, step in zip(offsets, index, spacing)])
        low = np.floor(shift).astype(int)
        frac = shift - low
        for deltas in product((0, 1), repeat=len(shape)):
            alpha = np.prod(np.where(deltas, frac, 1 - frac))
            if alpha > 0:
                shifts.append((index, low + deltas, alpha))
    pieces = []
    for start in range(0, count, 768):
        length = min(768, count - start)
        coord = np.column_stack(np.unravel_index(np.arange(start, start + length), shape))
        points = np.column_stack([axis[i].clip(field[0], field[-1])
                                  for axis, i, field in zip(positions, coord.T, fields)])
        amplitudes = interp(points)
        rows, cols, coefficients = [], [], []
        for index, shift, alpha in shifts:
            target = coord + shift
            weight = alpha * amplitudes[(slice(None),) + index]
            valid = (weight > 0) & np.all((target >= 0) & (target < shape), axis=1)
            if valid.any():
                rows.append(np.flatnonzero(valid))
                cols.append(np.ravel_multi_index(target[valid].T, shape))
                coefficients.append(weight[valid])
        matrix = coo_matrix((np.concatenate(coefficients), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(length, count)).tocsr() if rows else csr_matrix((length, count))
        pieces.append(matrix)
    return _normalize_columns(vstack(pieces, format="csr").T)


def _calibrated_lut(shape, spacing, center, path):
    with np.load(Path(path), allow_pickle=False) as lut:
        volumetric = "z_mm" in lut or "dz_mm" in lut
        dims = "zyx" if volumetric else "yx"
        required = {f"{prefix}{axis}_mm" for prefix in ("", "d") for axis in dims}
        required |= {"kernels", "provenance", "doi_included"}
        if missing := required - set(lut.files):
            raise ValueError(f"List-mode SV-PSF LUT lacks fields {', '.join(sorted(missing))}")
        doi = lut["doi_included"]
        if doi.shape or doi.dtype.kind != "b" or doi.item() or "reference_factors" in lut:
            raise ValueError("List mode requires a residual SV-PSF without DOI or sinogram reference factors")
        provenance = lut["provenance"]
        if provenance.shape or provenance.dtype.kind not in "US" or not provenance.item().strip():
            raise ValueError("List-mode SV-PSF LUT requires a provenance description")
        fields = tuple(_axis(lut, f"{axis}_mm") for axis in dims)
        shifts = tuple(_axis(lut, f"d{axis}_mm") for axis in dims)
        kernels = np.asarray(lut["kernels"], dtype=float)
    if (kernels.shape != tuple(map(len, fields + shifts)) or np.any(kernels < 0)
            or not np.isfinite(kernels).all()
            or not np.allclose(kernels.sum(axis=tuple(range(len(dims), 2*len(dims)))), 1., rtol=1e-6)):
        raise ValueError("SV-PSF kernels must match the source and offset axes, be nonnegative and sum to one")
    grid = shape if volumetric else shape[1:]
    steps = spacing[:len(dims)][::-1]
    positions = [origin + (np.arange(n) - (n - 1) / 2) * step
                 for origin, n, step in zip(center[:len(dims)][::-1], grid, steps)]
    matrix = _lut_projection(grid, positions, steps, fields, shifts, kernels)
    return ListmodePSF(shape, (matrix,), "calibrated 3D SV-PSF" if volumetric else "calibrated transaxial SV-PSF",
                       planar=not volumetric)


def prepare_listmode_psf(shape, spacing, center, settings, detector_config=None):
    """Build the image-space operator on the selected grid."""
    from Gimntools.GimnRec.listmode import _volume_grid

    shape_xyz, spacing, center, _ = _volume_grid(shape, spacing, center)
    shape = tuple(shape_xyz[::-1])
    if settings.get("lut"):
        return _calibrated_lut(shape, spacing, center, settings["lut"])
    return _gaussian_field(shape, spacing, center, settings, detector_config)
