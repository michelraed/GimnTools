"""Prediction and LUT fitting for a two-head detector with finite segments."""

from dataclasses import asdict
from functools import lru_cache
from io import BytesIO
import json
from operator import index
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.spatial import cKDTree

from Gimntools import __version__
from Gimntools.GimnRec.detector_response import (
    _doi_matrix, depth_quadrature_order, face_quadrature_order, lut_reference_grid,
)


DEFAULT_LUT_FIT_TOLERANCE = 0.30


def analytical_projection_matrix(recon, angles_deg, settings, acquisition):
    """Return P(measured bin | emission in pixel) for the transaxial model.

    Columns represent uniform emission within each voxel. Equal pose dwell
    is used unless acquisition["pose_exposure"] gives relative exposures in
    sorted pose order. The matrix retains relative efficiency.
    """
    if settings.mu_per_mm is None or settings.lut_path is not None:
        raise ValueError("Analytical projection requires a 511 keV attenuation coefficient without a LUT")
    if not acquisition:
        raise ValueError("Build the unrebinned sinogram from ROOT before generating the analytical response")
    angles = np.asarray(angles_deg, dtype=float)
    if angles.shape != (recon.n_ang,) or not np.isfinite(angles).all():
        raise ValueError("Analytical projection requires a finite angle per sinogram column")
    matrix = _doi_matrix(recon, angles, settings, acquisition, analytic=True)
    efficiency = np.asarray(matrix.sum(axis=0)).ravel()
    if not np.isfinite(efficiency).all() or np.any(efficiency > 1):
        raise ValueError("Analytical quadrature exceeds unit probability. Check geometry and increase quadrature orders")
    return matrix


def analytical_reference_factors(recon, angles_deg, matrix, *, source_support=None):
    """Match uniform illumination of the modeled voxels on the requested grid."""
    source = (np.ones((recon.n_img, recon.n_img)) if source_support is None
              else np.asarray(source_support, dtype=float))
    uniform = recon._project(source, np.deg2rad(angles_deg))
    analytical_uniform = np.asarray(matrix.sum(axis=1)).reshape(uniform.shape)
    return np.divide(analytical_uniform, uniform, out=np.zeros_like(uniform),
                     where=uniform > 0)


def _fit_kernel(reference, target, support, regularization):
    """Constrained kernel fit with the nonnegative amplitude eliminated analytically."""
    scale = np.linalg.norm(target)
    if scale == 0:
        raise ValueError("Cannot fit a LUT at a position with zero predicted efficiency")
    target = target / scale
    initial, _ = nnls(reference, target)
    mass = initial.sum()
    if mass <= 0:
        raise ValueError("Reference projector has no overlap with the analytical prediction")
    initial /= mass

    def objective(kernel):
        projection = reference @ kernel
        power = projection @ projection
        amplitude = max(0.0, (projection @ target) / power) if power else 0.0
        residual = amplitude * projection - target
        grid = kernel.reshape(support, support)
        vertical, horizontal = np.diff(grid, axis=0), np.diff(grid, axis=1)
        gradient = np.zeros_like(grid)
        gradient[1:] += vertical
        gradient[:-1] -= vertical
        gradient[:, 1:] += horizontal
        gradient[:, :-1] -= horizontal
        value = residual @ residual + regularization * (
            np.sum(vertical**2) + np.sum(horizontal**2)
        )
        derivative = 2 * (amplitude * (reference.T @ residual) + regularization * gradient.ravel())
        return value, derivative

    result = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * initial.size,
        constraints={"type": "eq", "fun": lambda k: k.sum() - 1,
                     "jac": lambda k: np.ones_like(k)},
        options={"ftol": 1e-10, "maxiter": 300},
    )
    if not result.success:
        raise ValueError(f"Analytical LUT fit failed: {result.message}")
    kernel = np.maximum(result.x, 0)
    kernel /= kernel.sum()
    prediction = reference @ kernel
    power = prediction @ prediction
    amplitude = max(0.0, (prediction @ target) / power) if power else 0.0
    error = np.linalg.norm(amplitude * prediction - target)
    return kernel.reshape(support, support), amplitude * scale, error


def generate_analytical_lut(recon, angles_deg, settings, acquisition, path=None, *,
                            sample_count=5, half_support=2, kernel_stride=1,
                            regularization=1e-4, weights=None, fit_tolerance=DEFAULT_LUT_FIT_TOLERANCE,
                            analytical_matrix=None):
    """Fit and export a full DOI-inclusive LUT using the conventional Siddon model.

    The file is compatible with the regular LUT loader. Separate arrays hold
    the fitted amplitude, analytical efficiency and relative projection error.
    Positions and support offsets fall on the supplied reconstruction grid.
    With no output path, return the fitted map without creating a file.
    The caller can supply the analytical matrix already built for this grid.
    """
    sample_count, half_support, kernel_stride = map(index, (sample_count, half_support, kernel_stride))
    margin = half_support * kernel_stride
    n = recon.n_img
    if (not 1 <= sample_count <= 32 or not 0 <= half_support <= 12
            or kernel_stride < 1 or 2 * margin >= n
            or sample_count > n - 2 * margin):
        raise ValueError("Select a valid LUT mesh and kernel support within the image grid")
    if (not np.isfinite([regularization, fit_tolerance]).all()
            or regularization < 0 or fit_tolerance < 0):
        raise ValueError("LUT regularization and fit tolerance must be finite and nonnegative")
    if weights is None:
        sqrt_weights = np.ones(recon.n_dist * recon.n_ang)
    else:
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (recon.n_dist, recon.n_ang) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights)):
            raise ValueError("LUT weights need at least one positive finite weight on the sinogram grid")
        sqrt_weights = np.sqrt(weights.ravel())

    start = perf_counter()
    if analytical_matrix is None:
        analytical_matrix = analytical_projection_matrix(recon, angles_deg, settings, acquisition)
    if analytical_matrix.shape != (recon.n_dist * recon.n_ang, n * n):
        raise ValueError("Analytical projection matrix must match the sinogram and image grids")
    matrix = analytical_matrix.tocsc()
    angles_rad = np.deg2rad(angles_deg)
    efficiency_map = np.asarray(matrix.sum(axis=0)).reshape(n, n)
    source_support = efficiency_map > 0
    reference_factors = analytical_reference_factors(
        recon, angles_deg, matrix, source_support=source_support)
    weighted_factors = reference_factors.ravel() * sqrt_weights

    candidates = np.asarray(matrix.T @ sqrt_weights).reshape(n, n) > 0
    if margin:
        candidates[:margin] = candidates[-margin:] = False
        candidates[:, :margin] = candidates[:, -margin:] = False
    positions = np.argwhere(candidates)
    if not len(positions):
        raise ValueError("No observable source voxel has room for the requested LUT kernel. "
                         "Check the field of view, weights and kernel width")
    tree = cKDTree(positions)
    lower, upper = positions.min(axis=0), positions.max(axis=0)

    def axis_samples(first, last):
        if sample_count == 1:
            return np.array([np.clip(n // 2, first, last)])
        return np.unique(np.rint(np.linspace(first, last, sample_count)).astype(int))

    y_mesh = axis_samples(lower[0], upper[0])
    x_mesh = axis_samples(lower[1], upper[1])
    offset = np.arange(-half_support, half_support + 1) * kernel_stride
    width = len(offset)
    if width**2 * sqrt_weights.size > 5_000_000:
        raise ValueError("Analytical LUT fit exceeds the reference size limit. Reduce the kernel support")

    @lru_cache(maxsize=min(128, max(1, 2_000_000 // sqrt_weights.size)))
    def reference_column(iy, ix):
        image = np.zeros((n, n))
        image[iy, ix] = 1
        return recon._project(image, angles_rad).ravel() * weighted_factors

    @lru_cache(maxsize=1024)
    def fit(iy, ix):
        target = matrix[:, iy * n + ix].toarray().ravel() * sqrt_weights
        reference = np.column_stack([
            reference_column(iy + dy, ix + dx) for dy in offset for dx in offset
        ])
        if not np.any(reference.T @ target):
            return None
        return _fit_kernel(reference, target, width, regularization)

    def observable_fit(iy, ix):
        _, closest = tree.query((iy, ix))
        sy, sx = positions[closest]
        result = fit(sy, sx)
        if result is not None:
            return (sy, sx), result
        _, neighbours = tree.query((iy, ix), k=len(positions))
        for item in np.atleast_1d(neighbours)[1:]:
            sy, sx = positions[item]
            result = fit(sy, sx)
            if result is not None:
                return (sy, sx), result
        raise ValueError("Analytical and Siddon responses do not overlap at observable source voxels. "
                         "Increase the kernel width or the rays per sinogram bin")

    shape = (len(y_mesh), len(x_mesh))
    kernels = np.empty((*shape, width, width))
    amplitude, error, fit_efficiency = (np.empty(shape) for _ in range(3))
    fit_positions = np.empty((*shape, 2), dtype=int)
    for i, iy in enumerate(y_mesh):
        for j, ix in enumerate(x_mesh):
            (sy, sx), result = observable_fit(iy, ix)
            kernels[i, j], amplitude[i, j], error[i, j] = result
            fit_efficiency[i, j] = efficiency_map[sy, sx]
            fit_positions[i, j] = [sx, sy]
    nominal = np.stack(np.meshgrid(x_mesh, y_mesh), axis=-1)
    relocated = int(np.count_nonzero(np.any(nominal != fit_positions, axis=-1)))
    if relocated:
        warnings.warn(
            f"{relocated} LUT nodes were fitted at nearby observable voxels. "
            "See the recorded fitting locations and efficiency map.", stacklevel=2)
    if np.any(error > fit_tolerance):
        warnings.warn("Analytical LUT fit exceeds the error threshold at some positions. "
                      "Use the analytical projection matrix for quantitative work.", stacklevel=2)
    grid = recon.img_min + (np.arange(n) + 0.5) * recon.img_pixel
    delta = offset * recon.img_pixel
    provenance = json.dumps(dict(
        version=__version__, model="2D collinear first-interaction, finite rectangular segments",
        analytical_geometry=acquisition, reconstruction_grid=recon.geometry(),
        settings=asdict(settings), face_order_used=face_quadrature_order(recon, settings),
        depth_order_used=depth_quadrature_order(recon, settings, acquisition),
        regularization=regularization,
        weighted_fit=weights is not None, fit_tolerance=fit_tolerance,
        relocated_nodes=relocated, requested_samples_per_axis=sample_count,
        forward_reference="Siddon with analytical supported-field bin acceptance",
        references=["10.1088/0031-9155/54/7/004", "10.1088/0031-9155/48/8/302",
                    "10.1088/0031-9155/56/13/018", "10.1088/1361-6560/ab4914"],
    ))
    result = dict(x_mm=grid[x_mesh], y_mm=grid[y_mesh], dx_mm=delta, dy_mm=delta, kernels=kernels,
                  provenance=provenance, doi_included=True, fit_scale=amplitude,
                  relative_error=error, efficiency=efficiency_map[np.ix_(y_mesh, x_mesh)],
                  fit_efficiency=fit_efficiency, fit_positions_mm=grid[fit_positions],
                  source_support=source_support, source_efficiency=efficiency_map,
                  reference_factors=reference_factors,
                  reference_angles_deg=np.asarray(angles_deg),
                  reference_grid=lut_reference_grid(recon),
                  elapsed_seconds=perf_counter() - start)
    if path is not None:
        save_analytical_lut(path, result)
    return result


def pack_analytical_lut(result):
    """Encode the same NPZ used by both the automatic and file-based models."""
    with BytesIO() as output:
        np.savez_compressed(output, **result)
        return output.getvalue()


def save_analytical_lut(path, result):
    """Save a fitted map atomically from arrays or previously packed bytes."""
    path = Path(path)
    output = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(output.name)
    try:
        with output:
            if isinstance(result, bytes):
                output.write(result)
            else:
                np.savez_compressed(output, **result)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def kernel_summary(lut, iy, ix):
    """Centroid, asymmetry and marginal widths for a sampled LUT kernel."""
    from Gimntools.GimnRec.quality_metrics import fwhm_by_bins

    kernel = lut["kernels"][iy, ix]
    x, y = lut["dx_mm"], lut["dy_mm"]
    px, py = kernel.sum(axis=0), kernel.sum(axis=1)

    def width(profile, axis):
        peak = int(np.argmax(profile))
        if not peak or peak == profile.size - 1:
            return float("nan")
        half = 0.5 * profile[peak]
        if not np.any(profile[:peak] < half) or not np.any(profile[peak+1:] < half):
            return float("nan")
        return fwhm_by_bins(profile, axis, baseline="zero")[0]

    return dict(centroid=(float(px @ x), float(py @ y)),
                asymmetry=float(np.abs(kernel - kernel[::-1, ::-1]).sum() / 2),
                fwhm=(width(px, x), width(py, y)), x_profile=px, y_profile=py)
