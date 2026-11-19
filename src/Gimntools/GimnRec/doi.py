"""First-interaction depth quadrature for a finite crystal segment."""

from operator import index

import numpy as np
from scipy.special import roots_legendre


def first_interaction_quadrature(
    depth_mm,
    mu_per_mm,
    incidence_cosine,
    order,
    *,
    face_width_mm=None,
    entrance_offset_mm=None,
    transverse_slope=None,
):
    """Return normal depths, conditional weights and interaction probability.

    The incidence cosine is relative to the inward face normal, which points
    into the scintillator. Supply all three face arguments to model a square
    face with an oblique ray. The offset gives its two entrance coordinates
    relative to the face center, in mm. The slope gives the change of those
    coordinates per mm of normal depth, so its magnitude must agree with
    the incidence cosine. An immediate side exit gives empty quadrature
    arrays and zero probability.
    """
    order = index(order)
    if order < 1:
        raise ValueError("quadrature order must be positive")
    if not np.all(np.isfinite([depth_mm, mu_per_mm, incidence_cosine])):
        raise ValueError("depth, attenuation and incidence cosine must be finite")
    if depth_mm <= 0 or mu_per_mm <= 0 or not 0 < incidence_cosine <= 1:
        raise ValueError("depth and attenuation must be positive and cosine in (0, 1]")

    face_arguments = (face_width_mm, entrance_offset_mm, transverse_slope)
    if any(value is not None for value in face_arguments):
        if any(value is None for value in face_arguments):
            raise ValueError("face width, entrance offset and slope are required together")
        depth_mm = _exit_depth(
            depth_mm, incidence_cosine, face_width_mm,
            entrance_offset_mm, transverse_slope,
        )
        if depth_mm == 0:
            return np.empty(0), np.empty(0), 0.0

    nodes, weights = roots_legendre(order)
    probabilities = (nodes + 1.0) / 2.0
    # Integrate in conditional CDF coordinates. Using the optical thickness
    # keeps the inverse stable when interaction is very unlikely.
    with np.errstate(over="ignore", under="ignore"):
        thickness = np.float64(mu_per_mm) * depth_mm / incidence_cosine
    acceptance = -np.expm1(-thickness)
    if acceptance == 0:
        depths = depth_mm * probabilities
    else:
        optical_depths = -np.log1p(-probabilities * acceptance)
        if thickness <= 1:
            depths = depth_mm * (optical_depths / thickness)
        else:
            depths = (incidence_cosine * optical_depths) / mu_per_mm
    return depths, weights / 2.0, float(acceptance)


def _exit_depth(depth_mm, incidence_cosine, face_width_mm, offset, slope):
    offset = np.asarray(offset, dtype=float)
    slope = np.asarray(slope, dtype=float)
    if offset.shape != (2,) or slope.shape != (2,):
        raise ValueError("entrance offset and slope must have two coordinates")
    if not np.isfinite(face_width_mm) or face_width_mm <= 0:
        raise ValueError("face width must be finite and positive")
    half_width = face_width_mm / 2.0
    if not np.all(np.isfinite(offset)) or np.any(np.abs(offset) > half_width):
        raise ValueError("entrance offset must lie on the crystal face")
    if not np.all(np.isfinite(slope)) or not np.isclose(
        1.0 / np.hypot(1.0, np.hypot(*slope)), incidence_cosine, atol=0.0
    ):
        raise ValueError("transverse slope must agree with the incidence cosine")
    moving = slope != 0
    remaining = half_width - np.sign(slope[moving]) * offset[moving]
    with np.errstate(over="ignore"):
        side_exit = remaining / np.abs(slope[moving])
    return min(depth_mm, side_exit.min(initial=np.inf))
