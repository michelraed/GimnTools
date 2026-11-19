"""Numerical checks of the standalone first-interaction model."""

import numpy as np
import pytest
from scipy.integrate import quad

from Gimntools.GimnRec.doi import first_interaction_quadrature


@pytest.mark.parametrize("cosine", [1.0, 0.8, 0.25])
def test_truncated_exponential_moments(cosine):
    length, mu = 20.0, 0.08
    depths, weights, acceptance = first_interaction_quadrature(length, mu, cosine, 64)
    rate = mu / cosine
    assert acceptance == pytest.approx(1 - np.exp(-rate * length))
    assert np.all(np.isfinite(depths))
    assert np.all((0 < depths) & (depths < length))
    assert np.all(weights >= 0)
    for power in range(3):
        expected = quad(
            lambda z: z**power * rate * np.exp(-rate * z) / acceptance,
            0, length,
        )[0]
        assert weights @ depths**power == pytest.approx(expected, rel=1e-5)


def test_obliquity_and_crystal_depth_change_response():
    def evaluate(length, cosine):
        depths, weights, probability = first_interaction_quadrature(length, 0.08, cosine, 32)
        return weights @ depths, probability

    normal_mean, normal_probability = evaluate(20, 1.0)
    oblique_mean, oblique_probability = evaluate(20, 0.5)
    short_mean, short_probability = evaluate(10, 1.0)
    assert oblique_mean < normal_mean and oblique_probability > normal_probability
    assert short_mean < normal_mean and short_probability < normal_probability


@pytest.mark.parametrize("sign", [-1, 1])
def test_side_wall_limits_acceptance_and_is_symmetric(sign):
    depths, weights, acceptance = first_interaction_quadrature(
        20, 0.08, 0.8, 32,
        face_width_mm=3, entrance_offset_mm=(sign * 0.75, 0),
        transverse_slope=(sign * 0.75, 0),
    )
    assert depths.max() < 1.0
    assert weights.sum() == pytest.approx(1.0)
    assert acceptance == pytest.approx(-np.expm1(-0.1))


def test_outward_ray_at_edge_has_zero_acceptance():
    depths, weights, acceptance = first_interaction_quadrature(
        20, 0.08, 0.8, 8,
        face_width_mm=3, entrance_offset_mm=(1.5, 0), transverse_slope=(0.75, 0),
    )
    assert depths.size == weights.size == 0
    assert acceptance == 0


@pytest.mark.parametrize("mu", [1e-20, 100.0])
def test_extreme_optical_thickness(mu):
    depths, weights, acceptance = first_interaction_quadrature(20, mu, 1.0, 64)
    assert np.all(np.isfinite(depths))
    assert np.all((depths > 0) & (depths < 20))
    assert weights.sum() == pytest.approx(1.0)
    assert acceptance == pytest.approx(-np.expm1(-20 * mu), abs=0)


def test_inconsistent_geometry_rejected():
    with pytest.raises(ValueError, match="slope must agree"):
        first_interaction_quadrature(
            20, 0.08, 0.8, 8,
            face_width_mm=3, entrance_offset_mm=(0, 0), transverse_slope=(0, 0),
        )
