"""Matched spatially variant PSF and sensitivity on the list-mode grid."""

import numpy as np
import pytest

from Gimntools.GimnRec.listmode import _ray_matrix, reconstruct_listmode
from Gimntools.GimnRec.listmode_psf import prepare_listmode_psf


def test_gaussian_psf_varies_with_position_preserves_mass_and_is_adjoint():
    shape = (13, 21, 21)
    psf = prepare_listmode_psf(shape, (1.,) * 3, (0.,) * 3, dict(fwhm_mm=(1.8, 1.5), growth=.85))

    def variance(iz, iy, ix):
        point = np.zeros(shape)
        point[iz, iy, ix] = 1.
        blurred = psf.forward(point.ravel()).reshape(shape)
        assert blurred.sum() == pytest.approx(1.)
        assert np.all(blurred >= 0)
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        return np.sum(blurred * ((z - iz)**2 + (y - iy)**2 + (x - ix)**2))

    assert variance(6, 10, 16) > 1.2 * variance(6, 10, 10)
    rng = np.random.default_rng(8)
    x, y = rng.random((2, np.prod(shape)))
    np.testing.assert_allclose(psf.forward(x) @ y, x @ psf.transpose(y), rtol=1e-13)
    np.testing.assert_allclose(psf.transpose(np.ones(x.size)), 1.)


@pytest.mark.parametrize("volumetric", [False, True])
def test_lut_interpolates_calibration_and_offsets_in_mm(tmp_path, volumetric):
    path = tmp_path / "residual.npz"
    shape = (7, 9, 11)
    # At x=-4 use identity. At x=4 put half of the mass one voxel away.
    if volumetric:
        fields = dict(z_mm=[0.], dz_mm=[0., 1.25], dy_mm=[0.], dx_mm=[0.])
        kernels = np.array([[1., 0.], [.5, .5]]).reshape(1, 1, 2, 2, 1, 1)
    else:
        fields = dict(dy_mm=[0.], dx_mm=[0., .8])
        kernels = np.array([[1., 0.], [.5, .5]]).reshape(1, 2, 1, 2)
    payload = dict(x_mm=[-3.2, 3.2], y_mm=[0.], kernels=kernels, provenance="point-source fit", **fields)
    np.savez(path, doi_included=False, **payload)
    psf = prepare_listmode_psf(shape, (.8, 1., 1.25), (0.,) * 3, dict(lut=path))
    point = np.zeros(shape)
    point[3, 4, 5] = 1.
    output = psf.forward(point.ravel()).reshape(shape)
    assert output[3, 4, 5] == pytest.approx(.75)
    expected = (4, 4, 5) if volumetric else (3, 4, 6)
    assert output[expected] == pytest.approx(.25)
    np.testing.assert_allclose(psf.transpose(np.ones(np.prod(shape))), 1.)
    np.savez(path, doi_included=True, **payload)
    with pytest.raises(ValueError, match="without DOI"):
        prepare_listmode_psf(shape, (.8, 1., 1.25), (0.,) * 3, dict(lut=path))


def test_listmode_em_uses_psf_in_forward_adjoint_and_sensitivity():
    angles = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    head = np.column_stack([12 * np.cos(angles), 12 * np.sin(angles), 8 * np.sin(3 * angles)])
    rays = np.hstack((-head, head))
    shape, spacing = (9, 11, 13), (.9, 1., 1.1)
    grid = np.array(shape[::-1])
    matrix = _ray_matrix(rays, grid, np.array(spacing), -grid * spacing / 2)
    psf = prepare_listmode_psf(shape, spacing, (0.,) * 3, dict(fwhm_mm=(1.6, 2.1), growth=.6))
    ones = np.ones(len(rays))
    sensitivity = psf.transpose(matrix.T @ ones)
    vol, metadata = reconstruct_listmode(
        rays, shape, spacing, psf=psf, normalization_lors=rays, iterations=2, relaxation=1.,
    )
    assert metadata["sv_psf"] == "3D Gaussian SV-PSF"
    assert vol.ravel() @ sensitivity == pytest.approx(len(rays), rel=1e-10)
    assert np.isfinite(vol).all() and np.all(vol >= 0)
