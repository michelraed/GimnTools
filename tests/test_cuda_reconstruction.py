"""Run the same comparisons on a CUDA device or with NUMBA_ENABLE_CUDASIM=1."""

import sys

import numpy as np
import pytest

from Gimntools.GimnRec.detector_response import DetectorResponse
from Gimntools.GimnRec.listmode import reconstruct_listmode
from Gimntools.GimnRec.listmode_response import ListmodeResponse
from Gimntools.GimnRec.reconstruction_backend import reconstruction_accelerator
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


@pytest.fixture
def accelerator():
    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        pytest.skip("CUDA device or CUDA simulator required")
    return reconstruction_accelerator("cuda")


def test_backend_names_and_unavailable_cuda():
    assert reconstruction_accelerator("CPU") is None
    with pytest.raises(ValueError, match="CPU or CUDA"):
        reconstruction_accelerator("invalid")
    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            reconstruction_accelerator("cuda")


def test_legacy_numba_uses_context_symbols_matching_its_ctypes_abi():
    from types import SimpleNamespace
    from Gimntools.GimnRec.cuda_reconstruction import _use_legacy_context_symbols

    library = SimpleNamespace(cuCtxGetDevice=object(), cuCtxGetDevice_v2=object(),
                              cuCtxSynchronize=object(), cuCtxSynchronize_v2=object(),
                              cuMemAlloc_v2=object())
    driver = SimpleNamespace(lib=library, _find_api=lambda name: getattr(library, name + "_v2"))
    _use_legacy_context_symbols(driver)
    assert driver._find_api("cuCtxGetDevice") is library.cuCtxGetDevice
    assert driver._find_api("cuCtxSynchronize") is library.cuCtxSynchronize
    assert driver._find_api("cuMemAlloc") is library.cuMemAlloc_v2


@pytest.mark.skipif(sys.platform == "win32", reason="This test delivers a POSIX signal")
def test_native_cuda_failure_preserves_cpu_reconstruction(monkeypatch):
    from Gimntools.GimnRec import cuda_worker

    crash = """
import os
import pickle
import resource
import signal
import sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
pickle.dump(("ok", None), sys.stdout.buffer)
sys.stdout.buffer.flush()
pickle.load(sys.stdin.buffer)
os.kill(os.getpid(), signal.SIGSEGV)
"""
    monkeypatch.setattr(cuda_worker, "_WORKER_COMMAND", (sys.executable, "-X", "faulthandler", "-c", crash))
    sino = np.ones((1, 5, 6))
    angles = np.arange(6) * 30.
    gpu = siddon_reconstructor(sino, fov_mm=5., backend="CUDA", verbose=False)
    with pytest.raises(RuntimeError, match=r"CUDA worker stopped during sinogram ray tracing \(process status -11\)") as error:
        gpu.mlem(angles, iterations=1, verbose=False)
    assert "Fatal Python error: Segmentation fault" in str(error.value)
    assert not gpu._accelerator.close.alive
    cpu = siddon_reconstructor(sino, fov_mm=5., backend="CPU", verbose=False)
    assert np.isfinite(cpu.mlem(angles, iterations=1, verbose=False)).all()


def test_cuda_siddon_fbp_mlem_and_osem_match_cpu(accelerator):
    angles = np.array([0., 25., 56., 90., 122., 158.])
    rng = np.random.default_rng(45)
    sino = rng.uniform(1., 12., (1, 7, len(angles)))
    geometry = dict(fov_mm=9., sino_range_mm=8., n_img=9, verbose=False,
                    rays_per_bin=2, cor_mm=.27, dx_cor_mm=.38, dy_cor_mm=-.13)
    cpu = siddon_reconstructor(sino, **geometry)
    gpu = siddon_reconstructor(sino, backend="CUDA", **geometry)
    assert gpu.geometry()["backend"] == "cuda"
    plan = np.deg2rad(angles)
    image = rng.random((9, 9))
    np.testing.assert_allclose(gpu._project(image, plan), cpu._project(image, plan), atol=1e-11)
    np.testing.assert_allclose(gpu._backproject(sino[0], plan), cpu._backproject(sino[0], plan), atol=1e-10)
    np.testing.assert_allclose(gpu.fbp(angles), cpu.fbp(angles), atol=1e-11)
    settings = dict(iterations=2, psf_kernel=np.array([.1, .6, .3]),
                    background_sino=sino*.1, norm_sino=rng.uniform(.7, 1.2, sino.shape[1:]),
                    regularization_every=1, regularization_sigma=.4, relaxation=.8, verbose=False)
    np.testing.assert_allclose(gpu.mlem(angles, **settings), cpu.mlem(angles, **settings), atol=1e-11)
    np.testing.assert_allclose(gpu.osem(angles, subsets_n=2, **settings),
                               cpu.osem(angles, subsets_n=2, **settings), atol=1e-11)
    np.testing.assert_allclose(gpu.mlem(angles, iterations=1, legacy=True, verbose=False),
                               cpu.mlem(angles, iterations=1, legacy=True, verbose=False), atol=1e-11)


def test_cuda_sparse_detector_response_and_lut_match_cpu(accelerator):
    from scipy.sparse import csr_matrix

    sino = np.arange(1., 25.).reshape(1, 4, 6)
    angles = np.arange(6)*30.
    rng = np.random.default_rng(4)
    doi = csr_matrix(rng.uniform(.1, 1., (24, 16)))
    sv = csr_matrix(rng.uniform(.1, 1., (16, 16)))
    sv = sv.multiply(1 / np.asarray(sv.sum(axis=0)).ravel()).tocsr()
    results = []
    for backend in ("cpu", "cuda"):
        rec = siddon_reconstructor(sino, n_img=4, fov_mm=4., verbose=False, backend=backend)
        response = DetectorResponse.__new__(DetectorResponse)
        response.recon, response.angles, response.doi = rec, angles, doi
        response.sv, response.reference_factors = sv, None
        results.append(rec.osem(angles, subsets_n=2, iterations=2, response=response, verbose=False))
    np.testing.assert_allclose(*results, rtol=1e-12, atol=1e-12)


def test_cuda_listmode_trace_and_em_match_cpu(accelerator):
    from Gimntools.GimnRec.listmode import _ray_matrix

    angles = np.linspace(0., 2*np.pi, 30, endpoint=False)
    d = np.column_stack((8*np.cos(angles), 8*np.sin(angles), 3*np.sin(2*angles)))
    rays = np.column_stack((-d, d)) + np.array([.12, -.15, .3]*2)
    shape = (5, 5, 7)
    spacing = np.array([.85, 1., 1.2])
    grid_xyz = np.array(shape[::-1])
    corner = -grid_xyz * spacing/2
    traced = _ray_matrix(rays, grid_xyz, spacing, corner)
    device = accelerator.trace_lors(rays, grid_xyz, spacing, corner)
    np.testing.assert_allclose(device.toarray(), traced.toarray(), atol=1e-12)
    options = dict(iterations=2, subsets=2, psf={"fwhm_mm": (1., 1.3), "growth": .3})
    cpu, _ = reconstruct_listmode(rays, shape, spacing, **options)
    gpu, info = reconstruct_listmode(rays, shape, spacing, backend="cuda", **options)
    assert info["backend"] == "cuda"
    np.testing.assert_allclose(gpu, cpu, atol=1e-10)

    # Exercise the grouped measured/simulated detector model with the same
    # weighted system matrix rather than replacing crystal pairs with lines.
    sensitivity = np.asarray(traced.sum(axis=0)).ravel()
    response = ListmodeResponse(sensitivity, (traced,), np.arange(len(rays)),
                                "test crystal-pair response", {}, None, {})
    cpu, _ = reconstruct_listmode(rays, shape, spacing, detector_model=response, **options)
    gpu, _ = reconstruct_listmode(rays, shape, spacing, detector_model=response, backend="cuda", **options)
    np.testing.assert_allclose(gpu, cpu, atol=1e-10)
