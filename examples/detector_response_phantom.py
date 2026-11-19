"""Compare detector response modes on one synthetic off-axis point source.

The phantom uses a more finely sampled version of the same approximate
model. This exercises the 2D implementation. It is not scanner validation.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from Gimntools.GimnRec.detector_response import (
    DetectorResponse, ResponseSettings, acquisition_geometry,
)
from Gimntools.GimnRec.quality_metrics import fwhm_by_bins
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


def main():
    rng = np.random.default_rng(42)
    config = dict(
        data_from="PETSYS", crystal_type="segmented", rotate_along="y",
        radius=35.0, pla_thickness=2.0, det_z_rotation=90.0, rotation_angles=12,
        crystal_size=[20, 26.88, 26.88], sipm_pitch=[3.36, 3.36], sipm_pixel_n=[8, 8],
    )
    poses = -np.arange(12) * 15.0
    acquisition = acquisition_geometry("synthetic", config, {"left": 0.2, "right": 180.2},
                                      {0: "left", 15: "right"}, "y", poses)
    n = 33
    recon = siddon_reconstructor(np.zeros((1, 15, 180)), fov_mm=26.88,
                                  sino_range_mm=26.88, n_img=n, verbose=False)
    angles = (np.arange(recon.n_ang) + 0.5) * (180.0 / recon.n_ang)
    grid = recon.img_min + (np.arange(n) + 0.5) * recon.img_pixel
    iy0, ix0 = n // 2, n // 2 + 5
    source = np.zeros((n, n))
    source[iy0, ix0] = 1

    with TemporaryDirectory() as tmp:
        lut = str(Path(tmp) / "synthetic.npz")
        np.savez(lut, x_mm=[-8.0, 0.0, 8.0], y_mm=[0.0],
                 dx_mm=[-0.8, 0.0, 0.8], dy_mm=[0.0],
                 kernels=np.array([[[[0.35, 0.6, 0.05]],
                                    [[0.1, 0.8, 0.1]],
                                    [[0.05, 0.6, 0.35]]]]),
                 provenance="Synthetic PSF with off-axis displacement", doi_included=False)
        truth = DetectorResponse(recon, angles, ResponseSettings(0.08, 3, 5, 5, lut),
                                 acquisition)
        expected = truth.operators()[0](source)
        expected *= 15_000 / expected.sum()
        # Compile legacy Numba operators before timing steady-state EM.
        recon._project(source, np.deg2rad(angles))
        recon._backproject(expected, np.deg2rad(angles))
        background = np.full_like(recon.sinogram, 0.05)
        draws = rng.poisson(expected + background, size=(4, *background.shape))
        modes = {
            "Conventional": None,
            "DOI": ResponseSettings(0.08, 3),
            "SV-PSF": ResponseSettings(lut_path=lut),
            "DOI + SV-PSF": ResponseSettings(0.08, 3, lut_path=lut),
        }
        print("Synthetic off-axis point, 15,000 expected signal counts, 20 MLEM iterations")
        print("Mode               setup(s)  EM(s)/run  radial/tangential FWHM(mm)  peak error(mm)  noise")
        for name, settings in modes.items():
            start = perf_counter()
            response = None if settings is None else DetectorResponse(
                recon, angles, settings, acquisition)
            build_seconds = perf_counter() - start
            images = []
            start = perf_counter()
            for counts in draws:
                recon.sinogram[...] = counts
                volume = recon.mlem(angles, iterations=20, regularization_every=0,
                                    background_sino=background, response=response, verbose=False)
                image = volume[0]
                images.append(image / image.sum())
            seconds = (perf_counter() - start) / len(images)
            average = np.mean(images, axis=0)
            iy, ix = np.unravel_index(average.argmax(), average.shape)
            radial, _ = fwhm_by_bins(average[iy], grid)
            tangential, _ = fwhm_by_bins(average[:, ix], grid)
            error = np.hypot(grid[ix] - grid[ix0], grid[iy] - grid[iy0])
            noise = np.sqrt(np.mean(np.var(images, axis=0, ddof=1)))
            print(f"{name:18} {build_seconds:8.3f} {seconds:10.3f}"
                  f" {radial:18.2f} / {tangential:<8.2f} {error:15.2f} {noise:10.3g}")


if __name__ == "__main__":
    main()
