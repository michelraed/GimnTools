"""Generate a DOI-inclusive analytical LUT using a synthetic scanner geometry."""

import argparse
from pathlib import Path

import numpy as np

from Gimntools.GimnRec.analytical_response import generate_analytical_lut
from Gimntools.GimnRec.detector_response import ResponseSettings, acquisition_geometry
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mu-mm-inv", type=float, required=True,
                        help="511 keV linear attenuation coefficient in inverse mm")
    parser.add_argument("--order", type=int, default=3)
    args = parser.parse_args()
    config = dict(
        data_from="PETSYS", crystal_type="segmented", rotate_along="y",
        radius=35.0, pla_thickness=2.0, det_z_rotation=90.0, rotation_angles=12,
        crystal_size=[20, 26.88, 26.88], sipm_pitch=[3.36, 3.36], sipm_pixel_n=[8, 8],
    )
    acquisition = acquisition_geometry("synthetic", config,
                                       {"left": 0.2, "right": 180.2},
                                       {0: "left", 15: "right"}, "y",
                                       -np.arange(12) * 15.0)
    recon = siddon_reconstructor(np.zeros((1, 15, 180)), fov_mm=26.88,
                                  sino_range_mm=26.88, n_img=17, verbose=False)
    angles = (np.arange(recon.n_ang) + 0.5) * (180.0 / recon.n_ang)
    settings = ResponseSettings(args.mu_mm_inv, 3, args.order, args.order)
    result = generate_analytical_lut(recon, angles, settings, acquisition, args.output,
                                     sample_count=3, half_support=2)
    print(f"Saved {args.output} in {result['elapsed_seconds']:.2f} s")
    print("Relative projection fit error by sample")
    print(result["relative_error"])


if __name__ == "__main__":
    main()
