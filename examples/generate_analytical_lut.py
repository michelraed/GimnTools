"""Generate a DOI-inclusive analytical LUT using a synthetic scanner geometry."""

import argparse
from pathlib import Path

import numpy as np

from Gimntools.GimnRec.analytical_response import generate_analytical_lut
from Gimntools.GimnRec.detector_response import (
    CRYSTAL_MATERIALS, acquisition_geometry, crystal_response_settings,
)
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crystal-material", choices=CRYSTAL_MATERIALS, default="LYSO")
    parser.add_argument("--mu-mm-inv", type=float,
                        help="Override the 511 keV linear attenuation coefficient in inverse mm")
    parser.add_argument("--image-fov-mm", type=float, default=26.88)
    parser.add_argument("--image-size", type=int, default=17)
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
    recon = siddon_reconstructor(np.zeros((1, 15, 180)), fov_mm=args.image_fov_mm,
                                  sino_range_mm=26.88, n_img=args.image_size, verbose=False)
    angles = (np.arange(recon.n_ang) + 0.5) * (180.0 / recon.n_ang)
    settings = crystal_response_settings(
        config, material=args.crystal_material, mu=args.mu_mm_inv, face_order=args.order,
        depth_order=args.order, analytical=True,
    )
    result = generate_analytical_lut(recon, angles, settings, acquisition, args.output,
                                     sample_count=3, half_support=2)
    print(f"Saved {args.output} in {result['elapsed_seconds']:.2f} s")
    print("Relative projection fit error by sample")
    print(result["relative_error"])


if __name__ == "__main__":
    main()
