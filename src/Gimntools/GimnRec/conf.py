"""Default configuration example (GATE, 2 opposed detectors).

These dictionaries mirror the JSON config files in ``examples/CONFIG_GATE``.
They are kept as a Python-importable example; production code should load
its configuration from JSON via :mod:`Gimntools.GimnRec.data_loaders`.
"""

import numpy as np

# Angles used when generating the sinogram, per detector name.
detectors_angles = {"detector-2": 0,
                    "detector-1": -180}

# rsector-ID -> detector-name mapping for GATE data.
rsectors = {0: "detector-1",
            1: "detector-2"}

# Detector characteristics.
detector_config = {
    "pla_thickness": 2.0,
    "radius": 25,
    "crystal_size": np.asarray([20, 26.88, 26.88]),
    "sipm_pitch": np.asarray([3.36, 3.36]),
    "sipm_pixel_n": np.asarray([8, 8]),
    "data_from": "GATE",
    "crystal_type": "segmented",
    "rotation_angles": 12,
    "arround": 360,
    # Must be one of the branches implemented in utils.coincidence_to_lor:
    # "CrystalID" | "GateGlobalPositions" | "GeneralGlobalPosition" |
    # "PaulGlobalPosition". "monolithic" was NOT one of them and made
    # coincidence_to_lor return an all-zero sinogram (fixed in 0.4.0).
    "process_as": "CrystalID",
    "rotate_along": "z",
}
