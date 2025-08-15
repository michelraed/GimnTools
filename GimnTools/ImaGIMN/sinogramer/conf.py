import numpy as np

"""
Configuuration angles for generating sinogram
"""
detectors_angles = {"detector-2":0,
                    "detector-1":-180}

"""
rsectors list for Gate
"""
rsectors = { 0 : "detector-1",
             1 : "detector-2"   
}


"""
Detector characteristics
"""
detector_config = { 
                    "pla_thickness": 2.0,
                    "radius" : 25,
                    "crystal_size" : np.asarray([20,26.88,26.88]),
                    "sipm_pitch": np.asarray([3.36,3.36]),
                    "sipm_pixel_n":np.asarray([8,8]),
                    "data_from":"gate",
                    "crystal_type":"segmented",
                    "rotation_angles":12,
                    "arround":360,
                    "process_as":"monolithic",
                    "rotate_along":"z"}

