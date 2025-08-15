import numpy as np
from scipy.fft import rfft
from scipy.ndimage import gaussian_filter

from scipy.ndimage import rotate

def rot_center(theta):
    """
    @brief Calculates the center of rotation (COR) of a sinogram.
    This function computes the center of rotation from a 2D sinogram array by analyzing its Fourier transform.
    @param thetasum (array-like) 
        The 2D sinogram array with dimensions (z, theta).
    @return float
        The calculated center of rotation (COR).
    @details
        The function applies a real-valued Fast Fourier Transform (rFFT) to the flattened input array,
        extracts the real and imaginary parts at a specific frequency, computes the phase, and then
        determines the center of rotation based on the phase information.
    """
    T = rfft(theta.ravel())
    imag = T[theta.shape[0]].imag
    real = T[theta.shape[0]].real
    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    COR = theta.shape[-1]/2-phase*theta.shape[-1]/(2*np.pi)
    return COR
