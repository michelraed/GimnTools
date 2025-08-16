import scipy.fft as fft
import numpy as np

def ramLak(size):
    """
    Implements the Ram-Lak filter.
    
    The Ram-Lak filter is a ramp filter commonly used in computed tomography (CT) reconstruction.
    This function generates the Ram-Lak filter kernel of the specified size.

    @param size: Size of the filter kernel.
    @return: Ram-Lak filter kernel.
    """
    ramlak = np.zeros(size)
    value = 0.0
    ang = 0.5 / (size / 2)
    for i in range(size):
        if i <= int(size / 2):
            value = value + ang
        if i > int(size / 2):
            value = value - ang
        ramlak[i] = value

    return ramlak

def cossineFilter(size):
    """
    Implements the Cosine filter.
    
    The Cosine filter is a modified version of the Ram-Lak filter, which applies a cosine window to the Ram-Lak filter.
    This function generates the Cosine filter kernel of the specified size.

    @param size: Size of the filter kernel.
    @return: Cosine filter kernel.
    """
    ram = ramLak(size)
    out = np.zeros(ram.shape[0])
    for i in range(ram.shape[0]):
        out[i] = ram[i] * np.cos(np.pi * ram[i])
    return out


def apply_filter_to_sinogram(filterType, sinogram):
    """
    Applies the specified filter to a sinogram.
    
    This function takes a sinogram and a filter type (either Ram-Lak or Cosine filter), and applies the filter to each column of the sinogram in the frequency domain.
    The filtered sinogram is then transformed back to the spatial domain and returned.

    @param filterType: Type of filter to apply (either ramLak or cossineFilter).
    @param sinogram: Input sinogram to be filtered.
    @return: Filtered sinogram.
    """
    # Initialize the output array
    out = np.zeros(sinogram.shape)
    
    # Initialize the filter
    filt = filterType(sinogram.shape[0])
    
    # Apply the filter to each column of the sinogram in the frequency domain
    for angle in range(sinogram.shape[1]):
        fftd = fft.fft(sinogram[:, angle])
        for i in range(fftd.shape[0]):
            fftd[i] = fftd[i] * filt[i]
        fftd = fft.ifft(fftd)
        out[:,angle] = fftd
    return np.real(out)