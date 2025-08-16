from GimnTools.ImaGIMN.processing.tools.math import convolution,deconvolution
from GimnTools.ImaGIMN.processing.tools.kernels import butterworth_kernel, generate_gaussian_kernel_in_mm
import numpy as np



def decovolve_psf(image, FWHM, pixel_size_mm=1):
    """
    Performs deconvolution of an image using a Gaussian kernel based on the Full Width at Half Maximum (FWHM).
    
    The function calculates the standard deviation (sigma) from the FWHM, generates a Gaussian kernel, 
    applies a Fourier shift to the kernel, and then performs deconvolution on the image.

    @param image: Input image to be deconvolved.
    @param FWHM: Full Width at Half Maximum used to calculate the Gaussian kernel.
    @param pixel_size_mm: Size of the pixel in millimeters (default is 1).
    @return: Deconvolved image.
    """
    # Calculate sigma based on FWHM
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    # Generate the Gaussian kernel
    kernel = generate_gaussian_kernel_in_mm(image.shape[0], sigma, pixel_size_mm)
    # Shift the Gaussian kernel
    kernel = np.fft.fftshift(kernel)
    return np.abs(deconvolution(image, kernel))


def gaussian_filter(image, FWHM, pixel_size_mm=1):
    """
    Applies a Gaussian filter to an image using a kernel based on the Full Width at Half Maximum (FWHM).
    
    The function calculates the standard deviation (sigma) from the FWHM, generates a Gaussian kernel, 
    applies a Fourier shift to the kernel, and then performs convolution on the image.

    @param image: Input image to be filtered.
    @param FWHM: Full Width at Half Maximum used to calculate the Gaussian kernel.
    @param pixel_size_mm: Size of the pixel in millimeters (default is 1).
    @return: Filtered image.
    """
    # Calculate sigma based on FWHM
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    # Generate the Gaussian kernel
    kernel = generate_gaussian_kernel_in_mm(image.shape[0], sigma, pixel_size_mm)
    # Shift the Gaussian kernel
    kernel = np.fft.fftshift(kernel)
    
    return np.abs(convolution(image, kernel))


def butterworth_filter(image, order, cutoff, pixel_size_mm=1):
    """
    Applies a Butterworth filter to an image.
    
    The function generates a Butterworth kernel based on the specified order and cutoff frequency, 
    applies a Fourier shift to the kernel, and then performs convolution on the image.

    @param image: Input image to be filtered.
    @param order: Order of the Butterworth filter.
    @param cutoff: Cutoff frequency of the Butterworth filter in millimeters.
    @param pixel_size_mm: Size of the pixel in millimeters (default is 1).
    @return: Filtered image.
    """
    kernel = butterworth_kernel(image.shape[0], order, cutoff, pixel_size_mm)
    kernel = np.fft.fftshift(kernel)
    return np.abs(convolution(image, kernel))