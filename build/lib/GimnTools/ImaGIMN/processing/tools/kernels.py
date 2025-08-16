import numpy as np
from numba import njit


def gaussian(x, y, sigma):
    """
    Calculates the value of the Gaussian function at a point (x, y) with standard deviation sigma.

    @param x: x-coordinate of the point.
    @param y: y-coordinate of the point.
    @param sigma: Standard deviation of the Gaussian function.
    @return: Value of the Gaussian function at the point (x, y).
    """
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def generate_gaussian_kernel_in_mm(size, sigma_mm, pixel_size_mm):
    """
    Generates a Gaussian kernel of specified size in millimeters.
    
    @param size: Size of the kernel (number of rows/columns).
    @param sigma_mm: Standard deviation of the Gaussian function in millimeters.
    @param pixel_size_mm: Size of the pixel in millimeters.
    @return: Generated Gaussian kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x_mm = (i - center) * pixel_size_mm
            y_mm = (j - center) * pixel_size_mm
            kernel[i, j] = gaussian(x_mm, y_mm, sigma_mm)
    return kernel

@njit
def butterworth_kernel(size, order, cutoff, pixel_size_mm):
    """
    Generates a Butterworth kernel of specified size.
    
    @param size: Size of the kernel (number of rows/columns).
    @param order: Order of the Butterworth filter.
    @param cutoff: Cutoff frequency of the Butterworth filter in millimeters.
    @param pixel_size_mm: Size of the pixel in millimeters.
    @return: Generated Butterworth kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x_mm = (i - center) * pixel_size_mm
            y_mm = (j - center) * pixel_size_mm
            r = np.sqrt(x_mm**2 + y_mm**2)
            kernel[i, j] = 1 / (1 + (r / cutoff)**(2 * order))   
    return kernel

@njit
def butterworth_kernel_high_pass(size, order, cutoff, pixel_size_mm):
    """
    Generates a Butterworth high-pass kernel of specified size.
    
    @param size: Size of the kernel (number of rows/columns).
    @param order: Order of the Butterworth filter.
    @param cutoff: Cutoff frequency of the Butterworth filter in millimeters.
    @param pixel_size_mm: Size of the pixel in millimeters.
    @return: Generated Butterworth high-pass kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x_mm = (i - center) * pixel_size_mm
            y_mm = (j - center) * pixel_size_mm
            r = np.sqrt(x_mm**2 + y_mm**2)
            if r > cutoff:
                kernel[i, j] = 1 / (1 + (r / cutoff)**(2 * order)) 
            else:
                kernel[i, j] = 0

    return kernel

@njit
def butterworth_kernel_low_pass(size, order, cutoff, pixel_size_mm):
    """
    Generates a Butterworth low-pass kernel of specified size.
    
    @param size: Size of the kernel (number of rows/columns).
    @param order: Order of the Butterworth filter.
    @param cutoff: Cutoff frequency of the Butterworth filter in millimeters.
    @param pixel_size_mm: Size of the pixel in millimeters.
    @return: Generated Butterworth low-pass kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x_mm = (i - center) * pixel_size_mm
            y_mm = (j - center) * pixel_size_mm
            r = np.sqrt(x_mm**2 + y_mm**2)
            if r < cutoff:
                kernel[i, j] = 1 / (1 + (r / cutoff)**(2 * order)) 
            else:
                kernel[i, j] = 0
    return kernel

def gaussian_kernel_norm(n, m, sigma):
    """
    Generates a normalized 2D Gaussian kernel centered at the pixel (n, m) with standard deviation sigma.
    
    This function creates a Gaussian kernel that is centered at the specified pixel coordinates (n, m).
    The kernel is generated based on the standard deviation (sigma) and is normalized to ensure that the 
    sum of all values in the kernel equals 1.

    @param n: Height of the kernel (number of rows).
    @param m: Width of the kernel (number of columns).
    @param sigma: Standard deviation of the Gaussian function.
    @return: Normalized 2D Gaussian kernel.
    """
    # Calculate the center of the kernel
    center_x = n // 2
    center_y = m // 2

    # Create the kernel coordinates
    x = np.arange(n) - center_x
    y = np.arange(m) - center_y

    # Create the meshgrid of coordinates
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Calculate the Gaussian kernel
    kernel = np.exp(-(x_mesh**2 + y_mesh**2) / (2 * sigma**2))
    kernel /= (2 * np.pi * sigma**2)
    kernel = kernel / kernel.max()  # Normalize the kernel
    return kernel