from numpy import dtype
from copy import deepcopy
from numba import njit
import numpy as np


@njit  # <--- Numba decorator. Tells Numba to compile this function. Note: Use NumPy libraries internally; other libraries are not accepted by Numba.
def get_neighbors(imagem, u, v, window):
    """
    Extracts the neighboring pixels from an image centered at pixel (u, v) 
    within a specified window size.

    This function creates a matrix of neighbors from the image, centered at 
    the pixel (u, v), with a size defined by the window parameter.

    @param imagem: Input image from which to extract neighbors.
    @param u: Row index of the center pixel.
    @param v: Column index of the center pixel.
    @param window: Size of the window (should be odd).
    @return: A matrix containing the neighboring pixels.
    """
    # Define the matrix where neighbors will be stored
    tipo = imagem.dtype
    neighbors = np.zeros((window, window), dtype=tipo)
    x_pos = 0
    y_pos = 0 

    # Iterate over the generated matrix
    for i in range(u - window // 2, u + window // 2 + 1):
        x_pos = 0
        for j in range(v - window // 2, v + window // 2 + 1):
            neighbors[y_pos, x_pos] = imagem[i, j]  # Get values from the original matrix
            x_pos += 1
        y_pos += 1
    return neighbors

@njit  # <--- Numba decorator. Tells Numba to compile this function.
def apply_filter(kernel, image):
    """
    Implements a convolutional filter that applies a kernel to an image.

    This function applies the specified kernel to the input image using 
    convolution. The kernel must have the same dimensions in both x and y.

    @param kernel: The convolution kernel to apply.
    @param image: The input image to which the kernel will be applied.
    @return: The filtered image.
    """
    tipo = image.dtype
    x_len = image.shape[1]
    y_len = image.shape[0]
    out = np.zeros((y_len, x_len), dtype=tipo)
    window_y, window_x = kernel.shape
    if window_y == window_x:  # Check if the kernel has the same dimensions in x and y
        window = window_x
        for i in range(window // 2, (y_len - (window // 2))):
            for j in range(window // 2, (x_len - (window // 2))):
                vizinhos = get_neighbors(image, i, j, window)
                out[i, j] = ((vizinhos * kernel) / (np.abs(kernel).sum())).sum()  # Normalize the kernel
    else:
        print("Kernel dimensions in x and y are not equal")
    return out

@njit
def covolve_1d(kernel, vector):
    """
    Applies 1D convolution to a vector using the specified kernel.

    This function performs convolution on a 1D vector with the given kernel.

    @param kernel: The convolution kernel to apply.
    @param vector: The input vector to which the kernel will be applied.
    @return: The convolved vector.
    """
    out = np.zeros(vector.shape[0])
    tam_kernel = kernel.shape[0] // 2
    for i in range(tam_kernel, vector.shape[0] - tam_kernel):
        neighbors = vector[i - tam_kernel:i + tam_kernel + 1]
        out[i] = ((neighbors * kernel).sum()) / (np.abs(kernel).sum())
    return out

@njit
def applyseparable_filter(kernel_a, kernel_b, imagem):
    """
    Implements a separable filter using two 1D kernels.

    This function applies two 1D kernels to the input image in a separable manner,
    first along the rows and then along the columns.

    @param kernel_a: applies a 1D convolution kernel to a row.
    @param kernel_b: applies 1D convolution kernel to a column.
    @param imagem: The input image to which the kernels will be applied.
    @return: The filtered image.
    """
    tipo = imagem.dtype
    x_len = imagem.shape[1]
    y_len = imagem.shape[0]
    out = np.ones((y_len, x_len), dtype=tipo)

    for j in range(y_len):
        linha = imagem[j, :]
        out[j, :] = covolve_1d(kernel_b, linha)

    for i in range(x_len):
        coluna = out[:, i]
        out[:, i] = covolve_1d(kernel_a, coluna)

    return out

@njit
def median_filter(image, window):
    """
    Applies a median filter to the input image using a specified window size.

    @param image: Input image to which the median filter will be applied.
    @param window: Size of the window (should be odd).
    @return: A new image with the median filter applied.
    """
    # Get the dimensions of the input image
    height, width = image.shape
    
    # Create an output image initialized with zeros
    filtered_image = np.zeros_like(image)

    # Calculate the half window size
    half_window = window // 2

    # Iterate over each pixel in the image
    for u in range(half_window, height - half_window):
        for v in range(half_window, width - half_window):
            # Get the neighboring pixels using the get_neighbors function
            neighbors = get_neighbors(image, u, v, window)
            
            # Flatten the neighbors array and compute the median
            median_value = np.median(neighbors)
            
            # Assign the median value to the filtered image
            filtered_image[u, v] = median_value

    return filtered_image