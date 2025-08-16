import numpy as np
from numba import njit
from copy import deepcopy

def resize(input_img, mx, my, interpolation):
    """
    Resizes an image using different interpolation methods.

    @param input_img (numpy.ndarray): Input image (2D array of intensity values).
    @param mx (int): Number of columns in the output image.
    @param my (int): Number of rows in the output image.
    @param interpolation (str): Interpolation method to be used ('Nearest-Neighbor', 'Bilinear', 'Cubic Spline').

    @return output_img (numpy.ndarray): Resized image.
    """
    nx, ny = input_img.shape  # Dimensions of the original image
    coef = None

    if interpolation == "Cubic Spline":
        # Calculate cubic coefficients for interpolation
        coef = compute_cubic_spline_coefficients(input_img, 0.9)

    cx, cy = nx / 2, ny / 2  # Coordinates of the center of the original image
    scalex = nx / mx  # Scale factor for the x-axis
    scaley = ny / my  # Scale factor for the y-axis
    output_img = np.zeros((mx, my))  # Initialize the output image

    # Iterate over each pixel of the output image
    for xo in range(mx):
        for yo in range(my):
            dx = xo - mx / 2  # Relative offset to the center of the output image
            dy = yo - my / 2
            xa = cx + dx * scalex  # Calculate the x coordinate in the original image
            ya = cy + dy * scaley  # Calculate the y coordinate in the original image
            v = 0  # Initial value of the interpolated pixel

            # Choose the interpolation method based on the provided string
            if interpolation == "Nearest-Neighbor":
                v = get_interpolated_pixel_nearest_neighbor(input_img, xa, ya)
            elif interpolation == "Bilinear":
                v = get_interpolated_pixel_linear(input_img, xa, ya)
            elif interpolation == "Cubic Spline":
                v = get_interpolated_pixel_cubic_spline(coef, xa, ya)

            output_img[xo, yo] = v  # Assign the interpolated value to the corresponding pixel

    if interpolation == "Cubic Spline":
        # Normalize the image if cubic interpolation is used
        output_img = renormalize(output_img, input_img.dtype)

    return output_img

def get_interpolated_pixel_cubic_spline(coef, x, y):
    """
    Interpolates the value of a pixel using cubic interpolation.

    @param coef (numpy.ndarray): Calculated cubic coefficients.
    @param x (float): x coordinate in the original image's coordinate system.
    @param y (float): y coordinate in the original image's coordinate system.

    @return interpolated_value (float): Value of the interpolated pixel.
    """
    i = int(np.floor(x))  # x coordinate rounded down
    j = int(np.floor(y))  # y coordinate rounded down

    # Ensure the neighborhood is within the image bounds
    if i - 1 < 0 or i + 2 >= coef.shape[0] or j - 1 < 0 or j + 2 >= coef.shape[1]:
        neighbor = np.zeros((4, 4))  # Create a default neighborhood if out of bounds
        for ni in range(4):
            for nj in range(4):
                ii = i - 1 + ni
                jj = j - 1 + nj
                if 0 <= ii < coef.shape[0] and 0 <= jj < coef.shape[1]:
                    neighbor[ni, nj] = coef[ii, jj]
    else:
        neighbor = coef[i - 1:i + 3, j - 1:j + 3]  # Extract the 4x4 neighborhood around the pixel

    return get_sample_cubic_spline(x - i, y - j, neighbor)

def get_sample_cubic_spline(x, y, neighbor):
    """
    Calculates the pixel value using cubic interpolation.

    @param x (float): Fraction of the x coordinate within the pixel.
    @param y (float): Fraction of the y coordinate within the pixel.
    @param neighbor (numpy.ndarray): 4x4 matrix of cubic coefficients.

    @return interpolated_value (float): Value of the interpolated pixel.
    """
    arr1 = get_cubic_spline(x)  # Cubic coefficients for x
    arr2 = get_cubic_spline(y)  # Cubic coefficients for y
    return np.sum(neighbor * np.outer(arr1, arr2))  # Calculate the weighted sum using the coefficient matrix

@njit
def get_cubic_spline(t):
    """
    Calculates the cubic spline coefficients for a given normalized value t.

    @param t (float): Normalized value for coefficient calculation.

    @return coefficients (numpy.ndarray): Cubic spline coefficients.
    """
    v = np.zeros(4)
    t1 = 1.0 - t
    t2 = t * t
    v[0] = (t1 * t1 * t1) / 6.0
    v[1] = (2.0 / 3.0) + 0.5 * t2 * (t - 2)
    v[3] = (t2 * t) / 6.0
    v[2] = 1.0 - v[3] - v[1] - v[0]
    return v

@njit
def compute_cubic_spline_coefficients(input_img, a):
    """
    Calculates the cubic coefficients for the input image using a cubic filter.

    @param input_img (numpy.ndarray): Input image.
    @param a (float): Smoothing coefficient for the cubic filter.

    @return output (numpy.ndarray): Image with calculated cubic coefficients.
    """
    nx, ny = input_img.shape
    output = np.zeros((nx, ny))

    # Apply the cubic filter row by row
    for y in range(ny):
        row = input_img[:, y]
        row_out = do_cubic_filter(row, a)
        output[:, y] = row_out

    # Apply the cubic filter column by column
    for x in range(nx):
        col = output[x, :]
        col_out = do_cubic_filter(col, a)
        output[x, :] = col_out

    return output

@njit
def do_cubic_filter(signal, a):
    """
    Applies a cubic filter to a one-dimensional signal.

    @param signal (numpy.ndarray): One-dimensional signal to be filtered.
    @param a (float): Smoothing coefficient.

    @return temp (numpy.ndarray): Filtered signal.
    """
    n = len(signal)
    temp = np.zeros(n)

    # Causal filter
    temp[0] = signal[0]
    for i in range(1, n):
        temp[i] = signal[i] + a * temp[i-1]

    # Anti-causal filter
    temp[-1] = temp[-1]
    for i in range(n-2, -1, -1):
        temp[i] = a * (temp[i+1] - temp[i])

    return temp

def renormalize(image, dtype):
    """
    Normalizes the image to the value range of the specified data type.

    @param image (numpy.ndarray): Image to be normalized.
    @param dtype (type): Data type for normalization.

    @return normalized (numpy.ndarray): Normalized image.
    """
    image = deepcopy(image)
    normalized = np.zeros_like(image).astype(dtype)
    min_value = image.min()
    max_value = image.max()
    rescale_slope = (max_value - min_value) / np.iinfo(dtype).max
    rescale_intercept = min_value
    normalized = ((image - min_value) / rescale_slope).astype(dtype)
    return normalized

@njit
def get_interpolated_pixel_nearest_neighbor(image, x, y):
    """
    Interpolates the value of a pixel using the nearest neighbor method.

    @param image (numpy.ndarray): Input image.
    @param x (float): x coordinate of the pixel in the original image.
    @param y (float): y coordinate of the pixel in the original image.

    @return interpolated_value (float): Value of the interpolated pixel.
    """
    # Round coordinates to the nearest pixel and ensure they are within image bounds
    i = int(np.clip(np.round(x), 0, image.shape[0] - 1))
    j = int(np.clip(np.round(y), 0, image.shape[1] - 1))
    return image[i, j]

@njit
def get_interpolated_pixel_linear(image, x, y):
    """
    Interpolates the value of a pixel using bilinear interpolation.

    @param image (numpy.ndarray): Input image.
    @param x (float): x coordinate of the pixel in the original image.
    @param y (float): y coordinate of the pixel in the original image.

    @return interpolated_value (float): Value of the interpolated pixel.
    """
    # Step 1: Locate the four neighboring pixels
    i, j = int(np.floor(x)), int(np.floor(y))

    # Calculate how much the point is offset within the interpolation square
    dx = x - i  # Fraction in x
    dy = y - j  # Fraction in y

    # Step 2: Ensure indices do not exceed image bounds
    i1 = min(i, image.shape[0] - 2)  # Limits to ensure interpolation square exists
    j1 = min(j, image.shape[1] - 2)

    # Step 3: Get the values of the four surrounding pixels
    v00 = image[i1, j1]  # Top-left
    v01 = image[i1, j1 + 1]  # Top-right
    v10 = image[i1 + 1, j1]  # Bottom-left
    v11 = image[i1 + 1, j1 + 1]  # Bottom-right

    # Step 4: Linear interpolation in both directions (x and y)
    # First interpolate in the x direction
    v0 = v00 * (1 - dx) + v01 * dx  # Interpolate between v00 and v01
    v1 = v10 * (1 - dx) + v11 * dx  # Interpolate between v10 and v11

    # Then interpolate in the y direction between the values interpolated in the x direction
    v = v0 * (1 - dy) + v1 * dy

    return v