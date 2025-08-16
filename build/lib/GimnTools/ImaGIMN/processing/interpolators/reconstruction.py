from numba import njit
import numpy as np

@njit
def beta_spline_interpolation(pixel00, pixel01, pixel10, pixel11, dx, dy):
    """
    Implements beta-spline interpolation between 4 pixels.
    
    This function performs beta-spline interpolation using the 4 surrounding pixels and the fractional 
    coordinates (dx, dy) within the pixel. The interpolated value is calculated using the beta-spline 
    coefficients.

    @param pixel00: Value of the pixel at (0, 0) relative to the current pixel.
    @param pixel01: Value of the pixel at (0, 1) relative to the current pixel.
    @param pixel10: Value of the pixel at (1, 0) relative to the current pixel.
    @param pixel11: Value of the pixel at (1, 1) relative to the current pixel.
    @param dx: Fractional x-coordinate within the current pixel.
    @param dy: Fractional y-coordinate within the current pixel.
    @return: Interpolated value.
    """
    # Beta-spline coefficients
    c00 = 1/6 * (-dx**3 + 3*dx**2 - 3*dx + 1)
    c01 = 1/6 * (3*dx**3 - 6*dx**2 + 4)
    c10 = 1/6 * (-3*dx**3 + 3*dx**2 + 3*dx + 1)
    c11 = 1/6 * dx**3

    # Interpolation in x and y directions
    interp_x_top = pixel00 * c00 + pixel01 * c01
    interp_x_bottom = pixel10 * c10 + pixel11 * c11
    interpolated_value = interp_x_top * (1 - dy) + interp_x_bottom * dy

    return interpolated_value

@njit
def bilinear_interpolation(pixel00, pixel01, pixel10, pixel11, dx, dy):
    """
    Implements linear interpolation between 4 pixels.
    
    This function performs linear interpolation using the 4 surrounding pixels and the fractional 
    coordinates (dx, dy) within the pixel. The interpolated value is calculated using a bilinear 
    interpolation approach.

    @param pixel00: Value of the pixel at (0, 0) relative to the current pixel.
    @param pixel01: Value of the pixel at (0, 1) relative to the current pixel.
    @param pixel10: Value of the pixel at (1, 0) relative to the current pixel.
    @param pixel11: Value of the pixel at (1, 1) relative to the current pixel.
    @param dx: Fractional x-coordinate within the current pixel.
    @param dy: Fractional y-coordinate within the current pixel.
    @return: Interpolated value.
    """
    top = pixel00 * (1 - dx) + pixel01 * dx
    bottom = pixel10 * (1 - dx) + pixel11 * dx
    interpolated_value = top * (1 - dy) + bottom * dy
    return interpolated_value

@njit
def nearest_neighbor_interpolation(pixel00, pixel01, pixel10, pixel11, dx, dy):
    """
    Implements nearest neighbor interpolation between 4 pixels.
    
    This function performs nearest neighbor interpolation using the 4 surrounding pixels and the fractional 
    coordinates (dx, dy) within the pixel. The interpolated value is determined by rounding the fractional 
    coordinates to the nearest integer and selecting the corresponding pixel value.

    @param pixel00: Value of the pixel at (0, 0) relative to the current pixel.
    @param pixel01: Value of the pixel at (0, 1) relative to the current pixel.
    @param pixel10: Value of the pixel at (1, 0) relative to the current pixel.
    @param pixel11: Value of the pixel at (1, 1) relative to the current pixel.
    @param dx: Fractional x-coordinate within the current pixel.
    @param dy: Fractional y-coordinate within the current pixel.
    @return: Interpolated value.
    """
    # Round the fractional coordinates to the nearest integer
    x = round(dx)
    y = round(dy)
    
    # Select the corresponding pixel value based on the rounded coordinates
    if x == 0 and y == 0:
        return pixel00
    elif x == 0 and y == 1:
        return pixel01
    elif x == 1 and y == 0:
        return pixel10
    else:
        return pixel11