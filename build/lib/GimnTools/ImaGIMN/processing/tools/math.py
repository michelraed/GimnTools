import numpy as np

from scipy.ndimage import rotate as rt_scipy
from numba import njit

import itk
from itk import BSplineInterpolateImageFunction

""""
Here are implemented the mathematical entities used on the library

"""


def deconvolution(image, function):
    """
    Performs deconvolution of an image using a Point Spread Function (PSF).
    
    Deconvolution is performed in the frequency domain, where the image and the PSF are transformed using 
    the Fast Fourier Transform (FFT). The convolution is performed in the frequency domain and the resulting 
    image is transformed back to the spatial domain.

    @param image: Input image to be deconvolved.
    @param function: Point Spread Function (PSF) used for deconvolution.
    @return: Deconvolved image.
    """
    # Transform everything to the frequency domain
    function_fft = np.fft.fft2(function)
    image_fft = np.fft.fft2(image)
    # Perform convolution in the frequency domain:
    convolved = image_fft / (function_fft + 1e-10)
    # Transform back to the spatial domain:
    img = np.fft.ifft2(convolved)
    return np.abs(img)

def convolution(image, kernel):
    """
    Performs convolution of an image with a kernel using the Fast Fourier Transform (FFT).
    
    Convolution is performed in the frequency domain, where the image and the kernel are transformed using 
    the FFT. The convolution is then performed in the frequency domain and the resulting image is transformed 
    back to the spatial domain.

    @param image: Input image to be convolved.
    @param kernel: Kernel used for convolution.
    @return: Convolved image.
    """
    # Transform everything to the frequency domain
    psf_fft = np.fft.fft2(kernel)
    image_fft = np.fft.fft2(image)
    # Perform convolution in the frequency domain:
    convolved = image_fft * psf_fft
    # Transform back to the spatial domain:
    img = np.fft.ifft2(convolved)
    return np.abs(img)


def rotate(image, angle, interpolation_func, center=None, channels=False):
    """
    Rotates an image using a specified interpolation function.

    @param image (numpy.ndarray): Input image.
    @param angle (float): Angle of rotation in degrees.
    @param interpolation_func (function): Interpolation function to use.
    @param center (tuple, optional): Rotation center (default is the image center).
    @param channels (bool, optional): Indicates if the image has channels (default is False).

    @return Rotated image.
    """
    # Convert the rotation angle to radians
    theta = angle * np.pi / 180

    # Get the image shape
    height, width = image.shape[:2]

    if center is None:
        # Calculate the image center
        center_x = (width // 2) - 1
        center_y = (height // 2) - 1
    else:
        center_x, center_y = center

    # Calculate the rotation transformation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Create a coordinate grid for the new rotated image
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.stack([x_mesh, y_mesh], axis=-1)
    transformed_coords = np.dot(coords - np.array([center_x, center_y]), rotation_matrix.T) + np.array([center_x, center_y])

    if channels:
        rotated = interpolate_channels(image, transformed_coords, interpolation_func)
    else:
        rotated = interpolate(image, transformed_coords, interpolation_func)

    return rotated

@njit
def interpolate(image, transformed_coords, interpolation_func):
    """
    Interpolates the image values using a specified interpolation function.

    @param image (numpy.ndarray): Input image.
    @param transformed_coords (numpy.ndarray): Transformed coordinates.
    @param interpolation_func (function): Interpolation function to use.

    @return Interpolated image.
    """
    height, width = image.shape[:2]

    # Extract the transformed x and y coordinates
    transformed_x = transformed_coords[..., 0]
    transformed_y = transformed_coords[..., 1]

    # Apply the interpolation function to get the pixel values in the rotated image
    rotated_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            src_x = transformed_x[y, x]
            src_y = transformed_y[y, x]

            x0 = int(src_x)
            y0 = int(src_y)

            # Calculate the neighboring pixel coordinates
            x1 = x0 + 1
            y1 = y0 + 1

            # Calculate the coordinate differences
            dx = src_x - x0
            dy = src_y - y0

            # Get the neighboring pixel values
            pixel00 = image[max(0, min(height - 1, y0)), max(0, min(width - 1, x0))]
            pixel01 = image[max(0, min(height - 1, y0)), max(0, min(width - 1, x1))]
            pixel10 = image[max(0, min(height - 1, y1)), max(0, min(width - 1, x0))]
            pixel11 = image[max(0, min(height - 1, y1)), max(0, min(width - 1, x1))]

            # Calculate the interpolated value
            interpolated_value = interpolation_func(pixel00, pixel01, pixel10, pixel11, dx, dy)

            rotated_image[y, x] = interpolated_value

    return rotated_image

def interpolate_channels(image, transformed_coords, interpolation_func):
    """
    Interpolates the image values using a specified interpolation function, considering RGB channels.

    @param image (numpy.ndarray): Input image.
    @param transformed_coords (numpy.ndarray): Transformed coordinates.
    @param interpolation_func (function): Interpolation function to use.

    @return Interpolated image with channels.
    """
    height, width = image.shape[:2]

    # Extract the transformed x and y coordinates
    transformed_x = transformed_coords[..., 0]
    transformed_y = transformed_coords[..., 1]

    if len(image.shape) > 2:
        count = image.shape[2]
    else:
        count = 1

    # Apply the interpolation function to get the pixel values in the rotated image
    rotated_image = np.zeros_like(image)
    for channel in range(count):
        for y in range(height):
            for x in range(width):
                src_x = transformed_x[y, x]
                src_y = transformed_y[y, x]

                x0 = int(src_x)
                y0 = int(src_y)

                # Calculate the neighboring pixel coordinates
                x1 = x0 + 1
                y1 = y0 + 1

                # Calculate the coordinate differences
                dx = src_x - x0
                dy = src_y - y0

                # Get the neighboring pixel values
                pixel00 = image[max(0, min(height - 1, y0)), max(0, min(width - 1, x0)), channel]
                pixel01 = image[max(0, min(height - 1, y0)), max(0, min(width - 1, x1)), channel]
                pixel10 = image[max(0, min(height - 1, y1)), max(0, min(width - 1, x0)), channel]
                pixel11 = image[max(0, min(height - 1, y1)), max(0, min(width - 1, x1)), channel]

                # Calculate the interpolated value
                interpolated_value = interpolation_func(pixel00, pixel01, pixel10, pixel11, dx, dy)

                rotated_image[y, x, channel] = interpolated_value

    return rotated_image