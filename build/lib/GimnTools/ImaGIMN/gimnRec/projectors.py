import numpy as np
from GimnTools.ImaGIMN.processing.tools.math import rotate
from numba import njit


def radon_m(image, angles, interpolator, center=None):
    """
    Computes the sinogram of an image using the Radon transform method.

    @param image (numpy.ndarray): Input image.
    @param angles (numpy.ndarray): Angles at which projections will be calculated.
    @param interpolator (function): Interpolation function to be used.
    @param center (tuple, optional): Center of the image (default is the center of the image).

    @return sino (numpy.ndarray): Sinogram of the image.
    """
    sino = np.zeros([image.shape[0], angles.size])
    for i, angle in enumerate(angles):
        test = rotate(image, angle, interpolator, center=center)
        sino[:, i] = test.sum(axis=1)
    return sino

def projector(image, angles, interpolator, center=None):
    """
    Computes the sinogram of an image using the projection method.

    @param image (numpy.ndarray): Input image.
    @param angles (numpy.ndarray): Angles at which projections will be calculated.
    @param interpolator (function): Interpolation function to be used.
    @param center (tuple, optional): Center of the image (default is the center of the image).

    @return sino (numpy.ndarray): Sinogram of the image.
    """
    ang = angles
    sino = np.zeros([image.shape[0], len(angles)])
    for i, angle in enumerate(ang):
        test = rotate(image, angle, interpolator, center=center)
        sino[:, i] = test.sum(axis=1)
    return sino

@njit
def direct_radon (imagem , angles):
    """
    @brief Computes the Radon transform (sinogram) of a given image for specified projection angles.
    This function calculates the direct Radon transform of a 2D image using bilinear interpolation
    for a set of projection angles. 
    
    @param imagem 2D numpy array representing the input image. The image must be square (same number of rows and columns).
    @param angles 1D array-like of projection angles in degrees at which the Radon transform is computed.
    @return sinograma 2D numpy array containing the sinogram, where each column corresponds to the projection at a specific angle.
    @note The function assumes that the input image is square. If the image is not square, a warning is printed.
    @note Uses bilinear interpolation to estimate pixel values at non-integer coordinates during the projection process.
    @warning The function prints a message if the input image is not square, but continues execution.
    @see bilinear_interpolation
"""
    array = np.asarray(imagem)
    nphi = len(angles)
    dimensions = imagem.shape
    tamanho=dimensions[0]
    tamanho2=dimensions[1]
    colsino=np.zeros((tamanho),dtype=np.float64)
    if tamanho != tamanho2:
        print('the image dimensions must be equal')
    center=(tamanho-1)/2
    radius=center*center

    sinograma=np.zeros((tamanho,nphi))

    for k in range(nphi):
        angle = np.deg2rad(angles[k]-90)
        cos=np.cos(angle)
        sen=np.sin(angle)
        for m in range (tamanho):
            colsino[m]=0
            for n in range (tamanho):
                mc = m - center
                nc = n - center
                if mc*mc + nc*nc < radius:
                    x = center + mc*cos - nc*sen
                    y = center + mc*sen + nc*cos
                    v =bilinear_interpolation(imagem,x,y)
                    var =colsino[m]+v
                    colsino[m] =var
        sinograma[:,k]=colsino
    return sinograma


@njit
def bilinear_interpolation(image, x, y):
    """
    Perform bilinear interpolation for a given position (x,y) in the image.
    
    Args:
        image: 2D numpy array
        x: x-coordinate (column)
        y: y-coordinate (row)
    
    Returns:
        Interpolated pixel value
    """
    # Get integer coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, image.shape[1] - 1)  # Ensure within bounds
    y1 = min(y0 + 1, image.shape[0] - 1)
    
    # Get fractional parts
    dx = x - x0
    dy = y - y0
    
    # Get pixel values
    p00 = image[y0, x0]  # Top-left
    p01 = image[y0, x1]  # Top-right
    p10 = image[y1, x0]  # Bottom-left
    p11 = image[y1, x1]  # Bottom-right
    
    # Interpolate horizontally
    top = (1 - dx) * p00 + dx * p01
    bottom = (1 - dx) * p10 + dx * p11
    
    # Interpolate vertically
    return (1 - dy) * top + dy * bottom