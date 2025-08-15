import numpy as np
from scipy.ndimage import rotate as rt_scipy
from numba import njit
from GimnTools.ImaGIMN.processing.tools.math  import rotate


def iradon_m(sinogram, interpolator, angles, center=None):
    """
    Performs image reconstruction from a sinogram using the filtered backprojection method.

    @param sinogram (numpy.ndarray): Input sinogram.
    @param interpolator (function): Interpolation function to be used.
    @param angles (numpy.ndarray): Angles corresponding to the projections of the sinogram.
    @param center (tuple, optional): Center of the image (default is the center of the image).

    @return Reconstructed image.
    """
    size = sinogram.shape[0]
    angle_n = angles

    out = np.zeros([size, size])
    aux = np.zeros([size, size])

    for i, angle in enumerate(angle_n):
        col = sinogram[:, i]
        aux[:, 0:size] = col
        out += rotate(aux, angle - 90, interpolator, center=center)
    out = out / size
    return np.flip(out, axis=1)


def backprojector(sinogram, angles, interpolator, center=None):
    """
    Performs backprojection of a sinogram to reconstruct the image.

    @param sinogram (numpy.ndarray): Input sinogram.
    @param angles (numpy.ndarray): Angles corresponding to the projections of the sinogram.
    @param interpolator (function): Interpolation function to be used.
    @param center (tuple, optional): Center of the image (default is the center of the image).

    @return Reconstructed image.
    """
    size = sinogram.shape[0]

    out = np.zeros([size, size])
    aux = np.zeros([size, size])

    for i, angle in enumerate(angles):
        col = sinogram[:, i]
        aux[:, 0:size] = col
        out += rotate(aux, angle - 90, interpolator, center=center)

        #out += rotate(aux, angle - (np.pi / 2.0), interpolator, center=center)
    out = out / size
    return np.flip(out, axis=1)

#(parallel=True)
#
# Quando estamos trabalhando com parallel=True irá demorar muito mais
@njit
def inverse_radon(sinogram, angles):
    """
    @brief Reconstructs an image from its sinogram using the inverse backprojection method.
    This function implements the inverse backprojection (Inverse Radon Transform) to reconstruct a two-dimensional image
    from its sinogram, using a list of projection angles. For each pixel of the output image, the function
    sums the contributions from all projections, interpolating values from the sinogram when necessary.
    @param sinogram (np.ndarray): 2D matrix representing the sinogram, where each column corresponds to a projection at a specific angle.
    @param angles (np.ndarray): 1D vector containing the angles (in degrees) of the projections present in the sinogram.
    @return np.ndarray: Image reconstructed from the sinogram, with the same size as the vertical dimension of the sinogram.
    @note
        - The function assumes that the output image is square and that the number of rows in the sinogram matches the image size.
        - Linear interpolation is used for non-integer values when accessing the sinogram.
        - Normalization is performed by the ratio between the largest angle (in radians) and the number of angles.
    """
    nb_angles = sinogram.shape[1]  # Número de ângulos (largura do sinograma)
    size = sinogram.shape[0]       # Tamanho da imagem (altura e largura)
    
    # Inicializa a matriz de saída
    reconstructed_image = np.zeros((size, size), dtype=np.float64)
    center = (size - 1) / 2  # Centro da imagem
    
    # Loop sobre cada pixel da imagem de saída
    for i in range(size):
        for j in range(size):
            sum_value = 0.0
            for k in range(nb_angles):
                angle = np.deg2rad(angles[k]-90)
                cos_angle = np.cos(angle)  # Coseno do ângulo k
                sin_angle = np.sin(angle)  # Seno do ângulo k
                
                S = ((j-center)*cos_angle + (i-center)* sin_angle)+center

                
                # Pega a coluna k do sinograma
                col_sin = sinogram[:,k ]
                
                # Se V não for inteiro, faz interpolação
                if S< 0 or S >= len(col_sin):
                    continue  # Ignora se V estiver fora dos limites
                
                if S - int(S) != 0:
                    inter = get_interpolated_pixel_1d(col_sin, S)
                    sum_value += inter
                else:
                    sum_value += col_sin[int(S)]
            
            # Normaliza a soma sobre todos os ângulos
            reconstructed_image[i, j] = sum_value * (np.deg2rad(angles.max())/ nb_angles)
    
    return reconstructed_image

@njit
def get_interpolated_pixel_1d(vector, t):
    """
    @brief Performs 1D linear interpolation for a value t between the indices of a vector.
    This function calculates the linearly interpolated value between two adjacent elements of a vector,
    given a fractional index t. If t is out of the vector bounds, it returns 0.0.
    @param vector Input vector (1D array) containing the values to be interpolated.
    @param t Fractional index (float) for which the interpolated value will be calculated.
    @return Interpolated value corresponding to index t. Returns 0.0 if t is out of the vector bounds.
    """
    tt = int(t)
    
    if tt < 0 or tt + 1 >= len(vector):
        return 0.0  # Retorna zero se estiver fora dos limites
    
    T = t - tt
    
    # Interpolação linear entre os valores adjacentes de vector
    interpolated = (vector[tt + 1] - vector[tt]) * T + vector[tt]
    
    return interpolated