from numba import njit
import numpy as np

from GimnTools.ImaGIMN.gimnRec.reconstructors.rotation_reconstruction import rotation_reconstructor
from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor

from GimnTools.ImaGIMN.gimnRec.corrections import *



# @file reconstructor_system_matrix_cpu.py
# @brief Implements an image reconstructor using the system matrix and CPU processing.
# @details This module is designed to reconstruct sinogram data using different algorithms, including Maximum Likelihood Expectation Maximization (MLEM) and Ordered Subset Expectation Maximization (OSEM), leveraging the system matrix approach.



@njit
def compute_tv_gradient(image, epsilon=1e-8):
    rows, cols = image.shape
    tv_grad = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            val = 0.0
            # Vizinho esquerdo
            if i > 0:
                diff = image[i, j] - image[i-1, j]
                val += diff / np.sqrt(diff**2 + epsilon)
            # Vizinho direito
            if i < rows - 1:
                diff = image[i, j] - image[i+1, j]
                val += diff / np.sqrt(diff**2 + epsilon)
            # Vizinho superior
            if j > 0:
                diff = image[i, j] - image[i, j-1]
                val += diff / np.sqrt(diff**2 + epsilon)
            # Vizinho inferior
            if j < cols - 1:
                diff = image[i, j] - image[i, j+1]
                val += diff / np.sqrt(diff**2 + epsilon)
            tv_grad[i, j] = val
    return tv_grad


@njit
def system_matrix(nxd, nrd, nphi, angles, correction_center=None):
    """
    @brief Generates the system matrix corresponding to the projection around each pixel for the given angles.
    @details This function assumes a circular geometry for projection. If a different geometry is needed, modify the 'yp' variable in the calculation.
    
    @param[in] nxd Number of elements in the x-dimension of the image.
    @param[in] nrd Number of bins in the sinogram (distance bins or acquisition pixels).
    @param[in] nphi Number of projection angles (corresponding to the number of angle steps in the sinogram).
    @param[in] angles Array of angles in radians, typically generated as np.linspace(0, angle_in_radians, number_of_angles).
    @param[in] correction_center The center of rotation for projection correction. If None, it defaults to the center of the image.
    
    @return The generated system matrix, with shape (nrd * nphi, nxd * nxd).
    """
    angles = np.deg2rad(angles)
    system_matrix = np.zeros((nrd*nphi, nxd*nxd)) # numero de linhas =  numero de bins no sinograma
                                                  # numero de colunas=  numero de pixels na imagem
    if correction_center is None:
      correction_center =nxd*0.5
    """

    A ideia da system matrix é obter o sinograma para cada pixel da imagem e armazenar isso na forma de um vetor
    pois depois a projeção e retroprojeção é feita simplesmente pela multiplicação de matrizes
    """
    rot = np.pi
    for xv in range(nxd):
        for yv in range(nxd):
            for ph in range((nphi)):
                yp = -(xv-(correction_center))*np.sin(ph*np.pi/nphi+rot)+(yv-(correction_center))*np.cos(ph*np.pi/nphi+rot) # aqui se assume
                yp_bin = int(yp + nrd/2.0)                                                      #indica que o zero está na metade do eixo x
                if yp_bin+ph*nrd < nrd*nphi:
                  system_matrix[yp_bin+ph*nrd, xv+yv*nxd] = 1
    return system_matrix


class reconstructor_system_matrix_cpu(line_integral_reconstructor):
    """
    @class reconstructor_system_matrix_cpu
    @brief Image reconstructor based on the system matrix using CPU processing.
    @details This class uses a system matrix to project and backproject sinograms for reconstruction using MLEM and OSEM algorithms.
    """

    # nxd = number of elements in the x-dimension of the image
    # nrd = number of distance elements in the sinogram (bins) (number of pixels in acquisition)
    # nphi = number of angles in the sinogram (can be understood as the number of elements in the angle vector that compose the sinogram)
    
    nxd = 0  # Number of elements in the x-dimension of the image
    nrd = 0  # Number of bins in the sinogram (distance bins or acquisition pixels)
    nphi = 0  # Number of angles in the sinogram
    correction_center = 0  # Center of rotation for correction
    sens_img = 0  # Sensitivity image

    def __init__(self, sinogram=None, center_of_rotation=None, transpose=None):
        """
        @brief Constructor for the reconstructor_system_matrix_cpu class.
        @param[in] path Optional file path for data.
        @param[in] sinogram Sinogram data used for reconstruction.
        @param[in] sinogram_order Tuple defining the order of dimensions in the sinogram.
        @param[in] center_of_rotation Center of rotation for the image.
        @param[in] transpose Optional transpose parameter for image adjustment.
        """
        super(reconstructor_system_matrix_cpu, self).__init__(sinogram, center_of_rotation)

        self.nxd = self.sinogram.shape[2]
        self.nrd = int(self.nxd)
        self.nphi = self.sinogram.shape[1]
        self.slice = self.sinogram.shape[0]
        self.correction_center = center_of_rotation
        
        print("image x bins:",self.nxd, "sinogram radial bins:", self.nrd, "sinogram angles:", self.nphi)

    def forward_project(self, image, sys_mat):
        """
        @brief Projects the image forward using the system matrix to generate the sinogram.
        @param[in] image The image to be forward-projected.
        @param[in] sys_mat The system matrix for projection.
        @return Forward-projected sinogram.
        """
        # Reshape the image and apply the system matrix to generate the sinogram
        return np.reshape(np.matmul(sys_mat, np.reshape(image, (self.nxd * self.nxd, 1))), (self.nphi, self.nrd))

    def backproject(self, sino, sys_mat):
        """
        @brief Backprojects the sinogram using the system matrix to generate the image.
        @param[in] sino The sinogram to be backprojected.
        @param[in] sys_mat The system matrix for backprojection.
        @return Backprojected image.
        """
        # Reshape the sinogram and apply the transposed system matrix to generate the image
        return np.reshape(np.matmul(sys_mat.T, np.reshape(sino, (self.nrd * self.nphi, 1))), (self.nxd, self.nxd))

    def mlem(self, num_its, angles):
        """
        @brief Performs image reconstruction using the Maximum Likelihood Expectation Maximization (MLEM) algorithm.
        @param[in] num_its Number of iterations for MLEM.
        @param[in] angles Array of projection angles (in radians).
        @return Reconstructed image after MLEM.
        """
        # Initialize the reconstruction image with ones
        recon = np.ones((self.slice, self.nxd, self.nxd))
        slice_count = self.slice_n()
        
        for slice_z in range(self.sinogram.shape[0]):
            # Select the sinogram slice to be reconstructed
            sino_for_reconstruction = self.sinogram[slice_z]
            # Calculate the center of rotation for correction
            # Generate the system matrix
            sys_mat = system_matrix(nxd=self.nxd, nrd=self.nrd, nphi=self.nphi, angles=angles)
            # Generate a sinogram of ones for sensitivity image calculation
            sino_ones = np.ones_like(sino_for_reconstruction)
            # Compute the sensitivity image by backprojecting the sinogram of ones
            sens_image = self.backproject(sino_ones, sys_mat)

            # Perform the MLEM iteration
            for it in range(num_its):
                # print(it)
                # Forward project the current reconstruction estimate
                fpsino = self.forward_project(recon[slice_z, :, :], sys_mat)
                # Compute the ratio between the original sinogram and the forward projection

              
                ratio = sino_for_reconstruction / (fpsino + 1.0e-9)
                # Compute the correction factor by backprojecting the ratio
                correction = self.backproject(ratio, sys_mat) / (sens_image + 1e-9)
                # Update the reconstruction by multiplying with the correction factor
                recon[slice_z, :, :] *= correction    
            #recon [slice_z] = (recon[slice_z]/recon[slice_z].sum())*slice_count[slice_z]


        return recon


    def osem(self, num_its, num_subsets, angles, show_images=False):
        print("iterations: ", num_its, " subsets: ", num_subsets)
        print("sino shape: ", self.sinogram.shape)

        # Precompute system matrix
        sys_mat = system_matrix(self.nxd, self.nrd, self.nphi, angles, self.correction_center)
        
        # Create angle indices
        angle_indices = np.arange(self.nphi)
        subset_angle_indices = np.array_split(angle_indices, num_subsets)
        
        # Precompute subset matrices and sensitivity images
        subset_matrices = []
        subset_sens = []
        for angle_indices in subset_angle_indices:
            # Calculate row indices for this subset
            rows = []
            for angle_idx in angle_indices:
                rows.extend(range(angle_idx * self.nrd, (angle_idx + 1) * self.nrd))
            subset_matrix = sys_mat[rows, :]
            subset_matrices.append(subset_matrix)
            
            # Precompute subset sensitivity
            ones_vec = np.ones(subset_matrix.shape[0])
            sens_sub = subset_matrix.T @ ones_vec
            subset_sens.append(sens_sub.reshape(self.nxd, self.nxd))

        # Initialize reconstruction
        recon = np.ones((self.slice, self.nxd, self.nxd))

        for slice_z in range(self.slice):
            # print("slice : ", slice_z)
            sino = self.sinogram[slice_z]
            
            # Prepare sinogram subsets
            sino_subsets = []
            for angle_indices in subset_angle_indices:
                sino_sub = sino[angle_indices, :].flatten()
                sino_subsets.append(sino_sub)

            # OSEM iterations
            for it in range(num_its):
                # print("Iteration-", it+1)
                for ss in range(num_subsets):
                    sub_mat = subset_matrices[ss]
                    sub_sino = sino_subsets[ss]
                    sens_sub = subset_sens[ss]
                    
                    # Forward projection
                    fpsino = sub_mat @ recon[slice_z].ravel()
                    
                    # Compute ratio with clipping
                    ratio = sub_sino / (fpsino + 1e-12)
                    
                    # Backproject and update
                    back_ratio = sub_mat.T @ ratio
                    recon[slice_z] *= back_ratio.reshape(self.nxd, self.nxd) / (sens_sub + 1e-12)

            if show_images:
                plt.imshow(recon[slice_z], cmap='gray')
                plt.title(f"Slice {slice_z}")
                plt.show()

        return recon
   

    def osem_tv(self, num_its, num_subsets, angles, beta=0.1, tv_epsilon=1e-8, sens_image=None, show_images=False):
            """
            OSEM com regularização de Variação Total (TV) para reconstrução de última geração.
            
            Parâmetros:
            beta: Força da regularização (controla suavização)
            tv_epsilon: Pequeno valor para estabilidade numérica
            """
            recon = np.ones((self.slice, self.nxd, self.nxd))
            slice_count = self.slice_n()

            for slice_z in range(self.sinogram.shape[0]):
                # print(f"Processando slice {slice_z}")
                sino = self.sinogram[slice_z]
                self.correction_center = rot_center(sino)
                
                # Gerar matriz do sistema
                sys_mat = system_matrix(self.nxd, self.nrd, self.nphi, angles)
                
                # Calcular imagem de sensibilidade
                if sens_image is None:
                    sino_ones = np.ones_like(sino)
                    sens_img = self.backproject(sino_ones, sys_mat)
                else:
                    sens_img = sens_image

                # Preparar subconjuntos
                sino1d = sino.reshape(-1, 1)
                sub_sino = np.array_split(sino1d, num_subsets)
                sub_mat = np.array_split(sys_mat, num_subsets, axis=0)

                for it in range(num_its):
                    # print(f"Iteração {it+1}/{num_its}")
                    
                    # Loop por subconjuntos
                    for ss in range(num_subsets):
                        # Projeção direta
                        fp = sub_mat[ss] @ recon[slice_z].ravel()
                        # Calcular razão
                        ratio = sub_sino[ss] / (fp.reshape(-1, 1) + 1e-10)
                        ratio = np.clip(ratio, 0, 10)  # Limitar razões extremas
                        
                        # Retroprojeção e atualização
                        correction = sub_mat[ss].T @ ratio
                        recon[slice_z] *= correction.reshape(self.nxd, self.nxd)
                    
                    # Aplicar regularização TV após cada iteração completa
                    tv_grad = compute_tv_gradient(recon[slice_z], tv_epsilon)
                    recon[slice_z] /= (sens_img + beta * tv_grad + 1e-9)
                    
                    # Normalização
                    recon[slice_z] = (recon[slice_z] / recon[slice_z].sum()) * slice_count[slice_z]


            return recon

