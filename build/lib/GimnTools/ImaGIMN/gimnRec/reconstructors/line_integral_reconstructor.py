from GimnTools.ImaGIMN.image import image
from GimnTools.ImaGIMN.gimnRec.backprojectors import *
from GimnTools.ImaGIMN.gimnRec.projectors import *
from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import *
from GimnTools.ImaGIMN.processing.interpolators.reconstruction import *
from GimnTools.ImaGIMN.gimnRec.corrections import *



class line_integral_reconstructor(image):
    """
    @brief Creates a Reconstructor class that will inherit the image class.

    The sinogram order inside our program is:
    - rows: slice
    - columns: distances
    - Z: angles

    The sinogram order list must have the following names, but the order can change:
    - ("slice", "distances", "angles")
    """

    __sinogram_order_recon = ("slice", "distances", "angles")  # Sinogram order inside our program
    __center_of_rotation = None  # Center of rotation
    __sinogram_order = None  # Sinogram Order, used for repositioning the dimensions following the order needed to reconstruct
    __sinogram = None
    __reconstructed_mlem = None
    __reconstructed_osem = None
    __reconstructed_fbp = None

    def __init__(self, sinogram, center_of_rotation=None):
        """
        @brief Constructs the reconstructor class, it will initiate the super class of reconstructor, that inherits an image class.

        The image class will be responsible for opening the dicom and retrieving its pixels as a numpy object.

        @param path Path to the image file
        @param sinogram Sinogram data
        @param sinogram_order Order of the sinogram dimensions, default is ("slice", "angles", "distances")
        @param center_of_rotation Center of rotation
        @param transpose Flag to transpose the sinogram
        """
        super(line_integral_reconstructor, self).__init__(image=sinogram)
        self.__sinogram = sinogram

    @property
    def sinogram(self):
        """
        @brief Returns the sinogram pixels as a numpy object
        @return Sinogram as a numpy array
        """
        return self.__sinogram

    def set_sinogram(self, sinogram):
        """
        @brief Sets the sinogram data.

        @param sinogram Sinogram data to set
        """
        self.__sinogram = sinogram

    def set_img(self, img):
        """
        @brief Sets the image.

        @param img Image data to set
        """
        self.__img = img

    def set_center_of_rotation(self, center_of_rotation):
        """
        @brief Sets the value for the center of rotation.

        @param center_of_rotation Center of rotation value
        """
        self.__center_of_rotation = center_of_rotation


    def osem(self, iterations, subsets_n, angles, verbose=False, ):
        """
        @brief Reconstructs the sinogram using the Ordered Subset Expectation Maximization (OSEM) algorithm for a given number of iterations.

        This reconstruction is done using the rotations of the "reconstructed image" in order to obtain the projections.

        @param iterations Number of iterations
        @param subsets_n Number of subsets
        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param angles Angles for reconstruction, should be a numpy array of angles in radians
        @param verbose Flag to print the iteration number during reconstruction

        @return Reconstructed image using the OSEM algorithm
        """
        from matplotlib import pyplot as plt
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        slice_count = self.slice_n()

        sino_ones = np.ones_like(self.sinogram[0])
        sens_image = inverse_radon(sino_ones,angles)



        for slice_z in range(slices):

            sinogram = self.sinogram[slice_z, :, :]
            angles_subset = np.array_split(angles, subsets_n)
            subsets = np.array_split(sinogram, subsets_n, axis=1)
            reconstruction = np.ones([pixels, pixels])

            
            for it in range(iterations):
                
                for i, subset in enumerate(subsets):
                    #aqui ele vai pegar a imagem estimada, e projetar apenas nos angulos do subset desejado
                    rec_sub = direct_radon(reconstruction, angles_subset[i])
                    #calcula-se a razão do subset pelo subset estimado nesta iteração
                    coef = (subset / (rec_sub+1e-10))

                    # isso daqui ta pra quando um pixel cresce demais, ele volta pra um, pra nao estragar. 
                    coef[coef>1000]=1
                    backprojection = np.abs(inverse_radon(coef, angles_subset[i]))
                    reconstruction *= backprojection

            
            reconstruction = reconstruction/(sens_image+1e-9)

            #reconstruction = (reconstruction/reconstruction.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = reconstruction

        self.__reconstructed_osem = rec
        return rec

    def fbp(self, filter_type, angles):
        """

        @brief Reconstructs the sinogram using the Filtered Back-Projection (FBP) algorithm.

        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param filter_type Filter to be used in the FBP, can be: cossineFilter or ramLak
        @param angles Angles for reconstruction, should be a numpy array of angles in radians

        @return Reconstructed image using the FBP algorithm
        """
        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1

        slices = self.sinogram.shape[0]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        slice_count = self.slice_n()

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z, :, :]
            filtered = apply_filter_to_sinogram(filter_type, sinogram)
            img = np.abs(inverse_radon(filtered, angles=angles))
            imagem_estimada = (img/img.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = imagem_estimada
        self.__reconstructed_fbp = rec
        return rec

    def slice_n(self):
        """
        @brief Counts the total number of counts for each slice.

        @return Array containing the total counts for each slice
        """
        slice_count = np.zeros(self.sinogram.shape[0])
        for slice_sino in range(self.sinogram.shape[0]):
            slice_count[slice_sino] = self.sinogram[slice_sino, :, :].sum()
        return slice_count

    def normalize(self, image):
        """
        @brief Normalize the sinogram.

        @param image Image to normalize

        @return Normalized image
        """
        norm = np.zeros(image.shape)
        slices_n = self.slice_n()
        image = image[:]
        # Additional normalization logic can be added here
        return norm

    def mlem(self, iterations, angles, verbose=False):
        """
        @brief Reconstructs the sinogram using the Maximum Likelihood Expectation Maximization (MLEM) algorithm for a given number of iterations.

        This reconstruction is done using the rotations of the "reconstructed image" in order to obtain the projections.

        @param iterations Number of iterations
        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param angles Angles for reconstruction, should be a numpy array of angles in radians
        @param verbose Flag to print the iteration number during reconstruction

        @return Reconstructed image using the MLEM algorithm
        """
        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1
        
        slice_count = self.slice_n()
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        #Generate a sinogram of ones for sensitivity image calculation
        

        sino_ones = np.ones_like(self.sinogram[0])
        sens_image = inverse_radon(sino_ones,angles)
        # Reshape the sinogram into 1D for subset processing

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z]
            imagem_estimada = np.ones([pixels, pixels])
            for it in range(iterations):
                proje_estimada = direct_radon(np.abs(imagem_estimada),angles)   
                diff = np.nan_to_num(sinogram / (proje_estimada+1e-10))
                diff[diff>100] = 1
                imagem_estimada *= np.abs(np.nan_to_num(inverse_radon(diff,angles),copy=True,nan=1))
                imagem_estimada /= (sens_image+1e-9)
                
            #imagem_estimada = (imagem_estimada/imagem_estimada.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = imagem_estimada
            
            
        self.__reconstructed_mlem = rec
        return rec


    def compute_tv_gradient(self, image, epsilon=1e-8):
        """
        Calcula o gradiente da TV usando diferenças para frente, evitando o efeito wrap-around.
        """
        grad = np.zeros_like(image)
        # Diferenças para a direção x (vertical)
        diff_x = np.zeros_like(image)
        diff_x[:-1, :] = image[1:, :] - image[:-1, :]
        # Diferenças para a direção y (horizontal)
        diff_y = np.zeros_like(image)
        diff_y[:, :-1] = image[:, 1:] - image[:, :-1]

        # Gradiente normalizado (suavização)
        grad_x = diff_x / (np.abs(diff_x) + epsilon)
        grad_y = diff_y / (np.abs(diff_y) + epsilon)

        # Divergência dos gradientes
        div_x = np.zeros_like(image)
        div_y = np.zeros_like(image)
        div_x[1:, :] = grad_x[1:, :] - grad_x[:-1, :]
        div_y[:, 1:] = grad_y[:, 1:] - grad_y[:, :-1]

        grad = div_x + div_y
        return grad

    def osem_tv(self, iterations, subsets_n, angles, beta=0.2, tv_epsilon=1e-8, verbose=False):
        """
        Reconstrução usando OSEM com regularização TV.
        
        Primeiro é aplicada a atualização OSEM e, em seguida, um passo de
        descida de gradiente é feito para reduzir o termo TV.
        """
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.zeros((slices, pixels, pixels))
        slice_count = self.slice_n()

        # Cálculo da sensibilidade
        sino_ones = np.ones_like(self.sinogram[0])
        sens_image = inverse_radon(sino_ones, angles)
        sens_image = np.clip(sens_image, 1e-6, None)

        for slice_z in range(slices):
            current_counts = slice_count[slice_z]
            sinogram = self.sinogram[slice_z]
            # Inicialização: pode usar FBP para obter uma boa aproximação inicial
            reconstruction = inverse_radon(sinogram, angles)
            reconstruction = np.clip(reconstruction, 1e-6, None)
            # Dividir sinograma e ângulos em subconjuntos
            subsets = np.array_split(sinogram, subsets_n, axis=1)
            angle_subsets = np.array_split(angles, subsets_n)
            for it in range(iterations):
                # Atualização OSEM
                total_update = np.ones((pixels, pixels))
                for subset, angles_ss in zip(subsets, angle_subsets):
                    proj = direct_radon(reconstruction, angles_ss)
                    ratio = subset / (proj + 1e-10)
                    backproj = inverse_radon(ratio, angles_ss)
                    total_update *= backproj ** (1.0 / subsets_n)
                reconstruction *= total_update
                reconstruction = reconstruction / sens_image

                # Passo de regularização TV (descida de gradiente)
                grad_tv = self.compute_tv_gradient(reconstruction, tv_epsilon)
                reconstruction = reconstruction - beta * grad_tv
                reconstruction = np.clip(reconstruction, 1e-6, None)

                # Normalização para preservar o total de contagens da fatia
                factor = current_counts / (reconstruction.sum() + 1e-10)
                reconstruction *= factor

            rec[slice_z] = reconstruction
        return rec
