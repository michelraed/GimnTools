
try: 
# random image generator:
    from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor
    from GimnTools.ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction import reconstructor_system_matrix_cpu
    from GimnTools.ImaGIMN.gimnRec.reconstructors.rotation_reconstruction import rotation_reconstructor
    from GimnTools.ImaGIMN.processing.interpolators.reconstruction import beta_spline_interpolation, bilinear_interpolation, nearest_neighbor_interpolation
    from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import ramLak
    from GimnTools.ImaGIMN.gimnRec.projectors import radon_m, projector, direct_radon


    from skimage.transform import resize
    from skimage.data import shepp_logan_phantom, horse
    import scipy.io
    import numpy as np 
    from copy import deepcopy 
    from pathlib import Path

    from GimnTools.ImaGIMN.phantoms.random import generate_random_image

except ImportError as e:
    print(f"Erro na importação: {e}")
    print("Certifique-se de que a GimnTools está instalada:")
    print("pip install -e .")
    exit(1)

def random_image_test():
    """Generates a random image for testing purposes."""
    import matplotlib.pyplot as plt
    pixels  = 128 # size off the image
    number_of_figures = 10 # number of figures in the image
    try:
        # Generate a random image with specified number of figures using generate_random_image function
        # This function creates an image with random geometric figures like stars, ellipses, and polygons.
        # The figures are randomly placed and sized within the image.
        img = generate_random_image(pixels, number_of_figures)
        path = "./output/"

        # Ensure the output directory exists
        # If it doesn't exist, create it
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save the generated image to a file
        # The image is saved in PNG format with a specific filename
        filename = path + "random_image_test.png"

        # Display the image using matplotlib
        plt.figure(figsize=(6, 6))
        plt.clf()
        plt.axis('off')
        plt.imshow(img, cmap='jet')
        plt.title("Random Image Test")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        print(f"Random image saved to {filename}")
        return img
    except Exception as e:
        print(f"⚠️  Error generating random image: {e}")
        return False




def generate_shepp_logan_phantom(noise_level = 10):
    """
    Generates a Shepp-Logan phantom image, resizes it, and saves it as a PNG file.
    """
    try : 
        import matplotlib.pyplot as plt
        # Generate the Shepp-Logan phantom
        img = shepp_logan_phantom()

        # Resize the image to 128x128 pixels
        img = resize(img, (128, 128), anti_aliasing=True)

        noiser = noise_level
        resized_image = img * noiser
        noisy_image = np.random.poisson(resized_image).astype(np.float64)

        # Save the image to a file
        path = "./output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = path + "shepp_logan_phantom_test.png"

        plt.figure(figsize=(6, 6))
        plt.clf()
        plt.axis('off')
        plt.imshow(noisy_image, cmap='jet')
        plt.title("Shepp-Logan Phantom")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        print(f"Shepp-Logan phantom saved to {filename}")

        return noisy_image
    
    except Exception as e:
        print(f"⚠️  Error generating Shepp-Logan phantom: {e}")
        return False




from GimnTools.ImaGIMN.phantoms.derenzzo import DerenzoPhantomSlice

def derenzo_phantom_test(noise_factor=0.3):
    """
    Generates a Derenzo phantom image, resizes it, and saves it as a PNG file.
    """
    import matplotlib.pyplot as plt

    # Teste do código adaptado
    radius = 29.0 / 2
    num_sections = 4
    well_counts = (6, 6, 10, 15)
    well_diameters = (2.5, 2.0, 1.5, 1.0)
    well_separations = (2.5, 2.0, 1.5, 1.0)
    section_offsets = (0.00000001, 0.01, 0.01, 0.1)  # Offset personalizado para cada seção
    circle_value = 1
    well_value = 10
    image_size = 128


    try:
        my_phantom = DerenzoPhantomSlice(radius, num_sections, well_counts, well_diameters, well_separations, section_offsets,
                                    image_size=image_size, circle_value=circle_value, well_value=well_value)

        #my_phantom.save_image('derenzo_phantom_with_offsets.png')

        # Obter a matriz da imagem contendo apenas o círculo de fundo
        circle_image_matrix = my_phantom.draw_only_circle()
        # Obter a matriz completa do fantoma
        phantom_matrix = my_phantom.get_image_matrix()

        noisy_image = np.random.poisson(phantom_matrix*noise_factor).astype(np.float64)
        
        plt.imshow (noisy_image, cmap='jet')
        plt.title('Derenzo Phantom with Noise')
        plt.colorbar()
        print(f"✅ Imagem simulada de um fantoma derenzo de raio {radius}mm e poços com diametros {well_diameters} mm  com ruído foi criada e tem dimensões: {noisy_image.shape}")
        
        path = "./output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + "derenzo_phantom_test.png", bbox_inches='tight', pad_inches=0.1)
        return noisy_image
    
    except Exception as e:
        print(f"⚠️  Erro ao criar o fantoma Derenzo: {e}")
        return np.zeros((image_size, image_size))




def test_rotation_reconstructor():
    """
    Tests the rotation reconstructor with a sample image.
    """
    try:
        import matplotlib.pyplot as plt
        pixels = 128  # Size of the image
        number_of_angles = pixels  # Number of angles for the sinogram
        number_of_slices = 2  # Number of slices in the stack

        # Generate a derenzo phantom image with noise
        noisy_image = derenzo_phantom_test(noise_factor=0.3)
        angles = np.linspace(0, 180, number_of_angles, endpoint=True)
        sinogram = direct_radon(noisy_image, angles)
        stack = np.asarray([sinogram for _ in range(number_of_slices)])


        reconstructor = rotation_reconstructor()
        reconstructor.set_sinogram(stack)
        #reconstructor.set_sinogram(sino) 

        iterations = 3
        subsets = 8 #2, 4, 8, 16

        angles_here = angles+180
        recon_osem_rot = reconstructor.osem(iterations,subsets, bilinear_interpolation, angles=angles_here, verbose=False)
        recon_mlem_rot = reconstructor.mlem(iterations*subsets, bilinear_interpolation, angles=angles_here, verbose=False)
        recon_fbp_rot= reconstructor.fbp(bilinear_interpolation, ramLak, angles=angles_here)

        cmap = 'jet'
        # Display the reconstructed images        
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(recon_osem_rot[0], cmap=cmap)
        plt.title('OSEM Reconstruction (Rotation)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(recon_mlem_rot[0], cmap=cmap)
        plt.title('MLEM Reconstruction (Rotation)')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(recon_fbp_rot[0], cmap=cmap)
        plt.title('FBP Reconstruction (Rotation)')
        plt.axis('off') 
        plt.tight_layout()
        # Save the reconstructed images
        path = "./output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + "rotation_reconstructor_test.png", bbox_inches='tight', pad_inches=0.1)
        print(f"✅ Rotation reconstructor test completed successfully. Images saved to {path}rotation_reconstructor_test.png")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Error in rotation reconstructor test: {e}")
        return False



def test_line_integral_reconstructor():
    try: 
        import matplotlib.pyplot as plt
        pixels = 128  # Size of the image
        number_of_angles = pixels  # Number of angles for the sinogram
        number_of_slices = 2  # Number of slices in the stack

        # Generate a derenzo phantom image with noise
        noisy_image = derenzo_phantom_test(noise_factor=0.3)
        angles = np.linspace(0, 180, number_of_angles, endpoint=True)
        sinogram = direct_radon(noisy_image, angles)
        stack = np.asarray([sinogram for _ in range(number_of_slices)])


        reconstructor = rotation_reconstructor()
        reconstructor.set_sinogram(stack)
        #reconstructor.set_sinogram(sino) 

        iterations = 3
        subsets = 8 #2, 4, 8, 16

        #==========================================================
        #                 Reconstruir Line Integral
        #==========================================================
        #para este deve ser slice/bins/angle
        angles_m1 = np.linspace(0, 180, number_of_angles, endpoint=True)
        # angles_sm = np.linspace(180,0 , number_of_angles, endpoint=False)

        reconstructor = line_integral_reconstructor(stack)
        # recon_osem_li = reconstructor.osem(iterations,subsets, angles_m1, verbose=False)
        # recon_mlem_li = reconstructor.mlem(iterations*subsets, angles_m1, verbose=False)
        # recon_fbp_li = reconstructor.fbp( ramLak, angles)
        # reconstructor.set_sinogram(stack)

        iterations = [5]
        subsets = [32] 
        pilha = []

        for i in range(stack.shape[0]):
            for iter in iterations:
                for subset in subsets:
                    print(f'Processing - iteration: {iter}\tsubset: {subset}')
                    sino = stack[i:i+1, :, :]

                    reconstructor.set_sinogram(sino)
                    # Cria o dicionário com a estrutura especificada
                    recon = {}
                    recon["iterative_method"] = "OSEM/MLEM"
                    recon["iteration"] = iter
                    recon["subset"] = subset
                    recon["total_iterations"] = iter * subset
                    recon["sinogram"] = i + 1
                    recon ["image_OSEM"] = reconstructor.osem(iter, subset, angles_m1, verbose=False)
                    recon ["image_MLEM"] = reconstructor.mlem(iter * subset, angles_m1)   
                    recon["analytic_method"] = "FBP"
                    recon["fbp_image"] = reconstructor.fbp(ramLak, angles)
                    pilha.append(deepcopy(recon))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        cmap = 'jet'
        plt.imshow(pilha[0]["image_MLEM"][0], cmap=cmap)
        plt.title('Line Integral MLEM Reconstruction')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(pilha[0]["image_OSEM"][0], cmap=cmap)
        plt.title('Line Integral OSEM Reconstruction')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(pilha[0]["fbp_image"][0], cmap=cmap)
        plt.title('Line Integral FBP Reconstruction')
        plt.axis('off')
        plt.tight_layout()
        # Display the reconstructed image
        # Save the reconstructed images
        path = "./output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + "line_integral_reconstructor_test.png", bbox_inches='tight',
                        pad_inches=0.1) 
        print(f"✅ Line integral reconstructor test completed successfully. Images saved to {path}line_integral_reconstructor_test.png")
        return pilha
    except Exception as e:
        print(f"⚠️  Error in line integral reconstructor test: {e}")
        return False
    


def test_system_matrix_reconstructor():
    import matplotlib.pyplot as plt
    pixels = 128  # Size of the image
    number_of_angles = pixels  # Number of angles for the sinogram
    number_of_slices = 2  # Number of slices in the stack

    # Generate a derenzo phantom image with noise
    noisy_image = derenzo_phantom_test(noise_factor=0.3)
    angles = np.linspace(0, 180, number_of_angles, endpoint=True)
    sinogram = direct_radon(noisy_image, angles)
    stack = np.asarray([sinogram for _ in range(number_of_slices)])

    try:
        # neste cara os angulos tem que ser invertidos linspace(180,0)
        # Para este cara deve ser Slice/Angle/bins
        angles_sm = np.linspace(180,0 , number_of_angles, endpoint=False)
        stk = np.transpose(stack, (0, 2, 1)) 

        iterations = 3 
        subsets = 8 

        reconstructor = reconstructor_system_matrix_cpu(stk)
        recon_osem_sm = reconstructor.osem(iterations,subsets, angles_sm, show_images=False)
        recon_mlem_sm = reconstructor.mlem(iterations*subsets, angles_sm)

        cmap = 'jet'
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_image, cmap=cmap)
        plt.title('Original Derenzo Phantom with Noise')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(recon_osem_sm[0], cmap=cmap)
        plt.title('System Matrix OSEM Reconstruction')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(recon_mlem_sm[0], cmap=cmap) 
        plt.title('System Matrix MLEM Reconstruction')
        plt.axis('off')
        plt.tight_layout()
        # Save the reconstructed images
        path = "./output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + "system_matrix_reconstructor_test.png", bbox_inches='tight', pad_inches=0.1)
        print(f"✅ System matrix reconstructor test completed successfully. Images saved to {path}system_matrix_reconstructor_test.png")
        return True
    
    except Exception as e:
        print(f"⚠️  Error in system matrix reconstructor test: {e}")
        return False

def run_all_tests():
    """
    Runs all the tests and generates the corresponding images.
    """
    print("Running all tests...")
    
    a = []
    try:
        # Test random image generation
        a.append(random_image_test())
        
        # Test Shepp-Logan phantom generation
        a.append(generate_shepp_logan_phantom())
        
        # Test Derenzo phantom generation
        a.append(derenzo_phantom_test())
        
        # Test rotation reconstructor
        a.append(test_rotation_reconstructor())
        
        # Test line integral reconstructor
        a.append(test_line_integral_reconstructor())
        
        # Test system matrix reconstructor
        a.append(test_system_matrix_reconstructor())

        one_miss = False
        for i, result in enumerate(a):
            if result is not False:
                print(f"✅ Test {i+1} passed successfully.")
            else:
                print(f"❌ Test {i+1} failed.")
                one_miss = True
        if one_miss:
            print("⚠️  Some tests failed. Please check the output for details.")

    except Exception as e:
        print(f"⚠️  Error running tests: {e}")
    
if __name__ == "__main__":
    run_all_tests()
    print("All tests completed.")
    print("Check the output directory for generated images.")
    print("Thank you for using GimnTools!")
    print("If you have any issues, please report them on the GitHub repository.")
