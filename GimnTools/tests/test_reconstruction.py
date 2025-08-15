import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

try:
    # Importações da biblioteca
    from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor
    from GimnTools.ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction import reconstructor_system_matrix_cpu
    from GimnTools.ImaGIMN.gimnRec.reconstructors.rotation_reconstruction import rotation_reconstructor
    from GimnTools.ImaGIMN.processing.interpolators.reconstruction import bilinear_interpolation
    from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import ramLak
    from GimnTools.ImaGIMN.gimnRec.projectors import direct_radon
    from GimnTools.ImaGIMN.phantoms.random import generate_random_image
    from GimnTools.ImaGIMN.phantoms.derenzzo import DerenzoPhantomSlice
    from skimage.transform import resize
    from skimage.data import shepp_logan_phantom
except ImportError as e:
    raise ImportError(f"Erro na importação: {e}") from e

class TestGimnToolsReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        
        import matplotlib
        matplotlib.use('Agg')  # Usar backend não interativo para testes
        import matplotlib.pyplot as plt

        # Configuração inicial para todos os testes
        cls.test_dir = Path(tempfile.mkdtemp(prefix="gimntools_test_"))
        print(f"Test output directory: {cls.test_dir}")
        
        # Configurações reduzidas para testes rápidos
        cls.pixels = 32  # Tamanho reduzido para testes rápidos
        cls.number_of_angles = 32
        cls.number_of_slices = 1
        cls.iterations = 2
        cls.subsets = 4

    @classmethod
    def tearDownClass(cls):
        # Limpeza após todos os testes
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        print(f"Removed test directory: {cls.test_dir}")

    def test_random_image_generation(self):
        """Testa a geração de imagens aleatórias"""
        img = generate_random_image(self.pixels, 5)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (self.pixels, self.pixels))
        
        # Verificação básica de valores
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)
        
        # Teste de salvamento
        save_path = self.test_dir / "random_image.png"
        plt.imsave(save_path, img, cmap='jet')
        self.assertTrue(save_path.exists())

    def test_shepp_logan_phantom(self):
        """Testa a geração do fantoma Shepp-Logan"""
        img = shepp_logan_phantom()
        img = resize(img, (self.pixels, self.pixels), anti_aliasing=True)
        noisy_image = np.random.poisson(img * 10).astype(np.float64)
        
        self.assertEqual(noisy_image.shape, (self.pixels, self.pixels))
        self.assertGreater(noisy_image.max(), 0)
        
        # Teste de salvamento
        save_path = self.test_dir / "shepp_logan.png"
        plt.imsave(save_path, noisy_image, cmap='jet')
        self.assertTrue(save_path.exists())

    def test_derenzo_phantom(self):
        """Testa a geração do fantoma Derenzo"""
        phantom = DerenzoPhantomSlice(
            radius=14.5,
            num_sections=4,
            well_counts=(6, 6, 10, 15),
            well_diameters=(2.5, 2.0, 1.5, 1.0),
            well_separations=(2.5, 2.0, 1.5, 1.0),
            section_offsets=(0.00000001, 0.01, 0.01, 0.1),
            image_size=self.pixels,
            circle_value=1,
            well_value=10
        )
        
        noisy_image = np.random.poisson(phantom.get_image_matrix() * 0.3)
        
        self.assertEqual(noisy_image.shape, (self.pixels, self.pixels))
        self.assertGreater(noisy_image.max(), 0)
        
        # Teste de salvamento
        save_path = self.test_dir / "derenzo_phantom.png"
        plt.imsave(save_path, noisy_image, cmap='jet')
        self.assertTrue(save_path.exists())
        return noisy_image

    def test_rotation_reconstructor(self):
        """Testa o reconstructor por rotação"""
        img = self.test_derenzo_phantom()
        angles = np.linspace(0, 180, self.number_of_angles, endpoint=True)
        sinogram = direct_radon(img, angles)
        stack = np.asarray([sinogram for _ in range(self.number_of_slices)])

        reconstructor = rotation_reconstructor()
        reconstructor.set_sinogram(stack)
        
        # Teste FBP
        recon_fbp = reconstructor.fbp(bilinear_interpolation, ramLak, angles=angles)
        self.assertEqual(recon_fbp.shape, (self.number_of_slices, self.pixels, self.pixels))
        
        # Teste OSEM (execução reduzida)
        recon_osem = reconstructor.osem(
            iterations=1,
            subsets=2,
            interpolator=bilinear_interpolation,
            angles=angles,
            verbose=False
        )
        self.assertEqual(recon_osem.shape, (self.number_of_slices, self.pixels, self.pixels))

    def test_line_integral_reconstructor(self):
        """Testa o reconstructor por integral de linha"""
        img = self.test_derenzo_phantom()
        angles = np.linspace(0, 180, self.number_of_angles, endpoint=True)
        sinogram = direct_radon(img, angles)
        stack = np.asarray([sinogram for _ in range(self.number_of_slices)])

        reconstructor = line_integral_reconstructor(stack)
        
        # Teste FBP
        recon_fbp = reconstructor.fbp(ramLak, angles)
        self.assertEqual(recon_fbp.shape, (self.number_of_slices, self.pixels, self.pixels))
        
        # Teste MLEM (execução reduzida)
        recon_mlem = reconstructor.mlem(
            iterations=1,
            angles=angles
        )
        self.assertEqual(recon_mlem.shape, (self.number_of_slices, self.pixels, self.pixels))

    def test_system_matrix_reconstructor(self):
        """Testa o reconstructor por matriz do sistema"""
        img = self.test_derenzo_phantom()
        angles = np.linspace(180, 0, self.number_of_angles, endpoint=False)
        sinogram = direct_radon(img, angles)
        stack = np.transpose(
            np.asarray([sinogram for _ in range(self.number_of_slices)]),
            (0, 2, 1)
        )

        reconstructor = reconstructor_system_matrix_cpu(stack)
        
        # Teste OSEM (execução reduzida)
        recon_osem = reconstructor.osem(
            iterations=1,
            subsets=2,
            angles=angles,
            show_images=False
        )
        self.assertEqual(recon_osem.shape, (self.number_of_slices, self.pixels, self.pixels))

if __name__ == "__main__":
    unittest.main()