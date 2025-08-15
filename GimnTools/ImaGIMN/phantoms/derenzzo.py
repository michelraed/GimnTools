import warnings
import numpy as np
from PIL import Image


class DerenzoPhantomSlice:
    def __init__(self, radius, num_sections, well_counts, well_diameters, well_separations, section_offsets,
                 cyl_height=0, unit="mm", image_size=256, circle_value=0.1, well_value=1.0):
        """Inicializa o fantoma de Derenzo.

        Args:
            radius (float): Raio do fantoma em mm.
            num_sections (int): Número de seções do fantoma.
            well_counts (tuple): Quantidade de furos em cada seção.
            well_diameters (tuple): Diâmetro dos poços de cada seção em mm.
            well_separations (tuple): Espaçamento entre os furos de cada seção em mm.
            section_offsets (tuple): Offset de cada seção em relação ao raio do fantoma.
            cyl_height (int, optional): Altura do cilindro. Defaults to 0.
            unit (str, optional): Unidade de medida. Defaults to "mm".
            image_size (int, optional): Tamanho da imagem. Defaults to 256.
            circle_value (float, optional): Valor do círculo de fundo. Defaults to 0.1.
            well_value (float, optional): Valor dos poços. Defaults to 1.0.
        """
        self.num_sections = num_sections
        self.radius = radius
        self.well_counts = well_counts
        self.well_diameters = well_diameters 
        self.well_separations = well_separations
        self.section_offsets = section_offsets
        self.depth = cyl_height
        self.unit = unit
        self.image_size = image_size
        self.circle_value = circle_value
        self.well_value = well_value
        self.image = np.zeros((self.image_size, self.image_size))
        self.sections = []

        for sec_index in range(self.num_sections):
            well_count = self.well_counts[sec_index]
            well_sep = self.well_separations[sec_index]
            well_dia = self.well_diameters[sec_index]
            sec_offset = self.section_offsets[sec_index]
            
            rot_angle = sec_index * (360. / self.num_sections)
            section = DerenzoInnerSection(self.radius, well_count, well_sep, well_dia, section_offset=sec_offset, well_value=self.well_value)
            section.apply_rotation(rot_angle)
            self.sections.append(section)

        # Desenha o círculo de fundo
        self.draw_circle(self.image, (self.image_size // 2, self.image_size // 2), int(self.radius * self.image_size / (2 * self.radius)), self.circle_value)

        # Desenha os poços em cada seção
        for section in self.sections:
            section.draw_wells(self.image, self.radius, self.image_size)

    def save_image(self, filename):
        img = Image.fromarray((self.image * 255).astype(np.uint8), 'L')
        img.save(filename)
        print(f"Imagem salva como {filename}")

    def get_image_matrix(self):
        return self.image

    def draw_circle(self, image, center, radius, value):
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = value

    def draw_only_circle(self):
        # Create an empty image
        image = np.zeros((self.image_size, self.image_size))
        # Draw the background circle
        self.draw_circle(image, (self.image_size // 2, self.image_size // 2), int(self.radius * self.image_size / (2 * self.radius)), self.circle_value)
        return image


class DerenzoInnerSection:
    def __init__(self, phantom_radius, well_count, well_separation, well_diameter, section_offset=0.1, well_value=1.0):
        
        """ This Class will be responsible for drawing each section in the derenzo phantom.

        Args:
            phantom_radius (float): Phantom radius
            well_count (_type_): _description_
            well_separation (_type_): _description_
            well_diameter (_type_): _description_
            section_offset (float, optional): _description_. Defaults to 0.1.
            well_value (float, optional): _description_. Defaults to 1.0.
        """
        self.R = phantom_radius
        self.well_count = well_count
        self.well_sep = well_separation
        self.well_dia = well_diameter
        self.r = self.well_dia / 2.0
        self.section_offset = self.R * section_offset
        self.well_value = well_value
        self.place_wells_in_section()
        self.label_xy = np.array((0, -1.1 * self.R))

    @property
    def row_height(self):
        return self.well_sep * np.sqrt(3)

    @property
    def num_rows(self):
        h_section = self.R - (2 * self.section_offset + self.well_sep)
        return int(np.floor(h_section / self.row_height))

    @property
    def num_wells(self):
        return self.well_count

    @property
    def well_area(self):
        return np.pi * self.r**2

    @property
    def total_area(self):
        return self.num_wells * self.well_area

    @property
    def label(self):
        return "%.1f mm" % (self.well_sep)

    def place_wells_in_section(self):
        if self.num_rows <= 1:
            self.section_offset = 0.0
            if self.num_rows <= 1:
                warnings.warn(("Cannot fit multiple features in section with "
                               "feature size = %s" % (self.well_sep)))
        xs, ys = [], []
        for i in range(self.num_rows):
            rn = i + 1
            for x in np.arange(-rn, rn, 2) + 1:
                xs.append(x * self.well_sep)
                ys.append(-(self.section_offset + self.row_height * rn))
        self.locs = np.vstack((xs, ys)).T

    def apply_rotation(self, deg):
        self.rot_angle = deg
        th = -1 * deg * (np.pi / 180)
        rot_mat = np.array([(np.cos(th), -np.sin(th)),
                            (np.sin(th),  np.cos(th))])
        self.locs = np.array([np.dot(l, rot_mat) for l in self.locs])
        self.label_xy = np.dot(self.label_xy, rot_mat)

    def draw_wells(self, image, phantom_radius, image_size):
        center = (image_size // 2, image_size // 2)
        scale = image_size / (2 * phantom_radius)
        for xy in self.locs:
            x, y = xy
            x = int(center[0] + x * scale)
            y = int(center[1] + y * scale)
            r = int(self.r * scale)
            self.draw_circle(image, (x, y), r, self.well_value)

    def draw_circle(self, image, center, radius, value):
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = value


class SpheresPhantomSection:
    
    def __inti__(self):
        size = 128
        dimension = 25.0  # mm
        spacing = dimension / size
        radius = 5
        sections = 5
        crua = np.zeros((size, size, size))
        radius_set = np.asarray([3.0, 2.5 , 2.0 , 1 , 1.0 ])/2
        profundidade = 12.5
            
    def draw_circle(image, spacing, radius=1, center=(10, 0), value=10):
        size = image.shape
        circle = np.zeros(size)
        for i in range(size[1]):
            for j in range(size[0]):
                x = (i - size[1] / 2) * spacing - center[0]
                y = -(j - size[0] / 2) * spacing - center[1]
                r = (x ** 2 + y ** 2) ** 0.5
                if r <= radius:
                    circle[j, i] = value
        return image + circle

    def rotate_point(point, sections):
        point = np.asarray(point)
        angles = np.linspace(0, np.deg2rad(360), sections, endpoint=False)
        rotations = [np.array([(np.cos(th), -np.sin(th)), (np.sin(th), np.cos(th))]) for th in angles]
        positions = [np.dot(rot, point) for rot in rotations]
        return positions

    def generate_sphere_phantom(image, profundidade, spacing, radius_set, sections, radius):
        point = (radius, 0)
        positions = rotate_point(point, sections)
    
        image_center = np.asarray(image.shape) / 2

        for i, rad in enumerate(radius_set):
            for fatia in range(image.shape[0]):
                lim_sup_esf = image_center[0]*spacing - profundidade + rad
                lim_inf_esf = image_center[0]*spacing - profundidade - rad
                centro_esfera = image_center[0]*spacing - profundidade 
                if (image_center[0] - fatia) * spacing < lim_sup_esf and (image_center[0] - fatia) * spacing > lim_inf_esf:
                    s = (rad ** 2 - (((image_center[0] - fatia) * spacing)-centro_esfera )** 2)** 0.5
                    print(s)
                    image[fatia, :, :] = draw_circle(image[fatia, :, :], spacing, s, center=positions[i], value=10000)
        print (image.max())
        return image  # Return the modified image




if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    # Teste do código adaptado
    radius = 29.0 / 2
    num_sections = 4
    well_counts = (6, 6, 10, 15)
    well_diameters = (2.5, 2.0, 1.5, 1.0)
    well_separations = (2.5, 2.0, 1.5, 1.0)
    section_offsets = (0.00000001, 0.01, 0.01, 0.1)  # Offset personalizado para cada seção
    circle_value = 0.2
    well_value = 0.8
    image_size = 1024

    my_phantom = DerenzoPhantomSlice(radius, num_sections, well_counts, well_diameters, well_separations, section_offsets,
                                image_size=image_size, circle_value=circle_value, well_value=well_value)
    my_phantom.save_image('derenzo_phantom_with_offsets.png')

    # Obter a matriz da imagem contendo apenas o círculo de fundo
    circle_image_matrix = my_phantom.draw_only_circle()

    # Plotar a primeira imagem
    plt.subplot(1, 2, 1)
    plt.imshow(circle_image_matrix, cmap='gray')
    plt.title('Círculo de Fundo')
    plt.colorbar()

    # Obter a matriz completa do fantoma
    phantom_matrix = my_phantom.get_image_matrix()

    # Plotar a segunda imagem
    plt.subplot(1, 2, 2)
    plt.imshow(phantom_matrix, cmap='gray')
    plt.title('Fantoma Completo')
    plt.colorbar()

    # Ajustar layout para exibir as imagens lado a lado
    plt.tight_layout()
    plt.show()
