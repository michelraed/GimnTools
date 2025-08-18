# GimnTools

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gimntools.readthedocs.io/)
[![PyPI Version](https://img.shields.io/pypi/v/gimntools.svg)](https://pypi.org/project/gimntools/)

##  Introdução e Visão Geral

A GimnTools é uma biblioteca Python de código aberto projetada para o processamento de imagens médicas e reconstrução tomográfica. Ela oferece um conjunto de ferramentas para manipulação de imagens, geração de sinogramas, aplicação de filtros e implementação de algoritmos de reconstrução. O objetivo desta documentação é fornecer um guia completo para desenvolvedores e pesquisadores que desejam utilizar a GimnTools em seus projetos.

## ✨ Principais Funcionalidades

### 🔬 Reconstrução Tomográfica
- **Algoritmos de Reconstrução**: MLEM, OSEM, FBP (Filtered Back-Projection)
- **Reconstrução por Integrais de Linha**: Implementação otimizada com Numba
- **Matriz de Sistema**: Suporte para reconstrução baseada em matriz de sistema
- **Correções**: Atenuação, dispersão e normalização

### 📊 Processamento de Sinogramas
- **Geração de Sinogramas**: Suporte para cristais monolíticos e segmentados
- **Conversão de Coordenadas**: Transformação entre espaços de cristais e pixels
- **Configuração de Sistema**: Gerenciamento flexível de geometrias de detectores

### 🖼️ Processamento de Imagens
- **Filtros Espaciais**: Convolução, filtros de vizinhança otimizados
- **Filtros de Frequência**: Transformada de Fourier e filtragem espectral
- **Filtros Morfológicos**: Operações de abertura, fechamento, erosão e dilatação
- **Interpolação**: Métodos de reconstrução e reamostragem

### 🎯 Fantomas e Simulação
- **Fantoma de Derenzo**: Geração paramétrica de fantomas de teste
- **Geometrias Customizadas**: Criação de formas geométricas diversas
- **Fantomas Aleatórios**: Geração procedural para testes variados

### 📁 I/O de Imagens Médicas
- **DICOM**: Leitura e escrita usando SimpleITK e ITK
- **Normalização**: Conversão entre tipos de dados e escalas
- **Metadados**: Preservação de informações de espaçamento, origem e orientação

## 🚀 Instalação Rápida

### Requisitos
- Python 3.8 ou superior
- Sistema operacional: Windows, macOS ou Linux

### Instalação via pip
```bash
pip install GimnTools
```

### Instalação para Desenvolvimento
```bash
git clone https://github.com/usuario/GimnTools.git
cd GimnTools
pip install -e ".[dev]"
```

### Usando Scripts de Instalação
```bash
# Clone o repositório
git clone https://github.com/usuario/GimnTools.git
cd GimnTools

# Execute o script de instalação
python scripts/install.py --mode development --jupyter

# Ou use o Makefile (Linux/macOS)
make install-dev
```

## 🛠️ Scripts de Instalação

A biblioteca inclui scripts automatizados para facilitar o desenvolvimento:

### Scripts Disponíveis


#### Script de Instalação (`scripts/install.py`)
```bash
# Instalação básica
python scripts/install.py

# Instalação para desenvolvimento
python scripts/install.py --mode development --jupyter --docs

# Criar ambiente virtual
python scripts/install.py --venv
```


### Usando Makefile (Linux/macOS)

```bash
# Instalar para usuario	
make install

# Instalar para desenvolvimento
make install-dev

# Executar testes
make test

# Build completo
make build

# Deploy para PyPI de teste
make upload

# Limpar arquivos temporários
make clean
```

## 📁 Estrutura do Projeto

```
GimnTools/
├── GimnTools/               # Pacote principal
│   ├── __init__.py
│   └── ImaGIMN/            # Módulo principal
│       ├── image.py         # Classe base para imagens
│       ├── IO/             # Entrada e saída
│       │   ├── MIIO.py     # I/O de imagens médicas
│       │   └── GimnIO.py   # I/O geral
│       ├── sinogramer/     # Geração de sinogramas
│       │   ├── sinogramer.py
│       │   ├── conf.py
│       │   └── systemSpace.py
│       ├── gimnRec/        # Reconstrução
│       │   ├── reconstructors/
│       │   ├── backprojectors.py
│       │   ├── projectors.py
│       │   └── corrections.py
│       ├── processing/     # Processamento
│       │   ├── filters/    # Filtros
│       │   ├── interpolators/
│       │   ├── tools/      # Ferramentas matemáticas
│       │   └── ploter/     # Visualização
│       └── phantoms/       # Fantomas
│           ├── derenzzo.py
│           ├── random.py
│           └── geometries/
├── scripts/                # Scripts de automação
│   ├── build.py           # Script de build
│   ├── install.py         # Script de instalação
│   └── deploy.py          # Script de deploy
├── docs/                  # Documentação
├── tests/                 # Testes
├── examples/              # Exemplos
├── setup.py              # Configuração setuptools
├── pyproject.toml        # Configuração moderna
├── requirements.txt      # Dependências
├── requirements-dev.txt  # Dependências dev
├── Makefile             # Automação (Unix)
└── README.md            # Este arquivo
```

## 🔧 Dependências

### Principais
- **numpy** (≥1.20.0): Computação numérica
- **scipy** (≥1.7.0): Algoritmos científicos
- **matplotlib** (≥3.3.0): Visualização
- **numba** (≥0.54.0): Compilação JIT para performance
- **SimpleITK** (≥2.1.0): Processamento de imagens médicas
- **itk** (≥5.2.0): Toolkit de imagens
- **Pillow** (≥8.0.0): Manipulação de imagens
- **scikit-image** (≥0.18.0): Processamento de imagens
- **h5py** (≥3.1.0): Armazenamento HDF5
- **tqdm** (≥4.60.0): Barras de progresso

### Dependências de Desenvolvimento
- **pytest**: Framework de testes
- **sphinx**: Geração de documentação
- **black**: Formatação de código
- **flake8**: Linting
- **mypy**: Verificação de tipos

## Documentação  dos Módulos

### 1. `ImaGIMN.IO`

Responsável pelas operações de entrada e saída.

*   **`MIIO.py`**: Classe de alto nível para manipulação de I/O de imagens médicas, especialmente DICOM. Usa `SimpleITK` e `ITK` como backend. Fornece funcionalidades essenciais como leitura, escrita e normalização de imagens.
*   **`GimnIO.py`**: Utilitários de I/O de baixo nível, como criação de pastas e salvamento de arquivos JSON.

### 2. `ImaGIMN.gimnRec`

Contém os algoritmos e componentes para a reconstrução tomográfica.

*   **`reconstruction_filters.py`**: Implementa filtros usados na reconstrução FBP (Filtered Back-Projection), como Ram-Lak, Shepp-Logan, etc.
*   **`projectors.py`**: Contém funções para a projeção (forward projection), que simula a aquisição de dados (cria um sinograma a partir de uma imagem).
*   **`backprojectors.py`**: Contém funções para a retroprojeção (back-projection), o processo inverso da projeção, usado em algoritmos como FBP e iterativos.
*   **`corrections.py`**: Funções para aplicar correções necessárias durante a reconstrução, como a correção do centro de rotação.

#### 2.1. `ImaGIMN.gimnRec.reconstructors`

Implementações concretas de algoritmos de reconstrução.

*   **`line_integral_reconstructor.py`**: Reconstrutor principal que implementa algoritmos baseados em integral de linha, como FBP, MLEM e OSEM. É uma classe versátil e de uso geral.
*   **`rotation_reconstruction.py`**: Implementação que realiza a reconstrução baseada em rotações sucessivas da imagem. Serve como uma classe base ou alternativa para os reconstrutores.
*   **`system_matrix_reconstruction.py`**: Reconstrutor que utiliza uma matriz de sistema pré-calculada para realizar as projeções e retroprojeções. Pode ser mais lento para gerar a matriz, mas mais rápido para iterar. Versão para CPU.
*   **`system_matrix_reconstruction2.py`**: Uma segunda implementação (possivelmente experimental ou com otimizações diferentes) do reconstrutor baseado em matriz de sistema.

### 3. `ImaGIMN.processing`

Ferramentas para processamento e análise de imagens.

#### 3.1. `ImaGIMN.processing.filters`

*   **`spatial_filters.py`**: Contém filtros que operam diretamente na matriz de pixels. Inclui funções para convolução 2D, aplicação de filtros separáveis e extração de vizinhanças de pixels.
*   **`frequency_filters.py`**: Implementa filtros no domínio da frequência, como o filtro Gaussiano e o Butterworth. Útil para suavização e realce de bordas.
*   **`morphological_filters.py`**: Funções para morfologia matemática (Erosão, Dilatação, Abertura e Fechamento), usadas para análise de formas e estruturas na imagem.

#### 3.2. `ImaGIMN.processing.interpolators`

*   **`reconstruction.py`**: Fornece diferentes métodos de interpolação (Vizinho Mais Próximo, Bilinear, Spline Beta) que são cruciais para algoritmos de rotação e reconstrução, garantindo a precisão ao acessar valores de pixels em coordenadas não inteiras.

#### 3.3. `ImaGIMN.processing.tools`

*   **`kernels.py`**: Funções para gerar kernels (matrizes) usados em filtros de convolução, como kernels Gaussianos e Butterworth de diferentes tipos (passa-baixa, passa-alta).
*   **`math.py`**: Funções matemáticas essenciais, como convolução e deconvolução no domínio da frequência (via FFT) e rotação de imagem com diferentes interpoladores.
*   **`utils.py`**: Utilitários diversos, com destaque para a função de redimensionamento de imagem (`resize`) que suporta múltiplos métodos de interpolação.

#### 3.4. `ImaGIMN.processing.ploter`

*   **`ploter.py`**: Ferramenta de visualização para plotar múltiplos slices de um volume 3D (como um sinograma ou uma imagem reconstruída) em uma grade, facilitando a inspeção visual.

### 4. `ImaGIMN.phantoms`

*   **`derenzzo.py`**: Contém classes para a geração de phantoms de Derenzo, que são padrões de teste padrão em imagem médica para avaliar a resolução espacial de um sistema.


##  Exemplos de Uso Práticos

### Exemplo 1: Criação de imagens Aleatórias

```python
  
from GimnTools.ImaGIMN.phantoms.random import generate_random_image

"""Generates a random image for testing purposes."""
import matplotlib.pyplot as plt
pixels  = 128 # size off the image
number_of_figures = 10 # number of figures in the image

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


```

### Exemplo 2: Teste Shepp-Logan Phantom

```python
from skimage.data import shepp_logan_phantom

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

```

### Exemplo 3: Gerar Fantoma de Derenzo

```python

from GimnTools.ImaGIMN.phantoms.derenzzo import DerenzoPhantomSlice

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
```

### Exemplo 3: Reconstrução usando reconstrutor com projetor baseado na rotação da imagem

```python

from GimnTools.ImaGIMN.gimnRec.reconstructors.rotation_reconstruction import rotation_reconstructor
from GimnTools.ImaGIMN.processing.interpolators.reconstruction import beta_spline_interpolation, bilinear_interpolation, nearest_neighbor_interpolation
from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import ramLak
from GimnTools.ImaGIMN.gimnRec.projectors import radon_m, projector, direct_radon



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
        

```

### Exemplo 3: Reconstrutor baseado na integral da linha de projeção

```python
from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor
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
```

### Exemplo 3: Reconstrutor System Matrix

```python

from GimnTools.ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction import reconstructor_system_matrix_cpu
import matplotlib.pyplot as plt
pixels = 128  # Size of the image
number_of_angles = pixels  # Number of angles for the sinogram
number_of_slices = 2  # Number of slices in the stack

# Generate a derenzo phantom image with noise
noisy_image = derenzo_phantom_test(noise_factor=0.3)
angles = np.linspace(0, 180, number_of_angles, endpoint=True)
sinogram = direct_radon(noisy_image, angles)
stack = np.asarray([sinogram for _ in range(number_of_slices)])


# neste reconstrutor os angulos tem que ser invertidos linspace(180,0)
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

```


##  API Reference

### `ImaGIMN.IO.MIIO`

| Classe/Método | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `MIIO` | Classe para I/O de imagens DICOM. | - | `MIIO object` |
| `renormalize` | Normaliza a imagem para um tipo de dado. | `image` (ndarray), `dtype` (type) | `tuple` (ndarray, dict) |
| `save_dicom` | Salva um array como arquivo DICOM. | `image` (ndarray), `nome_arquivo` (str), `origin` (tuple), `spacing` (tuple), `save_json` (bool) | - |

### `ImaGIMN.phantoms.derenzzo.DerenzoPhantomSlice`

| Método/Propriedade | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `__init__` | Inicializa o phantom. | `radius`, `num_sections`, `well_counts`, `well_diameters`, `well_separations`, `section_offsets`, etc. | - |
| `save_image` | Salva o phantom como imagem. | `filename` (str) | - |
| `get_image_matrix` | Retorna a matriz de pixels do phantom. | - | `numpy.ndarray` |

### `ImaGIMN.processing.interpolators.reconstruction`

| Função | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `beta_spline_interpolation` | Interpolação Spline Beta. | `p00`, `p01`, `p10`, `p11` (float), `dx` (float), `dy` (float) | `float` |
| `bilinear_interpolation` | Interpolação Bilinear. | `p00`, `p01`, `p10`, `p11` (float), `dx` (float), `dy` (float) | `float` |
| `nearest_neighbor_interpolation` | Interpolação do Vizinho Mais Próximo. | `p00`, `p01`, `p10`, `p11` (float), `dx` (float), `dy` (float) | `float` |

### `ImaGIMN.processing.tools.kernels`

| Função | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `generate_gaussian_kernel_in_mm`| Gera um kernel Gaussiano. | `size` (int), `sigma_mm` (float), `pixel_size_mm` (float) | `ndarray` |
| `butterworth_kernel` | Gera um kernel Butterworth. | `size` (int), `order` (int), `cutoff` (float), `pixel_size_mm` (float) | `ndarray` |
| `butterworth_kernel_high_pass`| Gera um kernel Butterworth passa-alta. | `size` (int), `order` (int), `cutoff` (float), `pixel_size_mm` (float) | `ndarray` |
| `butterworth_kernel_low_pass` | Gera um kernel Butterworth passa-baixa.| `size` (int), `order` (int), `cutoff` (float), `pixel_size_mm` (float) | `ndarray` |

### `ImaGIMN.processing.tools.math`

| Função | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `deconvolution` | Deconvolução no domínio da frequência. | `image` (ndarray), `function` (ndarray) | `ndarray` |
| `convolution` | Convolução no domínio da frequência. | `image` (ndarray), `kernel` (ndarray) | `ndarray` |
| `rotate` | Rotaciona uma imagem. | `image` (ndarray), `angle` (float), `interpolation_func` (function), `center` (tuple) | `ndarray` |

### `ImaGIMN.processing.tools.utils`

| Função | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `resize` | Redimensiona uma imagem. | `input_img` (ndarray), `mx` (int), `my` (int), `interpolation` (str) | `ndarray` |

### `ImaGIMN.processing.ploter.ploter`

| Função | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `plot_slices` | Plota slices de um volume 3D. | `sinogram` (ndarray), `n_slices` (int), `x_extent` (tuple), `y_extent` (tuple), `rows` (int), `cols` (int), etc. | - |

### `ImaGIMN.gimnRec.reconstructors.rotation_reconstruction.rotation_reconstructor`

| Método | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `mlem` | Reconstrução MLEM baseada em rotação. | `iterations` (int), `interpolation` (function), `angles` (ndarray), `verbose` (bool) | `ndarray` |
| `osem` | Reconstrução OSEM baseada em rotação. | `iterations` (int), `subsets_n` (int), `interpolation` (function), `angles` (ndarray), `verbose` (bool) | `ndarray` |

### `ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction2.reconstructor_system_matrix_cpu`

*Nota: Esta classe herda de `line_integral_reconstructor`, então possui os mesmos métodos base.*

| Método | Descrição | Parâmetros | Retorno |
| --- | --- | --- | --- |
| `mlem_tv` | MLEM com regularização Total Variation. | `iterations` (int), `beta` (float), `angles` (ndarray), `verbose` (bool) | `ndarray` |


## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o projeto
2. **Clone** seu fork: `git clone https://github.com/seu-usuario/GimnTools.git`
3. **Crie uma branch**: `git checkout -b feature/nova-funcionalidade`
4. **Instale para desenvolvimento**: `make install-dev`
5. **Faça suas alterações** e adicione testes
6. **Execute testes**: `make test`
7. **Formate o código**: `make format`
8. **Commit**: `git commit -m "Adiciona nova funcionalidade"`
9. **Push**: `git push origin feature/nova-funcionalidade`
10. **Abra um Pull Request**

### Diretrizes de Contribuição
- Siga os padrões de código (black, flake8)
- Adicione testes para novas funcionalidades
- Atualize a documentação quando necessário
- Use mensagens de commit descritivas

## 📄 Licença

Este projeto está licenciado sob a licença Apache 2.0. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

- **Issues**: [https://github.com/usuario/GimnTools/issues](https://github.com/michelraed/GimnTools/issues/2)
- **Discussões**:[https://github.com/usuario/GimnTools/discussions](https://github.com/michelraed/GimnTools/discussions/1)
- **Email**: michelraed@gmail.com




**GimnTools** - Ferramentas para reconstrução e processamento de imagens médicas em Python.




