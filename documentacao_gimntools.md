# Documentação Técnica da Biblioteca GimnTools

**Autor**: Michel Raed
**Data**: 2025-06-19

## 1. Introdução e Visão Geral

A GimnTools é uma biblioteca Python de código aberto projetada para o processamento de imagens médicas e reconstrução tomográfica. Ela oferece um conjunto de ferramentas robustas para manipulação de imagens, geração de sinogramas, aplicação de filtros e implementação de algoritmos de reconstrução. O objetivo desta documentação é fornecer um guia completo e exaustivo para desenvolvedores e pesquisadores que desejam utilizar a GimnTools em seus projetos.

### Principais Funcionalidades

*   **Manipulação de Imagens**: A classe `image` permite carregar, salvar e manipular imagens médicas, incluindo metadados como espaçamento e origem.
*   **Geração de Sinogramas**: O módulo `sinogramer` oferece funcionalidades para criar sinogramas a partir de dados de projeção, com suporte para diferentes configurações de detectores.
*   **Reconstrução Tomográfica**: A biblioteca implementa algoritmos de reconstrução como MLEM, OSEM e FBP, disponíveis no módulo `gimnRec`.
*   **Processamento de Imagem**: Inclui uma variedade de filtros espaciais, de frequência e morfológicos, além de ferramentas de interpolação e utilitários matemáticos.
*   **Geração de Phantoms**: Ferramentas para criar phantoms digitais, como o Derenzo, para testes e calibração de algoritmos.
*   **Entrada e Saída (I/O)**: Suporte para leitura e escrita de imagens no formato DICOM, com funcionalidades de normalização.

## 2. Instalação e Requisitos

Para instalar a GimnTools, é recomendável o uso de um ambiente virtual para evitar conflitos de dependência.

```bash
python -m venv gimnenv
source gimnenv/bin/activate
```

A instalação da biblioteca pode ser feita diretamente a partir do código-fonte:

```bash
git clone <URL_DO_REPOSITORIO_GIMNTOOLS>
cd GimnTools
pip install .
```

### Requisitos

A GimnTools depende das seguintes bibliotecas Python:

*   `numpy`
*   `SimpleITK`
*   `itk`
*   `numba`
*   `matplotlib`
*   `Pillow`

## 3. Guia de Início Rápido

Este guia apresenta um exemplo simples de como carregar uma imagem, gerar um sinograma e reconstruí-la.

```python
from GimnTools.ImaGIMN.image import image
from GimnTools.ImaGIMN.sinogramer.sinogramer import Sinogramer
from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor
import numpy as np

# 1. Carregar ou criar uma imagem de phantom
# (O ideal é carregar um phantom ou imagem real)
from GimnTools.ImaGIMN.phantoms.derenzzo import DerenzoPhantomSlice
phantom = DerenzoPhantomSlice(radius=64, num_sections=4, well_counts=(4,6,8,10), well_diameters=(8,6,4,2), well_separations=(16,12,8,4), section_offsets=(0.2,0.2,0.2,0.2), image_size=128)
pixels_phantom = phantom.get_image_matrix()

# 2. Gerar projeções (simuladas aqui com a transformada de Radon do scikit-image, por exemplo)
# A GimnTools espera um sinograma como entrada para a reconstrução.
# Esta etapa normalmente viria de um scanner real ou de um simulador mais complexo.
from skimage.transform import radon
angles = np.linspace(0., 180., 180, endpoint=False)
sinogram = radon(pixels_phantom, theta=angles, circle=True)
sinogram = np.expand_dims(sinogram.T, axis=0) # Formato (slices, distancias, angulos)


# 3. Reconstruir a imagem usando OSEM
reconstructor = line_integral_reconstructor(sinogram)
angles_rad = np.deg2rad(angles)
reconstructed_image_obj = reconstructor.osem(iterations=10, subsets_n=4, angles=angles_rad)

# 4. Salvar a imagem reconstruída
# O método osem retorna a classe 'image', então podemos salvar diretamente
reconstructed_image_obj.save_dicom("imagem_reconstruida.dcm")

print("Processo concluído. Imagem reconstruída salva.")
```

## 4. Documentação Detalhada dos Módulos

### 4.1. `ImaGIMN.IO`

Responsável pelas operações de entrada e saída.

*   **`MIIO.py`**: Classe de alto nível para manipulação de I/O de imagens médicas, especialmente DICOM. Usa `SimpleITK` e `ITK` como backend. Fornece funcionalidades essenciais como leitura, escrita e normalização de imagens.
*   **`GimnIO.py`**: Utilitários de I/O de baixo nível, como criação de pastas e salvamento de arquivos JSON.

### 4.2. `ImaGIMN.gimnRec`

Contém os algoritmos e componentes para a reconstrução tomográfica.

*   **`reconstruction_filters.py`**: Implementa filtros usados na reconstrução FBP (Filtered Back-Projection), como Ram-Lak, Shepp-Logan, etc.
*   **`projectors.py`**: Contém funções para a projeção (forward projection), que simula a aquisição de dados (cria um sinograma a partir de uma imagem).
*   **`backprojectors.py`**: Contém funções para a retroprojeção (back-projection), o processo inverso da projeção, usado em algoritmos como FBP e iterativos.
*   **`corrections.py`**: Funções para aplicar correções necessárias durante a reconstrução, como a correção do centro de rotação.

#### 4.2.1. `ImaGIMN.gimnRec.reconstructors`

Implementações concretas de algoritmos de reconstrução.

*   **`line_integral_reconstructor.py`**: Reconstrutor principal que implementa algoritmos baseados em integral de linha, como FBP, MLEM e OSEM. É uma classe versátil e de uso geral.
*   **`rotation_reconstruction.py`**: Implementação que realiza a reconstrução baseada em rotações sucessivas da imagem. Serve como uma classe base ou alternativa para os reconstrutores.
*   **`system_matrix_reconstruction.py`**: Reconstrutor que utiliza uma matriz de sistema pré-calculada para realizar as projeções e retroprojeções. Pode ser mais lento para gerar a matriz, mas mais rápido para iterar. Versão para CPU.
*   **`system_matrix_reconstruction2.py`**: Uma segunda implementação (possivelmente experimental ou com otimizações diferentes) do reconstrutor baseado em matriz de sistema.

### 4.3. `ImaGIMN.processing`

Ferramentas para processamento e análise de imagens.

#### 4.3.1. `ImaGIMN.processing.filters`

*   **`spatial_filters.py`**: Contém filtros que operam diretamente na matriz de pixels. Inclui funções para convolução 2D, aplicação de filtros separáveis e extração de vizinhanças de pixels.
*   **`frequency_filters.py`**: Implementa filtros no domínio da frequência, como o filtro Gaussiano e o Butterworth. Útil para suavização e realce de bordas.
*   **`morphological_filters.py`**: Funções para morfologia matemática (Erosão, Dilatação, Abertura e Fechamento), usadas para análise de formas e estruturas na imagem.

#### 4.3.2. `ImaGIMN.processing.interpolators`

*   **`reconstruction.py`**: Fornece diferentes métodos de interpolação (Vizinho Mais Próximo, Bilinear, Spline Beta) que são cruciais para algoritmos de rotação e reconstrução, garantindo a precisão ao acessar valores de pixels em coordenadas não inteiras.

#### 4.3.3. `ImaGIMN.processing.tools`

*   **`kernels.py`**: Funções para gerar kernels (matrizes) usados em filtros de convolução, como kernels Gaussianos e Butterworth de diferentes tipos (passa-baixa, passa-alta).
*   **`math.py`**: Funções matemáticas essenciais, como convolução e deconvolução no domínio da frequência (via FFT) e rotação de imagem com diferentes interpoladores.
*   **`utils.py`**: Utilitários diversos, com destaque para a função de redimensionamento de imagem (`resize`) que suporta múltiplos métodos de interpolação.

#### 4.3.4. `ImaGIMN.processing.ploter`

*   **`ploter.py`**: Ferramenta de visualização para plotar múltiplos slices de um volume 3D (como um sinograma ou uma imagem reconstruída) em uma grade, facilitando a inspeção visual.

### 4.4. `ImaGIMN.phantoms`

*   **`derenzzo.py`**: Contém classes para a geração de phantoms de Derenzo, que são padrões de teste padrão em imagem médica para avaliar a resolução espacial de um sistema.

## 5. Exemplos de Uso Práticos

### Exemplo 1: Filtros Morfológicos (Abertura)

```python
from GimnTools.ImaGIMN.image import image
from GimnTools.ImaGIMN.processing.filters.morphological_filters import doOpen
import numpy as np

# Criar uma imagem de exemplo com ruído
img_data = np.zeros((128, 128))
img_data[30:100, 30:100] = 1 # Objeto principal
# Adicionar ruído de sal
noise = np.random.randint(0, 128, size=(20, 2))
img_data[noise[:, 0], noise[:, 1]] = 1

# Aplicar o filtro de abertura para remover pequenos ruídos
opened_image = doOpen(img_data)

# Salvar o resultado
img_obj = image(opened_image)
img_obj.save_dicom("imagem_aberta.dcm")
print("Filtro de abertura aplicado e imagem salva.")
```

### Exemplo 2: Reconstrução com Matriz de Sistema (MLEM)

```python
from GimnTools.ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction import reconstructor_system_matrix_cpu, system_matrix
import numpy as np

# Supondo que 'sino_slice' é um slice 2D de um sinograma (distancias, angulos)
# e 'angles_deg' é o array de ângulos em graus.
# Ex: sino_slice = sinogram[0, :, :]
# Ex: angles_deg = np.linspace(0., 180., 180, endpoint=False)

# --- Bloco de criação de dados de exemplo --- 
from skimage.transform import radon, iradon
from GimnTools.ImaGIMN.phantoms.derenzzo import DerenzoPhantomSlice
phantom = DerenzoPhantomSlice(radius=64, num_sections=4, well_counts=(4,6,8,10), well_diameters=(8,6,4,2), well_separations=(16,12,8,4), section_offsets=(0.2,0.2,0.2,0.2), image_size=128)
pixels_phantom = phantom.get_image_matrix()
angles_deg = np.linspace(0., 180., 180, endpoint=False)
sinogram_sk = radon(pixels_phantom, theta=angles_deg, circle=True)
sino_slice = sinogram_sk.T
# --- Fim do bloco de exemplo ---


rec = reconstructor_system_matrix_cpu(sinogram=np.expand_dims(sino_slice, axis=0))
angles_rad = np.deg2rad(angles_deg)

reconstructed_img = rec.mlem(iterations=5, angles=angles_rad, verbose=True)

# Salvar o resultado
from GimnTools.ImaGIMN.image import image
img_obj = image(reconstructed_img)
img_obj.save_dicom("imagem_mlem_sm.dcm")
print("Reconstrução MLEM com matriz de sistema concluída.")
```

### Exemplo 3: Redimensionar Imagem com Interpolação

```python
from GimnTools.ImaGIMN.image import image
from GimnTools.ImaGIMN.processing.tools.utils import resize

# Carregar a imagem
img_obj = image("caminho/para/sua/imagem.dcm")
original_pixels = img_obj.pixels[0] # Pegar um slice

# Redimensionar a imagem para 256x256 usando interpolação bilinear
resized_pixels = resize(original_pixels, 256, 256, 'Bilinear')

# Salvar o resultado
resized_obj = image(resized_pixels)
resized_obj.save_dicom("imagem_redimensionada.dcm")
print("Imagem redimensionada e salva.")
```

## 6. API Reference

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

## 7. Contribuições e Desenvolvimento

Agradecemos o interesse em contribuir para a GimnTools! Para contribuir, por favor, siga os seguintes passos:

1.  Faça um fork do repositório.
2.  Crie uma nova branch para a sua feature (`git checkout -b feature/nova-feature`).
3.  Faça o commit das suas alterações (`git commit -am 'Adiciona nova feature'`).
4.  Faça o push para a branch (`git push origin feature/nova-feature`).
5.  Abra um Pull Request.

