Aqui está o README reorganizado de forma lógica para uso acadêmico, mantendo todo o conteúdo original:

```markdown
# Sistema de Tomografia por Emissão de Pósitrons (PET)

## Componentes Principais

### 1. Função `generateRsectorAngle`
- **Objetivo**: Gera ângulos e índices de setores para detectores.
- **Parâmetros**: 
  - `detectors` (int): Número de detectores.
- **Retorno**:
  - `detector` (dict): Mapeia nomes de detectores (e.g., "detector-1") para ângulos em graus.
  - `rsectors` (dict): Mapeia nomes de detectores para índices de setores (0-based).
- **Funcionamento**:
  - Distribui ângulos uniformemente entre 0° e 360° (exclusivo).
  - Atribui índices sequenciais aos detectores.

---

### 2. Função `convert_crystal_to_xy` (Otimizada com Numba)
- **Objetivo**: Converte ID de cristal em coordenadas (x, y) em uma grade 8×8.
- **Parâmetros**: 
  - `crystal_id` (int): ID do cristal (0 a 63).
- **Retorno**: 
  - `np.array([x, y])` (int): Coordenadas x (coluna) e y (linha).
- **Lógica**:
  - **x**: Calculado como `7 - (crystal_id % 8)` (colunas invertidas).
  - **y**: Calculado como `crystal_id // 8` (linhas sequenciais).

---

### 3. Classe `SiPM`
**Objetivo**: Modela um arranjo de pixels SiPM com posicionamento preciso.

#### Inicialização:
```python
def __init__(self, cols, rows, PichX, PichY)
```
- **Parâmetros**:
  - `cols/rows`: Número de colunas/linhas na matriz SiPM
  - `PichX/PichY`: Espaçamento entre pixels (pitch) nos eixos X/Y
- **Atributos**:
  - `positionsX/positionsY`: Listas com coordenadas dos pixels

#### Método `initialize()`:
- Calcula posições dos pixels com lógica assimétrica:
  - **Eixo X**: 
    - Pixel central na origem (índices 3-4)
    - Deslocamentos negativos com ajuste fino (-PichX/2)
    - Deslocamentos positivos com ajuste fino (+PichX/2)
  - **Eixo Y**: 
    - Similar ao X mas com sinal invertido
    - Espelhamento vertical (valores Y negativos)

---

### 4. Função `DetectorReference` (Otimizada com Numba)
**Objetivo**: Converte coordenadas de evento para referencial do detector.

```python
@njit
def DetectorReference(event, crystal_size, pla_thickness, radius)
```
- **Parâmetros**:
  - `event`: Coordenadas 3D originais [x,y,z]
  - `crystal_size`: Dimensões do cristal [x,y,z]
  - `pla_thickness`: Espessura da camada PLA
  - `radius`: Raio do bore do scanner

- **Processamento**:
  1. Inverte ordem das coordenadas (`np.flip`)
  2. Aplica offset de metade do cristal no eixo X
  3. Calcula centro do detector considerando raio + PLA
  4. Retorna posição final no referencial do detector

---

### 5. Classe `Sinogramer`
Gerencia criação e manipulação de sinogramas.

#### Inicialização
```python
def __init__(self, system_conf, matrix=np.asarray([24, 32, 32]))
```
- **Parâmetros**:
  - `system_conf` (dict): Configuração do sistema (e.g., tipo de cristal, dimensões).
  - `matrix` (np.array): Dimensões padrão do sinograma.
- **Comportamento**:
  - **Cristais Monolíticos**: Inicializa sinograma com dimensões padrão.
  - **Cristais Segmentados**: Calcula dimensões dinamicamente com base em:
    - `n_angles = (2 * sipm_pixel_n[0] - 1) * rotation_angles`
    - `n_distances = (2 * sipm_pixel_n[0] - 1)`
    - `n_slices = (2 * sipm_pixel_n[1] - 1)`

#### Métodos Principais

##### `get_sinogram()`
- Retorna o array do sinograma atual.

##### `set_matrix(matrix)`
- Redefine as dimensões do sinograma e reinicializa com zeros.

##### `get_slice_mono(pair_pos_axis)` / `get_slice_seg(pair_pos_axis)`
- Calculam o índice do slice (fatia) no sinograma:
  - **Monolítico**: Normaliza posições do eixo z usando dimensões do cristal.
  - **Segmentado**: Soma direta das posições do eixo z.

##### `get_pixel_position(distance, angle)`
- Mapeia distância e ângulo para coordenadas (x, y) no sinograma:
  - **Ângulo → y**: Proporcional ao ângulo total (`arround`).
  - **Distância → x**: Normalizado entre `[-crystal_size/2, +crystal_size/2]`.

##### `fill_sino(distance, angle, slice_z)`
- Incrementa o sinograma na posição calculada por:
  - `slice_z`: Índice da fatia.
  - `get_pixel_position()`: Posição (x, y) no plano.

##### `global_to_sinogram(p1, p2, rotate_along)`
- Converte coordenadas 3D de dois pontos (`p1`, `p2`) em parâmetros de sinograma:
  - **Entrada**:
    - `p1`, `p2`: Pontos 3D (x, y, z).
    - `rotate_along`: Eixo de rotação ("x", "y", ou "z").
  - **Saída**:
    - `r`: Distância do centro à linha entre os pontos.
    - `phi`: Ângulo de projeção em graus.
  - **Cálculo**:
    - Usa ponto médio e inclinação da linha no plano perpendicular ao eixo de rotação.

---

### 6. Classe `systemSpace` (Herda de `Sinogramer` e `SiPM`)
**Objetivo**: Gerencia geometria do sistema e transformações de coordenadas.

#### Inicialização:
```python
def __init__(detector_angles, rotate_along, detector_parameters)
```
- **Parâmetros**:
  - `detector_angles`: Ângulos de posicionamento dos detectores
  - `rotate_along`: Eixo principal de rotação do gantry
  - `detector_parameters`: Configurações físicas do sistema

- **Funcionalidades**:
  - Inicializa rotações individuais dos detectores
  - Configura rotação global do gantry (inicialmente 0°)
  - Herda e inicializa funcionalidades de sinograma e SiPM

#### Métodos Principais:

##### Transformações de Coordenadas:
| Método | Descrição |
|--------|-----------|
| `getGlobalPreRotation()` | Retorna posição após rotação do detector |
| `getGlobalRotation()` | Aplica rotação do gantry |
| `getGlobalPosition()` | Posição final após todas as transformações |

##### Controle de Rotação:
| Método | Ação |
|--------|------|
| `rotateGantry(angle)` | Incrementa rotação do gantry |
| `rotateBy(angle)` | Define rotação absoluta do gantry |
| `getAngleOfRotation()` | Retorna ângulo atual do gantry |

##### Geração de Sinograma:
```python
def CalculateSinogram(event1, det1, event2, det2)
```
1. Converte eventos para coordenadas globais
2. Calcula parâmetros do sinograma:
   - Distância radial (`distance`)
   - Ângulo de projeção (`angle`)
3. Retorna tupla `(distance, angle)`

---

---

## Configuração do Sistema

### 1. Configuração Angular dos Detectores
```python
detectors_angles = {
    "detector-2": 0,     # Posicionado a 0°
    "detector-1": -180   # Posicionado a 180° oposto (equivalente a 180°)
}
```

### 2. Mapeamento de Setores (Rsectors)
```python
rsectors = {
    0: "detector-1",  # Setor 0 associado ao detector-1
    1: "detector-2"   # Setor 1 associado ao detector-2
}
```

### 3. Configuração Física do Detector (`detector_config`)

#### Parâmetros Geométricos Principais
| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `pla_thickness` | 2.0 mm | Espessura da camada PLA |
| `radius` | 25 mm | Raio do bore do scanner |
| `crystal_size` | [20, 26.88, 26.88] mm | Dimensões do cristal (X,Y,Z) |
| `sipm_pitch` | [3.36, 3.36] mm | Espaçamento entre pixels SiPM (X,Y) |
| `sipm_pixel_n` | [8, 8] | Matriz de pixels SiPM 8×8 |

#### Parâmetros Operacionais
| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `data_from` | "gate" | Origem dos dados (simulação GATE) |
| `crystal_type` | "segmented" | Tipo de cristal (segmentado) |
| `rotation_angles` | 12 | Número de ângulos de rotação |
| `arround` | 360° | Ângulo total de cobertura |
| `process_as` | "monolithic" | Modo de processamento (como monolítico) |
| `rotate_along` | "z" | Eixo principal de rotação do sistema |

### Características do Sistema
1. **Geometria de Detectores**:
   - Dois detectores posicionados opostamente (0° e 180°)
   - Configuração típica para sistema PET de 2 anéis

2. **Cristal e Sensores**:
   - Cristal segmentado com 26.88×26.88 mm² por segmento
   - Matriz SiPM 8×8 com pitch de 3.36 mm
   - Área efetiva por detector: 26.88×26.88 mm²

3. **Parâmetros de Aquisição**:
   - 12 projeções angulares (30° entre aquisições)
   - Rotação completa de 360°
   - Processamento como monolítico apesar de cristal segmentado

### Diagrama Conceitual
```
       Sistema de Tomografia
       ---------------------
          Z (eixo de rotação)
               ↑
               |
       +-------|-------+
       |       |       |
       |  D1   |  D2   |  ← Detectores (180° opostos)
       | (180°)| (0°)  |
       +-------|-------+
               |←→ Raio = 25 mm
               ● Centro de Rotação
```
```
