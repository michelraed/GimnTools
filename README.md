# GimnTools

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gimntools.readthedocs.io/)
[![PyPI Version](https://img.shields.io/pypi/v/gimntools.svg)](https://pypi.org/project/gimntools/)

**GimnTools** é uma biblioteca Python especializada em processamento de imagens médicas e reconstrução tomográfica. Desenvolvida para pesquisadores e profissionais da área de imagens médicas, oferece ferramentas avançadas para manipulação de sinogramas, reconstrução de imagens e geração de fantomas de teste.

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
- **Suporte DICOM**: Leitura e escrita usando SimpleITK e ITK
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

## 🛠️ Scripts de Build e Deploy

A biblioteca inclui scripts automatizados para facilitar o desenvolvimento:

### Scripts Disponíveis

#### 1. Script de Build (`scripts/build.py`)
```bash
# Build completo
python scripts/build.py --all

# Build apenas wheel
python scripts/build.py --build-type wheel

# Limpeza + build + testes
python scripts/build.py --clean --check --install --test
```

#### 2. Script de Instalação (`scripts/install.py`)
```bash
# Instalação básica
python scripts/install.py

# Instalação para desenvolvimento
python scripts/install.py --mode development --jupyter --docs

# Criar ambiente virtual
python scripts/install.py --venv
```

#### 3. Script de Deploy (`scripts/deploy.py`)
```bash
# Deploy para PyPI de teste
python scripts/deploy.py --repository testpypi

# Deploy para PyPI oficial (requer confirmação)
python scripts/deploy.py --repository pypi --version 1.0.1

# Apenas build (sem upload)
python scripts/deploy.py --build-only
```

### Usando Makefile (Linux/macOS)

```bash
# Instalar para desenvolvimento
make install-dev

# Executar testes
make test

# Formatar código
make format

# Build completo
make build

# Deploy para PyPI de teste
make upload-test

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

## 📚 Documentação

A documentação completa está disponível em:
- **Online**: [https://gimntools.readthedocs.io/](https://gimntools.readthedocs.io/)
- **Local**: Execute `make docs` para gerar localmente

### Tópicos da Documentação
- **Guia do Usuário**: Introdução e conceitos básicos
- **API Reference**: Documentação detalhada de todas as funções
- **Tutoriais**: Exemplos passo a passo
- **Guia do Desenvolvedor**: Como contribuir para o projeto


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

- **Issues**: [GitHub Issues](https://github.com/usuario/GimnTools/issues)
- **Discussões**: [GitHub Discussions](https://github.com/usuario/GimnTools/discussions)
- **Email**: contato@gimntools.com

## 🏆 Reconhecimentos

GimnTools foi desenvolvido para apoiar pesquisas em imagens médicas e tomografia. Agradecemos à comunidade científica e aos desenvolvedores de bibliotecas como SimpleITK, ITK e NumPy que tornaram este projeto possível.

## 📊 Status do Projeto

- ✅ **Estável**: Funcionalidades core implementadas e testadas
- 🔄 **Em Desenvolvimento**: Novas funcionalidades sendo adicionadas
- 📖 **Documentação**: Em constante atualização
- 🧪 **Testes**: Cobertura em expansão

---

**GimnTools** - Ferramentas avançadas para imagens médicas em Python.
