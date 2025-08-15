# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-06-19

### Adicionado
- Versão inicial da biblioteca GimnTools
- Classe `image` para manipulação de imagens médicas
- Módulo `sinogramer` para geração e manipulação de sinogramas
  - Suporte para cristais monolíticos e segmentados
  - Conversão de coordenadas de cristais
- Módulo `gimnRec` para reconstrução tomográfica
  - Algoritmos MLEM, OSEM e FBP
  - Reconstrução por integrais de linha
  - Matriz de sistema para reconstrução
- Módulo `processing` para processamento de imagens
  - Filtros espaciais com otimização Numba
  - Filtros de frequência
  - Filtros morfológicos
  - Ferramentas matemáticas e kernels
  - Sistema de visualização (ploter)
- Módulo `phantoms` para geração de fantomas
  - Fantoma de Derenzo parametrizável
  - Fantomas aleatórios
  - Geometrias customizadas
- Módulo `IO` para entrada e saída
  - Suporte completo para DICOM via SimpleITK e ITK
  - Normalização automática de tipos de dados
  - Preservação de metadados médicos

### Ferramentas de Desenvolvimento
- Script automatizado de build (`scripts/build.py`)
- Script de instalação (`scripts/install.py`)
- Script de deploy (`scripts/deploy.py`)
- Makefile para automação de tarefas
- Configuração completa do ambiente de desenvolvimento

### Documentação
- Documentação técnica completa em português
- README abrangente com exemplos
- Guia de instalação detalhado
- Exemplos práticos de uso
- API Reference completa

### Configuração do Projeto
- `setup.py` para distribuição via PyPI
- `pyproject.toml` com configurações modernas
- `requirements.txt` e `requirements-dev.txt`
- Configuração de ferramentas de qualidade (black, flake8, mypy)
- Configuração de testes com pytest
- Licença MIT

### Dependências
- numpy (≥1.20.0) - Computação numérica
- scipy (≥1.7.0) - Algoritmos científicos  
- matplotlib (≥3.3.0) - Visualização
- numba (≥0.54.0) - Compilação JIT
- SimpleITK (≥2.1.0) - Processamento de imagens médicas
- itk (≥5.2.0) - Toolkit de imagens
- Pillow (≥8.0.0) - Manipulação de imagens
- scikit-image (≥0.18.0) - Processamento adicional
- h5py (≥3.1.0) - Armazenamento HDF5
- tqdm (≥4.60.0) - Barras de progresso

## [Unreleased]

### Planejado
- Implementação de algoritmos de reconstrução iterativa avançados
- Suporte para reconstrução 4D (temporal)
- Interface gráfica para visualização interativa
- Plugins para software de análise médica
- Otimizações de performance com CUDA/OpenCL
- Suporte para formatos adicionais (NIfTI, ANALYZE)
- Documentação em inglês
- Tutoriais em vídeo
- API REST para processamento em nuvem

### Em Consideração
- Integração com ITK-SNAP
- Suporte para processamento distribuído
- Interface com MATLAB
- Bindings para R
- Containerização Docker
- Deployment em Kubernetes
