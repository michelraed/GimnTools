# Validação da reconstrução

## Aquisições PETSYS

Foram carregados os seis sinogramas NPZ fornecidos, preservando origem,
geometria, contagens e poses do arquivo. A reconstrução OSEM CPU usou
75 × 75 pixels transaxiais de 0,84 mm e 15 fatias.

| Arquivo | Poses | Contagens |
| --- | ---: | ---: |
| 1Fonte_Centro_rotang_12 | 12 | 1.732.512 |
| 1Fonte_OFf_2_rotang_12 | 12 | 1.367.264 |
| 1Fonte_OFF_rotang_12 | 12 | 1.133.366 |
| 2Fontes_OFF_rotang_12 | 12 | 2.565.935 |
| 2Fontes_eder_24072025_rotang_24 | 24 | 5.173.724 |
| Capilares_rotang_24 | 24 | 449.652 |

O NPZ `2Fontes_OFF_listmode` tem as mesmas 2.565.935 coincidências
que seu sinograma correspondente. A corrida LM-OSEM em CPU usou
75 × 75 × 75 voxels de 0,84 mm, raio físico PETSYS de 35 mm, cinco
iterações e quatro subsets. Foram traçadas todas as coincidências ao
longo de 12 poses e a diferença relativa do volume frente ao resultado
CPU anterior foi 0. A carga dos seis sinogramas foi repetida em Tk e
VTK/PySide6, e a reconstrução foi repetida no lançador com VTK.

## Testes automatizados

- O teste de lote exporta dois ROOTs PETSYS com poses distintas, verifica
  que os nomes existentes são preservados e reconstrói os NPZs em fila.
  Um ROOT inválido no lote afeta apenas aquela entrada.
- Os testes de GATE validam run IDs esparsos, ramos com ângulos em graus
  e radianos e o tratamento de coordenadas globais.
- A calibração sintética preserva canais com contagem medida igual a
  zero na sensibilidade, incluindo perdas e incerteza binomial. O teste
  exige o mapa original na grade correta, rejeita contagens acima dos
  primários e proíbe a amostragem analítica durante a preparação da LUT.
- As suítes cobrem paridade de UI e compartilhamento do banco de dados,
  controles de detectores, adjunção, normalização e PSF.
- Os quatro testes dedicados a CUDA passaram com CUDASIM. FBP, MLEM,
  OSEM, DOI, SV-PSF e list mode foram comparados com a CPU. Compilação
  offline com NVVM gerou PTX para os kernels Siddon 2D, Siddon 3D e CSR,
  com compute capability 8.6.

```bash
python -m pytest
NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_cuda_reconstruction.py
```

O [benchmark](RECONSTRUCTION_PERFORMANCE.md) descreve tempos e memória
medidos. As matrizes desse benchmark foram construídas com contagens
sintéticas para exercitar o formato. A reconstrução dos NPZs reais
usou os operadores de referência disponíveis para essas aquisições.

## Validação que requer recursos externos

A API OpenGATE 10.1.2 permitiu montar uma simulação de teste. O download
do pacote de tabelas eletromagnéticas G4EMLOW foi bloqueado pelo
servidor CERN neste ambiente. A geração por transporte e a calibração
com o detector experimental permanecem por validar. Para caracterizar
uma resposta física medida, são necessários calibrações independentes,
perfis de fontes pontuais em várias posições, mapas de eficiência e
medições de contraste e ruído. Os NPZs entregues permitem validar
importação e reconstrução, mas ainda não estabelecem ganho de resolução
de um novo modelo.
