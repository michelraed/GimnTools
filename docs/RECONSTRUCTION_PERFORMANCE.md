# Desempenho da reconstrução

Use **Backend → CPU** para carregar e executar sem drivers adicionais.
Para CUDA com GPU NVIDIA e driver compatível com CUDA 12, instale
`python -m pip install -e '.[cuda]'` e escolha **CUDA**. O mesmo seletor é
usado para sinogramas e list mode. A CLI física usa `--backend cuda`.

Os kernels Numba CUDA calculam as interseções de Siddon e os produtos
CSR de projeção e retroprojeção. Os coeficientes e a precisão de
acumulação seguem o operador da CPU. As estruturas esparsas mais
usadas e suas transpostas permanecem na GPU até o limite de 1 GiB ou
um terço da memória livre. A atualização EM, a preparação de algumas
correções analíticas e a filtragem FFT da FBP são feitas na CPU. A cada
produto há transferência do vetor de entrada e de saída. Para lotes
grandes ou suporte esparso, a escolha mais rápida depende do custo de
transferência e da VRAM disponível.

As chamadas CUDA executam em um processo separado. Se o runtime encerrar
esse processo durante a reconstrução, a interface exibe a etapa afetada e
permite repetir a aquisição em CPU. O diagnóstico nativo é registrado no
log da interface e no terminal de onde ela foi aberta.

## Medições na nuvem

Hardware Intel Xeon Platinum 8573C, Linux, quatro threads Numba,
acumulação em float64. A primeira chamada geométrica, incluindo
a inicialização Numba, levou 5,637 s. As matrizes físicas do benchmark
foram preenchidas com contagens sintéticas por canal e não incluem custo
de transporte Geant4. O baseline Siddon usa seu próprio modelo geométrico.
As duas tarefas têm números de coeficientes e física diferentes.

| Grade | Poses | Leitura LUT (s) | 40 updates físicos (s) | 40 updates Siddon (s) |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 12 | 0,008 | 0,020 | 0,070 |
| 64 × 64 | 24 | 0,011 | 0,039 | 0,081 |
| 128 × 128 | 12 | 0,019 | 0,055 | 0,155 |
| 128 × 128 | 24 | 0,038 | 0,089 | 0,223 |
| 256 × 256 | 12 | 0,078 | 0,149 | 0,409 |
| 256 × 256 | 24 | 0,130 | 0,308 | 0,696 |

O maior cenário usou 372 MiB de pico acumulado no processo. O maior erro
relativo de adjunção foi (4,94 \times 10^{-15}). A leitura da
sensibilidade armazenada e o cálculo explícito por somas esparsas foram
cronometrados separadamente. O script gera CSV e JSON com esses tempos,
construção, carregamento, forward, backprojection e tempo por update.

```bash
NUMBA_NUM_THREADS=4 python examples/benchmark_physical_response.py resultados
```

Este ambiente verificou paridade no simulador CUDA e compilação PTX dos
três kernels para compute capability 8.6. Medições de VRAM e velocidade
em GPU física permanecem pendentes.
