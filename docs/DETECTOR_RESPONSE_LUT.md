# LUT de resposta física

O arquivo NPZ versionado `gimntools-physical-pairs-v1` guarda contagens
CSR inteiras, primários emitidos por pose e voxel, sensibilidade por pose,
geometria JSON com assinatura SHA-256 e proveniência. As linhas seguem
pose ordenada pelo ângulo e par de canais de cabeças distintas. As
colunas são voxels em ordem C no volume Z, Y, X. Cada canal é dado pela
cabeça, multiplicada pelo número de pixels daquela cabeça, mais o índice
do cristal, conforme a ordem dos eixos registrada no plano. Contagens
acumuladas de uma pose e voxel podem ser iguais ou menores que os
primários simulados. Use `emitted > 0` para distinguir voxels calibrados
de voxels nunca amostrados ao analisar o mapa de perdas ou incerteza.

## Calibrar e reutilizar

Instale a ferramenta a partir da branch de desenvolvimento com
`python -m pip install -e .`. Use `.[opengate]` para a preparação
opcional com OpenGATE e os dados GEANT4 da versão correspondente.
Gere o plano com a grade que será usada na reconstrução. Para aquisições
sem janela registrada, defina-a no plano e na reconstrução com
`--energy-window MIN_KEV MAX_KEV`.

```bash
gimntools_physics plan aquisicao.npz plano.json --matrix 75 75 75 --voxel .84 .84 .84
gimntools_physics opengate-voxel plano.json amostra-0-2812.npz \
  --pose 0 --voxel-index 2812 --primaries 100000 --seed 317133
```

O comando de geração cobre uma pose e um voxel. Repita para todos os
voxels e poses que deseja calibrar e use uma semente independente por
trabalho. Escolha o número de primários com base na eficiência e no erro
de amostragem medidos. A saída do passo acima é um exemplo de entrada
parcial. Para importar de outro simulador, entregue um NPZ por trabalho
com `emitted` inteiro de forma `(poses, voxels)` e vetores inteiros
`pose`, `voxel`, `first`, `second`, uma linha por coincidência
aceita. O conversor Python `write_physical_pairs` também aceita
multiplicidades agregadas no parâmetro `count`.

Registre a descrição do transporte e do digitizador no JSON de
proveniência com `engine`, `version` e `readout`. O importador
também aceita um JSON vazio quando os arquivos OpenGATE já contêm esses
campos. Para calibrações externas, inclua a janela de energia, política
de leitura, sementes e referências da simulação. Use somente resultados
independentes das coincidências que serão reconstruídas.

```bash
gimntools_physics build plano.json aquisicao.physical.npz proveniencia.json amostra-*.npz
gimntools_physics inspect aquisicao.physical.npz
gimntools_physics reconstruct aquisicao.npz resultado.mha \
  --matrix 75 75 75 --voxel .84 .84 .84 \
  --physical-lut aquisicao.physical.npz --backend cpu --iterations 5 --subsets 4
```

Na interface, carregue o NPZ em **Reconstruction → List mode**, defina a
grade e marque **Use physical crystal-pair matrix**. Se o arquivo tiver
o mesmo nome da aquisição com a extensão `.physical.npz`, o seletor
detecta o arquivo na mesma pasta. A validação compara a grade, os ângulos
ordenados, as dimensões e o material do detector, o alinhamento e a
janela de energia. A GUI mostra o tempo de preparação e reconstrução e a
memória do operador já selecionado. A LUT e a sensibilidade podem ser
reutilizadas para qualquer aquisição compatível sem simulador instalado.
