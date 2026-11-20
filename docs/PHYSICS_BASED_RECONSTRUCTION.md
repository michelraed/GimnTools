# Resposta física para PET de dois painéis

A reconstrução por sinograma continua disponível com FBP, MLEM, OSEM, DOI
analítico e PSF espacialmente variante. O caminho calibrado por pares opera
na aba **Reconstruction → List mode**, com aquisição discreta PETSYS ou
GATE CrystalID. Os volumes e as poses são tridimensionais. As fases do
GATE conservam o identificador de run. O PETSYS conserva os ângulos
gravados no arquivo. Posições realinhadas do PETSYS são decodificadas no
reticulado de leitura e os mesmos deslocamentos entram na geometria do
operador.

## Do transporte à reconstrução

Considere uma pose $p$, um par de canais $i$ em faces opostas e um
voxel $j$. A calibração independente registra $C_{pij}$ coincidências
aceitas a partir de $N_{pj}$ primários. Para um voxel calibrado, a
probabilidade de detecção no par é $H_{pij}=C_{pij}/N_{pj}$. A
probabilidade de perda é $1-\sum_i H_{pij}$. A sensibilidade física
por voxel soma as probabilidades de todos os canais por pose e as
pondera pela exposição relativa $e_p$.

A projeção usa $H_p x$ e a retroprojeção usa $H_p^T y$. Os updates
MLEM/OSEM ponderam a retroprojeção da razão entre as contagens observadas
e previstas pela sensibilidade. A exposição global usa a fração de
eventos de cada subset. A exposição por pose usa o valor do NPZ quando
disponível. O registro experimental sem uma exposição explícita assume
poses de igual peso. A mesma construção é usada em CPU e CUDA.

Durante os updates, uma matriz esparsa guarda as linhas com observações e
a sensibilidade carregada mantém as contribuições dos canais com zero
eventos medidos. Se um canal medido inteiro estiver ausente da calibração,
a preparação informa a pose e o número de pares afetados. Opcionalmente,
uma PSF residual 3D compõe o operador com o seu próprio adjunto. Para
iniciar a comparação de modelos, deixe a PSF residual desligada até
estimar a largura que já está incluída na resposta do detector.

O operador analítico existente integra trajetórias de cristal com
amostragem quase-Monte Carlo. Arquivos físicos carregam contagens
independentes e pulam essa integração. O algoritmo convencional e o
pipeline analítico continuam disponíveis. A LUT física inclui a
sensibilidade junto com os coeficientes. Carregar esse arquivo é o modo
reutilizável em ambos os backends.

## Escopo físico

A ferramenta OpenGATE de calibração cria uma geometria de dois painéis
com cristais segmentados, translações, orientação, pitch e composição
nominal LYSO ou BGO. A fonte ocupa o volume de um voxel e emite dois
fótons de 511 keV em direções opostas. O digitizador soma energia
depositada por cabeça e identifica o cristal com maior depósito.
Esse critério é um modelo de leitura. A calibração quantitativa ainda
depende da resposta real do scanner, incluindo óptica, limiares,
espalhamento, aleatórios, tempo morto, resolução energética e materiais
do objeto e da montagem. O gerador atual transporta fótons em ar
e nos cristais. A aproximação por fatias dos modelos analíticos de
sinograma limita a descrição da obliquidade e do espalhamento entre cortes.

Tudo que depende de transporte é preparado à parte. Uma matriz física
requer coordenadas discretas no NPZ e uma calibração correspondente. Um
sinograma convencional já agrega os pares e não é a entrada deste modo.
Veja [o formato e o procedimento de calibração](DETECTOR_RESPONSE_LUT.md)
e [os resultados medidos](VALIDATION_REPORT.md).
