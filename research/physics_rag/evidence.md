# Evidências e limites

| Decisão | Evidência verificada | Hipótese ou limite | Verificação local |
| --- | --- | --- | --- |
| Tratar a resposta como função da posição | Gravel et al., resumo e introdução, DOI 10.1088/1361-6560/ab4914 | Limite angular e paralaxe dependem da geometria e da amostragem | Os testes de alinhamento e de resposta cruzam canais reais com o operador montado |
| Manter projeção e retroprojeção pareadas | Definição algébrica do adjunto da matriz esparsa | Contagens e exposição precisam representar a mesma seleção do digitizador | `test_physical_pairs.py`, erro de adjunção e normalização |
| Usar transporte independente em etapa separada | README OpenGATE e API 10.1.2, fonte genérica e coleção de hits | A seleção por maior depósito é um modelo de digitização | Teste determinístico do digitizador e provisionamento da geometria |
| Manter um operador em dois backends | README parallelproj e YRT-PET, CPU/CUDA | Ganho de execução depende do dispositivo e da residência de dados | `test_cuda_reconstruction.py`, CUDASIM, compilação PTX 8.6 |

O índice é local e opcional. Use `python research/physics_rag/indexer.py --fetch`
para indexar os README públicos fixados em `sources.json`. Pode colocar
extratos de fontes com acesso legítimo, incluindo `gravel.txt`, em
`research/physics_rag/.cache/texts`. O Git ignora esse cache.
Busque termos, títulos e DOI com
`python research/physics_rag/retriever.py 'parallax limited angle'`.
Os resultados incluem a passagem, o link e a revisão quando aplicável.
