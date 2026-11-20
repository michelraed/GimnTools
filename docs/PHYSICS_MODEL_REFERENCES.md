# Fontes consultadas

- Gravel P, Surti S, Krishnamoorthy S, Karp JS, Matej S. *Spatially-Variant Image-Based Modeling of PSF Deformations with Application to a Limited Angle Geometry from a Dual-Panel Breast-PET Imager*. Phys Med Biol 64, 225015 (2019). O resumo e a introdução do manuscrito NIHMS1586178 associam deformação da PSF à paralaxe e à cobertura angular e descrevem seu uso dentro da reconstrução. O scanner e a aquisição estudados são distintos dos do GimnTools. [DOI](https://doi.org/10.1088/1361-6560/ab4914)
- O README oficial do OpenGATE descreve a extensão PET sobre Geant4. Foram verificadas as APIs de fonte, materiais e atores da distribuição 10.1.2 e a criação dos objetos da geometria de teste sem inicializar o transporte. [OpenGATE](https://github.com/OpenGATE/opengate/tree/041e281dc4fc41e579c50b365b9b51d124e8aa8a)
- A documentação dos projetos parallelproj e YRT-PET demonstra a separação entre projetores e backends. Seus README descrevem CPU/CUDA e, no caso de YRT-PET, OSEM por histograma e list mode. Os kernels desta implementação usam o Siddon do GimnTools. [parallelproj](https://github.com/KUL-recon-lab/parallelproj/tree/04f9970595f15dc971805cc15c3c9ef9f628de44) · [YRT-PET](https://github.com/YaleBioImaging/yrt-pet/tree/bd569224367dec34e67286d549080a032c854fb4)
- O índice textual inclui os README de Geant4, STIR e OMEGA. Eles servem para localizar documentação e alternativas de implementação, sem equivaler à validação cruzada de reconstruções. [Geant4](https://github.com/Geant4/geant4/tree/f3d5293d384757b8a228a099898b2b87cfa4023c) · [STIR](https://github.com/UCL/STIR/tree/fe93e81616e79a544e0b9f99a262fae06450d744) · [OMEGA](https://github.com/villekf/OMEGA/tree/72a3923bfba67c0eebd525ba561e07a6e6bf79d5)

O leitor PDF fornecido e os arquivos de código citados estão no
[catalogador de evidências](../research/physics_rag/evidence.md). Os
endpoints bibliográficos de PubMed, Europe PMC, Crossref e os sites
de referência física retornaram HTTP 403 neste ambiente. Os artigos
prioritários ainda sem texto verificado não fundamentam equações desta
implementação.
