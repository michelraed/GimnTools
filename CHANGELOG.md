# Changelog

## 0.4.3 — 2026-10-15 — revisão estatística, geométrica e de integridade

- Separação explícita dos prompts e do fundo aditivo no MLEM/OSEM. `--background-scale` obrigatório na CLI PETSYS.
- Calibração COR e encoder integrada à geometria, sem alterar bins medidos de EM. A função de reamostragem angular auxiliar agora respeita a paridade de 180 graus.
- Rebin radial e suavização angular limitados ao ramo FBP (CLI PETSYS/GATE, GUI); nova seleção real de algoritmos em PETSYS.
- Correção da grade retangular de SiPMs e do mapeamento CrystalID parametrizado; suporte a `runID` não consecutivo no GATE.
- Regras estritas para dimensões de hardware, intervalos de energia e parâmetros inválidos.
- Metadados NPZ sem objetos serializados. Para abrir: `np.load(..., allow_pickle=False)` e `json.loads(str(f['geometry_json']))`. **Incompatibilidade intencional** com o antigo formato NPZ `geometry` de objeto Python.
- Métricas FWHM cartesiana e em eixos principais separadas.
- Isolamento mais robusto de aquisições e resultados assíncronos da GUI; importação sob demanda de dependências ROOT e SimpleITK.
- Regressões expandidas e documentação de escopo científico.

**Avisos de compatibilidade**: novos resultados iterativos podem diferir materialmente dos anteriores, em especial com fundo, interpolação, suavização e offsets calibrados. Essa diferença é intencional. Reprocessar datasets anteriores e registrar versão, flags, calibrações e parâmetros antes de comparar valores.

## <= 0.4.2

Código de base distribuído anteriormente pelo projeto. Ver documentação no histórico original.
