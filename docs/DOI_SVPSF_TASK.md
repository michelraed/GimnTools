# DOI-aware + spatially variant response reconstruction — Codex task

## Scope / scientific assumptions
Implement optional, physically motivated detector response models in **GimnTools** for its slice-wise 2D PET MLEM/OSEM pipeline; connect UI and CLI. Preserve existing FBP, MLEM/OSEM, PSF and default output bit-for-bit where feasible. No clinical or complete 3D DOI claims.

Physical setup (verify config and coordinate axes, do not guess): two opposing **8×8 LYSO crystal blocks**, 3×3×20 mm³ *per segment*, pitch 3.36 mm, no event-level measured DOI, rotating **object** with nominal 12 acquisition positions, about 35 mm center-to-entrance face, PETsys/GATE data, energy window 450–650 keV. The paper quotes a conflicting 75-mm face-to-face distance: surface this discrepancy; never silently choose an alternative. `examples/CONFIG_PETSYS/detector_config_petsys.json` uses `crystal_size: [20,26.88,26.88]` for **block** dimensions and rotation axis `y`; do not mistake block size for per-crystal size.

## First: audit current code and document findings
- `src/Gimntools/GimnRec/projectors.py`: Siddon matched operators, finite radial bin sub-rays.
- `reconstructor.py`: MLEM/OSEM, current sinogram-domain radial PSF `C`, sensitivity maps.
- `sinogramer.py`, `data_loaders.py`, `systemSpace.py`, `conf.py`: crystal IDs, geometry, object rotation / binning and information lost on binning.
- `filters.py`: existing Gaussian kernel and exact adjoint convolution.
- `reconstruction_v2.py`, `pipelines.py`, `cli/reconstruct_{gate,petsys}.py`, `ui/app.py`: public entry points and GUI wiring.
- `tests/test_math_audit.py`, `tests/test_psf_and_regularization.py`, `tests/test_fbp_ui_wiring.py`: invariants.
Record geometry conventions, where detector pair IDs are retained, whether a binned (r,φ) sinogram suffices for the proposed operator, and your chosen implementation strategy in docs before changing mathematics.

## Mathematical design
1. Add optional **DOI-marginalized finite-crystal** response for measured crystal pairs. For first-interaction depth `z∈[0,L]`, path incidence `α`, use normalized truncated exponential conditional on interaction, `p(z|α)= (μ/cos α)exp[-μz/cos α] / [1-exp(-μL/cos α)]` (or numerically stable equivalent). Include transverse entrance position / finite 3-mm crystal face and detection/acceptance weighting; integrate both crystals with selectable quadrature. Make crystal geometry, `μ_511` (user-configurable and documented, verified rather than invented), quadrature parameters explicit. First-interaction model is a *stated approximation*: Compton redistribution, wrong-crystal assignment, non-collinearity and positron range must not be implied covered.
2. Prefer a matched **pair-aware tube-of-response / system operator** `A` for a given detector pair and mechanical angle. The existing `Siddon` 2D reconstruction uses `(radial bin,angle bin)` sinograms. If pair-specific information is lost in binning, **do not fabricate it**: implement a detector-pair-aware route or a clearly labeled effective sinogram approximation with defensible rebinning / aggregation. Avoid naive `z tan α` as full LOR response.
3. Provide optional **spatially variant *residual*** response (2D effective bins and/or pair-dependent kernel) from imported GATE LUTs; document LUT schema, units, axes, interpolation, normalization, edge behavior, asymmetric kernels and provenance. Be explicit that a GATE kernel containing DOI must not be composed with DOI twice. If independent residual calibration cannot be established, support the feature as exploratory with warnings.
4. Compose forward and exact-transpose backward operators including any rebinning, sparse mapping or LUT; EM sensitivity = `Aᵀ 1` (or appropriately weighted `Aᵀ n` for physical normalization), recomputed/cached by complete model + geometry + binning + grid + subset + calibration fingerprint. Use nonnegative coefficients and matching normalization. Never mix image-space postfiltering with the physical PSF model. Preserve legacy radial Gaussian mode separately.
5. Constraints: NumPy/Numba/SciPy as existing; controlled memory and runtime; avoid building an unnecessarily dense full system matrix; clear warnings on invalid geometry/unsupported 3D/TOF assumptions.

## UI / CLI
In Reconstruction tab of `ui/app.py`, provide a `Detector Response Modeling` group with `Conventional`, `Existing radial Gaussian`, `DOI-aware`, `DOI + residual SV response`; expose geometry, μ, quadrature and LUT-file controls only when applicable. Show active settings, status/warnings and (if feasible) kernel preview. Pass options end-to-end through GUI, public APIs, pipelines and both CLIs; save metadata + input/LUT provenance to outputs. Keep defaults identical to before.

## Validation (required)
- Unit tests: truncated DOI PDF integrates to 1, increasing obliquity / depth changes response as expected, nonnegative finite normalized kernels, central symmetry, off-axis variation, crystal IDs and coordinate transforms, sample rebinning conservation.
- Dot-product adjointness `<Ax,y>≈<x,Aᵀy>` for **each** mode including DOI, LUT, non-symmetric kernels, binning and OSEM subsets. Check sensitivity and EM likelihood/convergence on controlled synthetic cases.
- Legacy tests all pass; GUI controls actually propagate; baseline reconstruction reproducible. Controlled point/source phantoms and radial/tangential FWHM, position error, noise and updates comparisons. Do not label reconstructed-source FWHM as intrinsic detector PSF; finite 2-mm Na-22 source and 0.8-mm capillaries are not pointlike.
- Clearly mark **numerical verification** vs **independent GATE** / **experimental validation** not performed without datasets.

## Execution / acceptance
Implement in reviewable phases: **(A) audit + standalone DOI quadrature and tests, (B) matched projector/backprojector and sensitivity with tests, (C) optional SV-LUT and tests, (D) GUI/CLI + smoke/regression tests, (E) docs/example and performance benchmark**. Stop and report an explicit blocker rather than silently simplifying physically important assumptions. Never modify `backup/main-pre-doi-svpsf-20261118`. Develop on `codex/doi-svpsf-20261118` and make commits.

**When implementation and tests are successful, finish by integrating the validated feature into `main` (push/merge).** Create a clean PR targeting `main` and merge it when checks pass and permissions allow, without force-push. The final `main` tree and integration diff must NOT contain the AI-only instruction files `AGENTS.md`, `docs/DOI_SVPSF_TASK.md`, or any generated agent prompt/plan/transcript/scratchpad files. Keep real user-facing scientific documentation, examples, code and tests. Remove excluded paths from the integration worktree/PR or use a clean integration branch before merging. Verify the final `main` contents explicitly. If any required test fails, important physical-model ambiguity remains, or the merge is blocked by permissions/rules, do not modify `main`; report the blocker and leave a PR for review. In final response report changed files, test results, scientific limitations, integration commit/PR, and any remaining real-data calibration requirements.
