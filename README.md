# GimnToolsV2

**Research-oriented preclinical PET reconstruction — PETSYS and GATE**  
**Version:** 0.4.3 · **Python:** 3.10+ · **License:** MIT · **Author:** Michel David Raed

GimnToolsV2 is an open-source Python toolkit for creating sinograms from
coincidence events and reconstructing **slice-wise 2-D PET images** from
rotating-detector PET acquisitions. It supports experimental **PETSYS** data
and **GATE** Monte Carlo simulation ROOT files, with a Python API, terminal
commands and a Tkinter interface.

> **Scientific status:** Research software. The included numerical and GUI
> regression tests do not prove scanner-specific accuracy, quantitative
> activity calibration, NEMA compliance or fitness for clinical diagnosis.
> This is a slice-wise reconstruction framework, **not** a validated full-3-D,
> oblique-LOR or time-of-flight PET reconstruction system.

## Features

- PETSYS coincidence-to-LOR processing and GATE ROOT event processing.
- Sinogram creation with configurable detector geometry, angle maps, crystal
  dimensions, rotation axes, energy windows and discretization.
- Siddon ray-tracing projector and matched backprojector.
- **FBP**, **MLEM** and **OSEM**, selectable individually or together.
- Additive nonnegative background model for PETSYS MLEM/OSEM; measured prompt
  count bins remain unchanged by interpolation and angular smoothing.
- Geometric centre-of-rotation (COR) / encoder-angle correction in projector
  coordinates, with support for a joint PETSYS calibration JSON.
- Optional radial PSF model, image-space Gaussian smoothing, geometric
  sensitivity normalization and configurable image matrix / voxel size.
- MHD and NRRD export through SimpleITK; NPZ arrays and JSON metadata.
- Graphical workflow including sinogram display, reconstruction, image views,
  image profiles and spatial-resolution analysis.

## Requirements

Python **3.10 or newer**; supported scientific dependencies are installed by
`pip`: NumPy, SciPy, Pandas, Matplotlib, Numba, uproot and SimpleITK. A C/C++
compiler is usually unnecessary when compatible prebuilt dependency wheels
exist. Initial calls to Numba-accelerated reconstruction can take longer
because kernels are JIT compiled.

For the GUI, Tkinter must also be present on the host system. On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-tk
```

On Windows and macOS, Tkinter is typically bundled with standard CPython
installers. On headless Linux servers use the CLI; do not run the GUI.

## Install

**From PyPI, after publication** (the uploaded wheel is already installable
locally even if the PyPI project does not exist yet):

```bash
python -m pip install GimnToolsV2==0.4.3
```

**From the project directory** (editable development checkout):

```bash
python -m pip install -e .
```

**From the wheel** distributed in this release:

```bash
python -m pip install ./dist/gimntoolsv2-0.4.3-py3-none-any.whl
```

We recommend an isolated environment to avoid incompatible scientific
library versions:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# Windows PowerShell (run from the repository root)
# py -3 -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# python -m pip install -e .
```

Confirm that the package and scripts are installed:

```bash
python -c "from Gimntools.GimnRec import __version__; print(__version__)"
reconstruct_gate --help
reconstruct_petsys --help
gimntools_ui   # optional: requires a graphical desktop and Tkinter
```

> Distribution/project name: `GimnToolsV2`. **Python import is case-sensitive:**
> `Gimntools.GimnRec`. Console scripts: `reconstruct_petsys`,
> `reconstruct_gate`, and `gimntools_ui`.

## Quick start: Python API

```python
import numpy as np
from Gimntools.GimnRec import siddon_reconstructor

# A real sinogram with axes (slice, radial_bin, angle_bin).
# Replace this array with your measured or simulated sinogram.
sino = np.zeros((1, 64, 12), dtype=np.float64)
sino[0, 32, :] = 100.0  # synthetic input ONLY: not scanner data

# Acquisition convention must match the scanner / simulator.
angles_deg = (np.arange(sino.shape[2]) + 0.5) * (180.0 / sino.shape[2])

rec = siddon_reconstructor.for_volume(
    sino, n_img=64, voxel_mm=0.42, sino_range_mm=26.88,
)
volume = rec.osem(
    angles_deg, iterations=3, subsets_n=3,
    regularization_every=0,
    verbose=False,
)
print(volume.shape, rec.geometry())
```

This intentionally simplistic artificial input is only a smoke-test; it is
**not** a physically realistic PET acquisition or a validation phantom.
All reconstruction geometry must be expressed in **millimetres** and physical
projection angles in **degrees** at the public API boundary.

### Geometry and conventions

- Sinogram shape: `(n_slices, n_radial_bins, n_angles)`.
- `sino_range_mm`: total physical radial extent of the sinogram bins.
- `n_img`, `voxel_mm`: output matrix width and in-plane voxel size.
- Reconstruction FOV: `n_img * voxel_mm` when both are supplied.
- `rays_per_bin`: optional sampling of sub-rays within each radial bin.
- `cor_mm`: constant radial axis offset. PETSYS can also use calibrated
  angular-dependent `dx_cor_mm`, `dy_cor_mm` and `phi_offset_deg`.
- Rotation axes depend on `detector_config['rotate_along']`; raw ROOT branch
  conventions and detector mapping must match the corresponding acquisition.

## Command line: GATE

```bash
reconstruct_gate \
  --measurement /path/to/simulation.root \
  --detector-config examples/CONFIG_GATE/detector_config_gate.json \
  --detectors-angles examples/CONFIG_GATE/detectors_angles.json \
  --rsectors examples/CONFIG_GATE/rsectors_gate.json \
  --process-as GateGlobalPositions \
  --methods osem \
  --n-img 128 \
  --osem-iters 10 --osem-subsets 4 \
  --gauss-every 0 \
  --save-format nrrd --output-dir ./output_gate --no-plots
```

GATE modes include `CrystalID`, `GateGlobalPositions`,
`GeneralGlobalPosition`, and `PaulGlobalPosition` as available in the relevant
branch schema. Override `--rotate-along x|y|z` only when it matches the
simulation geometry. Discover options with `reconstruct_gate --help`.

## Command line: PETSYS

```bash
reconstruct_petsys \
  --measurement /path/to/prompts.root \
  --background /path/to/background.root \
  --background-scale 1.0 \
  --detector-config examples/CONFIG_PETSYS/detector_config_petsys.json \
  --detectors-angles /path/to/measured_detector_angles.json \
  --rsectors examples/CONFIG_PETSYS/rsectors_petsys.json \
  --calibration /path/to/measured_calibration.json \
  --methods osem \
  --n-img 128 \
  --osem-iters 10 --osem-subsets 4 \
  --gauss-every 0 \
  --save-format mhd --output-dir ./output_petsys --no-plots
```

`--background-scale` must represent the physically justified scaling from
background acquisition to prompt acquisition (e.g. `T_prompts / T_background`
if acquisition conditions and estimator permit simple time scaling).
**Do not simply subtract a background histogram for MLEM/OSEM.** These
algorithms use the nonnegative prompt count histogram and include the
background estimate **additively** in the forward model. Background
uncertainty is not propagated by the implemented likelihood.

The PETSYS `calibration_TEMPLATE_DO_NOT_USE_FOR_REAL_DATA.json` is a schema
example with zero offsets, **not** a calibration of any scanner. Determine
COR and encoder offsets from a proper source/phantom calibration before
quantitative processing. Sample detector mappings and hardware dimensions
must be replaced with actual acquisition values.

### Physical and statistical limitations

The iterative model is based on independent Poisson prompt bins:

\[
Y_i \sim \operatorname{Poisson}(\mu_i),\qquad
\mu_i = n_i (Hx)_i + b_i,\qquad x_j\ge 0.
\]

Here `H` is the geometric projector (optionally with an approximate radial
PSF), `n_i` is a per-bin normalization factor and `b_i` is a background
estimate in prompt-equivalent counts. Using `norm_sino = 1` implicitly
assumes unit response and does **not** produce absolute activity units.

For FBP, a linear background-subtracted sinogram may be used. Optional
radial rebinning and angular smoothing are confined to the FBP branch,
because preprocessing count bins as continuous intensities generally breaks
independent-Poisson assumptions for EM reconstruction. The PSF here is an
**approximate 1-D radial model**, not a full spatially varying detector
response. Image smoothing during EM is heuristic regularization, not an
unpenalized MLE solution. OSEM with several subsets does not generally
provide monotonic whole-cycle log-likelihood ascent.

The project does **not** provide independently validated scanner-specific
corrections for attenuation, scatter, randoms, absolute normalization,
dead time, TOF, or full 3-D oblique LOR geometry. Preserve raw data and
reconstruction metadata for reproducibility.

## GUI

```bash
gimntools_ui
```

GUI tabs support GATE and PETSYS input, sinogram visualization, EM/FBP
reconstruction, orthogonal display, profile analysis and FWHM analysis.
Loading or acquisition changes should be checked before exporting results.
Tkinter needs an available graphical display. First Numba compilation can be
slow. In case of GUI problems, first reproduce the computation with the CLI.

## Outputs and data provenance

- `*.mhd` + `*.raw` or `*.nrrd`: reconstructed image data via SimpleITK.
- `*.npz`: NumPy arrays (including raw/FBP sinograms, reconstructed arrays
  and JSON-encoded metadata depending on workflow).
- `geometry_json` and (where provided) `calibration_used_json` are text
  arrays, not Python object pickles. Read with:

```python
import json
import numpy as np

with np.load('output.npz', allow_pickle=False) as d:
    geom = json.loads(str(d['geometry_json']))
    print(geom)
```

These keys are applicable to newly generated NPZ outputs; specific keys and
file names depend on the selected CLI/pipeline. Record software version,
scan duration, energy window, binning, calibration, background scale, matrix,
voxel size, algorithm, iterations, subsets and all post-processing options.

## Testing

For contributors with the source checkout:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
```

For GUI tests on a Linux CI server, install Xvfb and run:

```bash
xvfb-run -a python -m pytest tests/ -q
```

The prior repaired codebase reported **183 passing automated tests** using a
virtual graphical display. That is a historical regression result and not
an assertion of scanner-specific quantitative validation. Re-run the suite
with your environment and dependencies before any tagged publication.

## Repository tree

```text
GimnToolsV2/
├── pyproject.toml               # build metadata, dependencies and CLI entry points
├── README.md                    # complete technical user guide
├── LICENSE, NOTICE              # MIT terms and scientific attribution request
├── CITATION.cff                 # machine-readable publication citation
├── CODE_OF_CONDUCT.md           # participation and moderation policy
├── CONTRIBUTING.md              # contribution / reproducibility rules
├── CHANGELOG.md                 # changes affecting numerical results
├── PUBLISHING_PYPI.md           # release steps (in Portuguese)
├── examples/                    # scripts and editable JSON templates (NO scanner data)
├── tests/                       # regression tests (source checkout / sdist)
└── src/Gimntools/GimnRec/      # package, geometry, pipelines, CLI and GUI
```

The public repository intentionally excludes recorded PET datasets, phantom
image files, calibration fit outputs, executed notebooks, logs and audits.
Generated wheels contain only importable package modules, packaging metadata
and license notices. The source distribution contains tests and examples.

## Citation and reuse

**Please cite GimnToolsV2 in publications, theses, reports and conference
presentations that use its outputs**, indicating the exact version. The
recommended software citation is:

> Raed, M. D. (2026). *GimnToolsV2: Tomographic Reconstruction Toolkit for
> Preclinical PET* (Version 0.4.3) [Computer software].

BibTeX (add a repository URL and archival DOI when these exist):

```bibtex
@software{raed_gimntoolsv2_2026,
  author = {Raed, Michel David},
  title = {GimnToolsV2: Tomographic Reconstruction Toolkit for Preclinical PET},
  year = {2026},
  version = {0.4.3},
  note = {Computer software; MIT License}
}
```

See [CITATION.cff](CITATION.cff). Once the project has a GitHub release, you
can connect the repository to Zenodo to receive a persistent DOI; **do not
cite an invented DOI**. In an article Methods section, also report the
reconstruction pipeline and parameters needed to reproduce the results.

**Legal distinction:** MIT is a genuine permissive open-source license;
any redistribution must retain its copyright and license notices.
A **scholarly paper citation is strongly requested but is not a separate
legal condition of the MIT License**. An additional enforceable citation
condition would require a different, specialist-reviewed license and could
prevent classification as standard open-source software. See [NOTICE](NOTICE).

## Contributing, conduct and license

Contributions, reproducible bug reports and numerical validation phantoms
are welcome; read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All project code is distributed
under the [MIT License](LICENSE). Copyright © 2026 Michel David Raed.

## Algorithm references

- Shepp LA, Vardi Y. Maximum likelihood reconstruction for emission
  tomography. *IEEE Trans Med Imaging*. 1982;1:113–122.
  [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)
- Hudson HM, Larkin RS. Accelerated image reconstruction using ordered
  subsets of projection data. *IEEE Trans Med Imaging*. 1994;13:601–609.
  [doi:10.1109/42.363108](https://doi.org/10.1109/42.363108)
- Siddon RL. Fast calculation of the exact radiological path for a
  three-dimensional CT array. *Med Phys*. 1985;12:252–255.
  [doi:10.1118/1.595715](https://doi.org/10.1118/1.595715)
- Goertzen AL et al. NEMA NU 4-2008 comparison of preclinical PET imaging
  systems. *J Nucl Med*. 2012;53:1300–1309.
  [doi:10.2967/jnumed.111.099382](https://doi.org/10.2967/jnumed.111.099382)
