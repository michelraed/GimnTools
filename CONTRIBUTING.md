# Contributing to GimnToolsV2

Thank you for your interest in contributing. GimnToolsV2 is a research-oriented
tomographic reconstruction toolkit for preclinical PET; contributions that
improve correctness, documentation, and test coverage are especially welcome.

## Getting started

1. Fork the repository and clone your fork.
2. Install in editable mode with the dev extras:

   ```bash
   pip install -e ".[dev]"
   ```

3. On Linux, the GUI additionally requires the system Tk package:

   ```bash
   sudo apt install python3-tk   # Debian/Ubuntu
   ```

## Ground rules

- **Scientific honesty.** Do not claim a correction "fixes" an artifact unless
  you can demonstrate it (synthetic phantom, known-position source, or measured
  data with ground truth). Distinguish clearly between *measured* effects and
  *hypotheses* in commit messages and documentation. This is a project policy,
  not a suggestion.
- **Backward compatibility of results.** Changes that alter numerical output
  must (a) be justified, (b) be documented in `CHANGELOG.md`, and (c) where
  feasible, keep a `legacy=True` style escape hatch so users can reproduce old
  behaviour for comparison (see `siddon_reconstructor.mlem/osem`).
- **Language.** All code, comments, docstrings, and documentation are in
  English. (Exception: the SQLite column names `pesquisador`/`experimento` in
  `storage.py` are kept for compatibility with existing databases.)
- **No fabricated data.** Examples must either generate synthetic data
  explicitly labelled as synthetic, or point to user-supplied files and fail
  loudly (e.g. `FileNotFoundError`) when the files are absent.

## Style

- Python >= 3.8, PEP 8 where practical; keep the existing formatting of a file
  you are editing rather than reformatting wholesale.
- NumPy-style docstrings for public functions.
- Physical quantities carry units in variable names or docstrings (`fov_mm`,
  `sigma_bins`, `phi_offset_deg`).

## Tests

There is a substantial automated regression suite; however, coverage remains
incomplete, particularly for scanner-specific calibration and physical
quantitation. New contributions should add independent regression tests.
Useful invariants to test include:

- Projector/backprojector adjointness: `<Ax, y> == <x, A^T y>` within
  tolerance.
- `legacy=True` reproduces the previous MLEM/OSEM output bit-for-bit on a fixed
  synthetic sinogram.
- `build_subsets(n, s, order)` partitions `range(n)` exactly once.
- Round-trip of the JSON config loaders in `data_loaders.py`.

Run whatever tests exist with:

```bash
pytest
```

## Submitting changes

1. Create a topic branch.
2. Make focused commits with clear messages (what + why; note verification
   status).
3. Update `CHANGELOG.md` under an "Unreleased" heading.
4. Open a pull request describing the motivation, the change, and how you
   verified it.

## Reporting issues

Please include: the data format (PETSYS/GATE), the `detector_config` used, the
exact command or code, the full traceback, and — for reconstruction-quality
issues — an image or profile illustrating the problem. Empty-sinogram issues
are most often caused by an `rsectors` map that does not match the real chipIDs
in the file; see the Troubleshooting section of `README.md` first.
