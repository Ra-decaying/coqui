# Changelog

## [Pre-release] - 2026-01-23

### Added

- New Python API example scripts for mean-field, MBPT, downfolding, interaction, and Wannier workflows under `examples/python_interface/`.
- `TEST_INPUT_DIR` environment variable to point to the test input files.
- Feature to write screened interaction in ISDF basis to HDF5 in MBPT with `dump_w_to_h5`.
- New QE and PySCF converter examples and input files for solids and molecules.
- ISDF for generating the auxiliary basis in the python interface (`coqui.run_isdf`).
- SrVO3 unit test files for Wannier and QE workflows under `tests/unit_test_files/`.

### Improved

- Refactored DIIS routines and improved the restart logic. Rename `diis_start` to `diis_warmup`.
- Improved SCF convergence criteria.
- ISDF auxiliary basis can now be written on reduced or full FFT mesh via `write_zeta_on_fft_mesh` parameter.
- Refactored Python interfaces for interaction and mean-field modules.
- Updated `README.md`, example documentation and docstring.

### Fixed

- Various bug fixes in PySCF interface and import logic.

### Removed

- Deprecated and redundant example input files and pseudopotentials.

### Notes

- This is the first formal changelog entry, summarizing all changes on `develop` since the last stable `main` release.
