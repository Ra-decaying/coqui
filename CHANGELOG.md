
# Changelog

## CoQui v0.2.0 [2026-02-04]

### Added

- Fully self-consistent GW+EDMFT solver and interface, including the new `dmft` Python submodule (`src/python/dmft/`), drivers, utilities (bath fitting, chemical potential, checkpoints, SCF driver, etc.)
- TRIQS CT-SEG interface (planned to be moved to the TRIQS library in the future).
- New examples for cRPA and GW+EDMFT.
- Unit test for Wannier90 MLWF workflow using SrVO3.
- `mlwf_h5_from_wannier90_output` utility to read Wannier90 output files and write MLWF data into CoQui h5 format.

### Improved

- Major refactor for downfolding routines to improve the API, enhance modularity, and increase code readability.
- Downfolded Coulomb interactions are now stored under `downfolded_model` h5 group within the input h5 group, e.g. `scf/iter{}` or `embed/iter{}`.
- `IAFT` initialization from CoQui checkpoint h5.
- Improved TRIQS ModEST interface for quantum embedding.
- Improved SCF runtime controls: added `iter_alg.enable` to optionally disable iterative mixing/DIIS paths, added `mu_tolerance` for explicit chemical-potential convergence control, and added stricter checkpoint/input dataset validation for `greens_func_source`/`greens_func_iteration` workflows.
- Examples for MLWF library and standalone modes.
- Examples for Fourier transform on imaginary time/frequency grids on Python.
- Chol-GW for molecules refactoring.
- Improved divergence treatment for dielectric function, and a customized option for metallic systems. The previous default `gygi` is now an alias for `gygi_smallest_q` for backward compatibility.

### Fixed

- Wannier90 compilation issue due to recent updates in Wannier90 codebase. 

### API Updates

- Renamed MBPT/downfolding input keys from `input_type` and `input_iter` to `greens_func_source` and `greens_func_iteration`.
- Renamed `downfold_local_coulomb` to `downfold_coulomb` to reflect support for non-local downfolded interactions.

## CoQui v0.1.0 [2026-01-23]

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
