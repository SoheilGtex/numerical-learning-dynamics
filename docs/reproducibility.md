# Reproducibility Guide

## Requirements

Python **3.11 or newer** is required. The project uses NumPy, pandas, Matplotlib, SciPy, pytest, and Ruff as declared in `pyproject.toml`.

## Normal development installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\\Scripts\\activate`.

This mode uses the compatible lower-bound dependency ranges in `pyproject.toml` and is intended for ordinary development and compatibility testing.

## Release-environment reproduction

For the closest reproduction of the packaged numerical artifacts, use an interpreter matching the recorded release Python version in `results/reproducibility/environment.txt`, create a clean environment, and install the exact snapshot:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-release.txt
python -m pip install -e . --no-deps
python scripts/reproduce_all.py --verify
```

The interpreter command may differ by platform; it should match the recorded release version rather than assuming a universally available executable name. `requirements-release.txt` was generated from the clean release environment with `python -m pip freeze --exclude-editable`. It contains exact installed package versions and does not include the project itself as an external package.

## Quality checks

```bash
ruff check .
pytest
python scripts/verify_repository.py
```

## Individual experiments

```bash
python scripts/run_baseline.py
python scripts/run_solver_comparison.py
python scripts/run_conditioning_study.py
python scripts/run_perturbation_study.py
python scripts/run_regularization_study.py
python scripts/run_uncertainty_study.py
python scripts/run_forecasting_study.py
```

## Full reproduction

```bash
python scripts/reproduce_all.py
```

The script runs the accepted experiments in order, stops on failure, and does not duplicate experiment logic. Use `python scripts/reproduce_all.py --verify` to check expected outputs after execution.

## Generated outputs

CSV tables are written to `results/tables/`; PNG figures are written to `results/figures/`. The experiment-level manifest is `results/reproducibility/experiment_manifest.json`. Run `python scripts/capture_environment.py` to refresh `results/reproducibility/environment.txt`; checksums are generated after all documentation and experiments are final. `requirements-release.txt` is the exact dependency snapshot for the packaged release.

## Expected reproducibility properties

The generators use explicit `numpy.random.default_rng(seed)` calls and deterministic seed policies. Re-running under the same release environment should reproduce the same values. Floating-point results may differ across Python/NumPy/LAPACK/BLAS/platform configurations. In well-conditioned experiments these differences are normally small, but deliberately ill-conditioned cases can strongly amplify low-level numerical differences, so coefficient-level outputs may differ materially across environments. Exact numerical values quoted in this release refer to the recorded release environment. The checksum file identifies the exact packaged release artifacts; it is not a cross-platform scientific reproducibility criterion. It answers whether files are exactly the files packaged here, not whether every supported machine will regenerate byte-identical floating-point artifacts.
