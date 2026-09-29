# Project Summary

## Problem

Numerical Learning Dynamics transforms a classroom prediction question about final Numerical Analysis grades into a controlled synthetic numerical-modeling program. The repository studies how least-squares solutions behave under rank deficiency, ill-conditioning, perturbation, regularization, uncertainty, information horizons, and specified contamination.

## Methods

- Phase 1: deterministic synthetic least-squares baseline.
- Phase 2: explicit Normal Equations, QR, SVD/pseudoinverse, and rank-deficiency comparison.
- Phase 3: condition-number, scaling, response-perturbation, and design-perturbation analysis.
- Phase 4: centered SVD Ridge, validation/GCV selection, normalized bias–variance analysis, and classical OLS intervals.
- Phase 5: four nested synthetic forecasting checkpoints, OLS/Ridge/Huber comparison, 20-replication ablation, and 20-replication response/feature contamination.

## Strongest results

The controlled rank-deficient experiment produced nearly identical fitted values from different coefficient representations, demonstrating non-identifiability. Conditioning increased coefficient and perturbation sensitivity while fitted values could remain stable. Ridge reduced variance and sensitivity in exchange for bias. OLS confidence and prediction intervals showed distinct widths and coverage under their stated assumptions. In Phase 5, mean held-out Ridge RMSE decreased from 1.558670 at the early checkpoint to 1.296868 at the later checkpoint; OLS was marginally lowest at the pre-semester checkpoint. Huber was most resistant to the specified 20% response contamination, but did not dominate feature contamination.

## Technical skills demonstrated

The repository demonstrates scientific Python packaging, stable linear algebra, SVD-based regularization, transparent IRLS, deterministic simulation, strict holdout design, metric implementation, test-driven numerical checks, reproducibility metadata, artifact traceability, and conservative CI configuration.

## Limitations

The data are synthetic and illustrative. No actual student records are included, and no causal educational or deployment conclusions are justified. Early-checkpoint coefficients can be projections under omitted variables. Huber is not a general solution to leverage contamination. Interval claims are restricted to classical OLS assumptions. Results may vary slightly across platforms.

## Reproducibility

Install with `python -m pip install -e ".[dev]"` for normal compatibility testing, or use the exact `requirements-release.txt` snapshot with a matching Python interpreter for the closest reproduction of the packaged artifacts. Run `python scripts/reproduce_all.py`; then run `python scripts/capture_environment.py` and `python scripts/verify_repository.py`. The experiment manifest, environment snapshot, and SHA-256 artifact manifest are stored under `results/reproducibility/`. Deliberately ill-conditioned cases can materially amplify cross-environment floating-point differences at coefficient level; the packaged CSV files are the canonical release record.
