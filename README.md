# Numerical Learning Dynamics

## A Numerical Modeling Study of Early Student-Performance Forecasting

This project originated from a prediction problem posed by **Dr. Jamshid Saeidian during a Numerical Analysis class at Kharazmi University**:

> Can a student's final Numerical Analysis grade be predicted from early-semester information?

The subsequent mathematical formulation, experiments, software implementation, and analysis were developed independently. This wording does not imply supervision, endorsement, coauthorship, or research collaboration.

## Research question

Can a student's final Numerical Analysis grade be predicted from early-semester information? This repository studies the question as a controlled numerical-modeling problem rather than as an empirical student-record analysis.

## Project origin

The project originated from the classroom prediction question described above. The subsequent mathematical formulation, experiments, software implementation, and analysis were developed independently.

## Current phase

Scientific Phases 1–5 are complete. Phase 6 packages the accepted work as a reproducible research-software release with a technical report, provenance metadata, artifact manifests, automated verification, and CI. Phase 1 provides the synthetic least-squares baseline; Phase 2 provides explicit numerical least-squares solvers; Phase 3 provides conditioning and perturbation analysis; Phase 4 provides Tikhonov/Ridge regularization, lambda selection, bias–variance–stability analysis, and classical OLS uncertainty quantification; Phase 5 provides early-semester held-out forecasting, ablation, and contamination robustness.

The model is

$$y = X\beta + \epsilon,$$

where the design matrix contains an explicit intercept and five early-semester predictors. The baseline estimates parameters by minimizing $\|X\beta-y\|_2^2$ with `numpy.linalg.lstsq`.

## What Phase 1 demonstrates

- deterministic synthetic modeling with known ground-truth coefficients;
- ordinary least-squares estimation and prediction;
- parameter-recovery evaluation;
- reproducible train/test experiments;
- MAE, RMSE, $R^2$, and relative coefficient-error metrics.

## Installation and execution

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_baseline.py
pytest
ruff check .
```

The complete Phase 1–5 reproduction sequence is:

```bash
python scripts/run_baseline.py
python scripts/run_solver_comparison.py
python scripts/run_conditioning_study.py
python scripts/run_perturbation_study.py
python scripts/run_regularization_study.py
python scripts/run_uncertainty_study.py
python scripts/run_forecasting_study.py
```

The Phase 6 packaging checks are:

```bash
python scripts/capture_environment.py
python scripts/verify_repository.py
```

For exact release reproduction, use the recorded Python version and [`requirements-release.txt`](requirements-release.txt), then install the project with `python -m pip install -e . --no-deps`. Normal CI intentionally uses the flexible `pyproject.toml` ranges for compatibility testing. Deliberately ill-conditioned experiments can materially amplify cross-environment floating-point differences at coefficient level; exact values quoted here refer to the recorded release environment, while the packaged CSV files remain the canonical release record.

## Research progression

The project progresses from least squares to solver stability, rank deficiency, conditioning, perturbation sensitivity, Tikhonov regularization, bias–variance, uncertainty, held-out forecasting, ablation, and robustness.

## Repository structure

Source code is under `src/numerical_learning/`; experiment entry points are under `scripts/`; tests are under `tests/`; mathematical and release documentation is under `docs/`; generated tables, figures, and reproducibility metadata are under `results/`.

## Quality and verification

Run `ruff check .`, `pytest`, and `python scripts/verify_repository.py`. GitHub Actions repeats installation, Ruff, pytest, and repository verification. The experiment manifest, environment snapshot, and SHA-256 artifact manifest provide release provenance.

## Limitations

The observations are mathematically controlled synthetic data, not records from Kharazmi University or any other students. Therefore, the results support software and numerical verification only; they do not support educational conclusions, causal inference, or claims of statistical significance.

Completed analyses include the reproducible synthetic least-squares baseline, solver comparison, conditioning, perturbation, Ridge/Tikhonov regularization, lambda selection, bias–variance/stability analysis, classical OLS uncertainty verification, early-semester held-out forecasting, feature ablation, and specified response/feature contamination robustness. Still future work includes bootstrap uncertainty, conformal prediction, nonlinear models, and state-space/Kalman methods.

## Phase 2: numerical least-squares solvers

Phase 2 extends the baseline with explicit Normal Equations, reduced unpivoted QR, an SVD/pseudoinverse solver, and the existing `numpy.linalg.lstsq` reference. It adds controlled well-conditioned, Phase 1, near-rank-deficient, and exactly rank-deficient cases, together with solver diagnostics and a small raw-versus-standardized conditioning diagnostic. Failures are retained as results rather than hidden. The implementation does not claim that any solver is universally superior; conclusions are limited to the executed synthetic cases.

In the exact rank-deficient case, SVD and `numpy.linalg.lstsq` produce minimum-norm solutions with nearly zero residuals, while the chosen generating coefficient vector is a different valid representation. This demonstrates that excellent fitted values do not imply unique parameter recovery: rank-deficient designs admit multiple coefficient vectors with the same observations.

## Phase 3: conditioning and perturbation study

Phase 3 studies a compact deterministic sweep from well-conditioned to near-rank-deficient designs in exact and modest-noise modes. It measures condition numbers, Gram-matrix conditioning, solver-dependent coefficient and fitted-value behavior, response perturbation amplification, design-matrix sensitivity, and raw-versus-standardized Phase 1 coordinates. The conditioning-sweep RMSE is an in-sample fitted-value error because the same observations are used for fitting and evaluation; Phase 3 does not test generalization or out-of-sample forecasting. All observations remain synthetic; results are numerical-analysis evidence, not educational findings.

The executed sweep used nine values of $\delta$ from $1$ through $10^{-8}$, producing measured design condition numbers from approximately $3.50$ to $2.38\times10^8$. In the exact model at $\delta=10^{-8}$, Normal Equations had coefficient error $3.59\times10^{-1}$ and in-sample fitted-value RMSE $5.98\times10^{-9}$, while QR, SVD, and `lstsq` had coefficient errors near $10^{-9}$ and fitted-value RMSE near $10^{-15}$. In the ten-replicate modest-noise summary at the same condition level, coefficient error was much larger than at $\delta=1$, while median in-sample fitted-value RMSE remained approximately $0.143$ for all solvers. Thus, in these synthetic systems, parameter estimates can become extremely unstable while in-sample fitted values remain nearly unchanged.

For every nonzero $\delta$, $\mathrm{span}\{x_1,x_2,x_3\}=\mathrm{span}\{x_1,z,x_3\}$ because $x_2=x_1+\delta z$. The column space is therefore unchanged mathematically while the coordinate representation becomes ill-conditioned as $\delta\to0$. Since fitted values are projections $\hat y=P_Xy$, stable solvers can retain nearly unchanged fitted values while the coefficient coordinates become unstable; exact invariance should not be assumed in floating-point arithmetic. The response-perturbation amplification median increased from approximately $0.226$ at $\delta=1$ to approximately $9.84\times10^6$ at $\delta=10^{-8}$ for QR, SVD, and `lstsq`. Using the spectral matrix 2-norm, the corresponding design-perturbation study increased from approximately $0.334$ to $5.414\times10^3$ between $\delta=1$ and $\delta=10^{-6}$. For the Phase 1 training design, the raw condition number was $345.99$ and the standardized condition number was $1.332$.

## Phase 4: Ridge regularization and OLS uncertainty

Phase 4 uses centered SVD Ridge with an unpenalized intercept, training-only standardization, a dimension-aware $\lambda=\alpha\sigma_{\max}^2$ grid, validation and GCV selection, and a synthetic-only oracle coefficient diagnostic. Practical selection uses training/validation information only; the untouched test partition is evaluated only after each selection decision. The experiments cover moderate ($\delta=1$), strongly ill-conditioned ($\delta=10^{-3}$), and very strongly ill-conditioned ($\delta=10^{-6}$) regimes. In the strong regime, validation selected $\alpha\approx2.64\times10^{-8}$, GCV selected $\alpha\approx1.44\times10^{-7}$, and the synthetic-only coefficient oracle selected $\alpha\approx2.34\times10^{-5}$. In the very strong regime, the validation minimum was the unregularized point, while the oracle selected $\alpha\approx2.64\times10^{-8}$; this is an observed outcome, not a claim that validation is universally superior.

The bias–variance experiment used 30 deterministic Gaussian-noise replicates. For fixed $X$ and zero-mean noise, Ridge is linear in $y$, so the exact conditional estimator expectation is computed by fitting to $E[y\mid X]=X\beta^\star$; the bias curve is not the unstable finite-replicate sample-mean error. Empirical coefficient variance is measured around this expected estimator. The experiment uses the same dimensionless alpha grid as the path study but converts each alpha to an absolute lambda using the singular-value scale of its own standardized training design. The bias–variance figure plots relative bias squared, coefficient variance, and coefficient MSE after division by $\|\beta^\star\|_2^2$, making all displayed quantities dimensionless and directly comparable. In the very strong regime, the corrected unregularized conditional relative bias is approximately zero while coefficient variance is approximately $2.97\times10^9$; at $\alpha=100$, conditional relative bias is approximately $0.919$ and coefficient variance approximately $8.15\times10^{-4}$. In the regularization stability study, response-perturbation amplification decreased from approximately $9.03$ at $\alpha=0$ to approximately $0.125$ near $\alpha=2.07\times10^{-2}$ for the selected very-strong design, while regularization changed the fitted objective and introduced bias.

The uncertainty experiment selects existing design rows by measured leverage: median ($h=0.075289$), upper quartile ($h=0.088993$), and maximum ($h=0.132119$). Under 200 Gaussian OLS replications, empirical 95% mean-response CI coverage ranged from $0.935$ to $0.955$, and future-observation prediction-interval coverage ranged from $0.940$ to $0.960$. Mean-response interval widths ranged from approximately $0.654$ to $0.866$, while prediction-interval widths ranged from approximately $2.470$ to $2.535$. Prediction intervals include future noise and are therefore wider than conditional-mean confidence intervals. These analytical intervals are for classical OLS; they are not presented as exact Ridge intervals because Ridge is biased.

## Phase 5: early-semester forecasting, ablation, and robustness

Phase 5 uses a new controlled synthetic generator with four nested information checkpoints: pre-semester, early semester, mid semester, and later semester. The 20-seed clean study reuses each seed's train/validation/test partition across checkpoints and evaluates OLS, validation-selected Ridge, and transparent Huber IRLS. These checkpoints represent feature availability, not time-series dependence.

Mean held-out RMSE was:

| Checkpoint | OLS | Ridge | Huber |
|---|---:|---:|---:|
| Pre-semester | 1.839638 | 1.841547 | 1.839715 |
| Early semester | 1.568967 | 1.558670 | 1.568399 |
| Mid semester | 1.384252 | 1.379206 | 1.385090 |
| Later semester | 1.301504 | 1.296868 | 1.297964 |

Ridge had the lowest mean RMSE at the early-, mid-, and later-semester checkpoints. At the pre-semester checkpoint, OLS had the lowest mean RMSE by a very small numerical margin; all three estimators were close. The adjacent Ridge RMSE reductions were **0.283**, **0.179**, and **0.082**, respectively. These are finite-sample synthetic results, not significance claims.

The strongest replicated Ridge ablation result was removal of the combined assessment group, which increased RMSE by **0.195** on average. This is a predictive contribution measure conditional on the other features, not a causal or educational-importance claim. In the 20-seed robustness study, 20% response contamination of training targets produced mean clean-test RMSE degradation of **0.315** for OLS, **0.156** for Ridge, and **0.072** for Huber, with degradation standard deviations of **0.177**, **0.107**, and **0.085**, respectively. Under 20% feature contamination applied to raw training predictors before preprocessing, mean degradation was **0.089** for OLS, **0.103** for Ridge, and **0.082** for Huber, with standard deviations of **0.081**, **0.067**, and **0.074**. Huber was more resistant to the specified response contamination, but it did not universally dominate under feature contamination; it does not automatically solve high-leverage predictor corruption.

All Phase 5 results remain controlled synthetic evidence. They do not establish actual student forecasting performance, empirical realism, causality, deployment readiness, or universal superiority of any model.

See [`docs/problem_formulation.md`](docs/problem_formulation.md) and [`docs/mathematical_notes.md`](docs/mathematical_notes.md) for the formal model, derivations, assumptions, and numerical interpretation.

The main [technical report](docs/technical_report.md), [reproducibility guide](docs/reproducibility.md), and [results index](docs/results_index.md) provide detailed interpretation, exact commands, and artifact provenance. The [project summary](docs/project_summary.md) is a concise reviewer-oriented overview. The [research-integrity statement](docs/research_integrity.md) records the synthetic-data and claim boundaries, and [`CITATION.cff`](CITATION.cff) contains the software citation metadata.
