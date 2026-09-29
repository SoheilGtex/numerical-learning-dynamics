# Research Integrity Statement

## Data

All datasets in this repository are controlled synthetic simulations. No actual Kharazmi University student records or other student-level records are included.

## Origin

The project originated from a classroom question posed by Dr. Jamshid Saeidian during a Numerical Analysis class at Kharazmi University. The subsequent mathematical formulation, simulation design, software implementation, experiments, and interpretation were developed independently. This does not imply supervision, endorsement, coauthorship, or formal collaboration.

## Claims

The repository makes no causal educational claims, no empirical claim about Kharazmi students, no publication claim, and no novelty claim. Results are evidence about the executed synthetic numerical experiments only.

## Oracle information

The synthetic Phase 1–4 generators expose known coefficients and generating parameters. This enables oracle diagnostics such as coefficient-error-based Ridge selection that would not be available on ordinary real data. Practical selection remains separated from oracle diagnostics.

## Test set

Phases 4–5 preserve strict test holdout for practical model selection. Validation information is used for Ridge selection; test outcomes are evaluated only after selection. Contamination experiments alter training observations only and keep validation/test observations clean.

## Uncertainty

Classical OLS confidence and prediction intervals rely on the specified Gaussian linear-model assumptions. Their empirical coverage verification does not automatically transfer to Ridge or Huber, and no such analytical interval claim is made.
