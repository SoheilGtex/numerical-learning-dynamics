"""Controlled synthetic data for the Phase 1 linear model."""

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = (
    "prior_gpa",
    "prerequisite_grade",
    "attendance_rate",
    "first_quiz",
    "assignment_performance",
)
TRUE_COEFFICIENTS = np.array([1.0, 0.18, 0.16, 2.0, 0.12, 0.16], dtype=float)


@dataclass(frozen=True)
class SyntheticDataset:
    """A synthetic regression dataset and its known generating parameters."""

    X: np.ndarray
    y: np.ndarray
    true_coefficients: np.ndarray
    feature_names: tuple[str, ...]


def generate_synthetic_data(
    n_samples: int = 160,
    seed: int = 2026,
    noise_std: float = 0.6,
) -> SyntheticDataset:
    """Generate reproducible features and targets from ``y = X beta + noise``.

    Features use natural grade/rate ranges. The returned ``X`` excludes the
    intercept; ``true_coefficients`` includes the intercept as its first entry.
    Targets are not clipped so the linear data-generating relationship remains
    exact, including in low-noise tests.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    X = np.column_stack(
        (
            rng.uniform(8.0, 20.0, n_samples),
            rng.uniform(8.0, 20.0, n_samples),
            rng.uniform(0.60, 1.0, n_samples),
            rng.uniform(5.0, 20.0, n_samples),
            rng.uniform(5.0, 20.0, n_samples),
        )
    )
    design = np.column_stack((np.ones(n_samples), X))
    noise = rng.normal(0.0, noise_std, n_samples)
    y = design @ TRUE_COEFFICIENTS + noise
    return SyntheticDataset(
        X=X,
        y=y,
        true_coefficients=TRUE_COEFFICIENTS.copy(),
        feature_names=FEATURE_NAMES,
    )
