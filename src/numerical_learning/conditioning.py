"""Utilities for controlled conditioning and perturbation experiments."""

from dataclasses import dataclass

import numpy as np

from .linear_models import add_intercept
from .solvers import condition_number


@dataclass(frozen=True)
class Standardization:
    """Column means and scales for predictor standardization."""

    means: np.ndarray
    scales: np.ndarray


def make_near_collinear_predictors(
    n_samples: int = 80,
    delta: float = 1.0,
    seed: int = 31415,
) -> np.ndarray:
    """Construct predictors whose second column is ``x1 + delta * z``."""
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    x3 = rng.normal(size=n_samples)
    return np.column_stack((x1, x1 + delta * z, x3))


def standardize_predictors(X: np.ndarray) -> tuple[np.ndarray, Standardization]:
    """Standardize predictor columns and return the transformation parameters."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    if np.any(scales == 0.0):
        raise ValueError("cannot standardize a constant predictor")
    return (X - means) / scales, Standardization(means=means, scales=scales)


def raw_to_standardized_coefficients(
    beta_raw: np.ndarray, transformation: Standardization
) -> np.ndarray:
    """Map raw-coordinate coefficients to standardized predictor coordinates."""
    beta_raw = np.asarray(beta_raw, dtype=float)
    if beta_raw.ndim != 1 or beta_raw.size != transformation.means.size + 1:
        raise ValueError("coefficient dimension does not match transformation")
    gamma = np.empty_like(beta_raw)
    gamma[1:] = beta_raw[1:] * transformation.scales
    gamma[0] = beta_raw[0] + np.dot(beta_raw[1:], transformation.means)
    return gamma


def standardized_to_raw_coefficients(
    gamma: np.ndarray, transformation: Standardization
) -> np.ndarray:
    """Map standardized-coordinate coefficients back to raw coordinates."""
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 1 or gamma.size != transformation.means.size + 1:
        raise ValueError("coefficient dimension does not match transformation")
    beta = np.empty_like(gamma)
    beta[1:] = gamma[1:] / transformation.scales
    beta[0] = gamma[0] - np.dot(beta[1:], transformation.means)
    return beta


def relative_perturbation(
    values: np.ndarray,
    magnitude: float,
    seed: int,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Add a deterministic perturbation with prescribed relative 2-norm."""
    values = np.asarray(values, dtype=float)
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.normal(size=values.shape)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != values.shape:
        raise ValueError("direction must have the same shape as values")
    direction_norm = np.linalg.norm(direction.ravel())
    values_norm = np.linalg.norm(values.ravel())
    if direction_norm == 0.0 or values_norm == 0.0:
        raise ValueError("values and direction must have non-zero norms")
    return values + magnitude * values_norm * direction / direction_norm


def relative_matrix_perturbation(
    matrix: np.ndarray,
    magnitude: float,
    seed: int,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Add a deterministic matrix perturbation measured in spectral 2-norm."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.normal(size=matrix.shape)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != matrix.shape:
        raise ValueError("direction must have the same shape as matrix")
    direction_norm = np.linalg.norm(direction, ord=2)
    matrix_norm = np.linalg.norm(matrix, ord=2)
    if direction_norm == 0.0 or matrix_norm == 0.0:
        raise ValueError("matrix and direction must have non-zero spectral norms")
    return matrix + magnitude * matrix_norm * direction / direction_norm


def design_condition_diagnostics(X: np.ndarray) -> dict[str, float | int]:
    """Return condition, Gram condition, rank, and singular-value diagnostics."""
    X = np.asarray(X, dtype=float)
    design = add_intercept(X)
    singular_values = np.linalg.svd(design, compute_uv=False)
    gram = design.T @ design
    gram_condition = condition_number(gram)
    return {
        "condition_number": condition_number(design),
        "gram_condition_number": gram_condition,
        "numerical_rank": int(np.linalg.matrix_rank(design)),
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
    }
