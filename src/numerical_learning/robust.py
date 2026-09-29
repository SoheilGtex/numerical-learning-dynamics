"""Transparent Huber regression using iteratively reweighted least squares."""

from dataclasses import dataclass

import numpy as np

from .linear_models import add_intercept


@dataclass(frozen=True)
class HuberResult:
    """Huber coefficients and explicit convergence diagnostics."""

    coefficients: np.ndarray
    converged: bool
    iterations: int
    scale: float
    delta: float
    residual_norm: float


def mad_scale(residuals: np.ndarray) -> float:
    """Return the MAD-based robust residual scale, with a zero-scale guard."""
    residuals = np.asarray(residuals, dtype=float)
    median = np.median(residuals)
    return float(np.median(np.abs(residuals - median)) / 0.6744897501960817)


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Return IRLS weights for the Huber loss."""
    residuals = np.asarray(residuals, dtype=float)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    absolute = np.abs(residuals)
    return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, np.finfo(float).eps))


def fit_huber_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tuning_constant: float = 1.345,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    fit_intercept: bool = True,
) -> HuberResult:
    """Fit Huber regression using stable weighted least squares at each iteration."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or X.shape[0] == 0:
        raise ValueError("X and y must have compatible non-empty dimensions")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    if tuning_constant <= 0 or not np.isfinite(tuning_constant):
        raise ValueError("tuning_constant must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    design = add_intercept(X) if fit_intercept else X
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    scale = mad_scale(y - design @ coefficients)
    if scale <= np.finfo(float).eps:
        scale = max(float(np.std(y - design @ coefficients)), 1.0) * np.finfo(float).eps
    delta = tuning_constant * scale
    converged = False
    for iteration in range(1, max_iterations + 1):
        residuals = y - design @ coefficients
        weights = huber_weights(residuals, delta)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        new_coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        change = np.linalg.norm(new_coefficients - coefficients) / max(
            1.0, np.linalg.norm(coefficients)
        )
        coefficients = new_coefficients
        if change < tolerance:
            converged = True
            break
    residuals = y - design @ coefficients
    return HuberResult(
        coefficients=coefficients,
        converged=converged,
        iterations=iteration,
        scale=scale,
        delta=delta,
        residual_norm=float(np.linalg.norm(residuals)),
    )
