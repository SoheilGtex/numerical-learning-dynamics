"""Classical Gaussian linear-model uncertainty calculations for OLS only."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class Interval:
    """Two-sided interval endpoints and point estimate."""

    estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence_level: float


def leverage_values(design: np.ndarray) -> np.ndarray:
    """Return diagonal hat-matrix values using a stable reduced QR factorization."""
    design = np.asarray(design, dtype=float)
    if design.ndim != 2 or design.shape[0] < design.shape[1]:
        raise ValueError("design must have at least as many rows as columns")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("design must have full column rank")
    q, _ = np.linalg.qr(design, mode="reduced")
    return np.sum(q**2, axis=1)


def estimate_noise_variance(X: np.ndarray, y: np.ndarray, coefficients: np.ndarray) -> float:
    """Estimate sigma^2 with RSS divided by n minus fitted parameter count."""
    design = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if design.ndim != 2 or y.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("X, y, and coefficients have invalid dimensions")
    if design.shape[0] != y.size or design.shape[1] != coefficients.size:
        raise ValueError("dimensions do not agree")
    degrees_of_freedom = y.size - coefficients.size
    if degrees_of_freedom <= 0:
        raise ValueError("n must be greater than the number of fitted parameters")
    rss = float(np.sum((design @ coefficients - y) ** 2))
    return rss / degrees_of_freedom


def ols_covariance(X: np.ndarray, sigma_squared: float) -> np.ndarray:
    """Estimate OLS coefficient covariance using an SVD, without inversion."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] <= X.shape[1]:
        raise ValueError("X must have more rows than columns")
    if not np.isfinite(sigma_squared) or sigma_squared < 0:
        raise ValueError("sigma_squared must be finite and non-negative")
    _, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1] or np.any(singular_values == 0):
        raise ValueError("X must have full column rank")
    covariance = sigma_squared * (Vt.T / singular_values**2) @ Vt
    return 0.5 * (covariance + covariance.T)


def _validate_interval_inputs(
    x0: np.ndarray, coefficients: np.ndarray, covariance: np.ndarray, sigma_squared: float, n: int, confidence_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    x0 = np.asarray(x0, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    p = coefficients.size
    if x0.ndim != 1 or coefficients.ndim != 1 or covariance.shape != (p, p) or x0.size != p:
        raise ValueError("interval dimensions do not agree")
    if n <= p:
        raise ValueError("n must be greater than the number of fitted parameters")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not np.isfinite(sigma_squared) or sigma_squared < 0:
        raise ValueError("sigma_squared must be finite and non-negative")
    return x0, coefficients, covariance, sigma_squared, n, confidence_level


def mean_confidence_interval(
    x0: np.ndarray, coefficients: np.ndarray, covariance: np.ndarray, sigma_squared: float, n: int, confidence_level: float = 0.95
) -> Interval:
    """Construct a Student-t interval for the conditional mean response."""
    x0, coefficients, covariance, sigma_squared, n, confidence_level = _validate_interval_inputs(
        x0, coefficients, covariance, sigma_squared, n, confidence_level
    )
    estimate = float(x0 @ coefficients)
    variance = max(0.0, float(x0 @ covariance @ x0))
    standard_error = np.sqrt(variance)
    critical = float(t.ppf(0.5 + confidence_level / 2.0, n - coefficients.size))
    half_width = critical * standard_error
    return Interval(estimate, estimate - half_width, estimate + half_width, standard_error, confidence_level)


def prediction_interval(
    x0: np.ndarray, coefficients: np.ndarray, covariance: np.ndarray, sigma_squared: float, n: int, confidence_level: float = 0.95
) -> Interval:
    """Construct a Student-t interval for a future noisy observation."""
    x0, coefficients, covariance, sigma_squared, n, confidence_level = _validate_interval_inputs(
        x0, coefficients, covariance, sigma_squared, n, confidence_level
    )
    estimate = float(x0 @ coefficients)
    variance = max(0.0, float(sigma_squared + x0 @ covariance @ x0))
    standard_error = np.sqrt(variance)
    critical = float(t.ppf(0.5 + confidence_level / 2.0, n - coefficients.size))
    half_width = critical * standard_error
    return Interval(estimate, estimate - half_width, estimate + half_width, standard_error, confidence_level)


def coefficient_confidence_intervals(
    coefficients: np.ndarray, covariance: np.ndarray, n: int, confidence_level: float = 0.95
) -> np.ndarray:
    """Return coefficient-wise classical OLS t-intervals."""
    coefficients = np.asarray(coefficients, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (coefficients.size, coefficients.size) or n <= coefficients.size:
        raise ValueError("invalid coefficient covariance or degrees of freedom")
    critical = float(t.ppf(0.5 + confidence_level / 2.0, n - coefficients.size))
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return np.column_stack((coefficients - critical * standard_errors, coefficients + critical * standard_errors))
