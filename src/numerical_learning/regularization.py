"""Explicit centered SVD Ridge/Tikhonov regularization utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeResult:
    """Ridge coefficients and diagnostics for an unpenalized intercept."""

    coefficients: np.ndarray
    lambda_: float
    singular_values: np.ndarray
    inversion_factors: np.ndarray
    shrinkage_factors: np.ndarray
    effective_df: float
    residual_norm: float
    regularized_system_condition_number: float


def _validate_inputs(X: np.ndarray, y: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2-D and y must be 1-D")
    if X.shape[0] != y.size or X.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero row count")
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and non-negative")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    return X, y


def ridge_filter_factors(singular_values: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficient inversion and hat-matrix shrinkage factors."""
    singular_values = np.asarray(singular_values, dtype=float)
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and non-negative")
    denominator = singular_values**2 + lambda_
    if np.any(denominator == 0):
        raise ValueError("lambda_ and singular values cannot both be zero")
    inversion = singular_values / denominator
    shrinkage = singular_values**2 / denominator
    return inversion, shrinkage


def solve_ridge_svd(X: np.ndarray, y: np.ndarray, lambda_: float) -> RidgeResult:
    """Solve centered Ridge with an unpenalized intercept using an SVD."""
    X, y = _validate_inputs(X, y, lambda_)
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    X_centered = X - x_mean
    y_centered = y - y_mean
    U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
    inversion, shrinkage = ridge_filter_factors(singular_values, lambda_)
    slopes = Vt.T @ (inversion * (U.T @ y_centered))
    intercept = y_mean - x_mean @ slopes
    coefficients = np.concatenate(([intercept], slopes))
    residual = np.column_stack((np.ones(X.shape[0]), X)) @ coefficients - y
    if singular_values.size == 0:
        regularized_condition = 1.0
    else:
        regularized_condition = float(
            (singular_values[0] ** 2 + lambda_)
            / (singular_values[-1] ** 2 + lambda_)
        )
    return RidgeResult(
        coefficients=coefficients,
        lambda_=float(lambda_),
        singular_values=singular_values,
        inversion_factors=inversion,
        shrinkage_factors=shrinkage,
        effective_df=float(1.0 + shrinkage.sum()),
        residual_norm=float(np.linalg.norm(residual)),
        regularized_system_condition_number=regularized_condition,
    )


def ridge_gcv_score(X: np.ndarray, y: np.ndarray, lambda_: float) -> float:
    """Compute GCV using the centered Ridge effective degrees of freedom."""
    X, y = _validate_inputs(X, y, lambda_)
    result = solve_ridge_svd(X, y, lambda_)
    denominator = 1.0 - result.effective_df / X.shape[0]
    if denominator <= 0 or not np.isfinite(denominator):
        return float("inf")
    return float((result.residual_norm**2 / X.shape[0]) / denominator**2)


def select_lambda_by_validation(validation_rmse: np.ndarray) -> int:
    """Return the exact validation-error minimizer index.

    Only validation errors are accepted; test targets and synthetic ground
    truth are structurally unavailable to this practical selector.
    """
    validation_rmse = np.asarray(validation_rmse, dtype=float)
    if validation_rmse.ndim != 1 or validation_rmse.size == 0:
        raise ValueError("validation_rmse must be a non-empty one-dimensional array")
    if not np.isfinite(validation_rmse).all():
        raise ValueError("validation_rmse must contain finite values")
    return int(np.argmin(validation_rmse))


def select_lambda_by_gcv(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> int:
    """Return the GCV minimizer index using training data only."""
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("lambdas must be a non-empty one-dimensional array")
    scores = np.asarray([ridge_gcv_score(X, y, value) for value in lambdas])
    return int(np.nanargmin(scores))


def select_lambda_oracle(coefficient_errors: np.ndarray) -> int:
    """Return a synthetic-only oracle index from precomputed coefficient errors."""
    coefficient_errors = np.asarray(coefficient_errors, dtype=float)
    if coefficient_errors.ndim != 1 or coefficient_errors.size == 0:
        raise ValueError("coefficient_errors must be a non-empty one-dimensional array")
    if not np.isfinite(coefficient_errors).all():
        raise ValueError("coefficient_errors must contain finite values")
    return int(np.argmin(coefficient_errors))


def relative_coefficient_bias(expected_coefficients: np.ndarray, true_coefficients: np.ndarray) -> float:
    """Return exact conditional relative bias from an expected estimator vector."""
    expected_coefficients = np.asarray(expected_coefficients, dtype=float)
    true_coefficients = np.asarray(true_coefficients, dtype=float)
    if expected_coefficients.shape != true_coefficients.shape:
        raise ValueError("expected and true coefficients must have matching shapes")
    denominator = np.linalg.norm(true_coefficients)
    if denominator == 0:
        raise ValueError("true coefficients must have non-zero norm")
    return float(np.linalg.norm(expected_coefficients - true_coefficients) / denominator)


def coefficient_variance_around_expected(
    estimates: np.ndarray, expected_coefficients: np.ndarray
) -> float:
    """Return mean squared coefficient deviation around the supplied expected vector."""
    estimates = np.asarray(estimates, dtype=float)
    expected_coefficients = np.asarray(expected_coefficients, dtype=float)
    if estimates.ndim != 2 or estimates.shape[1:] != expected_coefficients.shape:
        raise ValueError("estimates must be 2-D with coefficient-shaped rows")
    return float(np.mean(np.sum((estimates - expected_coefficients) ** 2, axis=1)))


def dimension_aware_lambda_grid(X: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert dimensionless alpha values to lambdas using centered X."""
    X = np.asarray(X, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if X.ndim != 2 or alphas.ndim != 1:
        raise ValueError("X must be 2-D and alphas must be 1-D")
    if np.any(~np.isfinite(alphas)) or np.any(alphas < 0):
        raise ValueError("alphas must be finite and non-negative")
    sigma_max = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)[0]
    lambdas = alphas * sigma_max**2
    return alphas, lambdas
