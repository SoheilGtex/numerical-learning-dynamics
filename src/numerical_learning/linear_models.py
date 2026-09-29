"""Linear least-squares baseline using NumPy's trusted implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeastSquaresResult:
    """Fitted coefficients and diagnostics returned by the baseline fit."""

    coefficients: np.ndarray
    residual_sum_of_squares: float
    rank: int
    singular_values: np.ndarray


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend one leading intercept column to the supplied matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    return np.column_stack((np.ones(X.shape[0]), X))


def fit_least_squares(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> LeastSquaresResult:
    """Fit ordinary least squares with ``numpy.linalg.lstsq``.

    By default, ``X`` contains predictors without an intercept and this
    function adds one. Set ``fit_intercept=False`` only when ``X`` already
    contains the complete design matrix.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations")
    if X.shape[0] == 0:
        raise ValueError("X and y must not be empty")
    design = add_intercept(X) if fit_intercept else X
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        design, y, rcond=None
    )
    residual_sum_of_squares = float(residuals[0]) if residuals.size else float(
        np.sum((design @ coefficients - y) ** 2)
    )
    return LeastSquaresResult(
        coefficients=coefficients,
        residual_sum_of_squares=residual_sum_of_squares,
        rank=int(rank),
        singular_values=singular_values,
    )


def predict(X: np.ndarray, coefficients: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """Generate predictions using fitted coefficients."""
    X = np.asarray(X, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if X.ndim != 2 or coefficients.ndim != 1:
        raise ValueError("X must be 2-D and coefficients must be 1-D")
    design = add_intercept(X) if fit_intercept else X
    if design.shape[1] != coefficients.size:
        raise ValueError("coefficient count does not match the design matrix")
    return design @ coefficients
