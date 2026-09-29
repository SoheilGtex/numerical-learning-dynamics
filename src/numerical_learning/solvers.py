"""Explicit least-squares solvers for the Phase 2 numerical study."""

from dataclasses import dataclass

import numpy as np

from .linear_models import add_intercept


@dataclass(frozen=True)
class SolverResult:
    """Common solver output with numerical diagnostics."""

    coefficients: np.ndarray
    residual_norm: float
    rank: int | None
    singular_values: np.ndarray | None
    status: str = "success"
    tolerance: float | None = None


def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate regression inputs before a solver-specific computation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if X.shape[0] != y.size:
        raise ValueError("X and y must contain the same number of observations")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    return X, y


def _design(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Apply the project-wide explicit-intercept convention."""
    return add_intercept(X) if fit_intercept else X


def _result(
    design: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    rank: int | None,
    singular_values: np.ndarray | None,
    status: str = "success",
    tolerance: float | None = None,
) -> SolverResult:
    """Build a result and compute its residual norm."""
    residual_norm = float(np.linalg.norm(design @ coefficients - y))
    return SolverResult(
        coefficients=coefficients,
        residual_norm=residual_norm,
        rank=rank,
        singular_values=singular_values,
        status=status,
        tolerance=tolerance,
    )


def solve_normal_equations(
    X: np.ndarray, y: np.ndarray, fit_intercept: bool = True
) -> SolverResult:
    """Solve least squares through ``(X.T @ X) beta = X.T @ y``.

    The normal-equations matrix is solved directly; its inverse is never
    formed. A singular or numerically unsolvable system raises ``LinAlgError``
    rather than silently switching to another method.
    """
    X, y = _validate_inputs(X, y)
    design = _design(X, fit_intercept)
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise np.linalg.LinAlgError(
            f"normal equations require full column rank; detected rank {rank}"
        )
    gram = design.T @ design
    rhs = design.T @ y
    coefficients = np.linalg.solve(gram, rhs)
    singular_values = np.linalg.svd(design, compute_uv=False)
    return _result(
        design,
        y,
        coefficients,
        rank=rank,
        singular_values=singular_values,
    )


def solve_qr(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> SolverResult:
    """Solve full-column-rank least squares through reduced unpivoted QR."""
    X, y = _validate_inputs(X, y)
    design = _design(X, fit_intercept)
    if design.shape[0] < design.shape[1]:
        raise np.linalg.LinAlgError("unpivoted QR requires an overdetermined system")
    Q, R = np.linalg.qr(design, mode="reduced")
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank < design.shape[1]:
        raise np.linalg.LinAlgError(
            f"unpivoted QR requires full column rank; detected rank {rank}"
        )
    coefficients = np.linalg.solve(R, Q.T @ y)
    return _result(
        design,
        y,
        coefficients,
        rank=rank,
        singular_values=singular_values,
        tolerance=tolerance,
    )


def solve_svd(
    X: np.ndarray,
    y: np.ndarray,
    rcond: float | None = None,
    fit_intercept: bool = True,
) -> SolverResult:
    """Solve least squares with an explicit SVD pseudoinverse.

    If ``rcond`` is ``None``, singular values at or below
    ``eps * max(m, n) * sigma_max`` are discarded. Otherwise, the threshold is
    ``rcond * sigma_max``. The returned rank and tolerance expose this choice.
    """
    X, y = _validate_inputs(X, y)
    design = _design(X, fit_intercept)
    U, singular_values, Vt = np.linalg.svd(design, full_matrices=False)
    if singular_values.size == 0:
        raise np.linalg.LinAlgError("SVD returned no singular values")
    if rcond is not None and rcond < 0:
        raise ValueError("rcond must be non-negative or None")
    factor = np.finfo(float).eps * max(design.shape) if rcond is None else rcond
    tolerance = float(factor * singular_values[0])
    retained = singular_values > tolerance
    rank = int(np.sum(retained))
    inverse = np.zeros_like(singular_values)
    inverse[retained] = 1.0 / singular_values[retained]
    coefficients = Vt.T @ (inverse * (U.T @ y))
    return _result(
        design,
        y,
        coefficients,
        rank=rank,
        singular_values=singular_values,
        tolerance=tolerance,
    )


def condition_number(X: np.ndarray) -> float:
    """Return the 2-norm condition number, or infinity for numerical rank loss."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    singular_values = np.linalg.svd(X, compute_uv=False)
    if np.linalg.matrix_rank(X) < min(X.shape):
        return float("inf")
    return float(singular_values[0] / singular_values[-1])
