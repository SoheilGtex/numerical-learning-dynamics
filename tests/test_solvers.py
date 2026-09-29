import numpy as np
import pytest

from numerical_learning.linear_models import add_intercept
from numerical_learning.solvers import (
    condition_number,
    solve_normal_equations,
    solve_qr,
    solve_svd,
)


def test_well_conditioned_solvers_agree() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    beta = np.array([1.0, 0.5, -1.0, 2.0])
    y = add_intercept(X) @ beta + rng.normal(0.0, 0.01, 40)
    results = [
        solve_normal_equations(X, y),
        solve_qr(X, y),
        solve_svd(X, y),
    ]
    reference = np.linalg.lstsq(add_intercept(X), y, rcond=None)[0]
    for result in results:
        np.testing.assert_allclose(result.coefficients, reference, rtol=1e-10, atol=1e-10)


def test_noiseless_known_solution_is_recovered() -> None:
    X = np.array(
        [
            [-1.0, 0.0, 2.0],
            [0.5, 1.0, -1.0],
            [2.0, -0.5, 0.0],
            [-0.25, 2.0, 1.0],
            [1.5, 0.25, -2.0],
            [0.0, -1.5, 0.5],
        ]
    )
    beta = np.array([2.0, -0.5, 1.25, 3.0])
    y = add_intercept(X) @ beta
    for solver in (solve_normal_equations, solve_qr, solve_svd):
        np.testing.assert_allclose(solver(X, y).coefficients, beta, atol=1e-10)


def test_rank_deficiency_is_explicit_and_svd_is_finite() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    X = np.column_stack((x, 2.0 * x, x**2))
    beta = np.array([1.0, 0.5, 2.0, -0.25])
    y = add_intercept(X) @ beta
    svd_result = solve_svd(X, y)
    assert svd_result.rank == 3
    assert np.isfinite(svd_result.coefficients).all()
    assert svd_result.residual_norm < 1e-10
    assert not np.allclose(svd_result.coefficients, beta)
    np.testing.assert_allclose(
        add_intercept(X) @ svd_result.coefficients,
        y,
        atol=1e-10,
    )
    with pytest.raises(np.linalg.LinAlgError, match="full column rank"):
        solve_normal_equations(X, y)
    with pytest.raises(np.linalg.LinAlgError, match="full column rank"):
        solve_qr(X, y)


def test_svd_tolerance_controls_effective_rank() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=30)
    z = rng.normal(size=30)
    X = np.column_stack((x, x + 1e-12 * z))
    y = add_intercept(X) @ np.array([1.0, 2.0, -1.0])
    default_result = solve_svd(X, y)
    truncated_result = solve_svd(X, y, rcond=1e-8)
    assert default_result.rank == 3
    assert truncated_result.rank == 2
    assert truncated_result.tolerance > default_result.tolerance


def test_invalid_inputs_and_unsupported_qr_case() -> None:
    with pytest.raises(ValueError, match="same number"):
        solve_svd(np.ones((3, 2)), np.ones(2))
    with pytest.raises(ValueError, match="two-dimensional"):
        solve_qr(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="must not be empty"):
        solve_normal_equations(np.empty((0, 2)), np.empty(0))
    with pytest.raises(np.linalg.LinAlgError, match="overdetermined"):
        solve_qr(np.ones((2, 3)), np.ones(2))


def test_condition_number_reports_singularity() -> None:
    assert np.isfinite(condition_number(np.eye(3)))
    assert condition_number(np.array([[1.0, 2.0], [2.0, 4.0]])) == np.inf
