from inspect import signature

import numpy as np
import pytest

from numerical_learning.conditioning import (
    raw_to_standardized_coefficients,
    standardize_predictors,
    standardized_to_raw_coefficients,
)
from numerical_learning.linear_models import add_intercept
from numerical_learning.regularization import (
    coefficient_variance_around_expected,
    dimension_aware_lambda_grid,
    relative_coefficient_bias,
    ridge_filter_factors,
    ridge_gcv_score,
    select_lambda_by_gcv,
    select_lambda_by_validation,
    select_lambda_oracle,
    solve_ridge_svd,
)
from numerical_learning.solvers import solve_svd


def test_zero_ridge_matches_unregularized_svd() -> None:
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y = np.array([2.0, 3.0, 5.0, 6.0])
    ridge = solve_ridge_svd(X, y, 0.0)
    svd = solve_svd(X, y)
    np.testing.assert_allclose(ridge.coefficients, svd.coefficients, atol=1e-12)


def test_intercept_is_not_penalized_and_large_lambda_approaches_mean() -> None:
    X = np.arange(1.0, 7.0).reshape(3, 2)
    y = np.full(3, 7.5)
    result = solve_ridge_svd(X, y, 1e12)
    assert result.coefficients[0] == pytest.approx(7.5, rel=1e-8)
    assert np.linalg.norm(result.coefficients[1:]) < 1e-8


def test_shrinkage_and_effective_df_decrease_with_lambda() -> None:
    singular = np.array([5.0, 1.0])
    small_inv, small_shrink = ridge_filter_factors(singular, 1e-3)
    large_inv, large_shrink = ridge_filter_factors(singular, 1e3)
    assert np.all((small_shrink >= 0) & (small_shrink <= 1))
    assert np.all(large_shrink < small_shrink)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 2.0, 2.5, 3.0])
    assert solve_ridge_svd(X, y, 1e3).effective_df < solve_ridge_svd(X, y, 1e-3).effective_df
    assert np.all(large_inv < small_inv)


def test_ridge_standardized_coordinates_preserve_raw_predictions() -> None:
    X = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 7.0], [5.0, 10.0]])
    y = np.array([2.0, 4.0, 5.0, 8.0])
    standardized, transform = standardize_predictors(X)
    standardized_result = solve_ridge_svd(standardized, y, 0.5)
    raw_from_standardized = standardized_to_raw_coefficients(
        standardized_result.coefficients, transform
    )
    np.testing.assert_allclose(
        add_intercept(standardized) @ standardized_result.coefficients,
        add_intercept(X) @ raw_from_standardized,
    )
    np.testing.assert_allclose(
        raw_to_standardized_coefficients(raw_from_standardized, transform),
        standardized_result.coefficients,
    )


def test_invalid_lambda_is_rejected() -> None:
    X = np.ones((3, 1))
    y = np.ones(3)
    with pytest.raises(ValueError):
        solve_ridge_svd(X, y, -1.0)
    with pytest.raises(ValueError):
        solve_ridge_svd(X, y, np.inf)


def test_gcv_is_finite_and_deterministic() -> None:
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
    y = np.array([2.0, 3.0, 5.0, 6.0, 8.0])
    lambdas = np.array([0.0, 0.1, 1.0, 10.0])
    first = np.array([ridge_gcv_score(X, y, value) for value in lambdas])
    second = np.array([ridge_gcv_score(X, y, value) for value in lambdas])
    assert np.isfinite(first).all()
    assert np.all(first >= 0)
    np.testing.assert_array_equal(first, second)
    assert select_lambda_by_gcv(X, y, lambdas) == int(np.argmin(first))


def test_dimension_aware_grid_and_selector_separation() -> None:
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    alphas = np.array([0.0, 0.5, 2.0])
    returned_alphas, lambdas = dimension_aware_lambda_grid(X, alphas)
    sigma_max = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)[0]
    np.testing.assert_array_equal(returned_alphas, alphas)
    np.testing.assert_allclose(lambdas, alphas * sigma_max**2)
    assert lambdas[0] == 0.0
    assert select_lambda_by_validation(np.array([3.0, 1.0, 2.0])) == 1
    assert select_lambda_oracle(np.array([2.0, 3.0, 1.0])) == 2


def test_same_alpha_grid_scales_to_each_training_design() -> None:
    alphas = np.array([0.0, 0.5, 2.0])
    X_first = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    X_second = 3.0 * X_first
    _, lambdas_first = dimension_aware_lambda_grid(X_first, alphas)
    _, lambdas_second = dimension_aware_lambda_grid(X_second, alphas)
    assert lambdas_first[0] == 0.0
    assert lambdas_second[0] == 0.0
    np.testing.assert_allclose(lambdas_second, 9.0 * lambdas_first)
    assert "test_targets" not in signature(select_lambda_by_validation).parameters
    assert "test_targets" not in signature(select_lambda_by_gcv).parameters
    assert "beta_star" not in signature(select_lambda_by_validation).parameters
    assert "beta_star" not in signature(select_lambda_by_gcv).parameters


def test_conditional_ols_bias_is_zero_and_ridge_bias_is_nonzero() -> None:
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
    beta = np.array([1.5, -0.75, 2.0])
    y_expected = add_intercept(X) @ beta
    ols = solve_ridge_svd(X, y_expected, 0.0)
    ridge = solve_ridge_svd(X, y_expected, 1.0)
    np.testing.assert_allclose(ols.coefficients, beta, atol=1e-12)
    assert relative_coefficient_bias(ols.coefficients, beta) < 1e-12
    assert relative_coefficient_bias(ridge.coefficients, beta) > 0


def test_variance_is_centered_on_supplied_expected_estimator() -> None:
    expected = np.array([1.0, 2.0])
    estimates = np.array([[2.0, 2.0], [0.0, 2.0], [1.0, 3.0]])
    assert coefficient_variance_around_expected(estimates, expected) == pytest.approx(1.0)
    assert coefficient_variance_around_expected(estimates, estimates.mean(axis=0)) != pytest.approx(1.0)
