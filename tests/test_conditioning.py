import numpy as np
import pytest

from numerical_learning.conditioning import (
    design_condition_diagnostics,
    make_near_collinear_predictors,
    raw_to_standardized_coefficients,
    relative_matrix_perturbation,
    relative_perturbation,
    standardize_predictors,
    standardized_to_raw_coefficients,
)
from numerical_learning.linear_models import add_intercept
from numerical_learning.solvers import condition_number


def test_diagonal_condition_number() -> None:
    assert condition_number(np.diag([1.0, 2.0])) == pytest.approx(2.0)


def test_collinearity_parameter_controls_conditioning() -> None:
    condition_loose = design_condition_diagnostics(
        make_near_collinear_predictors(delta=1e-1, seed=4)
    )["condition_number"]
    condition_tight = design_condition_diagnostics(
        make_near_collinear_predictors(delta=1e-5, seed=4)
    )["condition_number"]
    assert condition_tight > 100.0 * condition_loose


def test_gram_condition_is_squared_for_full_rank_matrix() -> None:
    X = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [-1.0, 1.0]])
    design = add_intercept(X)
    diagnostics = design_condition_diagnostics(X)
    gram_condition = np.linalg.cond(design.T @ design, 2)
    assert diagnostics["numerical_rank"] == design.shape[1]
    assert gram_condition == pytest.approx(
        diagnostics["condition_number"] ** 2, rel=1e-12
    )


def test_scaling_coefficient_transformation_preserves_predictions() -> None:
    X = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 7.0], [5.0, 10.0]])
    standardized, transformation = standardize_predictors(X)
    beta = np.array([2.0, -0.5, 1.25])
    gamma = raw_to_standardized_coefficients(beta, transformation)
    recovered = standardized_to_raw_coefficients(gamma, transformation)
    np.testing.assert_allclose(
        add_intercept(X) @ beta,
        add_intercept(standardized) @ gamma,
    )
    np.testing.assert_allclose(recovered, beta)


def test_perturbation_is_reproducible_and_zero_is_identity() -> None:
    values = np.arange(6.0).reshape(3, 2) + 1.0
    first = relative_perturbation(values, 1e-4, seed=8)
    second = relative_perturbation(values, 1e-4, seed=8)
    np.testing.assert_array_equal(first, second)
    direction = np.ones_like(values)
    np.testing.assert_array_equal(
        relative_perturbation(values, 0.0, seed=8, direction=direction), values
    )


def test_matrix_perturbation_uses_spectral_relative_norm() -> None:
    matrix = np.array([[3.0, 1.0], [0.0, 2.0], [1.0, -1.0]])
    direction = np.array([[1.0, 2.0], [-1.0, 1.0], [2.0, 0.5]])
    perturbed = relative_matrix_perturbation(
        matrix, 1e-4, seed=8, direction=direction
    )
    relative_change = np.linalg.norm(perturbed - matrix, ord=2) / np.linalg.norm(
        matrix, ord=2
    )
    assert relative_change == pytest.approx(1e-4, rel=1e-12)
