import numpy as np
import pytest

from numerical_learning.linear_models import fit_least_squares
from numerical_learning.robust import fit_huber_irls, mad_scale


def test_huber_matches_ols_in_quadratic_residual_regime() -> None:
    X = np.arange(12.0).reshape(6, 2)
    y = 1.0 + X @ np.array([0.4, -0.2]) + np.linspace(-0.01, 0.01, 6)
    ols = fit_least_squares(X, y)
    huber = fit_huber_irls(X, y, tuning_constant=10.0)
    np.testing.assert_allclose(huber.coefficients, ols.coefficients, atol=1e-6)
    assert huber.converged


def test_huber_reduces_effect_of_response_outlier() -> None:
    X = np.arange(20.0).reshape(10, 2)
    clean = 2.0 + X @ np.array([0.3, -0.1])
    y = clean.copy(); y[0] += 30.0
    ols = fit_least_squares(X, y)
    huber = fit_huber_irls(X, y)
    clean_ols = fit_least_squares(X, clean)
    assert np.linalg.norm(huber.coefficients - clean_ols.coefficients) < np.linalg.norm(ols.coefficients - clean_ols.coefficients)


def test_huber_is_deterministic_and_exposes_nonconvergence() -> None:
    X = np.arange(20.0).reshape(10, 2); y = 1.0 + X @ np.array([0.2, 0.3])
    first = fit_huber_irls(X, y); second = fit_huber_irls(X, y)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    limited = fit_huber_irls(X, y, max_iterations=1)
    assert limited.iterations == 1
    assert isinstance(limited.converged, bool)
    assert np.isfinite(first.coefficients).all()


def test_huber_rejects_invalid_parameters_and_mad_handles_zero() -> None:
    with pytest.raises(ValueError):
        fit_huber_irls(np.ones((3, 1)), np.ones(3), tuning_constant=0)
    with pytest.raises(ValueError):
        fit_huber_irls(np.ones((3, 1)), np.ones(3), max_iterations=0)
    assert mad_scale(np.ones(4)) == 0.0
