import numpy as np
import pytest

from numerical_learning.linear_models import add_intercept, fit_least_squares
from numerical_learning.uncertainty import (
    estimate_noise_variance,
    mean_confidence_interval,
    ols_covariance,
    prediction_interval,
)


def _fit_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0])
    fit = fit_least_squares(X, y)
    design = add_intercept(X)
    sigma_squared = estimate_noise_variance(design, y, fit.coefficients)
    covariance = ols_covariance(design, sigma_squared)
    return X, y, fit.coefficients, sigma_squared, covariance


def test_variance_covariance_are_valid() -> None:
    _, _, _, sigma_squared, covariance = _fit_case()
    assert np.isfinite(sigma_squared)
    assert sigma_squared >= 0
    assert covariance.shape == (3, 3)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) >= -1e-12)
    with pytest.raises(ValueError):
        estimate_noise_variance(np.ones((3, 3)), np.ones(3), np.ones(3))


def test_intervals_are_deterministic_and_prediction_is_wider() -> None:
    _, _, coefficients, sigma_squared, covariance = _fit_case()
    x0 = np.array([1.0, 2.5, 3.5])
    mean = mean_confidence_interval(x0, coefficients, covariance, sigma_squared, 6)
    prediction = prediction_interval(x0, coefficients, covariance, sigma_squared, 6)
    assert prediction.upper - prediction.lower > mean.upper - mean.lower
    repeat = mean_confidence_interval(x0, coefficients, covariance, sigma_squared, 6)
    assert mean == repeat
    wider = mean_confidence_interval(x0, coefficients, covariance, sigma_squared, 6, 0.99)
    assert wider.upper - wider.lower > mean.upper - mean.lower


def test_noiseless_intervals_can_collapse() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    design = add_intercept(X)
    coefficients = np.array([1.0, 2.0])
    y = design @ coefficients
    sigma_squared = estimate_noise_variance(design, y, coefficients)
    covariance = ols_covariance(design, sigma_squared)
    interval = prediction_interval(np.array([1.0, 2.5]), coefficients, covariance, sigma_squared, 5)
    assert interval.lower == pytest.approx(interval.upper)
