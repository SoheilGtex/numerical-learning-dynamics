import numpy as np
import pytest

from numerical_learning.data import generate_synthetic_data
from numerical_learning.linear_models import fit_least_squares, predict


def test_near_noiseless_parameter_recovery() -> None:
    dataset = generate_synthetic_data(n_samples=100, seed=11, noise_std=0.0)
    fit = fit_least_squares(dataset.X, dataset.y)
    np.testing.assert_allclose(fit.coefficients, dataset.true_coefficients, atol=1e-10)


def test_prediction_shape() -> None:
    dataset = generate_synthetic_data(n_samples=20, seed=5)
    fit = fit_least_squares(dataset.X[:15], dataset.y[:15])
    predictions = predict(dataset.X[15:], fit.coefficients)
    assert predictions.shape == (5,)


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="same number"):
        fit_least_squares(np.ones((3, 2)), np.ones(2))
    with pytest.raises(ValueError, match="coefficient count"):
        predict(np.ones((2, 2)), np.ones(2))


def test_no_double_intercept_when_design_is_explicit() -> None:
    X = np.column_stack((np.ones(4), np.arange(4.0)))
    y = np.array([1.0, 3.0, 5.0, 7.0])
    fit = fit_least_squares(X, y, fit_intercept=False)
    np.testing.assert_allclose(fit.coefficients, [1.0, 2.0])
