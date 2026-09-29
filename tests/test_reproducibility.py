import numpy as np

from numerical_learning.data import generate_synthetic_data
from numerical_learning.evaluation import (
    coefficient_relative_error,
    mae,
    r2_score,
    rmse,
)
from numerical_learning.linear_models import fit_least_squares, predict


def run_experiment() -> tuple[np.ndarray, dict[str, float]]:
    dataset = generate_synthetic_data(n_samples=80, seed=2026, noise_std=0.6)
    split = 60
    fit = fit_least_squares(dataset.X[:split], dataset.y[:split])
    predictions = predict(dataset.X[split:], fit.coefficients)
    metrics = {
        "mae": mae(dataset.y[split:], predictions),
        "rmse": rmse(dataset.y[split:], predictions),
        "r2": r2_score(dataset.y[split:], predictions),
        "coefficient_relative_error": coefficient_relative_error(
            fit.coefficients, dataset.true_coefficients
        ),
    }
    return fit.coefficients, metrics


def test_repeated_experiment_is_identical() -> None:
    coefficients_a, metrics_a = run_experiment()
    coefficients_b, metrics_b = run_experiment()
    np.testing.assert_array_equal(coefficients_a, coefficients_b)
    assert metrics_a == metrics_b
