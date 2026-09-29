"""Reusable utilities for the Phase 5 controlled forecasting study."""

from dataclasses import dataclass

import numpy as np

from .evaluation import mae, r2_score, rmse
from .linear_models import fit_least_squares, predict
from .regularization import dimension_aware_lambda_grid, solve_ridge_svd
from .robust import HuberResult, fit_huber_irls
from .temporal_data import TemporalDataset, TemporalSplit

ALPHAS = np.concatenate(([0.0], np.logspace(-6, 2, 8)))


@dataclass(frozen=True)
class PreparedData:
    """Train-only standardized arrays for one checkpoint and split."""

    train_X: np.ndarray
    validation_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    validation_y: np.ndarray
    test_y: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    raw_train_X: np.ndarray
    raw_validation_X: np.ndarray
    raw_test_X: np.ndarray


def prepare_checkpoint(dataset: TemporalDataset, split: TemporalSplit, checkpoint: str) -> PreparedData:
    """Extract and standardize a checkpoint using training observations only."""
    X = dataset.checkpoint_matrix(checkpoint)
    train_X = X[split.train]
    means = train_X.mean(axis=0)
    scales = train_X.std(axis=0, ddof=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    standardize = lambda values: (values - means) / scales
    return PreparedData(
        train_X=standardize(train_X),
        validation_X=standardize(X[split.validation]),
        test_X=standardize(X[split.test]),
        train_y=dataset.y[split.train],
        validation_y=dataset.y[split.validation],
        test_y=dataset.y[split.test],
        means=means,
        scales=scales,
        raw_train_X=train_X.copy(),
        raw_validation_X=X[split.validation].copy(),
        raw_test_X=X[split.test].copy(),
    )


def fit_ridge_selected(data: PreparedData) -> tuple[np.ndarray, float, float]:
    """Select Ridge alpha from validation RMSE, then return final coefficients."""
    alphas, lambdas = dimension_aware_lambda_grid(data.train_X, ALPHAS)
    validation_errors = []
    for lambda_ in lambdas:
        result = solve_ridge_svd(data.train_X, data.train_y, float(lambda_))
        validation_errors.append(rmse(data.validation_y, predict(data.validation_X, result.coefficients)))
    index = int(np.argmin(validation_errors))
    result = solve_ridge_svd(data.train_X, data.train_y, float(lambdas[index]))
    return result.coefficients, float(alphas[index]), float(lambdas[index])


def fit_model(data: PreparedData, model: str) -> tuple[np.ndarray, float | None, float | None, bool, int | None]:
    """Fit OLS, validation-selected Ridge, or Huber on a prepared checkpoint."""
    if model == "OLS":
        result = fit_least_squares(data.train_X, data.train_y)
        return result.coefficients, None, None, True, None
    if model == "Ridge":
        coefficients, alpha, lambda_ = fit_ridge_selected(data)
        return coefficients, alpha, lambda_, True, None
    if model == "Huber":
        result: HuberResult = fit_huber_irls(data.train_X, data.train_y)
        return result.coefficients, None, None, result.converged, result.iterations
    raise ValueError(f"unknown model: {model}")


def evaluate_model(data: PreparedData, model: str) -> dict[str, float | str | None]:
    """Fit one model and evaluate it once on the untouched test arrays."""
    coefficients, alpha, lambda_, converged, iterations = fit_model(data, model)
    predictions = predict(data.test_X, coefficients)
    return {
        "model": model,
        "selected_alpha": alpha,
        "selected_lambda": lambda_,
        "test_mae": mae(data.test_y, predictions),
        "test_rmse": rmse(data.test_y, predictions),
        "test_r2": r2_score(data.test_y, predictions),
        "status": "success" if converged else "non_converged",
        "huber_iterations": iterations,
    }


def contaminate_training(
    data: PreparedData,
    fraction: float,
    seed: int,
    mode: str,
) -> PreparedData:
    """Corrupt only training arrays; validation and test arrays are copied unchanged."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    if mode not in {"response", "feature"}:
        raise ValueError("mode must be response or feature")
    corrupted_X = data.train_X.copy()
    corrupted_y = data.train_y.copy()
    count = round(fraction * len(corrupted_y))
    indices = np.random.default_rng(seed).permutation(len(corrupted_y))[:count]
    if mode == "response":
        signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        corrupted_y[indices] += signs * 8.0
    raw_train = data.raw_train_X.copy()
    raw_validation = data.raw_validation_X.copy()
    raw_test = data.raw_test_X.copy()
    if mode == "feature" and count:
        feature_indices = np.arange(count) % raw_train.shape[1]
        raw_train[indices, feature_indices] += 6.0
        means = raw_train.mean(axis=0)
        scales = np.where(raw_train.std(axis=0) == 0.0, 1.0, raw_train.std(axis=0))
        corrupted_X = (raw_train - means) / scales
        validation_X = (raw_validation - means) / scales
        test_X = (raw_test - means) / scales
    else:
        means = data.means.copy()
        scales = data.scales.copy()
        validation_X = data.validation_X.copy()
        test_X = data.test_X.copy()
    return PreparedData(
        train_X=corrupted_X,
        validation_X=validation_X,
        test_X=test_X,
        train_y=corrupted_y,
        validation_y=data.validation_y.copy(),
        test_y=data.test_y.copy(),
        means=means,
        scales=scales,
        raw_train_X=raw_train,
        raw_validation_X=raw_validation,
        raw_test_X=raw_test,
    )
