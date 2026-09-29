"""Evaluation metrics for predictions and synthetic parameter recovery."""

import numpy as np


def _residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Validate and return prediction residuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if y_true.shape != y_pred.shape:
        raise ValueError("metric inputs must have the same shape")
    if y_true.size == 0:
        raise ValueError("metric inputs must not be empty")
    return y_true - y_pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute error."""
    return float(np.mean(np.abs(_residuals(y_true, y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    residuals = _residuals(y_true, y_pred)
    return float(np.sqrt(np.mean(residuals**2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R-squared, rejecting constant targets as undefined."""
    residuals = _residuals(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(total, 0.0):
        raise ValueError("R^2 is undefined for a constant target")
    return float(1.0 - np.sum(residuals**2) / total)


def coefficient_relative_error(
    estimated: np.ndarray, true: np.ndarray
) -> float:
    """Return ``||estimated - true||_2 / ||true||_2``."""
    estimated = np.asarray(estimated, dtype=float)
    true = np.asarray(true, dtype=float)
    if estimated.shape != true.shape:
        raise ValueError("estimated and true coefficients must have the same shape")
    denominator = float(np.linalg.norm(true))
    if np.isclose(denominator, 0.0):
        raise ValueError("relative coefficient error is undefined for zero truth")
    return float(np.linalg.norm(estimated - true) / denominator)
