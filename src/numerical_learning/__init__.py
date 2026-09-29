"""Numerical Learning Dynamics: Phase 1 scientific-computing foundation."""

from .data import FEATURE_NAMES, SyntheticDataset, generate_synthetic_data
from .evaluation import coefficient_relative_error, mae, r2_score, rmse
from .linear_models import add_intercept, fit_least_squares, predict
from .regularization import RidgeResult, solve_ridge_svd
from .robust import HuberResult, fit_huber_irls
from .solvers import (
    SolverResult,
    condition_number,
    solve_normal_equations,
    solve_qr,
    solve_svd,
)
from .temporal_data import (
    CHECKPOINTS,
    TemporalDataset,
    TemporalSplit,
    generate_temporal_data,
    make_temporal_split,
)
from .uncertainty import (
    Interval,
    estimate_noise_variance,
    mean_confidence_interval,
    ols_covariance,
    prediction_interval,
)

__all__ = [
    "CHECKPOINTS",
    "FEATURE_NAMES",
    "HuberResult",
    "Interval",
    "RidgeResult",
    "SolverResult",
    "SyntheticDataset",
    "TemporalDataset",
    "TemporalSplit",
    "add_intercept",
    "coefficient_relative_error",
    "condition_number",
    "estimate_noise_variance",
    "fit_huber_irls",
    "fit_least_squares",
    "generate_synthetic_data",
    "generate_temporal_data",
    "mae",
    "make_temporal_split",
    "mean_confidence_interval",
    "ols_covariance",
    "predict",
    "prediction_interval",
    "r2_score",
    "rmse",
    "solve_normal_equations",
    "solve_qr",
    "solve_ridge_svd",
    "solve_svd",
]
