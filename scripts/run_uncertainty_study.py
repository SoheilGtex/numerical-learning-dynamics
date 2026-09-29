"""Run a deterministic Monte Carlo verification of classical OLS intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerical_learning.data import TRUE_COEFFICIENTS
from numerical_learning.linear_models import add_intercept, fit_least_squares
from numerical_learning.uncertainty import (
    estimate_noise_variance,
    leverage_values,
    mean_confidence_interval,
    ols_covariance,
    prediction_interval,
)

N_REPLICATIONS = 200
N_SAMPLES = 80
NOISE_STD = 0.6


def _design(seed: int = 9100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        (
            rng.uniform(8.0, 20.0, N_SAMPLES),
            rng.uniform(8.0, 20.0, N_SAMPLES),
            rng.uniform(0.60, 1.0, N_SAMPLES),
            rng.uniform(5.0, 20.0, N_SAMPLES),
            rng.uniform(5.0, 20.0, N_SAMPLES),
        )
    )
    return X, add_intercept(X)


def _select_leverage_points(design: np.ndarray) -> dict[str, tuple[int, float]]:
    """Select existing rows nearest the median, upper quartile, and maximum leverage."""
    leverage = leverage_values(design)
    targets = {
        "median_leverage": float(np.median(leverage)),
        "upper_quartile_leverage": float(np.quantile(leverage, 0.75)),
        "maximum_leverage": float(np.max(leverage)),
    }
    selected = {}
    for name, target in targets.items():
        index = int(np.argmin(np.abs(leverage - target)))
        selected[name] = (index, float(leverage[index]))
    return selected


def main() -> None:
    """Execute the OLS coverage experiment under its stated assumptions."""
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    X, design = _design()
    selected_points = _select_leverage_points(design)
    rng = np.random.default_rng(9101)
    records = []
    for replicate in range(N_REPLICATIONS):
        y = design @ TRUE_COEFFICIENTS + rng.normal(0.0, NOISE_STD, N_SAMPLES)
        fit = fit_least_squares(X, y)
        sigma_squared = estimate_noise_variance(design, y, fit.coefficients)
        covariance = ols_covariance(design, sigma_squared)
        for point_type, (index, leverage) in selected_points.items():
            x0 = design[index]
            mean_interval = mean_confidence_interval(x0, fit.coefficients, covariance, sigma_squared, N_SAMPLES)
            prediction = prediction_interval(x0, fit.coefficients, covariance, sigma_squared, N_SAMPLES)
            true_mean = float(x0 @ TRUE_COEFFICIENTS)
            future = true_mean + rng.normal(0.0, NOISE_STD)
            records.extend(
                [
                    {"replicate": replicate, "point_type": point_type, "leverage": leverage, "interval_type": "mean_response_ci", "nominal_coverage": 0.95, "covered": mean_interval.lower <= true_mean <= mean_interval.upper, "interval_width": mean_interval.upper - mean_interval.lower},
                    {"replicate": replicate, "point_type": point_type, "leverage": leverage, "interval_type": "future_prediction_interval", "nominal_coverage": 0.95, "covered": prediction.lower <= future <= prediction.upper, "interval_width": prediction.upper - prediction.lower},
                ]
            )
    raw = pd.DataFrame(records)
    summary = (
        raw.groupby(["point_type", "leverage", "interval_type", "nominal_coverage"], as_index=False)
        .agg(
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            median_interval_width=("interval_width", "median"),
            n_replications=("replicate", "nunique"),
            n_evaluations=("covered", "size"),
        )
    )
    summary.to_csv(table_dir / "uncertainty_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot = summary.pivot(index="point_type", columns="interval_type", values="empirical_coverage")
    plot.plot(kind="bar", ax=ax)
    ax.axhline(0.95, color="black", linestyle="--", label="nominal 95%")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlabel("Existing design row selected by leverage")
    ax.set_title("Classical OLS Interval Coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ols_interval_coverage.png", dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
