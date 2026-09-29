"""Run the Phase 1 deterministic train/test baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from numerical_learning.data import generate_synthetic_data
from numerical_learning.evaluation import (
    coefficient_relative_error,
    mae,
    r2_score,
    rmse,
)
from numerical_learning.linear_models import fit_least_squares, predict

SEED = 2026
N_SAMPLES = 160
NOISE_STD = 0.6
TRAIN_FRACTION = 0.75


def main() -> None:
    """Generate results/tables and results/figures from a fixed configuration."""
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_data(N_SAMPLES, SEED, NOISE_STD)
    train_size = int(TRAIN_FRACTION * N_SAMPLES)
    X_train, X_test = dataset.X[:train_size], dataset.X[train_size:]
    y_train, y_test = dataset.y[:train_size], dataset.y[train_size:]

    fit = fit_least_squares(X_train, y_train)
    y_pred = predict(X_test, fit.coefficients)
    metrics = {
        "seed": SEED,
        "sample_size": N_SAMPLES,
        "noise_std": NOISE_STD,
        "train_size": train_size,
        "test_size": N_SAMPLES - train_size,
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "coefficient_relative_error": coefficient_relative_error(
            fit.coefficients, dataset.true_coefficients
        ),
    }
    pd.DataFrame([metrics]).to_csv(table_dir / "baseline_metrics.csv", index=False)

    lower = min(float(y_test.min()), float(y_pred.min()))
    upper = max(float(y_test.max()), float(y_pred.max()))
    padding = 0.05 * (upper - lower)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.8, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    ax.set_title("Phase 1 Synthetic Baseline: Predicted vs Actual")
    ax.set_xlabel("Actual final grade")
    ax.set_ylabel("Predicted final grade")
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    fig.tight_layout()
    fig.savefig(figure_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    print(pd.Series(metrics).to_string())
    print(f"Saved {table_dir / 'baseline_metrics.csv'}")
    print(f"Saved {figure_dir / 'predicted_vs_actual.png'}")


if __name__ == "__main__":
    main()
