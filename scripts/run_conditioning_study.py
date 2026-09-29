"""Run the compact Phase 3 conditioning and solver-stability study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from numerical_learning.conditioning import (
    design_condition_diagnostics,
    make_near_collinear_predictors,
    standardize_predictors,
)
from numerical_learning.data import generate_synthetic_data
from numerical_learning.evaluation import coefficient_relative_error, mae, rmse
from numerical_learning.linear_models import add_intercept, fit_least_squares, predict
from numerical_learning.solvers import (
    SolverResult,
    solve_normal_equations,
    solve_qr,
    solve_svd,
)

SOLVERS = ("normal_equations", "qr", "svd", "lstsq")
DELTAS = tuple(10.0 ** exponent for exponent in range(0, -9, -1))
MODES = {"exact": 0.0, "modest_noise": 0.15}
NOISY_SEEDS = tuple(range(31415, 31425))


def _solve(name: str, X: np.ndarray, y: np.ndarray) -> SolverResult:
    if name == "normal_equations":
        return solve_normal_equations(X, y)
    if name == "qr":
        return solve_qr(X, y)
    if name == "svd":
        return solve_svd(X, y)
    fit = fit_least_squares(X, y)
    return SolverResult(
        coefficients=fit.coefficients,
        residual_norm=float(np.linalg.norm(add_intercept(X) @ fit.coefficients - y)),
        rank=fit.rank,
        singular_values=fit.singular_values,
    )


def _run_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    singular_rows: list[dict[str, object]] = []
    beta_true = np.array([1.0, 0.5, -1.2, 2.0])
    for mode, noise_std in MODES.items():
        seeds = (31415,) if mode == "exact" else NOISY_SEEDS
        for seed in seeds:
            for delta in DELTAS:
                X = make_near_collinear_predictors(delta=delta, seed=31415)
                design = add_intercept(X)
                rng = np.random.default_rng(seed + 1)
                y = design @ beta_true + rng.normal(0.0, noise_std, X.shape[0])
                diagnostics = design_condition_diagnostics(X)
                singular_values = np.linalg.svd(design, compute_uv=False)
                if seed == seeds[0]:
                    for index, value in enumerate(singular_values, start=1):
                        singular_rows.append(
                            {
                                "mode": mode,
                                "conditioning_parameter": delta,
                                "singular_value_index": index,
                                "singular_value": value,
                            }
                        )
                for solver in SOLVERS:
                    row: dict[str, object] = {
                        "mode": mode,
                        "seed": seed,
                        "conditioning_parameter": delta,
                        **diagnostics,
                        "solver": solver,
                        "status": "success",
                        "residual_norm": np.nan,
                        "coefficient_relative_error": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                    }
                    try:
                        result = _solve(solver, X, y)
                        predictions = predict(X, result.coefficients)
                        row["numerical_rank"] = result.rank
                        row["residual_norm"] = result.residual_norm
                        row["coefficient_relative_error"] = coefficient_relative_error(
                            result.coefficients, beta_true
                        )
                        row["mae"] = mae(y, predictions)
                        row["rmse"] = rmse(y, predictions)
                    except (np.linalg.LinAlgError, ValueError) as error:
                        row["status"] = f"failure: {error}"
                    rows.append(row)
    return rows, singular_rows


def _save_figures(rows: list[dict[str, object]], singular_rows: list[dict[str, object]], figure_dir: Path, X_phase1: np.ndarray) -> None:
    frame = pd.DataFrame(rows)
    successful = frame[(frame["status"] == "success") & (frame["mode"] == "modest_noise")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in SOLVERS:
        subset = successful[successful["solver"] == solver]
        grouped = subset.groupby("condition_number", as_index=False)["coefficient_relative_error"].median()
        ax.plot(grouped["condition_number"], grouped["coefficient_relative_error"], marker="o", label=solver)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Condition number of design matrix")
    ax.set_ylabel("Relative coefficient error")
    ax.set_title("Coefficient Error vs Condition Number")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "coefficient_error_vs_condition.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in SOLVERS:
        subset = successful[successful["solver"] == solver]
        grouped = subset.groupby("condition_number", as_index=False)["rmse"].median()
        ax.plot(grouped["condition_number"], grouped["rmse"], marker="o", label=solver)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Condition number of design matrix")
    ax.set_ylabel("RMSE")
    ax.set_title("Fitted-Value RMSE vs Condition Number")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "fitted_rmse_vs_condition.png", dpi=150)
    plt.close(fig)

    singular = pd.DataFrame(singular_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in singular.groupby("mode"):
        for delta, subset in group.groupby("conditioning_parameter"):
            if delta in (1.0, 1e-4, 1e-8):
                ax.semilogy(subset["singular_value_index"], subset["singular_value"], marker="o", label=f"{mode}, delta={delta:g}")
    standardized, _ = standardize_predictors(X_phase1)
    for values, label in ((np.linalg.svd(add_intercept(X_phase1), compute_uv=False), "raw Phase 1"), (np.linalg.svd(add_intercept(standardized), compute_uv=False), "standardized Phase 1")):
        ax.semilogy(np.arange(1, len(values) + 1), values, linestyle="--", marker="x", label=label)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Conditioning and Scaling Singular-Value Spectra")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "conditioning_singular_values.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute the conditioning sweep and write tables and figures."""
    plt.rcParams["axes.unicode_minus"] = False
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows, singular_rows = _run_rows()
    pd.DataFrame(rows).to_csv(table_dir / "conditioning_sweep.csv", index=False)
    pd.DataFrame(singular_rows).to_csv(table_dir / "singular_values_conditioning.csv", index=False)
    frame = pd.DataFrame(rows)
    noisy = frame[frame["mode"] == "modest_noise"]
    summary = (
        noisy.groupby(["mode", "conditioning_parameter", "solver"], as_index=False)
        .agg(
            median_condition_number=("condition_number", "median"),
            mean_coefficient_error=("coefficient_relative_error", "mean"),
            median_coefficient_error=("coefficient_relative_error", "median"),
            std_coefficient_error=("coefficient_relative_error", "std"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            success_rate=("status", lambda values: np.mean(values == "success")),
        )
    )
    summary.to_csv(table_dir / "conditioning_summary.csv", index=False)
    phase1 = generate_synthetic_data(n_samples=160, seed=2026, noise_std=0.6)
    _save_figures(rows, singular_rows, figure_dir, phase1.X[:120])
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {table_dir / 'conditioning_sweep.csv'}")
    print(f"Saved {table_dir / 'singular_values_conditioning.csv'}")
    print(f"Saved {table_dir / 'conditioning_summary.csv'}")


if __name__ == "__main__":
    main()
