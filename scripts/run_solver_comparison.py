"""Run the Phase 2 comparison of explicit least-squares solvers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from numerical_learning.data import generate_synthetic_data
from numerical_learning.evaluation import coefficient_relative_error, mae, rmse
from numerical_learning.linear_models import add_intercept, fit_least_squares
from numerical_learning.linear_models import predict as phase1_predict
from numerical_learning.solvers import (
    SolverResult,
    condition_number,
    solve_normal_equations,
    solve_qr,
    solve_svd,
)

SOLVER_NAMES = ("normal_equations", "qr", "svd", "lstsq")


def _cases() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the four deterministic systems as predictor matrices without intercepts."""
    rng = np.random.default_rng(2040)

    X_a = rng.normal(size=(50, 3))
    beta_a = np.array([1.0, 0.5, -1.2, 2.0])
    y_a = add_intercept(X_a) @ beta_a + rng.normal(0.0, 0.02, X_a.shape[0])

    phase1 = generate_synthetic_data(n_samples=160, seed=2026, noise_std=0.6)
    X_b = phase1.X[:120]
    beta_b = phase1.true_coefficients
    y_b = phase1.y[:120]

    x1 = rng.normal(size=60)
    delta = 1.0e-6
    x2 = x1 + delta * rng.normal(size=60)
    x3 = rng.normal(size=60)
    X_c = np.column_stack((x1, x2, x3))
    beta_c = np.array([1.0, 1.0, 2.0, -0.5])
    y_c = add_intercept(X_c) @ beta_c

    x1 = rng.normal(size=45)
    x2 = 2.0 * x1
    x3 = rng.normal(size=45)
    X_d = np.column_stack((x1, x2, x3))
    beta_d = np.array([1.0, 1.75, 1.0, -0.5])
    y_d = add_intercept(X_d) @ beta_d

    return {
        "well_conditioned": (X_a, y_a, beta_a),
        "phase1_raw": (X_b, y_b, beta_b),
        "near_rank_deficient": (X_c, y_c, beta_c),
        "rank_deficient": (X_d, y_d, beta_d),
    }


def _solve(name: str, X: np.ndarray, y: np.ndarray) -> SolverResult:
    """Dispatch a solver while preserving independent reference implementation."""
    if name == "normal_equations":
        return solve_normal_equations(X, y)
    if name == "qr":
        return solve_qr(X, y)
    if name == "svd":
        return solve_svd(X, y)
    if name == "lstsq":
        fit = fit_least_squares(X, y)
        return SolverResult(
            coefficients=fit.coefficients,
            residual_norm=float(np.linalg.norm(add_intercept(X) @ fit.coefficients - y)),
            rank=fit.rank,
            singular_values=fit.singular_values,
        )
    raise ValueError(f"unknown solver: {name}")


def _condition_diagnostic(X: np.ndarray, table_dir: Path) -> None:
    """Save raw and standardized condition numbers for the Phase 1 training design."""
    raw_design = add_intercept(X)
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    if np.any(scales == 0.0):
        raise ValueError("cannot standardize a constant predictor")
    scaled_design = add_intercept((X - means) / scales)
    pd.DataFrame(
        [
            {"representation": "raw", "condition_number": condition_number(raw_design)},
            {
                "representation": "standardized",
                "condition_number": condition_number(scaled_design),
            },
        ]
    ).to_csv(table_dir / "scaling_conditioning.csv", index=False)


def _comparison_rows(cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[dict[str, object]]:
    """Run every solver on every case and retain failures as explicit rows."""
    rows: list[dict[str, object]] = []
    for case, (X, y, beta_true) in cases.items():
        design = add_intercept(X)
        measured_condition = condition_number(design)
        design_rank = int(np.linalg.matrix_rank(design))
        for solver in SOLVER_NAMES:
            row: dict[str, object] = {
                "case": case,
                "solver": solver,
                "n_samples": X.shape[0],
                "n_parameters": design.shape[1],
                "condition_number": measured_condition,
                "rank": design_rank,
                "residual_norm": np.nan,
                "coefficient_relative_error": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "status": "success",
            }
            try:
                result = _solve(solver, X, y)
                row["rank"] = result.rank if result.rank is not None else design_rank
                row["residual_norm"] = result.residual_norm
                row["coefficient_relative_error"] = coefficient_relative_error(
                    result.coefficients, beta_true
                )
                predictions = phase1_predict(X, result.coefficients)
                row["mae"] = mae(y, predictions)
                row["rmse"] = rmse(y, predictions)
            except (np.linalg.LinAlgError, ValueError) as error:
                row["status"] = f"failure: {error}"
            rows.append(row)
    return rows


def _plot_coefficient_errors(rows: list[dict[str, object]], figure_dir: Path) -> None:
    """Plot finite coefficient errors by case and solver."""
    frame = pd.DataFrame(rows)
    cases = tuple(frame["case"].drop_duplicates())
    x = np.arange(len(cases))
    width = 0.8 / len(SOLVER_NAMES)
    fig, ax = plt.subplots(figsize=(9, 5))
    finite_values: list[float] = []
    for index, solver in enumerate(SOLVER_NAMES):
        values = []
        for case in cases:
            value = frame.loc[
                (frame["case"] == case) & (frame["solver"] == solver),
                "coefficient_relative_error",
            ].iloc[0]
            values.append(value)
            if pd.notna(value) and value > 0.0:
                finite_values.append(float(value))
        values_array = np.asarray(values, dtype=float)
        ax.bar(x + (index - 1.5) * width, values_array, width, label=solver)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    ax.set_ylim(min(finite_values) * 0.5, max(finite_values) * 2.0)
    ax.set_xticks(x, cases, rotation=20, ha="right")
    ax.set_title("Relative Coefficient Error by Numerical Case")
    ax.set_xlabel("Numerical case")
    ax.set_ylabel("Relative coefficient error (log scale)")
    ax.legend(title="Solver")
    fig.tight_layout()
    fig.savefig(figure_dir / "solver_coefficient_error.png", dpi=150)
    plt.close(fig)


def _plot_spectra(cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], figure_dir: Path) -> None:
    """Plot positive singular values for raw and rank-sensitive cases."""
    selected = ("phase1_raw", "near_rank_deficient", "rank_deficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in selected:
        X = cases[case][0]
        values = np.linalg.svd(add_intercept(X), compute_uv=False)
        positive = values[values > 0.0]
        rank = int(np.linalg.matrix_rank(add_intercept(X)))
        ax.semilogy(
            np.arange(1, positive.size + 1),
            positive,
            marker="o",
            label=f"{case} (rank {rank})",
        )
    ax.set_title("Singular-Value Spectra")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "singular_value_spectra.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute Phase 2 and write tables and figures."""
    plt.rcParams["axes.unicode_minus"] = False
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    cases = _cases()
    rows = _comparison_rows(cases)
    pd.DataFrame(rows).to_csv(table_dir / "solver_comparison.csv", index=False)
    _condition_diagnostic(cases["phase1_raw"][0], table_dir)
    _plot_coefficient_errors(rows, figure_dir)
    _plot_spectra(cases, figure_dir)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {table_dir / 'solver_comparison.csv'}")
    print(f"Saved {table_dir / 'scaling_conditioning.csv'}")
    print(f"Saved {figure_dir / 'solver_coefficient_error.png'}")
    print(f"Saved {figure_dir / 'singular_value_spectra.png'}")


if __name__ == "__main__":
    main()
