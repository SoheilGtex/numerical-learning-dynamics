"""Run Phase 3 response, design-matrix, and scaling sensitivity studies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerical_learning.conditioning import (
    make_near_collinear_predictors,
    relative_matrix_perturbation,
    relative_perturbation,
    standardize_predictors,
    standardized_to_raw_coefficients,
)
from numerical_learning.data import generate_synthetic_data
from numerical_learning.evaluation import rmse
from numerical_learning.linear_models import add_intercept, fit_least_squares, predict
from numerical_learning.solvers import (
    SolverResult,
    condition_number,
    solve_normal_equations,
    solve_qr,
    solve_svd,
)

SOLVERS = ("normal_equations", "qr", "svd", "lstsq")
RESPONSE_DELTAS = (1.0, 1e-3, 1e-6, 1e-8)
RESPONSE_MAGNITUDES = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
DESIGN_DELTAS = (1.0, 1e-3, 1e-6)
DESIGN_MAGNITUDES = (0.0, 1e-6, 1e-4, 1e-2)


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


def _base_system(delta: float) -> tuple[np.ndarray, np.ndarray]:
    X = make_near_collinear_predictors(delta=delta, seed=31415)
    beta = np.array([1.0, 0.5, -1.2, 2.0])
    return X, add_intercept(X) @ beta


def _response_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in RESPONSE_DELTAS:
        X, y = _base_system(delta)
        direction = np.random.default_rng(9001).normal(size=y.shape)
        condition = condition_number(add_intercept(X))
        for solver in SOLVERS:
            try:
                original = _solve(solver, X, y)
            except (np.linalg.LinAlgError, ValueError) as error:
                for magnitude in RESPONSE_MAGNITUDES:
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": np.nan, "relative_solution_perturbation": np.nan, "amplification_ratio": np.nan, "residual_norm": np.nan, "rmse": np.nan, "status": f"base failure: {error}"})
                continue
            for magnitude in RESPONSE_MAGNITUDES:
                perturbed_y = relative_perturbation(y, magnitude, seed=9001, direction=direction)
                relative_data = np.linalg.norm(perturbed_y - y) / np.linalg.norm(y)
                try:
                    perturbed = _solve(solver, X, perturbed_y)
                    relative_solution = np.linalg.norm(perturbed.coefficients - original.coefficients) / np.linalg.norm(original.coefficients)
                    amplification = relative_solution / relative_data if relative_data > 0 else np.nan
                    predictions = predict(X, perturbed.coefficients)
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": relative_data, "relative_solution_perturbation": relative_solution, "amplification_ratio": amplification, "residual_norm": perturbed.residual_norm, "rmse": rmse(perturbed_y, predictions), "status": "success"})
                except (np.linalg.LinAlgError, ValueError) as error:
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": relative_data, "relative_solution_perturbation": np.nan, "amplification_ratio": np.nan, "residual_norm": np.nan, "rmse": np.nan, "status": f"failure: {error}"})
    return rows


def _design_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in DESIGN_DELTAS:
        X, y = _base_system(delta)
        direction = np.random.default_rng(9002).normal(size=X.shape)
        condition = condition_number(add_intercept(X))
        for solver in SOLVERS:
            try:
                original = _solve(solver, X, y)
            except (np.linalg.LinAlgError, ValueError) as error:
                for magnitude in DESIGN_MAGNITUDES:
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": np.nan, "relative_solution_perturbation": np.nan, "amplification_ratio": np.nan, "residual_norm": np.nan, "rmse": np.nan, "status": f"base failure: {error}"})
                continue
            for magnitude in DESIGN_MAGNITUDES:
                perturbed_X = relative_matrix_perturbation(
                    X, magnitude, seed=9002, direction=direction
                )
                relative_data = np.linalg.norm(
                    perturbed_X - X, ord=2
                ) / np.linalg.norm(X, ord=2)
                try:
                    perturbed = _solve(solver, perturbed_X, y)
                    relative_solution = np.linalg.norm(perturbed.coefficients - original.coefficients) / np.linalg.norm(original.coefficients)
                    predictions = predict(perturbed_X, perturbed.coefficients)
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": relative_data, "relative_solution_perturbation": relative_solution, "amplification_ratio": relative_solution / relative_data if relative_data > 0 else np.nan, "residual_norm": perturbed.residual_norm, "rmse": rmse(y, predictions), "status": "success"})
                except (np.linalg.LinAlgError, ValueError) as error:
                    rows.append({"condition_regime": f"delta={delta:g}", "condition_number": condition, "solver": solver, "perturbation_magnitude": magnitude, "relative_data_perturbation": relative_data, "relative_solution_perturbation": np.nan, "amplification_ratio": np.nan, "residual_norm": np.nan, "rmse": np.nan, "status": f"failure: {error}"})
    return rows


def _scaling_rows() -> list[dict[str, object]]:
    dataset = generate_synthetic_data(n_samples=160, seed=2026, noise_std=0.6)
    X = dataset.X[:120]
    standardized, transformation = standardize_predictors(X)
    rows: list[dict[str, object]] = []
    perturbation_direction = np.random.default_rng(9010).normal(size=dataset.y[:120].shape)
    for representation, predictors in (("raw", X), ("standardized", standardized)):
        design = add_intercept(predictors)
        fit = fit_least_squares(predictors, dataset.y[:120])
        predictions = predict(predictors, fit.coefficients)
        perturbed_y = relative_perturbation(
            dataset.y[:120], 1e-6, seed=9010, direction=perturbation_direction
        )
        perturbed_fit = fit_least_squares(predictors, perturbed_y)
        base_raw = (
            fit.coefficients
            if representation == "raw"
            else standardized_to_raw_coefficients(fit.coefficients, transformation)
        )
        perturbed_raw = (
            perturbed_fit.coefficients
            if representation == "raw"
            else standardized_to_raw_coefficients(
                perturbed_fit.coefficients, transformation
            )
        )
        rows.append(
            {
                "representation": representation,
                "condition_number": condition_number(design),
                "singular_value_spread": float(
                    np.linalg.svd(design, compute_uv=False)[0]
                    / np.linalg.svd(design, compute_uv=False)[-1]
                ),
                "fitted_rmse": rmse(dataset.y[:120], predictions),
                "coefficient_coordinate": representation,
                "coefficient_stability": float(
                    np.linalg.norm(perturbed_raw - base_raw)
                    / np.linalg.norm(base_raw)
                ),
                "transformation_scale_mean": float(transformation.scales.mean()),
            }
        )
    return rows


def _save_figure(rows: list[dict[str, object]], figure_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    successful = frame[
        (frame["status"] == "success")
        & (frame["perturbation_magnitude"] > 0)
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in SOLVERS:
        subset = successful[successful["solver"] == solver]
        grouped = subset.groupby("condition_number", as_index=False)["amplification_ratio"].median()
        ax.plot(grouped["condition_number"], grouped["amplification_ratio"], marker="o", label=solver)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Condition number")
    ax.set_ylabel("Median response amplification")
    ax.set_title("Response Perturbation Amplification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "response_amplification_vs_condition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute perturbation and scaling studies."""
    plt.rcParams["axes.unicode_minus"] = False
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    response = _response_rows()
    design = _design_rows()
    pd.DataFrame(response).to_csv(table_dir / "response_perturbation.csv", index=False)
    pd.DataFrame(design).to_csv(table_dir / "design_perturbation.csv", index=False)
    pd.DataFrame(_scaling_rows()).to_csv(table_dir / "scaling_study.csv", index=False)
    _save_figure(response, figure_dir)
    print(f"Saved {table_dir / 'response_perturbation.csv'}")
    print(f"Saved {table_dir / 'design_perturbation.csv'}")
    print(f"Saved {table_dir / 'scaling_study.csv'}")


if __name__ == "__main__":
    main()
