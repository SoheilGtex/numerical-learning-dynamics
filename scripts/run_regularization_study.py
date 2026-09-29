"""Run the focused Phase 4 Ridge selection and stability study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerical_learning.conditioning import (
    Standardization,
    make_near_collinear_predictors,
    relative_perturbation,
    standardize_predictors,
    standardized_to_raw_coefficients,
)
from numerical_learning.evaluation import coefficient_relative_error, rmse
from numerical_learning.linear_models import add_intercept, predict
from numerical_learning.regularization import (
    coefficient_variance_around_expected,
    dimension_aware_lambda_grid,
    relative_coefficient_bias,
    select_lambda_by_gcv,
    select_lambda_by_validation,
    select_lambda_oracle,
    solve_ridge_svd,
)
from numerical_learning.solvers import condition_number

REGIMES = {"moderate": 1.0, "strong": 1e-3, "very_strong": 1e-6}
ALPHAS = np.concatenate(([0.0], np.logspace(-12, 2, 20)))
NOISE_STD = 0.3


def _split_data(X: np.ndarray, beta: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], Standardization]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    train_end = int(0.6 * X.shape[0])
    validation_end = int(0.8 * X.shape[0])
    splits = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }
    y = add_intercept(X) @ beta + rng.normal(0.0, NOISE_STD, X.shape[0])
    train_X = X[splits["train"]]
    standardized_train, transform = standardize_predictors(train_X)
    data = {"train_X": standardized_train, "train_y": y[splits["train"]]}
    for name in ("validation", "test"):
        data[f"{name}_X"] = (X[splits[name]] - transform.means) / transform.scales
        data[f"{name}_y"] = y[splits[name]]
    return data, transform


def _raw_coefficients(result_coefficients: np.ndarray, transform: Standardization) -> np.ndarray:
    return standardized_to_raw_coefficients(result_coefficients, transform)


def _fit_rows(regime: str, delta: float, seed: int = 1729) -> tuple[list[dict[str, object]], dict[str, object]]:
    beta = np.array([1.0, 0.5, -1.2, 2.0])
    X = make_near_collinear_predictors(n_samples=180, delta=delta, seed=31415)
    data, transform = _split_data(X, beta, seed)
    alphas, lambdas = dimension_aware_lambda_grid(data["train_X"], ALPHAS)
    condition = condition_number(add_intercept(data["train_X"]))
    rows: list[dict[str, object]] = []
    for alpha, lambda_ in zip(alphas, lambdas, strict=True):
        result = solve_ridge_svd(data["train_X"], data["train_y"], float(lambda_))
        raw_beta = _raw_coefficients(result.coefficients, transform)
        rows.append(
            {
                "regime": regime,
                "seed": seed,
                "delta": delta,
                "condition_number": condition,
                "alpha": alpha,
                "lambda": lambda_,
                "effective_df": result.effective_df,
                "regularized_system_condition_number": result.regularized_system_condition_number,
                "coefficient_relative_error": coefficient_relative_error(raw_beta, beta),
                "training_fitted_rmse": rmse(data["train_y"], predict(data["train_X"], result.coefficients)),
                "validation_rmse": rmse(data["validation_y"], predict(data["validation_X"], result.coefficients)),
                "residual_norm": result.residual_norm,
            }
        )
    frame = pd.DataFrame(rows)
    validation_index = select_lambda_by_validation(frame["validation_rmse"].to_numpy())
    gcv_index = select_lambda_by_gcv(data["train_X"], data["train_y"], lambdas)
    oracle_index = select_lambda_oracle(frame["coefficient_relative_error"].to_numpy())
    selections = {
        "validation": validation_index,
        "GCV": gcv_index,
        "oracle_coefficient": oracle_index,
    }
    selection_rows = []
    for method, index in selections.items():
        selected = frame.iloc[index]
        selected_result = solve_ridge_svd(
            data["train_X"], data["train_y"], float(selected["lambda"])
        )
        final_test_rmse = rmse(
            data["test_y"], predict(data["test_X"], selected_result.coefficients)
        )
        selection_rows.append(
            {
                "regime": regime,
                "selection_method": method,
                "selected_alpha": selected["alpha"],
                "selected_lambda": selected["lambda"],
                "validation_rmse": selected["validation_rmse"],
                "test_rmse": final_test_rmse,
                "coefficient_relative_error": selected["coefficient_relative_error"],
                "effective_df": selected["effective_df"],
                "regularized_system_condition_number": selected["regularized_system_condition_number"],
            }
        )
    return rows, {"rows": selection_rows, "data": data, "transform": transform, "beta": beta, "condition": condition}


def _bias_variance(
    regime: str,
    delta: float,
    alphas: np.ndarray,
    seed: int = 2027,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    beta = np.array([1.0, 0.5, -1.2, 2.0])
    beta_norm_squared = float(np.dot(beta, beta))
    X = make_near_collinear_predictors(n_samples=180, delta=delta, seed=31415)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    train_idx, test_idx = indices[:108], indices[144:]
    train_X, test_X = X[train_idx], X[test_idx]
    standardized_train, transform = standardize_predictors(train_X)
    standardized_test = (test_X - transform.means) / transform.scales
    _, lambdas = dimension_aware_lambda_grid(standardized_train, alphas)
    means = {float(value): [] for value in lambdas}
    expected_betas = {}
    raw_rows: list[dict[str, object]] = []
    test_rmses: dict[float, list[float]] = {float(value): [] for value in lambdas}
    y_expected = add_intercept(train_X) @ beta
    for lambda_ in lambdas:
        expected_result = solve_ridge_svd(standardized_train, y_expected, float(lambda_))
        expected_betas[float(lambda_)] = standardized_to_raw_coefficients(
            expected_result.coefficients, transform
        )
    for replicate in range(30):
        y_train = add_intercept(train_X) @ beta + rng.normal(0.0, NOISE_STD, train_X.shape[0])
        y_test = add_intercept(test_X) @ beta + rng.normal(0.0, NOISE_STD, test_X.shape[0])
        for lambda_ in lambdas:
            result = solve_ridge_svd(standardized_train, y_train, float(lambda_))
            raw_beta = standardized_to_raw_coefficients(result.coefficients, transform)
            means[float(lambda_)].append(raw_beta)
            test_rmse = rmse(y_test, predict(standardized_test, result.coefficients))
            test_rmses[float(lambda_)].append(test_rmse)
            raw_rows.append({"regime": regime, "replicate": replicate, "lambda": lambda_, "coefficient_0": raw_beta[0], "coefficient_1": raw_beta[1], "coefficient_2": raw_beta[2], "coefficient_3": raw_beta[3], "test_rmse": test_rmse})
    summary = []
    for alpha, lambda_ in zip(alphas, lambdas, strict=True):
        estimates = np.asarray(means[float(lambda_)])
        expected_beta = expected_betas[float(lambda_)]
        mean_estimate = estimates.mean(axis=0)
        errors = estimates - beta
        bias_vector = expected_beta - beta
        bias_squared = float(np.dot(bias_vector, bias_vector))
        variance = coefficient_variance_around_expected(estimates, expected_beta)
        coefficient_mse = float(np.mean(np.sum(errors**2, axis=1)))
        summary.append(
            {
                "regime": regime,
                "alpha": alpha,
                "lambda": lambda_,
                "relative_bias": relative_coefficient_bias(expected_beta, beta),
                "bias_squared": bias_squared,
                "coefficient_variance": variance,
                "coefficient_mse": coefficient_mse,
                "relative_bias_squared": bias_squared / beta_norm_squared,
                "normalized_coefficient_variance": variance / beta_norm_squared,
                "normalized_coefficient_mse": coefficient_mse / beta_norm_squared,
                "decomposition_gap": coefficient_mse - (variance + bias_squared),
                "monte_carlo_relative_mean_error": relative_coefficient_bias(mean_estimate, beta),
                "mean_test_rmse": float(np.mean(test_rmses[float(lambda_)])),
                "std_test_rmse": float(np.std(test_rmses[float(lambda_)])),
                "effective_df": solve_ridge_svd(
                    standardized_train, add_intercept(train_X) @ beta, float(lambda_)
                ).effective_df,
            }
        )
    return summary, raw_rows


def _stability(regime: str, delta: float, lambdas: np.ndarray) -> list[dict[str, object]]:
    beta = np.array([1.0, 0.5, -1.2, 2.0])
    X = make_near_collinear_predictors(n_samples=180, delta=delta, seed=31415)
    data, transform = _split_data(X, beta, 1729)
    direction = np.random.default_rng(6071).normal(size=data["train_y"].shape)
    condition = condition_number(add_intercept(data["train_X"]))
    perturbed_y = relative_perturbation(data["train_y"], 1e-6, 6071, direction)
    relative_response = np.linalg.norm(perturbed_y - data["train_y"]) / np.linalg.norm(data["train_y"])
    rows = []
    for alpha, lambda_ in zip(ALPHAS, lambdas, strict=True):
        base = solve_ridge_svd(data["train_X"], data["train_y"], float(lambda_))
        changed = solve_ridge_svd(data["train_X"], perturbed_y, float(lambda_))
        base_raw = standardized_to_raw_coefficients(base.coefficients, transform)
        changed_raw = standardized_to_raw_coefficients(changed.coefficients, transform)
        relative_solution = np.linalg.norm(changed_raw - base_raw) / np.linalg.norm(base_raw)
        rows.append({"alpha": alpha, "lambda": lambda_, "condition_number": condition, "regularized_system_condition_number": base.regularized_system_condition_number, "relative_response_perturbation": relative_response, "relative_solution_perturbation": relative_solution, "amplification_ratio": relative_solution / relative_response, "coefficient_relative_error": coefficient_relative_error(base_raw, beta), "fitted_rmse": rmse(data["train_y"], predict(data["train_X"], base.coefficients))})
    return rows


def _figures(path: pd.DataFrame, selection: pd.DataFrame, bias: pd.DataFrame, stability: pd.DataFrame, figure_dir: Path) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, group in path.groupby("regime"):
        ax.plot(group[group["alpha"] > 0]["alpha"], group[group["alpha"] > 0]["coefficient_relative_error"], marker="o", label=regime)
        zero = group[group["alpha"] == 0]
        ax.scatter(zero["alpha"], zero["coefficient_relative_error"], marker="x")
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_yscale("log")
    ax.set_xlabel("Dimensionless alpha (lambda = alpha sigma_max^2; zero shown at origin)")
    ax.set_ylabel("Relative coefficient error")
    ax.set_title("Ridge Regularization Path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "regularization_path.png", dpi=150)
    plt.close(fig)

    strong = path[path["regime"] == "strong"]
    fig, ax = plt.subplots(figsize=(8, 5))
    strong_positive = strong[strong["alpha"] > 0]
    ax.plot(strong_positive["alpha"], strong_positive["validation_rmse"], marker="o", label="validation RMSE")
    for method, style in (("validation", "--"), ("GCV", ":"), ("oracle_coefficient", "-.")):
        selected = selection[(selection["regime"] == "strong") & (selection["selection_method"] == method)].iloc[0]
        value = selected["selected_alpha"]
        ax.axvline(value, linestyle=style, label=f"{method} selected" + (" (synthetic-only)" if method == "oracle_coefficient" else ""))
        ax.scatter(value, selected["validation_rmse"], marker="s", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Dimensionless alpha")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation Selection and Final Test Evaluations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_selection_vs_regularization.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, group in bias.groupby("regime"):
        ax.plot(group["alpha"], group["relative_bias_squared"], marker="o", label=f"{regime} squared bias")
        ax.plot(group["alpha"], group["normalized_coefficient_variance"], marker="x", linestyle="--", label=f"{regime} normalized variance")
        ax.plot(group["alpha"], group["normalized_coefficient_mse"], marker=".", linestyle=":", label=f"{regime} normalized MSE")
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_yscale("log")
    ax.set_xlabel("Dimensionless alpha")
    ax.set_ylabel("Dimensionless squared coefficient magnitude")
    ax.set_title("Normalized Bias-Variance Trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "bias_variance_tradeoff.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stability[stability["alpha"] > 0]["alpha"], stability[stability["alpha"] > 0]["amplification_ratio"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimensionless alpha")
    ax.set_ylabel("Response-perturbation amplification")
    ax.set_title("Regularization Stability")
    fig.tight_layout()
    fig.savefig(figure_dir / "regularization_stability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute Phase 4 regularization experiments."""
    repository = Path(__file__).resolve().parents[1]
    table_dir = repository / "results" / "tables"
    figure_dir = repository / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path_rows, selections, bias_rows, raw_bias = [], [], [], []
    for regime, delta in REGIMES.items():
        rows, metadata = _fit_rows(regime, delta)
        path_rows.extend(rows)
        selections.extend(metadata["rows"])
        alphas = np.array([row["alpha"] for row in rows])
        summary, raw = _bias_variance(regime, delta, alphas)
        bias_rows.extend(summary)
        raw_bias.extend(raw)
    path = pd.DataFrame(path_rows)
    selection = pd.DataFrame(selections)
    bias = pd.DataFrame(bias_rows)
    stability = pd.DataFrame(_stability("very_strong", REGIMES["very_strong"], path[path["regime"] == "very_strong"]["lambda"].to_numpy()))
    path.to_csv(table_dir / "regularization_path.csv", index=False)
    selection.to_csv(table_dir / "lambda_selection.csv", index=False)
    bias.to_csv(table_dir / "bias_variance_summary.csv", index=False)
    pd.DataFrame(raw_bias).to_csv(table_dir / "bias_variance_raw.csv", index=False)
    stability.to_csv(table_dir / "regularization_stability.csv", index=False)
    _figures(path, selection, bias, stability, figure_dir)
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
