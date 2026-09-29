"""Run the Phase 5 early-semester forecasting and robustness study."""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerical_learning.evaluation import mae, rmse
from numerical_learning.forecasting import (
    PreparedData,
    contaminate_training,
    evaluate_model,
    fit_model,
    prepare_checkpoint,
)
from numerical_learning.linear_models import predict
from numerical_learning.temporal_data import (
    CHECKPOINTS,
    FEATURE_NAMES,
    generate_temporal_data,
    make_temporal_split,
)

MODELS = ("OLS", "Ridge", "Huber")
SEEDS = tuple(range(20))
ROBUSTNESS_SEEDS = tuple(range(20))
CONTAMINATION_FRACTIONS = (0.0, 0.05, 0.10, 0.20)


def _prepare_columns(dataset, split, columns):
    X = dataset.X[:, columns]
    train_X = X[split.train]
    means = train_X.mean(axis=0)
    scales = np.where(train_X.std(axis=0) == 0.0, 1.0, train_X.std(axis=0))
    standardize = lambda values: (values - means) / scales
    return PreparedData(
        train_X=standardize(train_X), validation_X=standardize(X[split.validation]),
        test_X=standardize(X[split.test]), train_y=dataset.y[split.train],
        validation_y=dataset.y[split.validation], test_y=dataset.y[split.test],
        means=means, scales=scales,
        raw_train_X=train_X.copy(), raw_validation_X=X[split.validation].copy(),
        raw_test_X=X[split.test].copy(),
    )


def _clean_forecasting() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = []
    for replicate, seed in enumerate(SEEDS):
        dataset = generate_temporal_data(seed=5200 + seed)
        split = make_temporal_split(len(dataset.y), seed=8800 + seed)
        for checkpoint in CHECKPOINTS:
            data = prepare_checkpoint(dataset, split, checkpoint)
            for model in MODELS:
                result = evaluate_model(data, model)
                raw.append({"replicate": replicate, "seed": seed, "checkpoint": checkpoint, "number_of_features": data.train_X.shape[1], **result})
    raw_frame = pd.DataFrame(raw)
    success = raw_frame[raw_frame.status == "success"]
    summary = success.groupby(["checkpoint", "model"], as_index=False).agg(
        mean_test_rmse=("test_rmse", "mean"), median_test_rmse=("test_rmse", "median"),
        std_test_rmse=("test_rmse", "std"), mean_test_mae=("test_mae", "mean"),
        mean_test_r2=("test_r2", "mean"), success_rate=("status", lambda values: float((values == "success").mean())),
        n_successful=("replicate", "nunique"),
    )
    attempts = raw_frame.groupby(["checkpoint", "model"], as_index=False).agg(
        n_attempted=("replicate", "nunique"),
        success_rate=("status", lambda values: float((values == "success").mean())),
    )
    summary = summary.drop(columns="success_rate").merge(attempts, on=["checkpoint", "model"], how="outer")
    by_checkpoint = success.groupby(["checkpoint", "model"], as_index=False).agg(
        number_of_features=("number_of_features", "first"), test_mae=("test_mae", "mean"),
        test_rmse=("test_rmse", "mean"), test_r2=("test_r2", "mean"),
    )
    ridge = by_checkpoint[by_checkpoint.model == "Ridge"].set_index("checkpoint").reindex(CHECKPOINTS)
    improvements = []
    for previous, current in pairwise(CHECKPOINTS):
        improvements.append({"previous_checkpoint": previous, "checkpoint": current, "model": "Ridge", "delta_rmse": ridge.loc[previous, "test_rmse"] - ridge.loc[current, "test_rmse"]})
    return raw_frame, summary, by_checkpoint, {"improvements": pd.DataFrame(improvements)}


def _ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = []
    full_columns = list(range(len(FEATURE_NAMES)))
    groups = {name: [index] for index, name in enumerate(FEATURE_NAMES)}
    groups.update({"prior_history": [0, 1], "attendance": [2, 7], "assessment": [3, 5, 6], "coursework": [4]})
    for seed in SEEDS:
        dataset = generate_temporal_data(seed=5200 + seed)
        split = make_temporal_split(len(dataset.y), seed=8800 + seed)
        full_data = _prepare_columns(dataset, split, full_columns)
        full_coefficients, _, _, _, _ = fit_model(full_data, "Ridge")
        full_rmse = rmse(full_data.test_y, predict(full_data.test_X, full_coefficients))
        full_mae = mae(full_data.test_y, predict(full_data.test_X, full_coefficients))
        for name, removed in groups.items():
            kept = [index for index in full_columns if index not in removed]
            data = _prepare_columns(dataset, split, kept)
            coefficients, _, _, _, _ = fit_model(data, "Ridge")
            predictions = predict(data.test_X, coefficients)
            raw.append({"seed": seed, "feature_or_group": name, "delta_rmse": rmse(data.test_y, predictions) - full_rmse, "delta_mae": mae(data.test_y, predictions) - full_mae})
    frame = pd.DataFrame(raw)
    summary = frame.groupby("feature_or_group", as_index=False).agg(mean_delta_rmse=("delta_rmse", "mean"), median_delta_rmse=("delta_rmse", "median"), std_delta_rmse=("delta_rmse", "std"), mean_delta_mae=("delta_mae", "mean"), fraction_harmful_when_removed=("delta_rmse", lambda values: float((values > 0).mean())))
    return frame, summary


def _robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in ROBUSTNESS_SEEDS:
        dataset = generate_temporal_data(seed=6200 + seed)
        split = make_temporal_split(len(dataset.y), seed=9300 + seed)
        data = prepare_checkpoint(dataset, split, "later_semester")
        clean_results = {model: evaluate_model(data, model) for model in MODELS}
        for mode in ("response", "feature"):
            for fraction in CONTAMINATION_FRACTIONS:
                contaminated = contaminate_training(data, fraction, seed=7400 + seed, mode=mode)
                for model in MODELS:
                    result = evaluate_model(contaminated, model)
                    clean_rmse = clean_results[model]["test_rmse"]
                    rows.append({"seed": seed, "contamination_type": mode, "contamination_fraction": fraction, "model": model, "test_rmse": result["test_rmse"], "test_mae": result["test_mae"], "test_r2": result["test_r2"], "rmse_degradation": result["test_rmse"] - clean_rmse, "fit_status": result["status"], "huber_iterations": result["huber_iterations"]})
    frame = pd.DataFrame(rows)
    successful = frame[frame.fit_status == "success"]
    summary = successful.groupby(["contamination_type", "contamination_fraction", "model"], as_index=False).agg(mean_test_rmse=("test_rmse", "mean"), mean_test_mae=("test_mae", "mean"), mean_test_r2=("test_r2", "mean"), mean_rmse_degradation=("rmse_degradation", "mean"), median_rmse_degradation=("rmse_degradation", "median"), std_rmse_degradation=("rmse_degradation", "std"))
    attempts = frame.groupby(["contamination_type", "contamination_fraction", "model"], as_index=False).agg(success_rate=("fit_status", lambda values: float((values == "success").mean())), n_attempted=("seed", "nunique"))
    summary = summary.merge(attempts, on=["contamination_type", "contamination_fraction", "model"], how="outer")
    return frame, summary


def _figures(by_checkpoint, improvements, ablation, robustness, directory):
    order = {name: index for index, name in enumerate(CHECKPOINTS)}
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in by_checkpoint.groupby("model"):
        group = group.sort_values("checkpoint", key=lambda values: values.map(order))
        ax.plot(group.checkpoint, group.test_rmse, marker="o", label=model)
    ax.set_ylabel("Mean held-out test RMSE"); ax.set_xlabel("Information horizon"); ax.set_title("Early-Semester Forecasting Generalization"); ax.legend(); fig.tight_layout(); fig.savefig(directory / "forecasting_rmse_vs_checkpoint.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in by_checkpoint.groupby("model"):
        group = group.sort_values("checkpoint", key=lambda values: values.map(order))
        ax.plot(group.checkpoint, group.test_r2, marker="o", label=model)
    ax.set_ylabel("Mean held-out test R²"); ax.set_xlabel("Information horizon"); ax.set_title("Held-Out Explained Variation by Checkpoint"); ax.legend(); fig.tight_layout(); fig.savefig(directory / "forecasting_r2_vs_checkpoint.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(improvements.checkpoint, improvements.delta_rmse); ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("RMSE reduction"); ax.set_xlabel("New information checkpoint"); ax.set_title("Incremental Information Gain (Ridge)"); fig.tight_layout(); fig.savefig(directory / "checkpoint_information_gain.png", dpi=150); plt.close(fig)
    plot = ablation.sort_values("mean_delta_rmse"); fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(plot.feature_or_group, plot.mean_delta_rmse); ax.axvline(0, color="black", linewidth=0.8); ax.set_xlabel("Mean ΔRMSE when removed"); ax.set_title("Predictive Feature Ablation (Ridge)"); fig.tight_layout(); fig.savefig(directory / "feature_ablation.png", dpi=150); plt.close(fig)
    for mode, filename in (("response", "response_contamination_robustness.png"), ("feature", "feature_contamination_robustness.png")):
        fig, ax = plt.subplots(figsize=(8, 5)); subset = robustness[robustness.contamination_type == mode]
        for model, group in subset.groupby("model"):
            ax.plot(group.contamination_fraction, group.mean_test_rmse, marker="o", label=model)
        ax.set_xlabel("Training contamination fraction"); ax.set_ylabel("Clean-test RMSE"); ax.set_title(f"Robustness to {mode} contamination"); ax.legend(); fig.tight_layout(); fig.savefig(directory / filename, dpi=150); plt.close(fig)


def main() -> None:
    """Run all Phase 5 clean forecasting and robustness experiments."""
    repository = Path(__file__).resolve().parents[1]
    tables = repository / "results" / "tables"; figures = repository / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    raw, summary, by_checkpoint, extras = _clean_forecasting(); ablation_raw, ablation_summary = _ablation(); robustness_raw, robustness_summary = _robustness()
    raw.to_csv(tables / "forecasting_raw.csv", index=False); summary.to_csv(tables / "forecasting_summary.csv", index=False); by_checkpoint.to_csv(tables / "forecasting_by_checkpoint.csv", index=False); extras["improvements"].to_csv(tables / "checkpoint_improvement.csv", index=False)
    ablation_raw.to_csv(tables / "feature_ablation_raw.csv", index=False); ablation_summary.to_csv(tables / "feature_ablation_summary.csv", index=False)
    robustness_raw.to_csv(tables / "robustness_raw.csv", index=False); robustness_summary.to_csv(tables / "robustness_summary.csv", index=False)
    _figures(by_checkpoint, extras["improvements"], ablation_summary, robustness_summary, figures)
    print(summary.to_string(index=False)); print("\nAblation\n", ablation_summary.to_string(index=False)); print("\nRobustness\n", robustness_summary.to_string(index=False))


if __name__ == "__main__":
    main()
