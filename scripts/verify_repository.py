"""Verify repository structure and non-empty reproducibility artifacts."""

import json
import re
from pathlib import Path

REQUIRED_FILES = (
    "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md", "pyproject.toml",
    "requirements-release.txt",
    "docs/technical_report.md", "docs/reproducibility.md", "docs/results_index.md",
    "docs/project_summary.md", "docs/research_integrity.md",
    "results/reproducibility/experiment_manifest.json",
    "results/reproducibility/environment.txt",
    "results/reproducibility/artifact_checksums.sha256",
)
REQUIRED_TABLES = (
    "baseline_metrics.csv", "solver_comparison.csv", "conditioning_summary.csv",
    "bias_variance_summary.csv", "uncertainty_coverage.csv", "forecasting_summary.csv",
    "feature_ablation_summary.csv", "robustness_summary.csv",
)
REQUIRED_FIGURES = (
    "predicted_vs_actual.png", "solver_coefficient_error.png", "bias_variance_tradeoff.png",
    "ols_interval_coverage.png", "forecasting_rmse_vs_checkpoint.png",
    "feature_ablation.png", "response_contamination_robustness.png",
    "feature_contamination_robustness.png",
)
MARKDOWN_FILES = (
    "README.md", "docs/technical_report.md", "docs/project_summary.md",
    "docs/reproducibility.md", "docs/results_index.md",
)


def nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"missing or empty: {path}")


def verify_markdown_links(repository: Path) -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for relative in MARKDOWN_FILES:
        source = repository / relative
        for target in link_pattern.findall(source.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            if not (source.parent / target_path).resolve().is_file():
                raise AssertionError(f"broken Markdown link: {relative} -> {target}")


def verify_results_index(repository: Path) -> None:
    index = (repository / "docs/results_index.md").read_text(encoding="utf-8")
    for path in re.findall(r"`((?:results|scripts)/[^`]+)`", index):
        nonempty(repository / path)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    verify_markdown_links(repository)
    verify_results_index(repository)
    for relative in REQUIRED_FILES:
        nonempty(repository / relative)
    for name in REQUIRED_TABLES:
        nonempty(repository / "results" / "tables" / name)
    for name in REQUIRED_FIGURES:
        nonempty(repository / "results" / "figures" / name)
    manifest = json.loads((repository / "results/reproducibility/experiment_manifest.json").read_text())
    experiments = manifest["experiments"]
    if manifest.get("release_requirements_file") != "requirements-release.txt":
        raise AssertionError("manifest does not reference release requirements")
    identifiers = [item["id"] for item in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate experiment IDs")
    for item in experiments:
        if item["phase"] not in {1, 2, 3, 4, 5} or item["synthetic"] is not True:
            raise AssertionError(f"invalid manifest entry: {item['id']}")
        nonempty(repository / item["script"])
        for output in item["primary_outputs"]:
            nonempty(repository / output)
    for table in (repository / "results" / "tables").glob("*.csv"):
        text = table.read_text(encoding="utf-8")
        if text.strip() and all(value in {"", "nan", "NaN"} for value in text.splitlines()[-1].split(",")):
            raise AssertionError(f"possible NaN-only table: {table}")
    print("Repository verification passed.")


if __name__ == "__main__":
    main()
