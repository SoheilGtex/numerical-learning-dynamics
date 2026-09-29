import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase6_documentation_and_scripts_exist() -> None:
    required = (
        "README.md", "CHANGELOG.md", "CITATION.cff",
        "docs/technical_report.md", "docs/reproducibility.md",
        "docs/results_index.md", "docs/project_summary.md",
        "docs/research_integrity.md", "scripts/reproduce_all.py",
        "scripts/verify_repository.py", "scripts/capture_environment.py",
        ".github/workflows/ci.yml",
    )
    assert all((ROOT / path).is_file() for path in required)
    requirements = ROOT / "requirements-release.txt"
    assert requirements.is_file() and requirements.stat().st_size > 0
    assert "requirements-release.txt" in (ROOT / "docs/reproducibility.md").read_text()


def test_citation_metadata_and_report_navigation() -> None:
    citation = (ROOT / "CITATION.cff").read_text()
    assert "cff-version: 1.2.0" in citation
    assert "message:" in citation
    assert "title: \"Numerical Learning Dynamics\"" in citation
    assert "given-names: Soheil" in citation
    assert "family-names: Salmani" in citation
    report = (ROOT / "docs/technical_report.md").read_text()
    assert "## 13. Conclusion" in report
    readme = (ROOT / "README.md").read_text()
    assert "docs/technical_report.md" in readme
    assert "CITATION.cff" in readme


def test_results_index_references_existing_paths() -> None:
    index = (ROOT / "docs/results_index.md").read_text()
    for path in ("results/tables/solver_comparison.csv", "results/tables/singular_values_conditioning.csv", "scripts/run_conditioning_study.py"):
        assert path in index
        assert (ROOT / path).is_file()


def test_experiment_manifest_is_valid_and_traceable() -> None:
    manifest = json.loads((ROOT / "results/reproducibility/experiment_manifest.json").read_text())
    assert manifest["synthetic_data_only"] is True
    experiments = manifest["experiments"]
    assert {entry["phase"] for entry in experiments} == {1, 2, 3, 4, 5}
    assert len({entry["id"] for entry in experiments}) == len(experiments)
    for entry in experiments:
        assert entry["synthetic"] is True
        assert (ROOT / entry["script"]).is_file()
        assert all((ROOT / output).is_file() for output in entry["primary_outputs"])


def test_required_result_artifacts_are_nonempty() -> None:
    paths = list((ROOT / "results/tables").glob("*.csv")) + list((ROOT / "results/figures").glob("*.png"))
    assert paths
    assert all(path.stat().st_size > 0 for path in paths)
    assert (ROOT / "results/reproducibility/artifact_checksums.sha256").is_file()
