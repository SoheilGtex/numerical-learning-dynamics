"""Run all accepted numerical experiments in their documented order."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    "run_baseline.py",
    "run_solver_comparison.py",
    "run_conditioning_study.py",
    "run_perturbation_study.py",
    "run_regularization_study.py",
    "run_uncertainty_study.py",
    "run_forecasting_study.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify outputs after reproduction")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    for script in SCRIPTS:
        print(f"Running {script} ...", flush=True)
        subprocess.run([sys.executable, str(repository / "scripts" / script)], cwd=repository, check=True)
    if args.verify:
        subprocess.run([sys.executable, str(repository / "scripts" / "verify_repository.py")], cwd=repository, check=True)
    print("Full reproduction completed successfully.")


if __name__ == "__main__":
    main()
