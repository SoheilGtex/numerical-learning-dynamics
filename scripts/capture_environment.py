"""Capture reproducibility-relevant software versions without private paths."""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = (
    "numpy", "pandas", "matplotlib", "scipy", "pytest", "ruff", "pip", "setuptools"
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    output = repository / "results" / "reproducibility" / "environment.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        *(f"{name}: {package_version(name)}" for name in PACKAGES),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
