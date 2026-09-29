"""Run the core development checks with the current virtual-environment interpreter."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = (
        ("-m", "ruff", "format", "--check", "."),
        ("-m", "ruff", "check", "."),
        ("-m", "mypy"),
        ("scripts/export_contracts.py", "--check"),
        ("-m", "pytest"),
        ("-m", "build", "--no-isolation"),
    )
    for command in commands:
        print(f"\nRunning: python {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
