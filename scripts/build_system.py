"""Build, evaluate, and verify the complete standalone NLP system."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "prepare_laptop_dataset.py",
    "explore_laptop_dataset.py",
    "prepare_preprocessed_data.py",
    "train_baseline.py",
    "train_improved_models.py",
    "evaluate_and_improve.py",
    "verify_system.py",
]


def main() -> None:
    for script_name in STEPS:
        print(f"\n>>> Running {script_name}")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("\nComplete system build and verification passed.")


if __name__ == "__main__":
    main()
