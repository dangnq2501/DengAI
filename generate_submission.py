#!/usr/bin/env python3
"""Train the final DengAI models and create the upload-ready CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dengai_reproduction import generate_submission  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final city-specific DengAI models and generate submission.csv."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Main/outputs/submission.csv"),
        help="Output CSV path relative to the repository root.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    result = generate_submission(REPO_ROOT, arguments.output)

    print("Submission generated successfully")
    print(f"  output: {result.output_path}")
    print(f"  rows: {result.rows} (sj={result.san_juan_rows}, iq={result.iquitos_rows})")
    print(f"  San Juan train MAE: {result.san_juan_train_mae:.3f}")
    print(f"  Iquitos train MAE: {result.iquitos_train_mae:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
