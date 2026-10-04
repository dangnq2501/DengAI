"""Build a DengAI submission from independently chosen city predictions.

The default hybrid keeps San Juan predictions from the retrained vendor V10
network and Iquitos predictions from the confirmed city-specific tree model.
All rows are joined through the competition keys and restored to the official
submission-template order; positional row replacement is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["city", "year", "weekofyear"]
TARGET_COLUMN = "total_cases"


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=project_dir / "data" / "submission_format.csv",
    )
    parser.add_argument(
        "--sj-submission",
        type=Path,
        default=project_dir / "artifacts" / "submission_vendor_v10_retrained.csv",
    )
    parser.add_argument(
        "--iq-submission",
        type=Path,
        default=project_dir / "artifacts" / "submission_tree_ensemble.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir
        / "artifacts"
        / "submission_hybrid_v10_sj_tree_iq.csv",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = KEY_COLUMNS + [TARGET_COLUMN]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"{path} columns are {frame.columns.tolist()}, expected {expected_columns}"
        )
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{path} contains duplicate competition keys")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{path} contains missing predictions")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError(f"{path} contains negative predictions")
    return frame


def build_hybrid(
    template: pd.DataFrame,
    sj_submission: pd.DataFrame,
    iq_submission: pd.DataFrame,
) -> pd.DataFrame:
    expected_columns = KEY_COLUMNS + [TARGET_COLUMN]
    if template.columns.tolist() != expected_columns:
        raise ValueError("Submission template has unexpected columns")
    selected = pd.concat(
        [
            sj_submission.loc[sj_submission["city"].eq("sj")],
            iq_submission.loc[iq_submission["city"].eq("iq")],
        ],
        ignore_index=True,
    )
    hybrid = template[KEY_COLUMNS].merge(
        selected,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if hybrid[TARGET_COLUMN].isna().any():
        missing = hybrid.loc[hybrid[TARGET_COLUMN].isna(), KEY_COLUMNS]
        raise ValueError(f"Hybrid is missing {len(missing)} template rows")
    if len(hybrid) != len(template):
        raise ValueError(f"Hybrid has {len(hybrid)} rows, expected {len(template)}")
    if not hybrid[KEY_COLUMNS].equals(template[KEY_COLUMNS]):
        raise ValueError("Hybrid row order differs from the official template")
    hybrid[TARGET_COLUMN] = hybrid[TARGET_COLUMN].astype(int)
    return hybrid


def main() -> None:
    args = parse_args()
    template = pd.read_csv(args.template)
    sj_submission = _load_submission(args.sj_submission)
    iq_submission = _load_submission(args.iq_submission)
    hybrid = build_hybrid(template, sj_submission, iq_submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(args.output, index=False)

    metadata = {
        "san_juan_source": str(args.sj_submission),
        "iquitos_source": str(args.iq_submission),
        "san_juan_source_sha256": _sha256(args.sj_submission),
        "iquitos_source_sha256": _sha256(args.iq_submission),
        "submission_sha256": _sha256(args.output),
        "cities": {
            city: {
                "rows": int(len(group)),
                "mean": float(group[TARGET_COLUMN].mean()),
                "median": float(group[TARGET_COLUMN].median()),
                "maximum": int(group[TARGET_COLUMN].max()),
            }
            for city, group in hybrid.groupby("city", sort=False)
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")
    print(
        hybrid.groupby("city")[TARGET_COLUMN]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
