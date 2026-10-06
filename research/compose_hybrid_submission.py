"""Merge city-specific submission files into one competition CSV.

Useful when research keeps San Juan on the tuned lag MLP but swaps Iquitos to a
stronger tree/boosting candidate from ``features_engineering/`` or the
leaderboard notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from feature_lag_mlp.config import KEY_COLUMNS, TARGET_COLUMN
from feature_lag_mlp.training import assemble_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_DIR / "data" / "submission_format.csv",
    )
    parser.add_argument(
        "--sj-submission",
        type=Path,
        default=PROJECT_DIR
        / "research"
        / "artifacts"
        / "submission_feature_lag_mlp_research.csv",
    )
    parser.add_argument(
        "--iq-submission",
        type=Path,
        default=PROJECT_DIR
        / "features_engineering"
        / "artifacts"
        / "submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "research"
        / "artifacts"
        / "submission_hybrid_mlp_sj_lgb_iq.csv",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_city_submission(path: Path, city: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = KEY_COLUMNS + [TARGET_COLUMN]
    if frame.columns.tolist() != expected:
        raise ValueError(f"{path} must have columns {expected}")
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{path} contains duplicate keys")
    city_frame = frame.loc[frame["city"].eq(city), expected].copy()
    if city_frame.empty:
        raise ValueError(f"{path} does not contain rows for city {city!r}")
    return city_frame


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv(args.template)
    sj = _load_city_submission(args.sj_submission, "sj")
    iq = _load_city_submission(args.iq_submission, "iq")
    submission = assemble_submission(template, [sj, iq])
    submission.to_csv(args.output, index=False)
    metadata = {
        "sj_submission": str(args.sj_submission),
        "iq_submission": str(args.iq_submission),
        "submission_sha256": _sha256(args.output),
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
