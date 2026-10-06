"""Validate submissions and quantify paired audit uncertainty with weekly blocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent / 'artifacts'
    template = pd.read_csv(root.parents[1] / 'data' / 'submission_format.csv')
    distribution = {}
    for role in ('control', 'candidate'):
        submission = pd.read_csv(root / f'submission_{role}.csv')
        assert submission.iloc[:, :3].equals(template.iloc[:, :3])
        assert len(submission) == 416 and not submission.isna().any().any()
        assert submission.total_cases.ge(0).all()
        assert pd.api.types.is_integer_dtype(submission.total_cases)
        distribution[role] = submission.groupby('city').total_cases.agg(['mean', 'max']).to_dict()
    audit = pd.read_csv(root / 'audit_predictions.csv')
    rng = np.random.default_rng(42)
    samples = np.zeros(5000)
    for city, group in audit.groupby('city'):
        control = group.loc[group.role.eq('control')].reset_index(drop=True)
        candidate = group.loc[group.role.eq('candidate')].reset_index(drop=True)
        assert control[['city', 'year', 'weekofyear', 'total_cases']].equals(
            candidate[['city', 'year', 'weekofyear', 'total_cases']])
        delta = ((candidate.prediction - candidate.total_cases).abs().to_numpy()
                 - (control.prediction - control.total_cases).abs().to_numpy())
        starts = rng.integers(len(delta), size=(5000, int(np.ceil(len(delta) / 13))))
        indices = ((starts[:, :, None] + np.arange(13)) % len(delta)).reshape(5000, -1)[:, :len(delta)]
        samples += delta[indices].mean(axis=1) * len(delta) / 416
    result = dict(submission_checks_passed=True, distribution=distribution,
                  block_weeks=13, bootstrap_replicates=5000, bootstrap_seed=42,
                  candidate_minus_control_interval=np.quantile(samples, [.025, .975]).tolist(),
                  caveat='Descriptive paired circular-block bootstrap from one audit period; historical selection effects are not captured.')
    (root / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
