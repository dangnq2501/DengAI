"""Reproducible MLP seed/checkpoint ensembles with a held-out long-horizon audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from feature_lag_mlp.config import PROFILE_CONFIGS, KEY_COLUMNS, TARGET_COLUMN
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.features import build_training_matrix, build_test_matrix
from feature_lag_mlp.training import assemble_submission, prediction_frame
from feature_lag_mlp.validation import prepared_folds


def integer_cases(raw):
    raw = np.asarray(raw)
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite predictions')
    return np.maximum(raw, 0).astype(int)  # preserve the control's truncation


def aggregate(bank):
    """bank has profile, seed, checkpoint, row axes; no labels used here."""
    result = {'control': bank[0, 0, -1]}
    for index, profile in enumerate(('improved', 'tuned')):
        result[profile + '_mean'] = bank[index, :, -1].mean(axis=0)
        result[profile + '_median'] = np.median(bank[index, :, -1], axis=0)
        result[profile + '_snapshots'] = bank[index].mean(axis=(0, 1))
    result['mixed_mean'] = bank[:, :, -1].mean(axis=(0, 1))
    result['mixed_snapshots'] = bank.mean(axis=(0, 1, 2))
    return result


def fit_bank(x, y, predict_x, city, seeds, epochs, steps, directory):
    from torch_backend import fit
    import torch
    checkpoints = sorted(set((max(1, epochs - 10), max(1, epochs - 5), epochs)))
    bank = []
    directory.mkdir(parents=True, exist_ok=True)
    for profile in ('improved', 'tuned'):
        runs = []
        for seed in seeds:
            path = directory / f'{profile}_{seed}.npz'
            if path.exists():
                with np.load(path) as saved:
                    runs.append(saved['predictions'])
                continue
            predictions, history, model = fit(x, y, predict_x, city,
                PROFILE_CONFIGS[profile][city], seed, epochs, steps, checkpoints)
            np.savez_compressed(path, predictions=predictions)
            pd.DataFrame(history).to_csv(path.with_suffix('.csv'), index=False)
            if directory.name == 'final':
                torch.save(model.state_dict(), directory / f'{profile}_{seed}.pt')
            runs.append(predictions)
            print(f'Finished {city}/{directory.name}/{profile}/seed={seed}', flush=True)
        bank.append(np.stack(runs))
    return np.stack(bank)


def split_data(train, test):
    """Remove one official-test-length block from each city before development."""
    development, audit = [], []
    for city in ('sj', 'iq'):
        frame = train.loc[train.city.eq(city)].sort_values('week_start_date')
        horizon = int(test.city.eq(city).sum())
        development.append(frame.iloc[:-horizon])
        audit.append(frame.iloc[-horizon:])
    return pd.concat(development, ignore_index=True), pd.concat(audit, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research2' / 'artifacts')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137, 2026])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1 or args.seeds[0] != 42:
        parser.error('positive epochs/steps and first seed 42 are required')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train, test, template = load_competition_data(ROOT / 'data')
    dev, audit = split_data(train, test)
    import torch
    manifest = dict(seeds=args.seeds, epochs=args.epochs, steps=args.steps,
                    backend='pytorch-approximation', torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('torch_backend.py').read_bytes()).hexdigest(),
                    inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'data').glob('*.csv'))})
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Run configuration changed: use a fresh output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    rows, prediction_rows, selections = [], [], {}
    # Complete development selection for both cities before looking at audit labels.
    for city in ('sj', 'iq'):
        for fold in prepared_folds(dev, city, 3, 52):
            bank = fit_bank(fold['train_x'], fold['train_y'], fold['validation_x'],
                            city, args.seeds, args.epochs, args.steps,
                            out / city / f"development_{fold['fold']}")
            for name, raw in aggregate(bank).items():
                pred = integer_cases(raw)
                rows.append(dict(city=city, fold=fold['fold'], candidate=name,
                                 mae=float(np.abs(pred - fold['validation_y']).mean()),
                                 start=fold['validation_start'], end=fold['validation_end']))
                prediction_rows.extend(dict(city=city, fold=fold['fold'], candidate=name,
                    row=i, target=float(y), raw=float(r), prediction=int(p))
                    for i, (y, r, p) in enumerate(zip(fold['validation_y'], raw, pred)))
            pd.DataFrame(rows).to_csv(out / 'development_scores.csv', index=False)
        scores = pd.DataFrame(rows).query('city == @city')
        ranking = scores.groupby('candidate').mae.agg(['mean', 'std', 'max']).sort_values('mean')
        ranking.to_csv(out / f'ranking_{city}.csv')
        selections[city] = str(ranking.index[0])
        print(f'{city} development selection: {selections[city]}\n{ranking}', flush=True)
    (out / 'selection.json').write_text(json.dumps(selections, indent=2) + '\n')
    pd.DataFrame(prediction_rows).to_csv(out / 'development_predictions.csv', index=False)
    audit_scores, audit_predictions = [], []
    submissions = {'candidate': [], 'control': []}
    for city in ('sj', 'iq'):
        x, y = build_training_matrix(dev, city)
        bank = fit_bank(x, y, build_test_matrix(dev, audit.drop(columns=TARGET_COLUMN), city),
                        city, args.seeds, args.epochs, args.steps, out / city / 'audit')
        predictions = aggregate(bank)
        city_audit = audit.loc[audit.city.eq(city)].reset_index(drop=True)
        for role, name in (('control', 'control'), ('candidate', selections[city])):
            pred = integer_cases(predictions[name])
            target = city_audit[TARGET_COLUMN].to_numpy()
            audit_scores.append(dict(city=city, role=role, candidate=name, rows=len(target),
                mae=float(np.abs(pred - target).mean()), prediction_mean=float(pred.mean()),
                target_mean=float(target.mean())))
            frame = city_audit[KEY_COLUMNS + [TARGET_COLUMN]].copy()
            frame['role'], frame['prediction'] = role, pred
            audit_predictions.append(frame)
        # Freeze the selected recipe regardless of audit performance.
        x, y = build_training_matrix(train, city)
        final = aggregate(fit_bank(x, y, build_test_matrix(train, test, city), city,
                          args.seeds, args.epochs, args.steps, out / city / 'final'))
        for role, name in (('control', 'control'), ('candidate', selections[city])):
            submissions[role].append(prediction_frame(test, city, integer_cases(final[name])))
    pd.DataFrame(audit_scores).to_csv(out / 'audit_scores.csv', index=False)
    pd.concat(audit_predictions).to_csv(out / 'audit_predictions.csv', index=False)
    for role, frames in submissions.items():
        assemble_submission(template, frames).to_csv(out / f'submission_{role}.csv', index=False)
    totals = {role: sum(r['mae'] * r['rows'] for r in audit_scores if r['role'] == role)
                   / len(audit) for role in ('control', 'candidate')}
    report = dict(selection=selections, audit_weighted_mae=totals,
                  relative_improvement=1 - totals['candidate'] / totals['control'],
                  leaderboard_verified=False,
                  note='Audit is held out from this search, but historical architecture choices may have used these years.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
