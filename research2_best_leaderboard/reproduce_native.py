"""Reproduce the original TensorFlow entry point without a backend substitution.

This is a control reproduction, not a new improved candidate. Match the profile,
seed and TensorFlow/Keras versions of the successful submission before comparing.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['reference', 'improved', 'tuned'], default='improved')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research2' / 'native_control')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    stem = f'native_{args.profile}_seed{args.seed}'
    if not args.prepare_only:
        if importlib.util.find_spec('tensorflow') is None:
            parser.exit(2, 'TensorFlow is missing from this Python environment. No model was trained.\n'
                        'Use the environment/version of the successful feature_lag_mlp run; '
                        'this script will not fall back to another backend.\n')
        if list(out.glob(f'{stem}*')) or (out / f'submission_{stem}.csv').exists():
            # A prepare-only metadata file is safe, but completed/partial trained runs are not overwritten.
            existing = [p for p in out.glob(f'{stem}*') if not p.name.endswith('_prepared.json')]
            if existing or (out / f'submission_{stem}.csv').exists():
                parser.error('Output already contains this run; choose a fresh --output-dir.')
    command = [sys.executable, str(ROOT / 'src' / 'train_feature_lag_mlp.py'),
               '--profile', args.profile, '--seed', str(args.seed),
               '--normalization', 'shared-sj', '--epochs', '40',
               '--steps-per-epoch', '200', '--batch-size', '16', '--lr-mode', 'max',
               '--experiment-name', stem, '--output-dir', str(out), '--verbose', '0']
    if args.prepare_only:
        command.append('--prepare-only')
    subprocess.run(command, check=True)
    if not args.prepare_only:
        import tensorflow as tf
        manifest = dict(python=sys.version, tensorflow=tf.__version__, keras=tf.keras.__version__,
            command=command, purpose='Native control reproduction; not claimed to beat the leaderboard baseline',
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / 'src' / 'feature_lag_mlp').glob('*.py'))})
        (out / f'{stem}_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
