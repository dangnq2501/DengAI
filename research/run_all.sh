#!/usr/bin/env bash
# End-to-end research pipeline using the conda env ``bkk``.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TF_CPP_MIN_LOG_LEVEL=2

BKK_PYTHON="$(conda run -n bkk which python)"
PYTHON=("${BKK_PYTHON}")
BACKEND="${BACKEND:-numpy}"

echo "==> blend search (temporal CV for MLP weight + rounding, backend=${BACKEND})"
"${PYTHON[@]}" research/blend_search.py \
  --output-dir research/artifacts \
  --folds 2 \
  --seeds 42 \
  --mlp-weights 0.5,0.6,0.7,0.8,0.9,1.0 \
  --rounding nearest,floor \
  --backend "${BACKEND}" \
  --verbose 0

echo "==> tuned MLP submission (5-seed ensemble + CV blend, backend=${BACKEND})"
"${PYTHON[@]}" research/train_feature_lag_submission.py \
  --output-dir research/artifacts \
  --blend-config research/artifacts/blend_search.json \
  --backend "${BACKEND}" \
  --verbose 0

echo "==> optional city hybrid (MLP SJ + features_engineering IQ)"
if [[ -f research/artifacts/submission_feature_lag_mlp_research.csv ]]; then
  "${PYTHON[@]}" research/compose_hybrid_submission.py \
    --sj-submission research/artifacts/submission_feature_lag_mlp_research.csv \
    --iq-submission features_engineering/artifacts/submission.csv \
    --output research/artifacts/submission_hybrid_mlp_sj_lgb_iq.csv
fi

echo "Done. Primary: research/artifacts/submission_feature_lag_mlp_research.csv"
