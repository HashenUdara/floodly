#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../ml"
python -m src.inference.predictor \
  --artifact ../artifacts/flood-risk-v3/model_bundle.joblib \
  --input ../data/raw/test.csv \
  --limit 1
