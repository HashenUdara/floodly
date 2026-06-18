# FloodLens Demo Flow

## First Milestone Demo

1. Show the repository structure.
2. Show `ml/configs/config.yaml` pointing to repo-level data.
3. Run the ML pipeline after placing `train.csv` and `test.csv` in `data/raw/`.
4. Show `artifacts/flood-risk-v3/model_bundle.joblib` and `metadata.json`.
5. Run `./scripts/run_local_prediction.sh`.
6. Show the JSON output with `record_id`, `flood_risk_score`, `risk_level`, and
   `model_version`.

## Later Demo Additions

- FastAPI prediction endpoint
- Dashboard risk ranking
- Prediction logging
- MLOps monitoring page
- Grounded flood copilot
