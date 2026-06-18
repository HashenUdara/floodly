# FloodLens Architecture

## Step 1: ML Foundation

The current implementation proves local ML inference from a saved artifact.

```text
data/raw/train.csv + data/raw/test.csv
  -> ml/pipelines/train_pipeline.py
  -> ml/src/data + ml/src/features + ml/src/models
  -> artifacts/flood-risk-v3/model_bundle.joblib
  -> ml/src/inference/predictor.py
```

## Planned Backend

FastAPI will load `ml/src/inference/predictor.py` and expose:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /batch-predict`

## Planned Frontend

The frontend will consume only backend APIs. It will not load model files or ML
code directly.
