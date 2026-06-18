# FloodLens Model Card

## Model

`flood-risk-v3` uses a LightGBM, XGBoost, and CatBoost ensemble with a Ridge
meta-model. The pipeline is adapted from the provided `flood_model_v3.py` and
the `Flood_Risk_MLOps_Guide` project.

## Intended Use

The model supports flood risk intelligence and decision support. It is not an
official emergency alert system.

## Current Artifact

The deployable artifact is:

```text
artifacts/flood-risk-v3/model_bundle.joblib
```

It contains fitted base models, the Ridge meta-model, target encoder, medians,
feature column order, config snapshot, district-risk statistics, and metadata.
