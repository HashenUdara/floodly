"""
Reusable inference entrypoint for FloodLens.

The FastAPI backend should import FloodRiskPredictor instead of duplicating
feature engineering or model loading logic.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import engineer_features


def risk_level(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"


class FloodRiskPredictor:
    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        self.bundle = joblib.load(self.artifact_path)
        self.config = self.bundle["config"]
        self.model_version = self.bundle["metadata"]["model_version"]

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config["project"]["id_col"]
        ids = df[id_col].copy() if id_col in df.columns else pd.Series([None] * len(df))
        features = self._prepare_features(df)
        scores = self._predict_scores(features)
        return pd.DataFrame(
            {
                "record_id": ids.values,
                "flood_risk_score": np.round(scores, 6),
                "risk_level": [risk_level(float(score)) for score in scores],
                "model_version": self.model_version,
            }
        )

    def predict_records(self, records: Iterable[dict]) -> List[dict]:
        frame = pd.DataFrame(list(records))
        return self.predict_frame(frame).to_dict(orient="records")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config["project"]["id_col"]
        target_col = self.config["project"]["target"]
        drop_cols = self.config["data"]["drop_cols"]
        encoder = self.bundle["encoder"]
        raw_feature_columns = self.bundle["raw_feature_columns"]
        categorical_columns = self.bundle["categorical_columns"]
        feature_columns = self.bundle["feature_columns"]
        medians = self.bundle["medians"]

        engineered = engineer_features(df.copy(), self.config)
        district_stats = self.bundle.get("district_risk_stats", {})
        if "district_risk_mean" in raw_feature_columns:
            district_map = district_stats.get("district_risk_mean", {})
            global_mean = district_stats.get("global_risk_mean", 0.5)
            engineered["district_risk_mean"] = (
                engineered["district"].astype(str).map(district_map).fillna(global_mean)
            )

        drop = [target_col, id_col] + [col for col in drop_cols if col in engineered.columns]
        X_raw = engineered.drop(columns=[col for col in drop if col in engineered.columns])
        X_raw = X_raw.reindex(columns=raw_feature_columns)

        X = encoder.transform(X_raw, categorical_columns)
        X = X.drop(columns=[col for col in categorical_columns if col in X.columns])
        X = X.reindex(columns=feature_columns)
        return X.fillna(medians).fillna(0)

    def _predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        model_order = self.bundle["model_order"]
        base_models = self.bundle["base_models"]
        base_predictions = []
        for model_name in model_order:
            models = base_models.get(model_name, [])
            if not models:
                raise ValueError(f"No fitted models found for {model_name}")
            fold_preds = np.column_stack([model.predict(features) for model in models])
            base_predictions.append(fold_preds.mean(axis=1))

        stacked = np.column_stack(base_predictions)
        scores = self.bundle["meta_model"].predict(stacked)
        post = self.config["postprocessing"]
        return np.clip(scores, post["clip_min"], post["clip_max"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local FloodLens inference.")
    parser.add_argument(
        "--artifact",
        default="../artifacts/flood-risk-v3/model_bundle.joblib",
        help="Path to a saved model bundle.",
    )
    parser.add_argument(
        "--input",
        default="../data/raw/test.csv",
        help="CSV file containing rows to score.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of rows to score.")
    args = parser.parse_args()

    predictor = FloodRiskPredictor(args.artifact)
    rows = pd.read_csv(args.input).head(args.limit)
    predictions = predictor.predict_frame(rows)
    print(json.dumps(predictions.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
