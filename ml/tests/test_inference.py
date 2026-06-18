import numpy as np

from pipelines.export_model import export_model_bundle
from src.features.engineering import build_feature_matrix
from src.inference.predictor import FloodRiskPredictor, risk_level
from tests.test_pipeline import _make_config, _make_train


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class MeanMetaModel:
    def predict(self, X):
        return np.mean(X, axis=1)


def test_risk_level_thresholds():
    assert risk_level(0.1) == "Low"
    assert risk_level(0.33) == "Medium"
    assert risk_level(0.66) == "High"


def test_exported_bundle_predicts_single_row(tmp_path):
    config = _make_config()
    train = _make_train(80)
    test = _make_train(5, seed=10).drop(columns=["flood_risk_score"])
    y_train = train["flood_risk_score"]

    X_train, _, encoder, medians = build_feature_matrix(train, test, y_train, config)
    metrics = {
        "oof_mae": 0.1,
        "oof_rmse": 0.2,
        "test_std": 0.01,
        "n_features": X_train.shape[1],
    }
    model_artifacts = {
        "base_models": {
            "lgb": [ConstantModel(0.2)],
            "xgb": [ConstantModel(0.5)],
            "cat": [ConstantModel(0.8)],
        },
        "meta_model": MeanMetaModel(),
        "model_order": ["lgb", "xgb", "cat"],
    }

    export_result = export_model_bundle(
        model_artifacts=model_artifacts,
        encoder=encoder,
        medians=medians,
        config=config,
        metrics=metrics,
        output_dir=tmp_path,
    )

    predictor = FloodRiskPredictor(export_result["bundle_path"])
    prediction = predictor.predict_frame(test.head(1)).iloc[0]

    assert prediction["record_id"] == test.iloc[0]["record_id"]
    assert 0 <= prediction["flood_risk_score"] <= 1
    assert prediction["risk_level"] == "Medium"
    assert prediction["model_version"] == "flood-risk-test"
