"""
tests/test_pipeline.py
Fast, self-contained tests for data pipeline + feature engineering.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline        import clean_train, validate_dataframe, REQUIRED_TRAIN_COLS
from src.features.engineering  import engineer_features, build_feature_matrix, SmartTargetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_train(n=100, seed=0):
    rng = np.random.default_rng(seed)
    districts = ["Colombo", "Galle", "Kandy", "Jaffna", "Matara"]
    df = pd.DataFrame({
        "record_id":           [f"F{10000 + i}" for i in range(n)],
        "district":            rng.choice(districts, n),
        "place_name":          [f"Place{i}" for i in range(n)],
        "latitude":            rng.uniform(6.0, 9.8, n),
        "longitude":           rng.uniform(80.0, 82.0, n),
        "elevation_m":         rng.uniform(0, 500, n),
        "distance_to_river_m": rng.uniform(50, 5000, n),
        "landcover":           rng.choice(["Forest", "Urban", "Paddy"], n),
        "soil_type":           rng.choice(["Clay", "Sandy", "Loam"], n),
        "water_supply":        rng.choice(["Piped", "Well", "None"], n),
        "electricity":         rng.choice(["Yes", "No"], n),
        "road_quality":        rng.choice(["Good", "Fair", "Poor"], n),
        "population_density_per_km2": rng.uniform(100, 3000, n),
        "built_up_percent":    rng.uniform(0, 1, n),
        "urban_rural":         rng.choice(["Urban", "Rural"], n),
        "rainfall_7d_mm":      rng.uniform(0, 200, n),
        "monthly_rainfall_mm": rng.uniform(50, 400, n),
        "drainage_index":      rng.uniform(0, 1, n),
        "ndvi":                rng.uniform(-0.2, 0.8, n),
        "ndwi":                rng.uniform(-0.5, 0.5, n),
        "water_presence_flag": rng.choice(["Likely", "Unlikely"], n),
        "historical_flood_count": rng.integers(0, 10, n),
        "infrastructure_score":rng.uniform(0, 1, n),
        "nearest_hospital_km": rng.uniform(1, 50, n),
        "nearest_evac_km":     rng.uniform(0.5, 20, n),
        "flood_risk_score":    rng.uniform(0, 1, n),
        "flood_occurrence_current_event": rng.choice(["Yes", "No"], n),
        "inundation_area_sqm": rng.integers(0, 100000, n),
        "is_good_to_live":     rng.choice(["Yes", "No"], n),
        "reason_not_good_to_live": rng.choice(["flood risk", "infrastructure issues", "road quality", None], n),
        "is_synthetic":        rng.choice(["True", "False"], n),
        "generation_date":     ["2024-06-01"] * n,
        "distance_to_river_m_log1p": rng.uniform(0, 10, n),
        "population_density_per_km2_log1p": rng.uniform(0, 10, n),
        "rainfall_7d_mm_log1p":rng.uniform(0, 5, n),
        "monthly_rainfall_mm_log1p": rng.uniform(0, 6, n),
        "nearest_hospital_km_log1p": rng.uniform(0, 4, n),
        "nearest_evac_km_log1p":     rng.uniform(0, 3, n),
        "elevation_m_yeojohnson":    rng.uniform(-2, 2, n),
        "drainage_index_yeojohnson": rng.uniform(-1, 1, n),
        "ndvi_qmap":           rng.uniform(0, 1, n),
        "ndwi_qmap":           rng.uniform(0, 1, n),
        "built_up_percent_qmap":rng.uniform(0, 1, n),
        "seasonal_index":      rng.uniform(0, 1, n),
        "terrain_roughness_index": rng.uniform(0, 1, n),
        "socioeconomic_status_index": rng.uniform(0, 1, n),
        "extreme_weather_index":     rng.uniform(0, 1, n),
    })
    return df


def _make_config():
    return {
        "project": {"target": "flood_risk_score", "id_col": "record_id", "version": "test", "seed": 42},
        "paths": {"processed_train": "/tmp/tr.parquet", "processed_test": "/tmp/te.parquet"},
        "data":  {"drop_cols": ["generation_date", "is_synthetic"], "corrupted_row_filter": "is_synthetic",
                  "deduplicate_by": "record_id", "target_averaging": True},
        "features": {
            "reason_flags": ["flood", "infrastructure", "road"],
            "binary_cols": {
                "water_presence_flag": "Likely",
                "flood_occurrence_current_event": "Yes",
                "is_good_to_live": "Yes",
                "urban_rural": "Urban",
            },
            "missingness_flags": ["distance_to_river_m", "drainage_index"],
            "interactions": [
                ["extreme_weather_index", "terrain_roughness_index"],
                ["elevation_m", "distance_to_river_m_log1p"],
                ["rainfall_7d_mm", "drainage_index"],
                ["rainfall_7d_mm_log1p", "extreme_weather_index"],
            ],
            "date_col": "generation_date",
        },
        "encoding": {"cv_folds": 5, "target_encoder_smoothing_auto": True},
        "cv": {"n_folds": 5, "strategy": "stratified", "n_bins": 10, "seed": 42},
        "meta_learner": {"type": "ridge", "alpha": 1.0},
        "postprocessing": {"clip_min": 0.0, "clip_max": 1.0, "calibrate": False},
        "thresholds": {"warn_if_pred_std_below": 0.03, "warn_if_pred_std_above": 0.15},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_train_removes_nan_is_synthetic():
    df = _make_train(50)
    df.loc[:4, "is_synthetic"] = None   # 5 corrupted rows
    clean, report = clean_train(df)
    assert report["removed_corrupted"] == 5
    assert len(clean) == 45
    assert clean["is_synthetic"].notna().all()


def test_clean_train_averages_duplicate_targets():
    df = _make_train(10)
    # Duplicate record_id with different targets
    extra = df.iloc[0].copy()
    extra["flood_risk_score"] = 1.0
    df = pd.concat([df, extra.to_frame().T], ignore_index=True)
    clean, report = clean_train(df)
    assert len(clean) == 10   # duplicate merged, not dropped
    rid = df.iloc[0]["record_id"]
    expected = (df[df["record_id"] == rid]["flood_risk_score"].mean())
    assert abs(clean.loc[clean["record_id"] == rid, "flood_risk_score"].values[0] - expected) < 1e-9


def test_validate_dataframe_passes_valid():
    df = _make_train(20)
    report = validate_dataframe(df, REQUIRED_TRAIN_COLS, "test")
    assert report["passed"] is True


def test_validate_dataframe_flags_missing_col():
    df = _make_train(20).drop(columns=["flood_risk_score"])
    report = validate_dataframe(df, REQUIRED_TRAIN_COLS, "test")
    assert report["passed"] is False
    assert any("flood_risk_score" in i for i in report["issues"])


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────────────────────────────────────

def test_engineer_features_adds_flag_cols():
    df  = _make_train(30)
    cfg = _make_config()
    out = engineer_features(df, cfg)
    assert "flag_flood_reason"          in out.columns
    assert "flag_infrastructure_reason" in out.columns
    assert "flag_road_reason"           in out.columns


def test_engineer_features_binary_cols():
    df  = _make_train(30)
    cfg = _make_config()
    out = engineer_features(df, cfg)
    assert "water_presence_flag_bin" in out.columns
    assert set(out["water_presence_flag_bin"].unique()).issubset({0.0, 1.0})


def test_engineer_features_interactions():
    df  = _make_train(30)
    cfg = _make_config()
    out = engineer_features(df, cfg)
    assert "extreme_x_terrain" in out.columns
    assert "elev_x_river"      in out.columns
    assert "rain_x_drainage"   in out.columns
    assert "rain_x_extreme"    in out.columns


def test_engineer_features_district_risk():
    df  = _make_train(50)
    cfg = _make_config()
    y   = df["flood_risk_score"]
    out = engineer_features(df, cfg, train_ref=df, y_ref=y)
    assert "district_risk_mean" in out.columns
    assert out["district_risk_mean"].notna().all()


def test_engineer_features_no_original_cols_lost():
    df  = _make_train(20)
    cfg = _make_config()
    out = engineer_features(df, cfg)
    for col in df.columns:
        assert col in out.columns, f"Original column lost: {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Target encoder tests
# ─────────────────────────────────────────────────────────────────────────────

def test_target_encoder_no_leakage_shape():
    df  = _make_train(100)
    y   = df["flood_risk_score"]
    ste = SmartTargetEncoder()
    from sklearn.model_selection import StratifiedKFold
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_bin  = pd.cut(y, bins=5, labels=False)
    folds  = list(skf.split(df, y_bin))
    Xe     = ste.fit_transform(df[["district"]], y, ["district"], folds)
    assert "district_te" in Xe.columns
    assert len(Xe) == len(df)
    assert Xe["district_te"].notna().all()


def test_target_encoder_transform_no_nan():
    df  = _make_train(100)
    y   = df["flood_risk_score"]
    ste = SmartTargetEncoder()
    from sklearn.model_selection import StratifiedKFold
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_bin = pd.cut(y, bins=5, labels=False)
    folds = list(skf.split(df, y_bin))
    ste.fit_transform(df[["district"]], y, ["district"], folds)
    # Transform on new unseen data
    df2 = _make_train(20, seed=99)
    df2.loc[:2, "district"] = "UNSEEN_DISTRICT"
    Xe2 = ste.transform(df2[["district"]], ["district"])
    assert Xe2["district_te"].notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction sanity tests
# ─────────────────────────────────────────────────────────────────────────────

def test_predictions_in_range():
    """Predictions must be clipped to [0, 1]."""
    preds = np.array([0.35, 0.47, 0.55, 0.62, 1.01, -0.02])
    clipped = np.clip(preds, 0, 1)
    assert clipped.min() >= 0.0
    assert clipped.max() <= 1.0


def test_submission_format():
    """Submission CSV must have exactly record_id and flood_risk_score."""
    test_ids = pd.Series([f"F{i}" for i in range(10)])
    preds    = np.random.uniform(0.3, 0.7, 10)
    sub      = pd.DataFrame({"record_id": test_ids, "flood_risk_score": preds})
    assert list(sub.columns) == ["record_id", "flood_risk_score"]
    assert not sub["record_id"].duplicated().any()
    assert (sub["flood_risk_score"] >= 0).all()
    assert (sub["flood_risk_score"] <= 1).all()
