"""
src/features/engineering.py
Feature engineering and target encoding — extracted from flood_model_v3.py
and made config-driven + reusable.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def build_district_risk_stats(train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """Return train-only district risk statistics used during inference."""
    dist_map = (
        pd.DataFrame({"d": train["district"], "y": y_train.values})
        .groupby("d")["y"]
        .mean()
    )
    return {
        "district_risk_mean": {str(k): float(v) for k, v in dist_map.items()},
        "global_risk_mean": float(y_train.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SmartTargetEncoder (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

class SmartTargetEncoder:
    """Smoothed target encoder with adaptive smoothing by cardinality."""

    def __init__(self):
        self.global_mean  = None
        self.encoding_map = {}

    @staticmethod
    def _smoothing(series: pd.Series) -> int:
        n = series.nunique()
        if   n > 500: return 200
        elif n > 100: return 50
        elif n > 20:  return 20
        elif n > 5:   return 10
        else:         return 5

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cols: List[str],
        cv_folds: list,
    ) -> pd.DataFrame:
        self.global_mean = y.mean()
        X = X.copy()
        Xe = X.copy()
        for col in cols:
            s = self._smoothing(X[col])
            Xe[f"{col}_te"] = np.nan
            for tr_idx, val_idx in cv_folds:
                stats = (
                    pd.DataFrame({"c": X[col].iloc[tr_idx], "t": y.iloc[tr_idx]})
                    .groupby("c")["t"]
                    .agg(["count", "mean"])
                )
                smoothed = (
                    (stats["count"] * stats["mean"] + s * self.global_mean)
                    / (stats["count"] + s)
                )
                Xe.loc[X.index[val_idx], f"{col}_te"] = (
                    X[col].iloc[val_idx].map(smoothed).fillna(self.global_mean)
                )
            full = (
                pd.DataFrame({"c": X[col], "t": y})
                .groupby("c")["t"]
                .agg(["count", "mean"])
            )
            self.encoding_map[col] = (
                (full["count"] * full["mean"] + s * self.global_mean)
                / (full["count"] + s)
            )
        return Xe

    def transform(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        X = X.copy()
        for col in cols:
            X[f"{col}_te"] = (
                X[col].map(self.encoding_map[col]).fillna(self.global_mean)
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (config-driven)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    config: dict,
    train_ref: pd.DataFrame = None,
    y_ref: pd.Series = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    train_ref / y_ref are used for leakage-safe district risk encoding.
    """
    df = df.copy()
    feat_cfg = config["features"]

    # ── Reason-text multi-hot flags ──────────────────────────────────────────
    reason = df["reason_not_good_to_live"].fillna("").str.lower()
    for keyword in feat_cfg.get("reason_flags", []):
        df[f"flag_{keyword}_reason"] = reason.str.contains(keyword).astype(np.float32)

    # ── Binary categorical encodings ────────────────────────────────────────
    for col, positive_val in feat_cfg.get("binary_cols", {}).items():
        safe_col = col.replace(" ", "_").replace("(", "").replace(")", "")
        df[f"{safe_col}_bin"] = (df[col] == positive_val).astype(np.float32)

    # ── District-level mean risk (train stats only, no leakage) ─────────────
    if train_ref is not None and y_ref is not None:
        dist_map = (
            pd.DataFrame({"d": train_ref["district"], "y": y_ref.values})
            .groupby("d")["y"]
            .mean()
        )
        df["district_risk_mean"] = df["district"].map(dist_map).fillna(float(y_ref.mean()))

    # ── Generation-date seasonality ─────────────────────────────────────────
    date_col = feat_cfg.get("date_col", "generation_date")
    if date_col in df.columns:
        try:
            gen_dt = pd.to_datetime(df[date_col], errors="coerce")
            df["gen_month"]     = gen_dt.dt.month.fillna(6).astype(np.float32)
            df["gen_month_sin"] = np.sin(2 * np.pi * df["gen_month"] / 12).astype(np.float32)
            df["gen_month_cos"] = np.cos(2 * np.pi * df["gen_month"] / 12).astype(np.float32)
        except Exception as e:
            logger.warning(f"Date parsing failed: {e}")

    # ── Missingness flags ────────────────────────────────────────────────────
    for col in feat_cfg.get("missingness_flags", []):
        if col in df.columns:
            df[f"miss_{col}"] = df[col].isnull().astype(np.float32)

    # ── Interaction features ─────────────────────────────────────────────────
    interactions = feat_cfg.get("interactions", [])
    if len(interactions) >= 1:
        a, b = interactions[0]
        df["extreme_x_terrain"] = (
            df[a].fillna(0) * df[b].fillna(0)
        ).astype(np.float32)
    if len(interactions) >= 2:
        a, b = interactions[1]
        df["elev_x_river"] = (
            df[a].fillna(df[a].median()) * df[b].fillna(0)
        ).astype(np.float32)
    if len(interactions) >= 3:
        a, b = interactions[2]
        df["rain_x_drainage"] = (
            df[a].fillna(0) * (1 - df[b].fillna(0.5))
        ).astype(np.float32)
    if len(interactions) >= 4:
        a, b = interactions[3]
        df["rain_x_extreme"] = (
            df[a].fillna(0) * df[b].fillna(0)
        ).astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full feature pipeline (engineer + encode + align + fillna)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, SmartTargetEncoder, pd.Series]:
    """
    Full feature pipeline: engineer → encode → align → fillna.
    Returns (X_train, X_test, encoder, medians).
    """
    proj    = config["project"]
    cv_cfg  = config["cv"]

    TARGET = proj["target"]
    ID     = proj["id_col"]
    DROP   = config["data"]["drop_cols"]

    logger.info("Engineering features...")
    train_eng = engineer_features(train, config, train_ref=train, y_ref=y_train)
    test_eng  = engineer_features(test,  config, train_ref=train, y_ref=y_train)

    logger.info("Preparing feature matrices...")
    drop_train = [TARGET, ID] + [c for c in DROP if c in train_eng.columns]
    drop_test  = [ID]         + [c for c in DROP if c in test_eng.columns]
    X_train_raw = train_eng.drop(columns=drop_train)
    X_test_raw  = test_eng.drop(columns=drop_test)

    # Align columns
    common = [c for c in X_train_raw.columns if c in X_test_raw.columns]
    X_train_raw = X_train_raw[common].copy()
    X_test_raw  = X_test_raw[common].copy()

    # Target encoding
    cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Target encoding {len(cat_cols)} categorical columns...")
    skf    = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=cv_cfg["seed"])
    y_bin  = pd.cut(y_train, bins=cv_cfg["n_bins"], labels=False)
    folds  = list(skf.split(X_train_raw, y_bin))

    ste     = SmartTargetEncoder()
    X_train = ste.fit_transform(X_train_raw, y_train, cat_cols, folds)
    X_test  = ste.transform(X_test_raw, cat_cols)

    X_train = X_train.drop(columns=cat_cols)
    X_test  = X_test.drop(columns=cat_cols)

    # NaN fill with train medians
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    ste.raw_feature_columns_ = common
    ste.categorical_columns_ = cat_cols
    ste.feature_columns_ = list(X_train.columns)
    ste.district_risk_stats_ = build_district_risk_stats(train, y_train)

    logger.info(f"Feature matrix ready: {X_train.shape[1]} features, {len(X_train):,} train rows")
    return X_train, X_test, ste, medians
