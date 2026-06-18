"""
src/models/ensemble.py
3-GBM ensemble with Ridge meta-learner — extracted and modularised from v3.
Supports per-fold logging, early stopping, and champion/challenger gating.
"""

import logging
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_lgb(params: dict) -> lgb.LGBMRegressor:
    p = {k: v for k, v in params.items() if k not in {"enabled", "early_stopping_rounds"}}
    return lgb.LGBMRegressor(**p)


def _build_xgb(params: dict) -> xgb.XGBRegressor:
    p = {k: v for k, v in params.items() if k not in {"enabled", "early_stopping_rounds"}}
    return xgb.XGBRegressor(**p)


def _build_cat(params: dict) -> CatBoostRegressor:
    p = {k: v for k, v in params.items() if k != "enabled"}
    return CatBoostRegressor(**p)


# ─────────────────────────────────────────────────────────────────────────────
# Core ensemble training
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Train 3-GBM ensemble with Ridge meta-learner using stratified k-fold CV.

    Returns:
        test_preds  : clipped [0,1] predictions for test set
        oof_preds   : out-of-fold predictions for train set
        metrics     : dict with oof_mae, oof_rmse, meta_weights, fold_maes, pred_stats
        artifacts   : fitted fold models and Ridge meta-model for inference export
    """
    cv_cfg   = config["cv"]
    m_cfg    = config["models"]
    meta_cfg = config["meta_learner"]
    post_cfg = config["postprocessing"]

    n_folds = cv_cfg["n_folds"]
    seed    = cv_cfg["seed"]
    N       = len(y_train)

    skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_binned = pd.cut(y_train, bins=cv_cfg["n_bins"], labels=False)

    oof_lgb  = np.zeros(N);  test_lgb  = np.zeros(len(X_test))
    oof_xgb  = np.zeros(N);  test_xgb  = np.zeros(len(X_test))
    oof_cat  = np.zeros(N);  test_cat  = np.zeros(len(X_test))
    fold_maes: Dict[str, list] = {"lgb": [], "xgb": [], "cat": []}
    fitted_models: Dict[str, list] = {"lgb": [], "xgb": [], "cat": []}

    t_start = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_binned)):
        fold_start = time.time()
        logger.info(f"  ── Fold {fold + 1}/{n_folds} ──")

        Xf_tr, Xf_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        yf_tr, yf_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # LightGBM
        if m_cfg["lgb"]["enabled"]:
            m1 = _build_lgb(m_cfg["lgb"])
            m1.fit(
                Xf_tr, yf_tr,
                eval_set=[(Xf_val, yf_val)],
                callbacks=[
                    lgb.early_stopping(m_cfg["lgb"].get("early_stopping_rounds", 100), verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            oof_lgb[val_idx] = m1.predict(Xf_val)
            test_lgb        += m1.predict(X_test) / n_folds
            fold_maes["lgb"].append(mean_absolute_error(yf_val, oof_lgb[val_idx]))
            fitted_models["lgb"].append(m1)

        # XGBoost
        if m_cfg["xgb"]["enabled"]:
            m2 = _build_xgb(m_cfg["xgb"])
            m2.fit(
                Xf_tr, yf_tr,
                eval_set=[(Xf_val, yf_val)],
                verbose=False,
            )
            oof_xgb[val_idx] = m2.predict(Xf_val)
            test_xgb        += m2.predict(X_test) / n_folds
            fold_maes["xgb"].append(mean_absolute_error(yf_val, oof_xgb[val_idx]))
            fitted_models["xgb"].append(m2)

        # CatBoost
        if m_cfg["catboost"]["enabled"]:
            m3 = _build_cat(m_cfg["catboost"])
            m3.fit(Xf_tr, yf_tr, eval_set=(Xf_val, yf_val))
            oof_cat[val_idx] = m3.predict(Xf_val)
            test_cat        += m3.predict(X_test) / n_folds
            fold_maes["cat"].append(mean_absolute_error(yf_val, oof_cat[val_idx]))
            fitted_models["cat"].append(m3)

        elapsed = time.time() - fold_start
        logger.info(
            f"    LGB MAE={fold_maes['lgb'][-1]:.5f}  "
            f"XGB MAE={fold_maes['xgb'][-1]:.5f}  "
            f"CAT MAE={fold_maes['cat'][-1]:.5f}  "
            f"({elapsed:.1f}s)"
        )

    # ── Ridge meta-learner ───────────────────────────────────────────────────
    oof_stack  = np.column_stack([oof_lgb,  oof_xgb,  oof_cat])
    test_stack = np.column_stack([test_lgb, test_xgb, test_cat])

    meta = Ridge(alpha=meta_cfg["alpha"])
    meta.fit(oof_stack, y_train)

    oof_preds   = meta.predict(oof_stack)
    test_preds  = meta.predict(test_stack)
    test_preds  = np.clip(test_preds, post_cfg["clip_min"], post_cfg["clip_max"])

    # ── Metrics ──────────────────────────────────────────────────────────────
    oof_mae  = mean_absolute_error(y_train, oof_preds)
    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))
    total_time = time.time() - t_start

    metrics = {
        "oof_mae":      round(oof_mae,  6),
        "oof_rmse":     round(oof_rmse, 6),
        "oof_std":      round(float(oof_preds.std()),  4),
        "test_std":     round(float(test_preds.std()), 4),
        "test_min":     round(float(test_preds.min()), 4),
        "test_max":     round(float(test_preds.max()), 4),
        "meta_weights": {
            "lgb": round(float(meta.coef_[0]), 4),
            "xgb": round(float(meta.coef_[1]), 4),
            "cat": round(float(meta.coef_[2]), 4),
        },
        "fold_mae_lgb": [round(x, 5) for x in fold_maes["lgb"]],
        "fold_mae_xgb": [round(x, 5) for x in fold_maes["xgb"]],
        "fold_mae_cat": [round(x, 5) for x in fold_maes["cat"]],
        "training_time_s": round(total_time, 1),
        "n_folds": n_folds,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
    }

    logger.info("─" * 60)
    logger.info(
        f"Meta weights — LGB:{meta.coef_[0]:.3f}  "
        f"XGB:{meta.coef_[1]:.3f}  CAT:{meta.coef_[2]:.3f}"
    )
    logger.info(f"OOF  RMSE={oof_rmse:.6f}  MAE={oof_mae:.6f}  std={oof_preds.std():.4f}")
    logger.info(f"Test std={test_preds.std():.4f}  min={test_preds.min():.4f}  max={test_preds.max():.4f}")
    logger.info(f"Total training time: {total_time:.1f}s")

    artifacts = {
        "base_models": fitted_models,
        "meta_model": meta,
        "model_order": ["lgb", "xgb", "cat"],
    }

    return test_preds, oof_preds, metrics, artifacts
