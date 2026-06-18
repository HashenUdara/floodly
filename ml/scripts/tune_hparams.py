"""
scripts/tune_hparams.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optuna-based HPO for the flood risk ensemble.
Searches LGB / XGB / CatBoost params + meta Ridge alpha.
Writes the best config back to configs/config_tuned.yaml.

Usage:
  python scripts/tune_hparams.py --n-trials 50
  python scripts/tune_hparams.py --n-trials 100 --timeout 3600
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline        import load_config
from src.features.engineering  import build_feature_matrix
from src.models.ensemble       import train_ensemble

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("hpo")


def load_features(config: dict):
    """Load and cache feature matrices once for all trials."""
    train = pd.read_parquet(config["paths"]["processed_train"])
    test  = pd.read_parquet(config["paths"]["processed_test"])
    y     = train[config["project"]["target"]].copy()
    X_tr, X_te, _, _ = build_feature_matrix(train, test, y, config)
    return X_tr, X_te, y


def objective(trial, X_train, y_train, X_test, base_config: dict) -> float:
    """Optuna objective — minimise OOF MAE."""
    config = copy.deepcopy(base_config)

    # LGB search space
    config["models"]["lgb"]["num_leaves"]        = trial.suggest_int("lgb_num_leaves", 127, 1023, log=True)
    config["models"]["lgb"]["learning_rate"]      = trial.suggest_float("lgb_lr", 0.01, 0.05, log=True)
    config["models"]["lgb"]["min_child_samples"]  = trial.suggest_int("lgb_min_child", 10, 50)
    config["models"]["lgb"]["subsample"]          = trial.suggest_float("lgb_subsample", 0.6, 1.0)
    config["models"]["lgb"]["colsample_bytree"]   = trial.suggest_float("lgb_colsample", 0.5, 1.0)
    config["models"]["lgb"]["reg_alpha"]          = trial.suggest_float("lgb_reg_alpha", 0.01, 0.5, log=True)
    config["models"]["lgb"]["reg_lambda"]         = trial.suggest_float("lgb_reg_lambda", 0.1, 2.0, log=True)

    # XGB search space
    config["models"]["xgb"]["max_depth"]          = trial.suggest_int("xgb_depth", 5, 10)
    config["models"]["xgb"]["learning_rate"]      = trial.suggest_float("xgb_lr", 0.01, 0.05, log=True)
    config["models"]["xgb"]["subsample"]          = trial.suggest_float("xgb_subsample", 0.6, 1.0)
    config["models"]["xgb"]["colsample_bytree"]   = trial.suggest_float("xgb_colsample", 0.5, 1.0)
    config["models"]["xgb"]["reg_alpha"]          = trial.suggest_float("xgb_reg_alpha", 0.01, 0.5, log=True)
    config["models"]["xgb"]["reg_lambda"]         = trial.suggest_float("xgb_reg_lambda", 0.1, 2.0, log=True)

    # CatBoost search space
    config["models"]["catboost"]["depth"]         = trial.suggest_int("cat_depth", 6, 10)
    config["models"]["catboost"]["learning_rate"] = trial.suggest_float("cat_lr", 0.01, 0.05, log=True)
    config["models"]["catboost"]["l2_leaf_reg"]   = trial.suggest_float("cat_l2", 1.0, 10.0, log=True)

    # Meta learner
    config["meta_learner"]["alpha"]               = trial.suggest_float("meta_alpha", 0.01, 10.0, log=True)

    # Use fewer folds in HPO for speed (5 instead of 10)
    config["cv"]["n_folds"] = 5

    _, _, metrics = train_ensemble(X_train, y_train, X_test, config)
    oof_mae = metrics["oof_mae"]

    logger.info(f"  Trial {trial.number:>3d}  OOF MAE={oof_mae:.6f}")
    return oof_mae


def run(args):
    config   = load_config(args.config)
    logger.info(f"Loading features (this runs once)...")
    X_tr, X_te, y = load_features(config)
    logger.info(f"Features ready: {X_tr.shape[1]} cols, {len(X_tr):,} rows")

    study = optuna.create_study(
        direction="minimize",
        study_name="flood_risk_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: objective(trial, X_tr, y, X_te, config),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"\nBest trial: #{best.number}  OOF MAE={best.value:.6f}")
    logger.info(f"Best params: {best.params}")

    # Write best config
    tuned_config = copy.deepcopy(config)
    p = best.params
    tuned_config["models"]["lgb"].update({
        "num_leaves": p["lgb_num_leaves"], "learning_rate": p["lgb_lr"],
        "min_child_samples": p["lgb_min_child"], "subsample": p["lgb_subsample"],
        "colsample_bytree": p["lgb_colsample"], "reg_alpha": p["lgb_reg_alpha"],
        "reg_lambda": p["lgb_reg_lambda"],
    })
    tuned_config["models"]["xgb"].update({
        "max_depth": p["xgb_depth"], "learning_rate": p["xgb_lr"],
        "subsample": p["xgb_subsample"], "colsample_bytree": p["xgb_colsample"],
        "reg_alpha": p["xgb_reg_alpha"], "reg_lambda": p["xgb_reg_lambda"],
    })
    tuned_config["models"]["catboost"].update({
        "depth": p["cat_depth"], "learning_rate": p["cat_lr"],
        "l2_leaf_reg": p["cat_l2"],
    })
    tuned_config["meta_learner"]["alpha"] = p["meta_alpha"]
    tuned_config["cv"]["n_folds"] = config["cv"]["n_folds"]  # restore full folds

    out_path = "configs/config_tuned.yaml"
    with open(out_path, "w") as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Tuned config saved: {out_path}")
    logger.info(f"\nRun the tuned config with:")
    logger.info(f"  python run_pipeline.py --config {out_path} --notes 'optuna tuned trial {best.number}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--n-trials",  type=int, default=50)
    parser.add_argument("--timeout",   type=int, default=None, help="Max seconds")
    run(parser.parse_args())
