"""
src/evaluation/tracker.py
Lightweight experiment tracker: logs every run to a JSON registry,
implements champion / challenger promotion, and drift alerts.
No external dependencies — works offline on Kaggle.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry/experiments.json"
CHAMPION_FILE = "registry/champion.json"


# ─────────────────────────────────────────────────────────────────────────────
# Registry I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_registry(path: str = REGISTRY_FILE) -> list:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_registry(runs: list, path: str = REGISTRY_FILE) -> None:
    with open(path, "w") as f:
        json.dump(runs, f, indent=2)


def _load_champion(path: str = CHAMPION_FILE) -> Optional[dict]:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_champion(run: dict, path: str = CHAMPION_FILE) -> None:
    with open(path, "w") as f:
        json.dump(run, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Run logging
# ─────────────────────────────────────────────────────────────────────────────

def log_run(
    metrics: dict,
    config: dict,
    submission_path: str = None,
    notes: str = "",
    registry_path: str = REGISTRY_FILE,
) -> dict:
    """
    Record a training run. Returns the run record.
    Fields: run_id, timestamp, metrics, config_snapshot, submission_path, notes.
    """
    runs    = _load_registry(registry_path)
    run_id  = f"run_{len(runs) + 1:04d}"
    ts      = datetime.utcnow().isoformat()

    # Slim config snapshot (just key hparams, not full config)
    config_snap = {
        "version":      config["project"]["version"],
        "n_folds":      config["cv"]["n_folds"],
        "seed":         config["cv"]["seed"],
        "lgb_leaves":   config["models"]["lgb"].get("num_leaves"),
        "lgb_lr":       config["models"]["lgb"].get("learning_rate"),
        "xgb_depth":    config["models"]["xgb"].get("max_depth"),
        "cat_depth":    config["models"]["catboost"].get("depth"),
        "meta_alpha":   config["meta_learner"]["alpha"],
    }

    run = {
        "run_id":          run_id,
        "timestamp":       ts,
        "metrics":         metrics,
        "config_snapshot": config_snap,
        "submission_path": submission_path,
        "notes":           notes,
        "promoted":        False,
    }

    runs.append(run)
    _save_registry(runs, registry_path)
    logger.info(f"Run logged: {run_id}  OOF_MAE={metrics.get('oof_mae')}")
    return run


# ─────────────────────────────────────────────────────────────────────────────
# Champion / challenger gate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_champion(
    run: dict,
    config: dict,
    registry_path: str = REGISTRY_FILE,
    champion_path: str = CHAMPION_FILE,
) -> bool:
    """
    Promote run to champion if it beats the current champion on OOF MAE.
    Also enforces prediction-std thresholds from config.
    Returns True if promoted.
    """
    thresholds = config.get("thresholds", {})
    metrics    = run["metrics"]
    oof_mae    = metrics.get("oof_mae", 9999)
    test_std   = metrics.get("test_std", 0)

    # Std guard — alert if predictions collapse or explode
    std_low  = thresholds.get("warn_if_pred_std_below", 0.03)
    std_high = thresholds.get("warn_if_pred_std_above", 0.15)
    if test_std < std_low:
        logger.warning(f"⚠  Prediction std={test_std:.4f} is below {std_low} — predictions may have collapsed!")
    if test_std > std_high:
        logger.warning(f"⚠  Prediction std={test_std:.4f} is above {std_high} — check for outliers.")

    champion = _load_champion(champion_path)

    if champion is None:
        # First run — auto-promote
        run["promoted"] = True
        _save_champion(run, champion_path)
        _mark_promoted(run["run_id"], registry_path)
        logger.info(f"✓ First run — auto-promoted as champion: {run['run_id']}")
        return True

    champ_mae = champion["metrics"].get("oof_mae", 9999)
    gate      = thresholds.get("promote_if_cv_mae_below", champ_mae)  # default: beat current

    if oof_mae < champ_mae:
        run["promoted"] = True
        _save_champion(run, champion_path)
        _mark_promoted(run["run_id"], registry_path)
        logger.info(
            f"✓ New champion: {run['run_id']}  "
            f"OOF_MAE {oof_mae:.6f} < {champ_mae:.6f} (prev champion)"
        )
        return True
    else:
        logger.info(
            f"✗ Not promoted: {run['run_id']}  "
            f"OOF_MAE {oof_mae:.6f} ≥ {champ_mae:.6f} (current champion)"
        )
        return False


def _mark_promoted(run_id: str, registry_path: str) -> None:
    runs = _load_registry(registry_path)
    for r in runs:
        if r["run_id"] == run_id:
            r["promoted"] = True
    _save_registry(runs, registry_path)


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard & run summary
# ─────────────────────────────────────────────────────────────────────────────

def print_leaderboard(registry_path: str = REGISTRY_FILE, top_n: int = 10) -> None:
    """Print a ranked summary of all runs."""
    runs = _load_registry(registry_path)
    if not runs:
        logger.info("No runs logged yet.")
        return

    rows = []
    for r in runs:
        rows.append({
            "run_id":    r["run_id"],
            "timestamp": r["timestamp"][:16],
            "oof_mae":   r["metrics"].get("oof_mae"),
            "oof_rmse":  r["metrics"].get("oof_rmse"),
            "test_std":  r["metrics"].get("test_std"),
            "champion":  "★" if r.get("promoted") else "",
            "notes":     r.get("notes", "")[:40],
        })

    df = pd.DataFrame(rows).sort_values("oof_mae").head(top_n)
    print("\n" + "═" * 80)
    print(f"  EXPERIMENT LEADERBOARD (top {top_n})")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80 + "\n")


def get_champion() -> Optional[dict]:
    return _load_champion(CHAMPION_FILE)
