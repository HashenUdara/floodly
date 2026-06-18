"""
src/submission/generator.py
Generates competition-ready CSV submissions with auto-versioning,
prediction validation, and a submission history log.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUBMISSION_LOG = "registry/submission_history.json"


# ─────────────────────────────────────────────────────────────────────────────
# Submission generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_submission(
    test_ids: pd.Series,
    preds: np.ndarray,
    config: dict,
    run_id: str = None,
    notes: str = "",
    out_dir: str = "submissions",
) -> str:
    """
    Build and save the submission CSV.
    Validates predictions, auto-names the file, logs to history.
    Returns the output file path.
    """
    proj  = config["project"]
    post  = config["postprocessing"]
    ID    = proj["id_col"]
    TARGET = proj["target"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Validate ─────────────────────────────────────────────────────────────
    preds = np.array(preds, dtype=np.float64)
    _validate_predictions(preds, post)

    # ── Post-process ─────────────────────────────────────────────────────────
    preds_clipped = np.clip(preds, post["clip_min"], post["clip_max"])

    # ── Build DataFrame ───────────────────────────────────────────────────────
    sub = pd.DataFrame({ID: test_ids.values, TARGET: preds_clipped})

    if len(sub) != len(test_ids):
        raise ValueError(f"Prediction length {len(preds)} != test IDs length {len(test_ids)}")
    if sub[ID].duplicated().any():
        raise ValueError("Duplicate record_ids in submission — check test set.")

    # ── Auto-name file ────────────────────────────────────────────────────────
    version  = proj.get("version", "v?")
    ts       = datetime.utcnow().strftime("%Y%m%dT%H%M")
    run_tag  = f"_{run_id}" if run_id else ""
    filename = f"submission_{version}{run_tag}_{ts}.csv"
    out_path = str(Path(out_dir) / filename)

    sub.to_csv(out_path, index=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "count": int(len(sub)),
        "mean":  round(float(preds_clipped.mean()), 6),
        "std":   round(float(preds_clipped.std()),  6),
        "min":   round(float(preds_clipped.min()),  6),
        "max":   round(float(preds_clipped.max()),  6),
        "p25":   round(float(np.percentile(preds_clipped, 25)), 6),
        "p75":   round(float(np.percentile(preds_clipped, 75)), 6),
    }

    logger.info(f"Submission saved: {out_path}")
    logger.info(f"  Stats: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}")

    # ── Log to history ────────────────────────────────────────────────────────
    _log_submission(out_path, stats, version, run_id, notes)

    return out_path


def _validate_predictions(preds: np.ndarray, post_cfg: dict) -> None:
    if np.any(np.isnan(preds)):
        raise ValueError(f"NaN values in predictions: {np.isnan(preds).sum()} rows")
    if np.any(np.isinf(preds)):
        raise ValueError(f"Inf values in predictions: {np.isinf(preds).sum()} rows")

    out_of_range = ((preds < post_cfg["clip_min"]) | (preds > post_cfg["clip_max"])).sum()
    if out_of_range > 0:
        logger.warning(f"⚠  {out_of_range} predictions outside [{post_cfg['clip_min']}, {post_cfg['clip_max']}] — will be clipped")


def _log_submission(path, stats, version, run_id, notes):
    log_path = Path(SUBMISSION_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    history.append({
        "timestamp":   datetime.utcnow().isoformat(),
        "path":        path,
        "version":     version,
        "run_id":      run_id,
        "stats":       stats,
        "notes":       notes,
        "lb_score":    None,   # fill manually after leaderboard feedback
    })
    log_path.write_text(json.dumps(history, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Record leaderboard score against a submission
# ─────────────────────────────────────────────────────────────────────────────

def record_lb_score(submission_path: str, lb_score: float) -> None:
    """Update the submission history with a real leaderboard score."""
    log_path = Path(SUBMISSION_LOG)
    if not log_path.exists():
        logger.warning("No submission history found.")
        return
    history = json.loads(log_path.read_text())
    for entry in history:
        if entry["path"] == submission_path:
            entry["lb_score"] = lb_score
            logger.info(f"LB score {lb_score} recorded for {submission_path}")
            break
    log_path.write_text(json.dumps(history, indent=2))


def print_submission_history() -> None:
    """Print a table of all submissions with LB scores."""
    log_path = Path(SUBMISSION_LOG)
    if not log_path.exists():
        print("No submissions logged yet.")
        return
    history = json.loads(log_path.read_text())
    rows = []
    for e in history:
        rows.append({
            "timestamp": e["timestamp"][:16],
            "version":   e["version"],
            "run_id":    e.get("run_id", ""),
            "mean":      e["stats"]["mean"],
            "std":       e["stats"]["std"],
            "lb_score":  e.get("lb_score", "—"),
            "notes":     (e.get("notes") or "")[:35],
        })
    df = pd.DataFrame(rows)
    print("\n" + "═" * 80)
    print("  SUBMISSION HISTORY")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80 + "\n")
