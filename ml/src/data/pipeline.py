"""
src/data/pipeline.py
Data ingestion, validation, cleaning, and versioning for the flood risk pipeline.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Data validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_TRAIN_COLS = [
    "record_id", "flood_risk_score", "district", "latitude", "longitude",
    "elevation_m", "distance_to_river_m", "rainfall_7d_mm", "is_synthetic",
]
REQUIRED_TEST_COLS = [
    "record_id", "district", "latitude", "longitude",
    "elevation_m", "distance_to_river_m", "rainfall_7d_mm",
]


def validate_dataframe(df: pd.DataFrame, required_cols: list, name: str) -> dict:
    """Run schema + range checks. Returns a report dict."""
    report = {"name": name, "rows": len(df), "issues": [], "passed": True}

    # Column presence
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        report["issues"].append(f"Missing columns: {missing}")
        report["passed"] = False

    # Target range (train only)
    if "flood_risk_score" in df.columns:
        out_of_range = ((df["flood_risk_score"] < 0) | (df["flood_risk_score"] > 1)).sum()
        if out_of_range > 0:
            report["issues"].append(f"flood_risk_score out of [0,1]: {out_of_range} rows")

    # Duplicate record_ids
    dupes = df["record_id"].duplicated().sum()
    if dupes > 0:
        report["issues"].append(f"Duplicate record_ids: {dupes}")

    # Extreme null rate
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.5].to_dict()
    if high_null:
        report["issues"].append(f"High null rate cols (>50%): {high_null}")

    logger.info(f"Validation [{name}]: {'PASS' if report['passed'] else 'FAIL'} — {len(report['issues'])} issues")
    for issue in report["issues"]:
        logger.warning(f"  ⚠  {issue}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Data cleaning (mirrors v3 logic, centralised)
# ─────────────────────────────────────────────────────────────────────────────

def clean_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    1. Drop rows where is_synthetic is NaN (corrupted rows).
    2. Average flood_risk_score for duplicate record_ids.
    Returns cleaned DataFrame + cleaning report.
    """
    original_len = len(df)
    report = {}

    # Step 1 — drop corrupted rows
    df = df[df["is_synthetic"].notna()].copy()
    removed_corrupted = original_len - len(df)
    report["removed_corrupted"] = removed_corrupted

    # Step 2 — average duplicate targets
    target_means = df.groupby("record_id")["flood_risk_score"].mean()
    df = df.drop_duplicates(subset="record_id").copy()
    df["flood_risk_score"] = df["record_id"].map(target_means)
    removed_dupes = original_len - removed_corrupted - len(df)
    report["merged_duplicates"] = removed_dupes
    report["final_rows"] = len(df)

    logger.info(
        f"Cleaning: {removed_corrupted} corrupted removed, "
        f"{removed_dupes} duplicates merged → {len(df):,} clean rows"
    )
    return df.reset_index(drop=True), report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset versioning (lightweight — hash + metadata sidecar)
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def version_dataset(df: pd.DataFrame, name: str, out_dir: str, meta: dict = None) -> str:
    """Save versioned parquet + JSON sidecar. Returns version tag."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    parquet_path = out_dir / f"{name}_{ts}.parquet"
    meta_path    = out_dir / f"{name}_{ts}.json"

    df.to_parquet(parquet_path, index=False)
    h = _file_hash(str(parquet_path))

    record = {
        "version": ts,
        "name": name,
        "rows": len(df),
        "cols": list(df.columns),
        "sha256_prefix": h,
        "saved_at": datetime.utcnow().isoformat(),
        **(meta or {}),
    }
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Versioned dataset saved: {parquet_path} (sha256:{h})")
    return ts


def list_versions(out_dir: str, name: str) -> list:
    """Return sorted list of version metadata for a given dataset name."""
    out_dir = Path(out_dir)
    metas = sorted(out_dir.glob(f"{name}_*.json"), reverse=True)
    return [json.load(open(m)) for m in metas]


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_data_pipeline(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full data pipeline: load → validate → clean → version → return.
    Returns (train_clean_df, test_df).
    """
    paths = config["paths"]

    logger.info("=" * 60)
    logger.info("DATA PIPELINE START")
    logger.info("=" * 60)

    # Load
    logger.info(f"Loading train: {paths['train']}")
    train_raw = pd.read_csv(paths["train"])
    logger.info(f"Loading test:  {paths['test']}")
    test = pd.read_csv(paths["test"])

    # Validate
    validate_dataframe(train_raw, REQUIRED_TRAIN_COLS, "train_raw")
    validate_dataframe(test,      REQUIRED_TEST_COLS,  "test")

    # Clean train
    train_clean, clean_report = clean_train(train_raw)

    # Version
    version_dataset(
        train_clean, "train_clean",
        out_dir=config["paths"].get("versioned_dir", "data/versioned"),
        meta={"clean_report": clean_report, "source": paths["train"]},
    )
    version_dataset(
        test, "test",
        out_dir=config["paths"].get("versioned_dir", "data/versioned"),
        meta={"source": paths["test"]},
    )

    # Save processed
    Path(paths["processed_train"]).parent.mkdir(parents=True, exist_ok=True)
    train_clean.to_parquet(paths["processed_train"], index=False)
    test.to_parquet(paths["processed_test"], index=False)
    logger.info("Processed data saved.")

    logger.info("DATA PIPELINE COMPLETE")
    return train_clean, test
