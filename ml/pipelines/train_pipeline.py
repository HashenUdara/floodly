"""
FloodLens ML training pipeline.

Runs the copied competition pipeline from a production-friendly repository
layout and exports a reusable inference bundle.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))
os.chdir(ML_ROOT)

from pipelines.export_model import export_model_bundle
from src.data.pipeline import load_config, run_data_pipeline
from src.evaluation.tracker import evaluate_champion, log_run, print_leaderboard
from src.features.engineering import build_feature_matrix
from src.models.ensemble import train_ensemble
from src.submission.generator import generate_submission


def setup_logging(log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log_file = Path(log_dir) / f"run_{ts}.log"
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    logging.info("Log file: %s", log_file)


def run(args) -> None:
    t0 = time.time()
    setup_logging()
    logger = logging.getLogger("train_pipeline")

    config_path = args.config or "configs/config.yaml"
    config = load_config(config_path)
    logger.info("Config loaded: %s (version=%s)", config_path, config["project"]["version"])

    if args.skip_data:
        logger.info("Loading cached processed data.")
        train = pd.read_parquet(config["paths"]["processed_train"])
        test = pd.read_parquet(config["paths"]["processed_test"])
    else:
        logger.info("Running data pipeline.")
        train, test = run_data_pipeline(config)

    target = config["project"]["target"]
    id_col = config["project"]["id_col"]
    y_train = train[target].copy()
    test_ids = test[id_col].copy()

    logger.info("Building feature matrices.")
    X_train, X_test, encoder, medians = build_feature_matrix(train, test, y_train, config)

    if args.dry_run:
        logger.info("Dry run complete. Train shape=%s Test shape=%s", X_train.shape, X_test.shape)
        return

    logger.info("Training ensemble.")
    test_preds, oof_preds, metrics, model_artifacts = train_ensemble(
        X_train, y_train, X_test, config
    )

    run_record = log_run(metrics=metrics, config=config, notes=args.notes or "")
    promoted = evaluate_champion(run_record, config)

    sub_path = generate_submission(
        test_ids=test_ids,
        preds=test_preds,
        config=config,
        run_id=run_record["run_id"],
        notes=args.notes or "",
        out_dir="submissions",
    )
    run_record["submission_path"] = sub_path

    export_result = export_model_bundle(
        model_artifacts=model_artifacts,
        encoder=encoder,
        medians=medians,
        config=config,
        metrics=metrics,
        output_dir=args.artifact_dir,
    )

    elapsed = time.time() - t0
    logger.info("Run complete: %s", run_record["run_id"])
    logger.info("OOF MAE=%s OOF RMSE=%s promoted=%s", metrics["oof_mae"], metrics["oof_rmse"], promoted)
    logger.info("Submission saved: %s", sub_path)
    logger.info("Model bundle exported: %s", export_result["bundle_path"])
    logger.info("Wall time: %.1fs", elapsed)
    print_leaderboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export FloodLens model bundle.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    parser.add_argument("--notes", type=str, default="", help="Run notes for registry.")
    parser.add_argument("--skip-data", action="store_true", help="Use cached parquet data.")
    parser.add_argument("--dry-run", action="store_true", help="Validate features without training.")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="../artifacts/flood-risk-v3",
        help="Directory for exported inference bundle.",
    )
    run(parser.parse_args())
