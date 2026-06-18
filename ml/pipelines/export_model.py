"""Export a trained FloodLens model bundle for local and API inference."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib


def export_model_bundle(
    *,
    model_artifacts: Dict[str, Any],
    encoder: Any,
    medians: Any,
    config: dict,
    metrics: dict,
    output_dir: str = "../artifacts/flood-risk-v3",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_version = f"flood-risk-{config['project']['version']}"
    metadata = {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_count": len(getattr(encoder, "feature_columns_", [])),
        "model_order": model_artifacts["model_order"],
    }

    bundle = {
        "metadata": metadata,
        "config": config,
        "base_models": model_artifacts["base_models"],
        "meta_model": model_artifacts["meta_model"],
        "model_order": model_artifacts["model_order"],
        "encoder": encoder,
        "medians": medians,
        "raw_feature_columns": getattr(encoder, "raw_feature_columns_", []),
        "categorical_columns": getattr(encoder, "categorical_columns_", []),
        "feature_columns": getattr(encoder, "feature_columns_", []),
        "district_risk_stats": getattr(encoder, "district_risk_stats_", {}),
    }

    bundle_path = out_dir / "model_bundle.joblib"
    metadata_path = out_dir / "metadata.json"
    joblib.dump(bundle, bundle_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return {
        "bundle_path": str(bundle_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
