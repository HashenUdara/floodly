"""Backward-compatible wrapper for the main training pipeline."""

from pipelines.train_pipeline import run

if __name__ == "__main__":
    import argparse

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
