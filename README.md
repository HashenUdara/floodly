# FloodLens

FloodLens is a flood risk intelligence foundation for the MLOps hackathon. The
first milestone focuses on turning the existing competition model work into a
clean repository with a reusable local inference artifact.

## First Milestone

1. Put `train.csv` and `test.csv` in `data/raw/`.
2. Install the ML environment:
   ```bash
   ./scripts/setup_ml.sh
   ```
3. Train and export the model bundle:
   ```bash
   cd ml
   source .venv/bin/activate
   python pipelines/train_pipeline.py --notes "flood-risk-v3 foundation"
   ```
4. Run one local prediction:
   ```bash
   ./scripts/run_local_prediction.sh
   ```

The exported inference bundle is written to
`artifacts/flood-risk-v3/model_bundle.joblib`.
