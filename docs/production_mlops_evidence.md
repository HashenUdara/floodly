# Production MLOps Evidence

This document is the Stage 2 evidence log for deployment, observability,
automation, and scalability. Keep it updated with final URLs and measured
results before submission.

## Implemented Evidence

- CI workflow now has separate ML, backend, and frontend jobs.
- Backend CI validates Alembic migrations against PostgreSQL/pgvector.
- Artifact smoke workflow validates release artifact layout and API inference.
- Manual retraining workflow creates a challenger artifact and applies quality
  gates against the current champion metadata.
- FastAPI CORS is environment-driven through `BACKEND_CORS_ORIGINS`.
- `GET /readiness` reports model, metadata, seed data, upload storage, and
  optional database readiness.
- HTTP telemetry is logged to `backend/logs/http_events.jsonl`.
- `GET /monitoring/system` exposes request volume, errors, latency, route
  breakdown, document indexing failures, and retrieval activity.
- Model Operations dashboard includes system reliability cards.
- `scripts/load_smoke.py` records generated timestamp, throughput, latency,
  failure rate, and per-route status breakdown for the key service routes.
- `scripts/collect_ops_evidence.py` collects health, readiness, system
  monitoring, load-smoke, document-search, and scaling notes into
  `docs/evidence/`.

## Final Deployment URLs

| Surface | URL | Status |
| --- | --- | --- |
| Frontend | TBD | Not verified |
| Backend API | TBD | Not verified |
| Readiness | TBD | Not verified |
| GitHub Actions CI | TBD | Not verified |

## Readiness Checks

Run:

```bash
curl http://127.0.0.1:8000/readiness
```

Expected:

- model artifact: `ok`
- model loaded: `ok`
- model metadata: `ok`
- seed test data: `ok`
- upload storage: `ok`
- database: `ok` when RAG is configured, `skipped` when RAG is disabled

## System Monitoring Checks

Run:

```bash
curl http://127.0.0.1:8000/monitoring/system
```

Expected metrics:

- total HTTP requests
- HTTP error count and error rate
- p50 and p95 latency
- route-level counts and latency
- document indexing failures
- retrieval activity count

## Load-Smoke Command

Run after backend is live and artifacts are available:

```bash
ml/.venv/bin/python scripts/load_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --requests 5 \
  --output docs/evidence/load_smoke_latest.json
```

Use `--skip-documents-search` if the Knowledge Library database is not
configured for the environment being tested.

## Production Evidence Command

Run after backend is live and artifacts are available:

```bash
ml/.venv/bin/python scripts/collect_ops_evidence.py \
  --base-url http://127.0.0.1:8000 \
  --requests 5
```

Expected outputs:

```text
docs/evidence/readiness_latest.json
docs/evidence/load_smoke_latest.json
docs/evidence/production_ops_summary.md
```

## Latest Load Result

TBD after running against the final deployed backend.

Record:

- total requests
- success count
- failure count
- throughput
- p50 latency
- p95 latency
- max latency

## Retraining Workflow

Workflow:

```text
.github/workflows/retrain.yml
```

Purpose:

- validate input data
- train a challenger artifact
- compare against champion metadata
- reject if MAE/RMSE or prediction-range gates fail
- upload challenger report and artifact as workflow artifacts

Promotion policy:

- challenger MAE must stay within tolerance
- challenger RMSE must stay within tolerance
- challenger predictions must remain in `[0, 1]`
- final promotion to GitHub Release remains a human approval step

## Stage 2 Remaining Proof

- Run the CI workflow on the final commit and capture the green status.
- Run artifact smoke after uploading `floodlens-artifacts.zip` to GitHub
  Releases.
- Run load-smoke against the deployed backend and paste the result here.
- Run `scripts/collect_ops_evidence.py` against the deployed backend and keep
  the generated files under `docs/evidence/`.
- Add final frontend and backend URLs.
- Verify Neon/Dokploy migration from a clean database.
