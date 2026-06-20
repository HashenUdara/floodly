# FloodLens Evidence Outputs

This folder stores judge-visible outputs for Stage 2 and Stage 3.

Generate production evidence:

```bash
ml/.venv/bin/python scripts/collect_ops_evidence.py \
  --base-url http://127.0.0.1:8000 \
  --requests 5
```

Generate RAG evaluation evidence:

```bash
ml/.venv/bin/python scripts/evaluate_retrieval.py \
  --upload-fixtures \
  --output docs/evidence/rag_evaluation_latest.json \
  --summary-output docs/evidence/rag_evaluation_summary.md
```

Expected generated files:

```text
load_smoke_latest.json
readiness_latest.json
production_ops_summary.md
rag_evaluation_latest.json
rag_evaluation_summary.md
```

Do not include API keys, database credentials, uploaded private documents, or
raw production incident details in evidence files.
