#!/usr/bin/env python3
"""Collect production-readiness evidence from a running FloodLens backend."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_first_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source), None)
    if not row:
        return None
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
            continue
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            parsed[key] = value
    return parsed


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    started = time.perf_counter()
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return {
                "ok": 200 <= response.status < 400,
                "status": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "body": parsed,
            }
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": error.code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "body": detail,
        }
    except urllib.error.URLError as error:
        return {
            "ok": False,
            "status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "body": str(error),
        }


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100) * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)


def run_load_smoke(
    *,
    base_url: str,
    record: dict[str, Any] | None,
    requests: int,
    include_documents: bool,
) -> dict[str, Any]:
    operations: list[tuple[str, str, str, dict[str, Any] | None]] = [
        ("GET /health", "GET", "/health", None),
        ("POST /batch-predict", "POST", "/batch-predict", {"limit": 5}),
    ]
    if record is not None:
        operations.append(("POST /predict", "POST", "/predict", {"record": record}))
    if include_documents:
        operations.append(
            (
                "POST /documents/search",
                "POST",
                "/documents/search",
                {"query": "flood response actions", "limit": 3},
            )
        )

    measurements = []
    started = time.perf_counter()
    for _ in range(max(requests, 1)):
        for route, method, path, payload in operations:
            result = request_json(method, f"{base_url}{path}", payload)
            measurements.append(
                {
                    "route": route,
                    "ok": result["ok"],
                    "status": result["status"],
                    "latency_ms": result["latency_ms"],
                }
            )
    elapsed = time.perf_counter() - started
    latencies = [float(item["latency_ms"]) for item in measurements]
    success_count = sum(1 for item in measurements if item["ok"])
    return {
        "generated_at": timestamp(),
        "base_url": base_url,
        "total_requests": len(measurements),
        "success_count": success_count,
        "failure_count": len(measurements) - success_count,
        "failure_rate": round((len(measurements) - success_count) / len(measurements), 4)
        if measurements
        else None,
        "throughput_rps": round(len(measurements) / elapsed, 2) if elapsed else None,
        "p50_latency_ms": round(float(statistics.median(latencies)), 2) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95),
        "route_summary": summarize_routes(measurements),
        "routes": measurements,
    }


def summarize_routes(measurements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in measurements:
        grouped.setdefault(str(item["route"]), []).append(item)
    summary: dict[str, dict[str, Any]] = {}
    for route, items in grouped.items():
        latencies = [float(item["latency_ms"]) for item in items]
        success = sum(1 for item in items if item["ok"])
        summary[route] = {
            "requests": len(items),
            "success_count": success,
            "failure_count": len(items) - success,
            "failure_rate": round((len(items) - success) / len(items), 4) if items else None,
            "p50_latency_ms": round(float(statistics.median(latencies)), 2) if latencies else None,
            "p95_latency_ms": percentile(latencies, 95),
            "statuses": sorted({item["status"] for item in items}, key=lambda value: str(value)),
        }
    return summary


def write_summary(path: Path, evidence: dict[str, Any]) -> None:
    readiness = evidence["readiness"]
    load_smoke = evidence["load_smoke"]
    system = evidence["system_monitoring"]
    lines = [
        "# FloodLens Production Ops Evidence",
        "",
        f"- Generated: `{evidence['generated_at']}`",
        f"- Backend URL: `{evidence['base_url']}`",
        f"- Health: `{evidence['health'].get('status')}`",
        f"- Readiness: `{readiness.get('body', {}).get('status', 'unknown') if isinstance(readiness.get('body'), dict) else 'unknown'}`",
        f"- Load-smoke requests: `{load_smoke['total_requests']}`",
        f"- Load-smoke failure rate: `{load_smoke['failure_rate']}`",
        f"- Load-smoke p95 latency: `{load_smoke['p95_latency_ms']} ms`",
        f"- HTTP telemetry events: `{system.get('body', {}).get('total_requests', 'unknown') if isinstance(system.get('body'), dict) else 'unknown'}`",
        "",
        "## Deployment Notes",
        "",
        f"- Batch scoring is capped at `{evidence['deployment_notes']['batch_limit']}` records per request for local/demo reliability.",
        f"- Scaling path: {evidence['deployment_notes']['scaling_path']}.",
        "- Rollback path: restore the previous GitHub Release artifact bundle and restart the backend.",
        "- Secret handling: API keys and database credentials must be supplied through environment variables, never committed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FloodLens production evidence.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output", default="docs/evidence/readiness_latest.json")
    parser.add_argument("--load-output", default="docs/evidence/load_smoke_latest.json")
    parser.add_argument("--summary-output", default="docs/evidence/production_ops_summary.md")
    parser.add_argument("--skip-documents-search", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    record = read_first_record(Path(args.test_csv))
    health = request_json("GET", f"{base_url}/health")
    readiness = request_json("GET", f"{base_url}/readiness")
    system = request_json("GET", f"{base_url}/monitoring/system")
    document_search = (
        {"ok": None, "status": "skipped", "body": "skipped by --skip-documents-search"}
        if args.skip_documents_search
        else request_json(
            "POST",
            f"{base_url}/documents/search",
            {"query": "flood response actions", "limit": 3},
        )
    )
    load_smoke = run_load_smoke(
        base_url=base_url,
        record=record,
        requests=args.requests,
        include_documents=not args.skip_documents_search,
    )
    evidence = {
        "generated_at": timestamp(),
        "base_url": base_url,
        "health": health,
        "readiness": readiness,
        "document_search": document_search,
        "load_smoke": load_smoke,
        "system_monitoring": system,
        "deployment_notes": {
            "batch_limit": 100,
            "scaling_path": "queue-backed batch scoring, persistent object storage, and async workers for larger workloads",
            "rollback": "restore the previous GitHub Release artifact bundle and restart the backend",
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    load_output = Path(args.load_output)
    load_output.parent.mkdir(parents=True, exist_ok=True)
    load_output.write_text(json.dumps(load_smoke, indent=2), encoding="utf-8")
    write_summary(Path(args.summary_output), evidence)
    print(json.dumps(evidence, indent=2))

    required_ok = [health["ok"], readiness["ok"], system["ok"], load_smoke["failure_count"] == 0]
    if not args.skip_documents_search:
        required_ok.append(document_search["ok"])
    if not all(required_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
