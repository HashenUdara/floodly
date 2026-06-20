"""Small HTTP load-smoke utility for FloodLens deployment evidence."""

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


def request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> tuple[bool, float, int | None]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    started = time.perf_counter()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()
            status = response.status
            ok = 200 <= status < 400
    except urllib.error.HTTPError as error:
        status = error.code
        ok = False
    except urllib.error.URLError:
        status = None
        ok = False
    return ok, (time.perf_counter() - started) * 1000, status


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100) * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)


def first_record(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
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


def summarize_routes(measurements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in measurements:
        grouped.setdefault(str(item["route"]), []).append(item)
    summary = {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FloodLens API load smoke checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--skip-documents-search", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    record = first_record(Path(args.test_csv))
    operations: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET /health", "GET", None),
        ("POST /predict", "POST", {"record": record}),
        ("POST /batch-predict", "POST", {"limit": 5}),
    ]
    if not args.skip_documents_search:
        operations.append(
            (
                "POST /documents/search",
                "POST",
                {"query": "flood response actions", "limit": 3},
            )
        )

    route_paths = {
        "GET /health": "/health",
        "POST /predict": "/predict",
        "POST /batch-predict": "/batch-predict",
        "POST /documents/search": "/documents/search",
    }

    measurements = []
    started = time.perf_counter()
    for _ in range(max(args.requests, 1)):
        for route, method, payload in operations:
            ok, latency_ms, status = request(
                method, f"{base_url}{route_paths[route]}", payload
            )
            measurements.append(
                {
                    "route": route,
                    "ok": ok,
                    "status": status,
                    "latency_ms": round(latency_ms, 2),
                }
            )
    elapsed = time.perf_counter() - started
    latencies = [item["latency_ms"] for item in measurements]
    success_count = sum(1 for item in measurements if item["ok"])
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "route_summary": summarize_routes(measurements),
        "routes": measurements,
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    if result["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
