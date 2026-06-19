"""HTTP telemetry and system monitoring summaries."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from app.core.settings import settings


EMPTY_SYSTEM_SUMMARY = {
    "total_requests": 0,
    "error_count": 0,
    "error_rate": 0,
    "p50_latency_ms": None,
    "p95_latency_ms": None,
    "routes": [],
    "latest_error_at": None,
    "document_indexing_failures": 0,
    "retrieval_events": 0,
}


class SystemMonitoringService:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def log_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "method": method,
            "route": route,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "error": status_code >= 400,
        }
        if model_version:
            event["model_version"] = model_version
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
        return event

    def read_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events = []
        with self.log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def summary(self, document_indexing_failures: int = 0) -> dict[str, Any]:
        events = self.read_events()
        if not events:
            return {**EMPTY_SYSTEM_SUMMARY, "document_indexing_failures": document_indexing_failures}

        latencies = [float(event["duration_ms"]) for event in events]
        errors = [event for event in events if event.get("error")]
        route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            route_groups[str(event.get("route") or "unknown")].append(event)

        return {
            "total_requests": len(events),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(events), 4),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "routes": [
                route_summary(route, route_events)
                for route, route_events in sorted(
                    route_groups.items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            ],
            "latest_error_at": max(
                (event["timestamp"] for event in errors if event.get("timestamp")),
                default=None,
            ),
            "document_indexing_failures": document_indexing_failures,
            "retrieval_events": sum(
                1 for event in events if event.get("route") == "POST /documents/search"
            ),
        }


def route_summary(route: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(event["duration_ms"]) for event in events]
    errors = [event for event in events if event.get("error")]
    return {
        "route": route,
        "count": len(events),
        "error_count": len(errors),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
    }


def percentile(values: list[float], value: int) -> float | None:
    if not values:
        return None
    if value == 50:
        return round(float(median(values)), 2)
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((value / 100) * len(ordered) + 0.5) - 1))
    return round(float(ordered[index]), 2)


@lru_cache(maxsize=1)
def get_system_monitoring_service() -> SystemMonitoringService:
    return SystemMonitoringService(settings.http_event_log_path)
