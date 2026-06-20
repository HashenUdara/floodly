"""Live rainfall and river context for business-facing command views."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException, status

from app.core.settings import settings
from app.services.decision_intelligence_service import DecisionIntelligenceService
from app.services.location_service import MonitoredLocationProvider, get_location_service
from app.services.decision_intelligence_service import get_decision_intelligence_service


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"


class OpenMeteoClient:
    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{url}?{urlencode(params, doseq=True)}",
            headers={"User-Agent": "FloodLens/1.0 emergency-planning-demo"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            payload = response.read().decode("utf-8")
        return json.loads(payload)


class LiveContextService:
    def __init__(
        self,
        provider: MonitoredLocationProvider,
        decision_service: DecisionIntelligenceService,
        cache_path: Path,
        ttl_seconds: int,
        api_client: OpenMeteoClient | None = None,
    ):
        self.provider = provider
        self.decision_service = decision_service
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self.api_client = api_client or OpenMeteoClient(settings.live_context_timeout_s)

    def summary(self) -> dict[str, Any]:
        districts = self.decision_service.district_summary()
        exposed = [self._district_live_payload(item) for item in districts[:5]]
        highest = exposed[0] if exposed else None
        statuses = [item["live_context_status"] for item in exposed]

        return {
            "status": _combined_status(statuses),
            "source": "open-meteo",
            "generated_at": _utc_now(),
            "cache_ttl_seconds": self.ttl_seconds,
            "rainfall_outlook": _outlook(highest),
            "highest_attention_area": highest,
            "weather_pressure": {
                "rainfall_pressure": highest.get("rainfall_pressure") if highest else "Unavailable",
                "river_pressure": highest.get("river_pressure") if highest else "Unavailable",
                "next_24h_rain_mm": highest.get("next_24h_rain_mm") if highest else None,
                "next_7d_rain_mm": highest.get("next_7d_rain_mm") if highest else None,
            },
            "exposed_districts": exposed,
            "warnings": _summary_warnings(exposed),
        }

    def districts(self, district: str | None = None) -> dict[str, Any]:
        summaries = self.decision_service.district_summary()
        if district:
            summaries = [item for item in summaries if item["district"] == district]
        payload = [self._district_live_payload(item) for item in summaries[:10]]
        return {
            "status": _combined_status([item["live_context_status"] for item in payload]),
            "source": "open-meteo",
            "generated_at": _utc_now(),
            "districts": payload,
        }

    def location(self, record_id: str) -> dict[str, Any]:
        location = self._monitored_location(record_id)
        context = self._location_context(location)
        return {
            **context,
            "record_id": location.get("record_id"),
            "district": location.get("district"),
            "place_name": location.get("place_name"),
            "baseline_risk_level": location.get("baseline_risk_level"),
            "operational_priority": location.get("operational_priority"),
            "recommended_action": location.get("recommended_action"),
        }

    def refresh(self, district: str | None = None, limit: int | None = None) -> dict[str, Any]:
        capped_limit = max(1, min(limit or settings.live_context_refresh_limit, settings.live_context_refresh_limit))
        locations = self.provider.locations(district=district, limit=capped_limit)
        refreshed = []
        for location in locations:
            refreshed.append(self._location_context(location, force_refresh=True))
        return {
            "status": _combined_status([item["live_context_status"] for item in refreshed]),
            "source": "open-meteo",
            "requested": capped_limit,
            "refreshed": len(refreshed),
            "generated_at": _utc_now(),
        }

    def _district_live_payload(self, summary: dict[str, Any]) -> dict[str, Any]:
        location = self._representative_location(str(summary["district"]))
        live = self._location_context(location) if location else _empty_live_context()
        need_review = summary["critical_priority_count"] + summary["elevated_priority_count"]
        return {
            "district": summary["district"],
            "monitored_places": summary["monitored_places"],
            "need_review_count": need_review,
            "high_risk_count": summary["high_risk_count"],
            "top_reason": summary["top_risk_drivers"][0]["driver"] if summary["top_risk_drivers"] else "No strong reason",
            "live_context_status": live["live_context_status"],
            "rainfall_pressure": live["rainfall_pressure"],
            "river_pressure": live["river_pressure"],
            "next_24h_rain_mm": live["next_24h_rain_mm"],
            "next_7d_rain_mm": live["next_7d_rain_mm"],
            "river_discharge_max_m3s": live["river_discharge_max_m3s"],
            "source_timestamp": live["source_timestamp"],
        }

    def _representative_location(self, district: str) -> dict[str, Any] | None:
        priority = self.decision_service.emergency_priority(district=district, limit=1)
        if priority:
            try:
                return self._monitored_location(str(priority[0]["record_id"]))
            except HTTPException:
                pass
        locations = self.provider.locations(district=district, limit=1)
        return locations[0] if locations else None

    def _monitored_location(self, record_id: str) -> dict[str, Any]:
        matches = self.provider.locations(search=record_id, limit=5)
        for location in matches:
            if str(location.get("record_id")) == record_id:
                return location
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location record not found: {record_id}",
        )

    def _location_context(self, location: dict[str, Any], force_refresh: bool = False) -> dict[str, Any]:
        record_id = str(location.get("record_id") or "")
        cache_key = f"location:{record_id}"
        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached:
                return cached

        latitude = _as_float(location.get("map_latitude") or location.get("latitude")) or 7.8731
        longitude = _as_float(location.get("map_longitude") or location.get("longitude")) or 80.7718
        warnings: list[str] = []

        calls = {
            "forecast": (
                OPEN_METEO_FORECAST_URL,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "precipitation,rain",
                    "hourly": "precipitation,precipitation_probability,rain",
                    "daily": "precipitation_sum,precipitation_probability_max",
                    "forecast_days": 7,
                    "timezone": "Asia/Colombo",
                    "precipitation_unit": "mm",
                },
            ),
            "elevation": (
                OPEN_METEO_ELEVATION_URL,
                {"latitude": latitude, "longitude": longitude},
            ),
            "flood": (
                OPEN_METEO_FLOOD_URL,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": "river_discharge,river_discharge_max",
                    "forecast_days": 7,
                },
            ),
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                label: executor.submit(self._safe_call, label, url, params, warnings)
                for label, (url, params) in calls.items()
            }
            forecast = futures["forecast"].result()
            elevation = futures["elevation"].result()
            flood = futures["flood"].result()

        parsed = _parse_live_payload(forecast=forecast, elevation=elevation, flood=flood)
        status_value = "live" if forecast else "partial" if elevation or flood else "unavailable"
        result = {
            "live_context_status": status_value,
            "source": "open-meteo",
            "source_timestamp": _utc_now(),
            "latitude": latitude,
            "longitude": longitude,
            **parsed,
            "warnings": warnings,
        }
        self._cache_set(cache_key, result)
        return result

    def _safe_call(
        self,
        label: str,
        url: str,
        params: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        try:
            payload = self.api_client.get_json(url, params)
            if payload.get("error"):
                warnings.append(f"{label} unavailable: {payload.get('reason') or 'provider error'}")
                return None
            return payload
        except Exception as exc:  # pragma: no cover - provider/network-specific detail
            warnings.append(f"{label} unavailable: {exc}")
            return None

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        cache = self._read_cache()
        item = cache.get("items", {}).get(key)
        if not item:
            return None
        expires_at = _parse_time(item.get("expires_at"))
        if not expires_at or expires_at <= datetime.now(timezone.utc):
            return None
        value = item.get("value")
        return value if isinstance(value, dict) else None

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        cache = self._read_cache()
        cache.setdefault("items", {})[key] = {
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)).isoformat(),
            "value": value,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _read_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {"items": {}}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"items": {}}


def _parse_live_payload(
    forecast: dict[str, Any] | None,
    elevation: dict[str, Any] | None,
    flood: dict[str, Any] | None,
) -> dict[str, Any]:
    hourly = (forecast or {}).get("hourly", {})
    daily = (forecast or {}).get("daily", {})
    current = (forecast or {}).get("current", {})
    hourly_precipitation = [_as_float(value) or 0 for value in hourly.get("precipitation", [])]
    daily_precipitation = [_as_float(value) or 0 for value in daily.get("precipitation_sum", [])]
    daily_probability = [_as_float(value) or 0 for value in daily.get("precipitation_probability_max", [])]
    flood_daily = (flood or {}).get("daily", {})
    river_values = [_as_float(value) for value in flood_daily.get("river_discharge", [])]
    river_max_values = [_as_float(value) for value in flood_daily.get("river_discharge_max", [])]
    elevation_values = (elevation or {}).get("elevation", [])

    next_24h = round(sum(hourly_precipitation[:24]), 1) if hourly_precipitation else None
    next_7d = round(sum(daily_precipitation[:7]), 1) if daily_precipitation else None
    river_current = _first_number(river_values)
    river_max = _max_number(river_max_values or river_values)

    return {
        "rain_now_mm": _as_float(current.get("rain")),
        "precipitation_now_mm": _as_float(current.get("precipitation")),
        "next_24h_rain_mm": next_24h,
        "next_7d_rain_mm": next_7d,
        "precipitation_probability_max": round(max(daily_probability[:7]), 1) if daily_probability else None,
        "rainfall_pressure": _rainfall_pressure(next_24h, next_7d),
        "river_discharge_m3s": river_current,
        "river_discharge_max_m3s": river_max,
        "river_pressure": _river_pressure(river_current, river_max),
        "elevation_m": _first_number([_as_float(value) for value in elevation_values]),
    }


def _empty_live_context() -> dict[str, Any]:
    return {
        "live_context_status": "unavailable",
        "source": "open-meteo",
        "source_timestamp": None,
        "rain_now_mm": None,
        "precipitation_now_mm": None,
        "next_24h_rain_mm": None,
        "next_7d_rain_mm": None,
        "precipitation_probability_max": None,
        "rainfall_pressure": "Unavailable",
        "river_discharge_m3s": None,
        "river_discharge_max_m3s": None,
        "river_pressure": "Unavailable",
        "elevation_m": None,
        "warnings": ["No representative monitored place found."],
    }


def _rainfall_pressure(next_24h: float | None, next_7d: float | None) -> str:
    if next_24h is None and next_7d is None:
        return "Unavailable"
    if (next_24h or 0) >= 100 or (next_7d or 0) >= 240:
        return "Severe"
    if (next_24h or 0) >= 60 or (next_7d or 0) >= 150:
        return "High"
    if (next_24h or 0) >= 25 or (next_7d or 0) >= 75:
        return "Watch"
    return "Normal"


def _river_pressure(current: float | None, maximum: float | None) -> str:
    value = maximum if maximum is not None else current
    if value is None:
        return "Unavailable"
    if value >= 750:
        return "Severe"
    if value >= 350:
        return "High"
    if value >= 100:
        return "Watch"
    return "Normal"


def _combined_status(statuses: list[str]) -> str:
    if not statuses:
        return "unavailable"
    if any(status == "live" for status in statuses):
        return "live"
    if any(status == "partial" for status in statuses):
        return "partial"
    return "unavailable"


def _outlook(highest: dict[str, Any] | None) -> str:
    if not highest or highest.get("rainfall_pressure") == "Unavailable":
        return "Live rainfall unavailable; use baseline risk until the provider responds."
    if highest.get("rainfall_pressure") in {"Severe", "High"}:
        return f"{highest['district']} has elevated rainfall pressure. Review priority places first."
    if highest.get("rainfall_pressure") == "Watch":
        return f"{highest['district']} is under rainfall watch. Keep district teams ready."
    return "No major rainfall pressure detected from the live provider."


def _summary_warnings(exposed: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if not exposed:
        warnings.append("No monitored districts available for live context.")
    if exposed and all(item["live_context_status"] == "unavailable" for item in exposed):
        warnings.append("Live context provider unavailable; baseline risk remains visible.")
    return warnings


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(values: list[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return round(value, 2)
    return None


def _max_number(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(max(clean), 2) if clean else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@lru_cache(maxsize=1)
def get_live_context_service() -> LiveContextService:
    return LiveContextService(
        provider=get_location_service(),
        decision_service=get_decision_intelligence_service(),
        cache_path=settings.live_context_cache_path,
        ttl_seconds=settings.live_context_cache_ttl_s,
    )
