"""Decision intelligence aggregations for monitored flood-risk places."""

from collections import Counter
from functools import lru_cache
from typing import Any

from app.services.location_service import MonitoredLocationProvider, get_location_service


DEFAULT_LIMIT = 25
MAX_LIMIT = 100
PROVIDER_DISTRICT_LIMIT = 500


class DecisionIntelligenceService:
    def __init__(self, provider: MonitoredLocationProvider):
        self.provider = provider

    def district_summary(self) -> list[dict[str, Any]]:
        summaries = []
        for district in self.provider.districts():
            locations = self.provider.locations(district=district, limit=PROVIDER_DISTRICT_LIMIT)
            if not locations:
                continue

            scores = [_as_float(location.get("baseline_risk_score")) or 0 for location in locations]
            priorities = Counter(location.get("operational_priority") for location in locations)
            top_drivers = Counter(
                driver
                for location in locations
                for driver in location.get("risk_drivers", [])
            )

            summaries.append(
                {
                    "district": district,
                    "monitored_places": len(locations),
                    "average_baseline_risk_score": round(sum(scores) / len(scores), 4),
                    "high_risk_count": sum(
                        1 for location in locations if location.get("baseline_risk_level") == "High"
                    ),
                    "critical_priority_count": priorities.get("Critical", 0),
                    "elevated_priority_count": priorities.get("Elevated", 0),
                    "top_risk_drivers": [
                        {"driver": driver, "count": count}
                        for driver, count in top_drivers.most_common(3)
                    ],
                }
            )

        return sorted(
            summaries,
            key=lambda item: (
                item["average_baseline_risk_score"],
                item["high_risk_count"],
                item["monitored_places"],
            ),
            reverse=True,
        )

    def high_risk_locations(
        self,
        district: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        locations = self._locations_for_scope(district)
        ranked = sorted(
            locations,
            key=lambda location: (
                _as_float(location.get("baseline_risk_score")) or 0,
                _priority_weight(str(location.get("operational_priority") or "")),
            ),
            reverse=True,
        )
        return [_location_risk_payload(location) for location in ranked[:_bounded_limit(limit)]]

    def emergency_priority(
        self,
        district: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        locations = self._locations_for_scope(district)
        scored = []
        for location in locations:
            priority_score = _emergency_priority_score(location)
            scored.append((priority_score, location))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        return [
            {
                "rank": rank,
                "record_id": location.get("record_id"),
                "district": location.get("district"),
                "place_name": location.get("place_name"),
                "asset_type": location.get("asset_type"),
                "emergency_priority_score": round(score, 4),
                "baseline_risk_score": location.get("baseline_risk_score"),
                "baseline_risk_level": location.get("baseline_risk_level"),
                "operational_priority": location.get("operational_priority"),
                "priority_reasons": _priority_reasons(location),
                "recommended_action": location.get("recommended_action"),
            }
            for rank, (score, location) in enumerate(ranked[:_bounded_limit(limit)], start=1)
        ]

    def _locations_for_scope(self, district: str | None) -> list[dict[str, Any]]:
        if district:
            if district not in set(self.provider.districts()):
                return []
            return self.provider.locations(district=district, limit=PROVIDER_DISTRICT_LIMIT)

        locations = []
        for item in self.provider.districts():
            locations.extend(self.provider.locations(district=item, limit=PROVIDER_DISTRICT_LIMIT))
        return locations


def _location_risk_payload(location: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": location.get("record_id"),
        "district": location.get("district"),
        "place_name": location.get("place_name"),
        "asset_type": location.get("asset_type"),
        "baseline_risk_score": location.get("baseline_risk_score"),
        "baseline_risk_level": location.get("baseline_risk_level"),
        "operational_priority": location.get("operational_priority"),
        "risk_drivers": location.get("risk_drivers", []),
        "recommended_action": location.get("recommended_action"),
    }


def _emergency_priority_score(location: dict[str, Any]) -> float:
    baseline_risk = _as_float(location.get("baseline_risk_score")) or 0
    population_exposure = _clamp((_as_float(location.get("population_density_per_km2")) or 0) / 1000)
    evacuation_gap = _clamp((_as_float(location.get("nearest_evac_km")) or 0) / 30)
    flood_history = _clamp((_as_float(location.get("historical_flood_count")) or 0) / 5)
    infrastructure_weakness = 1 - _clamp((_as_float(location.get("infrastructure_score")) or 50) / 100)

    return _clamp(
        baseline_risk * 0.45
        + population_exposure * 0.2
        + evacuation_gap * 0.15
        + flood_history * 0.1
        + infrastructure_weakness * 0.1
    )


def _priority_reasons(location: dict[str, Any]) -> list[str]:
    reasons = list(location.get("risk_drivers", [])[:2])

    if _clamp((_as_float(location.get("population_density_per_km2")) or 0) / 1000) >= 0.55:
        reasons.append("High population exposure")
    if _clamp((_as_float(location.get("nearest_evac_km")) or 0) / 30) >= 0.45:
        reasons.append("Limited evacuation access")
    if 1 - _clamp((_as_float(location.get("infrastructure_score")) or 50) / 100) >= 0.45:
        reasons.append("Infrastructure vulnerability")
    if _clamp((_as_float(location.get("historical_flood_count")) or 0) / 5) >= 0.4:
        reasons.append("Past flood exposure")

    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped[:3]


def _priority_weight(priority: str) -> int:
    return {
        "Critical": 4,
        "Elevated": 3,
        "Watch": 2,
        "Routine": 1,
    }.get(priority, 0)


def _bounded_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def get_decision_intelligence_service() -> DecisionIntelligenceService:
    return DecisionIntelligenceService(get_location_service())
