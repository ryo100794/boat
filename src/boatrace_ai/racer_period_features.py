from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import json
import math
from typing import Any, Iterable


RELEASE_LAG_DAYS = 30
NUMERIC_FIELDS = (
    "avg_st",
    "win_rate",
    "place2_rate",
    "place3_rate",
    "f_count",
    "l_count",
    "ability_index",
    "starts",
)
CATEGORICAL_FIELDS = ("racer_class", "origin", "branch")


def load_racer_period_lookup(conn: Any) -> dict[int, tuple[dict[str, Any], ...]]:
    try:
        rows = conn.execute(
            """
            SELECT year, half, racer_no, racer_class, raw_json
            FROM racer_period_stats
            ORDER BY racer_no, year, half
            """
        ).fetchall()
    except Exception as exc:
        if "racer_period_stats" in str(exc).lower():
            return {}
        raise
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = _json_payload(row["raw_json"])
        available_from = _available_from(payload.get("calculation_to"))
        racer_no = _integer(row["racer_no"])
        if racer_no is None or available_from is None:
            continue
        item: dict[str, Any] = {
            "available_from": available_from,
            "racer_class": payload.get("racer_class") or row["racer_class"],
        }
        for key in NUMERIC_FIELDS:
            value = _number(payload.get(key))
            if value is not None:
                item[key] = value
        for key in CATEGORICAL_FIELDS:
            value = str(payload.get(key) or item.get(key) or "").strip()
            if value:
                item[key] = value.replace("　", "")
        grouped[racer_no].append(item)
    return {
        racer_no: tuple(sorted(items, key=lambda item: item["available_from"]))
        for racer_no, items in grouped.items()
    }


def enrich_racer_period_rows(
    rows: Iterable[Any],
    lookup: dict[int, tuple[dict[str, Any], ...]],
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = {key: row[key] for key in row.keys()}
        race_date = _date_value(item.get("race_date"))
        racer_no = _integer(item.get("racer_no"))
        selected = (
            select_available_period(
                lookup.get(racer_no, ()),
                race_date=race_date,
            )
            if race_date is not None and racer_no is not None
            else None
        )
        item["_racer_period"] = selected
        enriched.append(item)
    return enriched


def select_available_period(
    periods: Iterable[dict[str, Any]],
    *,
    race_date: date,
) -> dict[str, Any] | None:
    selected = None
    for period in periods:
        if period["available_from"] > race_date:
            break
        selected = period
    return selected


def racer_period_feature_values(row: Any) -> dict[str, Any]:
    period = row.get("_racer_period") if isinstance(row, dict) else None
    if not isinstance(period, dict):
        return {"has_racer_period_stats": 0}
    race_date = _date_value(row.get("race_date"))
    age_days = (
        (race_date - period["available_from"]).days
        if race_date is not None
        else 0
    )
    features: dict[str, Any] = {
        "has_racer_period_stats": 1,
        "period_stats_age_days": max(0, age_days),
    }
    for key in NUMERIC_FIELDS:
        features[f"has_period_{key}"] = int(key in period)
        if key in period:
            features[f"period_{key}"] = period[key]
    for key in CATEGORICAL_FIELDS:
        features[f"has_period_{key}"] = int(bool(period.get(key)))
        if period.get(key):
            features[f"period_{key}"] = period[key]
    return features


def _available_from(value: Any) -> date | None:
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        calculation_to = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None
    return calculation_to + timedelta(days=RELEASE_LAG_DAYS)


def _date_value(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
