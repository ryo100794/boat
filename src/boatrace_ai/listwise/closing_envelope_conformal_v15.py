from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from itertools import permutations
from typing import Any


MODEL_NAME = "closing_envelope_conformal_v15"
METHOD = "selection_free_strict_prior_daily_q20_closing_ratio_v15"
TARGET_QUANTILE = 0.20
MIN_TRAINING_DAYS = 5
MIN_TRAINING_RACES = 30
COMBINATIONS_PER_RACE = 120

_CANONICAL_COMBINATIONS = tuple(
    "".join(str(lane) for lane in lanes)
    for lanes in permutations(range(1, 7), 3)
)
_CANONICAL_COMBINATION_SET = frozenset(_CANONICAL_COMBINATIONS)
_PREDICTION_FIELDS = (
    "predicted_closing_odds",
    "closing_forecasts",
    "forecast_closing_odds",
)
_ACTUAL_FIELDS = ("actual_closing_odds", "closing_odds")


def _iso_date(value: object, *, field: str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)") from exc


def _combination(value: object) -> str | None:
    text = str(value).replace("-", "")
    return text if text in _CANONICAL_COMBINATION_SET else None


def _mapping_from(
    race: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[Mapping[object, object] | None, str | None]:
    for field in fields:
        value = race.get(field)
        if isinstance(value, Mapping):
            return value, field
    return None, None


def _normalize_odds(
    values: Mapping[object, object] | None,
) -> tuple[dict[str, float], dict[str, int]]:
    normalized: dict[str, float] = {}
    invalid_keys = 0
    invalid_values = 0
    duplicate_keys = 0
    for raw_key, raw_value in (values or {}).items():
        key = _combination(raw_key)
        if key is None:
            invalid_keys += 1
            continue
        if key in normalized:
            duplicate_keys += 1
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            invalid_values += 1
            continue
        if isinstance(raw_value, bool) or not math.isfinite(number) or number <= 0.0:
            invalid_values += 1
            continue
        normalized[key] = number
    missing = len(_CANONICAL_COMBINATION_SET.difference(normalized))
    return normalized, {
        "source_values": len(values or {}),
        "valid_values": len(normalized),
        "missing_values": missing,
        "invalid_keys": invalid_keys,
        "invalid_values": invalid_values,
        "duplicate_keys": duplicate_keys,
    }


def _race_reason(
    predicted: Mapping[str, float],
    actual: Mapping[str, float],
    predicted_audit: Mapping[str, int],
    actual_audit: Mapping[str, int],
) -> str | None:
    if not predicted:
        return "missing_predicted_closing_odds"
    if not actual:
        return "missing_actual_closing_odds"
    if any(
        predicted_audit[key]
        for key in ("invalid_keys", "invalid_values", "duplicate_keys")
    ):
        return "invalid_predicted_closing_odds"
    if any(
        actual_audit[key]
        for key in ("invalid_keys", "invalid_values", "duplicate_keys")
    ):
        return "invalid_actual_closing_odds"
    if len(predicted) != COMBINATIONS_PER_RACE:
        return "incomplete_predicted_closing_odds"
    if len(actual) != COMBINATIONS_PER_RACE:
        return "incomplete_actual_closing_odds"
    if set(predicted) != set(actual):
        return "closing_combination_mismatch"
    return None


def extract_closing_ratio_observations_v15(
    races: Iterable[Mapping[str, Any]],
    *,
    evaluation_date: str | date,
) -> dict[str, Any]:
    """Extract all 120 closing ratios per complete strict-prior race.

    A race is atomic: one missing or invalid combination rejects the whole race.
    The function never reads outcomes, payouts, probabilities, or purchase choices.
    """
    evaluation = _iso_date(evaluation_date, field="evaluation_date")
    materialized: list[tuple[str, str, Mapping[str, Any]]] = []
    for index, race in enumerate(races):
        race_date = _iso_date(race.get("race_date"), field="race_date")
        if race_date >= evaluation:
            raise ValueError(
                "closing conformal observations must be strict-prior days: "
                f"race_date={race_date}, evaluation_date={evaluation}"
            )
        race_id = str(race.get("race_id") or "")
        materialized.append((race_date, race_id or f"<missing:{index}>", race))

    race_id_counts = Counter(race_id for _, race_id, _ in materialized)
    observations: list[dict[str, Any]] = []
    race_audit: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    for race_date, audit_race_id, race in sorted(
        materialized, key=lambda row: (row[0], row[1])
    ):
        raw_race_id = str(race.get("race_id") or "")
        predicted_raw, predicted_field = _mapping_from(race, _PREDICTION_FIELDS)
        actual_raw, actual_field = _mapping_from(race, _ACTUAL_FIELDS)
        predicted, predicted_audit = _normalize_odds(predicted_raw)
        actual, actual_audit = _normalize_odds(actual_raw)
        reason = None
        if not raw_race_id:
            reason = "missing_race_id"
        elif race_id_counts[audit_race_id] != 1:
            reason = "duplicate_race_id"
        else:
            reason = _race_reason(
                predicted, actual, predicted_audit, actual_audit
            )

        audit_row = {
            "race_date": race_date,
            "race_id": raw_race_id or None,
            "accepted": reason is None,
            "reason": reason,
            "predicted_field": predicted_field,
            "actual_field": actual_field,
            "predicted": predicted_audit,
            "actual": actual_audit,
        }
        race_audit.append(audit_row)
        if reason is not None:
            rejection_reasons[reason] += 1
            continue

        for combination in _CANONICAL_COMBINATIONS:
            observations.append({
                "race_date": race_date,
                "race_id": raw_race_id,
                "combination": combination,
                "predicted_closing_odds": predicted[combination],
                "actual_closing_odds": actual[combination],
                "closing_ratio": actual[combination] / predicted[combination],
            })

    accepted_races = sum(row["accepted"] for row in race_audit)
    input_races = len(race_audit)
    return {
        "observations": observations,
        "audit": {
            "evaluation_date": evaluation,
            "input_races": input_races,
            "accepted_races": accepted_races,
            "rejected_races": input_races - accepted_races,
            "expected_combinations_per_race": COMBINATIONS_PER_RACE,
            "accepted_observations": len(observations),
            "rejected_observation_slots": (
                input_races - accepted_races
            ) * COMBINATIONS_PER_RACE,
            "complete": input_races > 0 and accepted_races == input_races,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "races": race_audit,
        },
    }


def _finite_sample_lower_quantile(
    values: Iterable[float], *, quantile: float
) -> tuple[float, int, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values")
    rank = max(1, math.floor((len(ordered) + 1) * quantile + 1e-12))
    rank = min(rank, len(ordered))
    return ordered[rank - 1], rank, rank / (len(ordered) + 1)


def fit_closing_envelope_conformal_v15(
    races: Iterable[Mapping[str, Any]],
    *,
    evaluation_date: str | date,
    target_quantile: float = TARGET_QUANTILE,
    minimum_training_days: int = MIN_TRAINING_DAYS,
    minimum_training_races: int = MIN_TRAINING_RACES,
    minimum_training_observations: int | None = None,
) -> dict[str, Any]:
    """Fit a selection-free q20 closing-odds envelope on prior whole days."""
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must be between zero and one")
    if minimum_training_days < 1 or minimum_training_races < 1:
        raise ValueError("minimum training days and races must be positive")
    if minimum_training_observations is None:
        minimum_training_observations = (
            minimum_training_races * COMBINATIONS_PER_RACE
        )
    if minimum_training_observations < 1:
        raise ValueError("minimum_training_observations must be positive")

    extracted = extract_closing_ratio_observations_v15(
        races, evaluation_date=evaluation_date
    )
    observations = extracted["observations"]
    audit = extracted["audit"]
    by_day: dict[str, list[float]] = defaultdict(list)
    races_by_day: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        day = str(row["race_date"])
        by_day[day].append(float(row["closing_ratio"]))
        races_by_day[day].add(str(row["race_id"]))
    dates = sorted(by_day)
    daily_q20: dict[str, float] = {}
    daily_quantile_ranks: dict[str, int] = {}
    daily_observations: dict[str, int] = {}
    daily_races: dict[str, int] = {}
    for day in dates:
        value, rank, _ = _finite_sample_lower_quantile(
            by_day[day], quantile=target_quantile
        )
        daily_q20[day] = value
        daily_quantile_ranks[day] = rank
        daily_observations[day] = len(by_day[day])
        daily_races[day] = len(races_by_day[day])

    training_races = sum(daily_races.values())
    ready = (
        len(dates) >= minimum_training_days
        and training_races >= minimum_training_races
        and len(observations) >= minimum_training_observations
    )
    base = {
        "model_name": MODEL_NAME,
        "method": METHOD,
        "evaluation_date": audit["evaluation_date"],
        "target_quantile": target_quantile,
        "target_coverage": 1.0 - target_quantile,
        "selection_free": True,
        "combinations_per_race": COMBINATIONS_PER_RACE,
        "finite_sample_unit": "whole_prior_race_day",
        "training_days": len(dates),
        "training_races": training_races,
        "training_observations": len(observations),
        "training_dates": dates,
        "trained_through_date": dates[-1] if dates else None,
        "minimum_training_days": minimum_training_days,
        "minimum_training_races": minimum_training_races,
        "minimum_training_observations": minimum_training_observations,
        "daily_q20": daily_q20,
        "daily_quantile_ranks": daily_quantile_ranks,
        "daily_observations": daily_observations,
        "daily_races": daily_races,
        "missing_audit": audit,
        "ready": ready,
    }
    if not ready:
        return {
            **base,
            "haircut": None,
            "uncapped_haircut": None,
            "finite_sample_day_rank": None,
            "finite_sample_day_quantile": None,
            "reason": "insufficient_strict_prior_complete_closing_data",
        }

    uncapped, day_rank, day_quantile = _finite_sample_lower_quantile(
        daily_q20.values(), quantile=target_quantile
    )
    return {
        **base,
        "haircut": min(1.0, uncapped),
        "uncapped_haircut": uncapped,
        "finite_sample_day_rank": day_rank,
        "finite_sample_day_quantile": day_quantile,
        "ratio_min": min(float(row["closing_ratio"]) for row in observations),
        "ratio_max": max(float(row["closing_ratio"]) for row in observations),
        "reason": None,
    }


def apply_closing_envelope_haircut_v15(
    predicted_closing_odds: float | Mapping[str, object],
    artifact: Mapping[str, Any],
) -> float | dict[str, float]:
    """Apply a verified V15 haircut without mutating the forecast or artifact."""
    if artifact.get("method") != METHOD or not artifact.get("ready"):
        raise ValueError("a ready V15 closing envelope artifact is required")
    haircut = float(artifact.get("haircut"))
    if not math.isfinite(haircut) or not 0.0 < haircut <= 1.0:
        raise ValueError("artifact haircut must be finite and in (0, 1]")

    def apply(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("predicted closing odds must be finite and positive")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "predicted closing odds must be finite and positive"
            ) from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("predicted closing odds must be finite and positive")
        return number * haircut

    if isinstance(predicted_closing_odds, Mapping):
        return {
            str(key): apply(value)
            for key, value in sorted(
                predicted_closing_odds.items(), key=lambda item: str(item[0])
            )
        }
    return apply(predicted_closing_odds)


def artifact_fingerprint_v15(artifact: Mapping[str, Any]) -> str:
    """Return a stable fingerprint of the fitted values and input audit."""
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
