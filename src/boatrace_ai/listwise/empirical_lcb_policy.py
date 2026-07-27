from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from math import isfinite
from typing import Any, Callable, Mapping, Protocol

from ..adaptive_allocation import allocate_adaptive_day
from .closing_odds import decision_odds


STAKE_YEN = 100
MAX_TICKETS_PER_RACE = 3
MAX_DAILY_TICKETS = 30


class EmpiricalEVArtifact(Protocol):
    ready: bool
    trained_through_date: str | None

    def predict(
        self,
        raw_ev: float,
        probability_rank: int | None = None,
        forecast_odds: float | None = None,
    ) -> Mapping[str, object]: ...

    def as_dict(self) -> Mapping[str, object]: ...


def _iso_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _verify_prior_only_artifact(
    races: list[dict[str, Any]],
    artifact: EmpiricalEVArtifact,
) -> None:
    if not races:
        return
    trained_through = artifact.trained_through_date
    if trained_through is None:
        if artifact.ready:
            raise ValueError("ready artifact must declare trained_through_date")
        return
    trained_date = _iso_date(trained_through, "trained_through_date")
    first_evaluation_date = min(
        _iso_date(race["race_date"], "race_date") for race in races
    )
    if trained_date >= first_evaluation_date:
        raise ValueError(
            "artifact must be trained strictly before every evaluation date"
        )


def _blended_probabilities(
    race: Mapping[str, Any],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
) -> dict[str, float]:
    probabilities = probability_blender(
        race["model_probabilities"],
        race["market_probabilities"],
        model_weight=float(calibrator["model_weight"]),
        temperature=float(calibrator["temperature"]),
    )
    if not probabilities:
        raise ValueError("probability_blender returned no probabilities")
    return {str(key): float(value) for key, value in probabilities.items()}


def _ranked_combinations(probabilities: Mapping[str, float]) -> list[str]:
    return sorted(
        probabilities,
        key=lambda combination: (-float(probabilities[combination]), combination),
    )


def policy_edge_records(
    races: list[dict[str, Any]],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
) -> list[dict[str, Any]]:
    """Build realized tickets for fitting an artifact used on later dates."""
    records: list[dict[str, Any]] = []
    for race in races:
        probabilities = _blended_probabilities(race, calibrator, probability_blender)
        odds = decision_odds(race)
        multipliers = race.get("historical_return_multipliers") or {}
        ranked = _ranked_combinations(probabilities)
        ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
        actual = str(race["actual_combination"])
        actual_payout_yen = int(race["actual_payout_yen"])
        for combination in ranked:
            probability = float(probabilities[combination])
            forecast_odds = float(odds[combination])
            return_multiplier = float(multipliers.get(combination, 1.0))
            raw_ev = probability * forecast_odds * return_multiplier
            hit = combination == actual
            records.append(
                {
                    "race_date": str(race["race_date"]),
                    "race_id": str(race["race_id"]),
                    "combination": combination,
                    "probability_rank": ranks[combination],
                    "probability": probability,
                    "forecast_odds": forecast_odds,
                    "return_multiplier": return_multiplier,
                    "raw_estimated_ev": raw_ev,
                    "gross_return_per_yen": (
                        actual_payout_yen / STAKE_YEN if hit else 0.0
                    ),
                    "hit": hit,
                    "snapshot_id": race.get("snapshot_id"),
                }
            )
    return records


def _eligible_candidate(
    race: dict[str, Any],
    combination: str,
    probability: float,
    raw_ev: float,
    point: float,
    lcb95: float,
    odds: float,
    return_multiplier: float,
) -> dict[str, Any]:
    actual = str(race["actual_combination"])
    return {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "jcd": race.get("jcd"),
        "rno": int(race.get("rno") or 0),
        "combination": combination,
        "probability": probability,
        "estimated_odds": odds,
        # Allocation must use the same conservative edge that admitted the ticket.
        "estimated_ev": lcb95,
        "raw_estimated_ev": raw_ev,
        "empirical_ev": point,
        "empirical_ev_lcb95": lcb95,
        "historical_return_multiplier": return_multiplier,
        "actual_combination": actual,
        "actual_payout_yen": int(race["actual_payout_yen"]),
        "hit": combination == actual,
        "real_odds_snapshot_id": race.get("snapshot_id"),
        "real_odds_captured_at": race.get("captured_at"),
        "real_odds_deadline_at": race.get("odds_deadline_at"),
        "real_odds_combinations": len(race["odds"]),
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(candidate["empirical_ev_lcb95"]),
        float(candidate["empirical_ev"]),
        float(candidate["probability"]),
        str(candidate["combination"]),
    )


def _race_candidates(
    race: dict[str, Any],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
    artifact: EmpiricalEVArtifact,
) -> list[dict[str, Any]]:
    probabilities = _blended_probabilities(race, calibrator, probability_blender)
    odds = decision_odds(race)
    multipliers = race.get("historical_return_multipliers") or {}
    candidates: list[dict[str, Any]] = []
    ranked = _ranked_combinations(probabilities)
    ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
    for combination in ranked:
        probability = probabilities[combination]
        price = float(odds[combination])
        return_multiplier = float(multipliers.get(combination, 1.0))
        raw_ev = float(probability) * price * return_multiplier
        prediction = artifact.predict(raw_ev, ranks[combination], price)
        point_value = prediction.get("empirical_ev")
        lcb_value = prediction.get("empirical_ev_lcb95")
        if point_value is None or lcb_value is None:
            continue
        point = float(point_value)
        lcb95 = float(lcb_value)
        if not isfinite(point) or not isfinite(lcb95):
            continue
        if point <= 1.0 or lcb95 <= 1.0:
            continue
        candidates.append(
            _eligible_candidate(
                race,
                str(combination),
                float(probability),
                raw_ev,
                point,
                lcb95,
                price,
                return_multiplier,
            )
        )
    return sorted(candidates, key=_candidate_sort_key, reverse=True)[
        :MAX_TICKETS_PER_RACE
    ]


def _candidate_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "race_id": str(candidate["race_id"]),
        "combination": str(candidate["combination"]),
        "probability": float(candidate["probability"]),
        "forecast_odds": float(candidate["estimated_odds"]),
        "return_multiplier": float(candidate["historical_return_multiplier"]),
        "raw_estimated_ev": float(candidate["raw_estimated_ev"]),
        "empirical_ev": float(candidate["empirical_ev"]),
        "empirical_ev_lcb95": float(candidate["empirical_ev_lcb95"]),
        "allocation_ev": float(candidate["estimated_ev"]),
    }


def simulate_empirical_lcb_policy(
    races: list[dict[str, Any]],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
    artifact: EmpiricalEVArtifact,
    daily_budget_yen: int,
) -> dict[str, Any]:
    """Use a pre-fitted prior-only artifact; current/future teachers are not accepted."""
    if daily_budget_yen <= 0:
        raise ValueError("daily_budget_yen must be positive")
    _verify_prior_only_artifact(races, artifact)
    races_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        races_by_day[str(race["race_date"])].append(race)

    daily: list[dict[str, Any]] = []
    cumulative_profit = peak_profit = max_drawdown = 0
    totals = {
        "eligible_tickets": 0,
        "tickets": 0,
        "hit_tickets": 0,
        "stake_yen": 0,
        "return_yen": 0,
    }
    for race_date in sorted(races_by_day):
        day_races = races_by_day[race_date]
        candidates: list[dict[str, Any]] = []
        if artifact.ready:
            for race in day_races:
                candidates.extend(
                    _race_candidates(race, calibrator, probability_blender, artifact)
                )
            candidates = sorted(
                candidates, key=_candidate_sort_key, reverse=True
            )[:MAX_DAILY_TICKETS]
        result = allocate_adaptive_day(
            race_date,
            candidates,
            {str(race["race_id"]) for race in day_races},
            daily_budget_yen=daily_budget_yen,
            fractional_kelly=0.25,
            max_daily_exposure_fraction=0.30,
            min_daily_exposure_fraction=0.0,
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_daily_tickets=MAX_DAILY_TICKETS,
            allocation_mode="kelly_floor",
            stake_granularity_yen=STAKE_YEN,
            min_stake_yen=STAKE_YEN,
        )
        result["eligible_candidate_audit"] = [
            _candidate_audit(candidate) for candidate in candidates
        ]
        cumulative_profit += int(result["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        result["cumulative_profit_yen"] = cumulative_profit
        daily.append(result)
        totals["eligible_tickets"] += len(candidates)
        for key in ("tickets", "hit_tickets", "stake_yen", "return_yen"):
            totals[key] += int(result[key])

    stake_yen = totals["stake_yen"]
    return_yen = totals["return_yen"]
    return {
        "status": "ready" if artifact.ready else "calibration_not_ready",
        "calibration": dict(artifact.as_dict()),
        "evaluation_days": len(daily),
        "evaluated_races": len(races),
        "eligible_days": sum(
            int(bool(day["eligible_candidate_audit"])) for day in daily
        ),
        "no_bet_days": sum(int(day["tickets"] == 0) for day in daily),
        **totals,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }
