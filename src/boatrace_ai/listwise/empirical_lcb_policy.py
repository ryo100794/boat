from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from math import isfinite
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..adaptive_allocation import allocate_adaptive_day
from ..bankroll_bootstrap import bootstrap_daily_roi
from ..discrete_log_allocation import candidate_with_settlements
from .closing_odds import decision_odds


STAKE_YEN = 100
MAX_TICKETS_PER_RACE = 3
MAX_DAILY_TICKETS = 30
MIN_PROMOTION_EVALUATION_DAYS = 30
MIN_PROMOTION_TICKETS = 50
MIN_PROMOTION_ROI = 1.05
MIN_PROMOTION_ROI_CI95_LOWER = 1.0
MIN_PROMOTION_PROBABILITY_ROI_ABOVE_ONE = 0.95


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


RankingProvider = Callable[
    [Mapping[str, Any], Mapping[str, float]], Sequence[str]
]


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


def _ranking_order(
    race: Mapping[str, Any],
    probabilities: Mapping[str, float],
    ranking_provider: RankingProvider | None,
) -> list[str]:
    if ranking_provider is None:
        return _ranked_combinations(probabilities)
    ranked = [str(value) for value in ranking_provider(race, probabilities)]
    expected = set(probabilities)
    if len(ranked) != len(expected) or set(ranked) != expected:
        raise ValueError(
            "ranking_provider must return every probability combination once"
        )
    return ranked


def _limited_ranking(
    race: Mapping[str, Any],
    probabilities: Mapping[str, float],
    ranking_provider: RankingProvider | None,
    max_rank: int | None,
) -> list[str]:
    ranked = _ranking_order(race, probabilities, ranking_provider)
    if max_rank is None:
        return ranked
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 1:
        raise ValueError("max_rank must be a positive integer")
    return ranked[:max_rank]


def race_settlement_rows(
    race: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    additive = race.get("settlements") or race.get("trifecta_settlements")
    rows: list[dict[str, Any]] = []
    if additive:
        for row in additive:
            combination = row.get("combination", row.get("actual_combination"))
            payout_yen = row.get("payout_yen", row.get("actual_payout_yen"))
            if combination is None or payout_yen is None:
                continue
            rows.append(
                {
                    "race_id": str(race["race_id"]),
                    "combination": str(combination),
                    "payout_yen": int(payout_yen),
                }
            )
    if not rows:
        rows.append(
            {
                "race_id": str(race["race_id"]),
                "combination": str(race["actual_combination"]),
                "payout_yen": int(race["actual_payout_yen"]),
            }
        )
    return tuple(rows)


def race_settlement_map(race: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(row["combination"]): int(row["payout_yen"])
        for row in race_settlement_rows(race)
    }


def policy_edge_records(
    races: list[dict[str, Any]],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
    ranking_provider: RankingProvider | None = None,
    max_rank: int | None = None,
) -> list[dict[str, Any]]:
    """Build realized tickets for fitting an artifact used on later dates."""
    records: list[dict[str, Any]] = []
    for race in races:
        probabilities = _blended_probabilities(race, calibrator, probability_blender)
        odds = decision_odds(race)
        multipliers = race.get("historical_return_multipliers") or {}
        ranked = _limited_ranking(
            race, probabilities, ranking_provider, max_rank
        )
        ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
        settlements = race_settlement_map(race)
        decision_time = (
            race.get("odds_deadline_at")
            or race.get("captured_at")
            or f"{race['race_date']}T00:00:00+09:00"
        )
        decision_time_source = (
            "odds_deadline_at" if race.get("odds_deadline_at")
            else "captured_at" if race.get("captured_at")
            else "conservative_operating_day_start"
        )
        settlement_time = race.get("settlement_time")
        settlement_time_source = "observed"
        if not settlement_time:
            settlement_time = f"{race['race_date']}T23:59:59+09:00"
            settlement_time_source = "conservative_operating_day_end"
        for combination in ranked:
            probability = float(probabilities[combination])
            forecast_odds = float(odds[combination])
            return_multiplier = float(multipliers.get(combination, 1.0))
            raw_ev = probability * forecast_odds * return_multiplier
            payout_yen = settlements.get(combination)
            hit = payout_yen is not None
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
                        float(payout_yen) / STAKE_YEN if hit else 0.0
                    ),
                    "hit": hit,
                    "decision_time": str(decision_time),
                    "decision_time_source": decision_time_source,
                    "settlement_time": str(settlement_time),
                    "settlement_time_source": settlement_time_source,
                    "candidate_population": "all_pregate_probability_ranked",
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
    decision = {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "jcd": race.get("jcd"),
        "rno": int(race.get("rno") or 0),
        "combination": combination,
        "probability": probability,
        "estimated_odds": odds,
        "estimated_ev": lcb95,
        "raw_estimated_ev": raw_ev,
        "empirical_ev": point,
        "empirical_ev_lcb95": lcb95,
        "historical_return_multiplier": return_multiplier,
        "real_odds_snapshot_id": race.get("snapshot_id"),
        "real_odds_captured_at": race.get("captured_at"),
        "real_odds_deadline_at": race.get("odds_deadline_at"),
        "real_odds_combinations": len(race["odds"]),
    }
    return candidate_with_settlements(decision, race_settlement_rows(race))


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
    ranking_provider: RankingProvider | None = None,
    max_rank: int | None = None,
    buy_threshold: float = 1.0,
    purchase_gate_enabled: bool = True,
    purchase_gate_denial_reason: str = "warmup_not_ready",
    max_tickets_per_race: int = MAX_TICKETS_PER_RACE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probabilities = _blended_probabilities(race, calibrator, probability_blender)
    odds = decision_odds(race)
    multipliers = race.get("historical_return_multipliers") or {}
    candidates: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    ranked = _limited_ranking(
        race, probabilities, ranking_provider, max_rank
    )
    ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
    for combination in ranked:
        probability = probabilities[combination]
        price = float(odds[combination])
        return_multiplier = float(multipliers.get(combination, 1.0))
        raw_ev = float(probability) * price * return_multiplier
        prediction = artifact.predict(raw_ev, ranks[combination], price)
        decision = {
            "race_id": str(race["race_id"]),
            "race_date": str(race["race_date"]),
            "combination": str(combination),
            "probability_rank": ranks[combination],
            "probability": float(probability),
            "forecast_odds": price,
            "raw_estimated_ev": raw_ev,
            "calibrated_roi": prediction.get("empirical_ev"),
            "calibrated_roi_lcb95": prediction.get("empirical_ev_lcb95"),
            "buy_threshold": buy_threshold,
            "purchase_gate_approved": False,
            "denial_reason": None,
        }
        for key in (
            "calibration_level",
            "cell_ready",
            "cell_support",
            "cell_support_days",
            "rank_support",
            "rank_support_days",
            "input_in_training_range",
            "input_in_local_block_range",
            "local_block_candidates",
            "local_block_candidate_days",
            "local_block_ess",
            "local_block_raw_ev_min",
            "local_block_raw_ev_max",
            "local_support_ready",
            "local_support_reasons",
            "context_local_support_ready",
            "context_local_support_reasons",
            "required_context_local_candidates",
            "required_context_local_candidate_days",
        ):
            if key in prediction:
                decision[key] = prediction[key]
        if not purchase_gate_enabled:
            decision["denial_reason"] = purchase_gate_denial_reason
            decisions.append(decision)
            continue
        if prediction.get("purchase_lcb95_available") is False:
            if prediction.get("cell_ready") is False:
                decision["denial_reason"] = "context_cell_not_ready"
            elif prediction.get("context_local_support_ready") is False:
                decision["denial_reason"] = "context_local_bin_not_ready"
            elif prediction.get("input_in_training_range") is False:
                decision["denial_reason"] = "outside_training_range"
            elif prediction.get("input_in_local_block_range") is False:
                decision["denial_reason"] = "outside_local_block_range"
            elif prediction.get("local_support_ready") is False:
                decision["denial_reason"] = "local_support_not_ready"
            else:
                decision["denial_reason"] = "calibrated_lcb_unavailable"
            decisions.append(decision)
            continue
        point_value = prediction.get("empirical_ev")
        lcb_value = prediction.get("empirical_ev_lcb95")
        if point_value is None or lcb_value is None:
            decision["denial_reason"] = "calibrated_value_missing"
            decisions.append(decision)
            continue
        point = float(point_value)
        lcb95 = float(lcb_value)
        if not isfinite(point) or not isfinite(lcb95):
            decision["denial_reason"] = "calibrated_value_nonfinite"
            decisions.append(decision)
            continue
        if point <= buy_threshold:
            decision["denial_reason"] = "calibrated_roi_not_above_one"
            decisions.append(decision)
            continue
        if lcb95 <= buy_threshold:
            decision["denial_reason"] = "calibrated_roi_lcb95_not_above_one"
            decisions.append(decision)
            continue
        decision["purchase_gate_approved"] = True
        decision["approval_reason"] = (
            "context_and_global_local_support_ready_and_calibrated_roi_lcb95_above_one"
        )
        decisions.append(decision)
        candidate = _eligible_candidate(
            race,
            str(combination),
            float(probability),
            raw_ev,
            point,
            lcb95,
            price,
            return_multiplier,
        )
        for key in (
            "calibration_level",
            "positive_return_days",
            "return_hhi",
            "cell_support",
            "cell_support_days",
            "rank_support",
            "rank_support_days",
            "context_local_support_ready",
            "required_context_local_candidates",
            "required_context_local_candidate_days",
        ):
            if key in prediction:
                candidate[key] = prediction[key]
        candidates.append(candidate)
    return (
        sorted(candidates, key=_candidate_sort_key, reverse=True)[
            :max_tickets_per_race
        ],
        decisions,
    )


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
        "calibration_level": candidate.get("calibration_level"),
        "positive_return_days": candidate.get("positive_return_days"),
        "return_hhi": candidate.get("return_hhi"),
        "cell_support": candidate.get("cell_support"),
        "cell_support_days": candidate.get("cell_support_days"),
        "rank_support": candidate.get("rank_support"),
        "rank_support_days": candidate.get("rank_support_days"),
        "context_local_support_ready": candidate.get(
            "context_local_support_ready"
        ),
        "required_context_local_candidates": candidate.get(
            "required_context_local_candidates"
        ),
        "required_context_local_candidate_days": candidate.get(
            "required_context_local_candidate_days"
        ),
    }


def empirical_bankroll_promotion_eligible(
    bankroll: Mapping[str, Any],
) -> bool:
    """Require enough days and day-clustered evidence before promotion."""
    return bool(
        bankroll.get("status") == "ready"
        and int(bankroll.get("evaluation_days") or 0)
        >= MIN_PROMOTION_EVALUATION_DAYS
        and int(bankroll.get("tickets") or 0) >= MIN_PROMOTION_TICKETS
        and float(bankroll.get("roi") or 0.0) >= MIN_PROMOTION_ROI
        and float(bankroll.get("roi_ci95_lower") or 0.0)
        > MIN_PROMOTION_ROI_CI95_LOWER
        and float(bankroll.get("probability_roi_above_one") or 0.0)
        >= MIN_PROMOTION_PROBABILITY_ROI_ABOVE_ONE
    )


def simulate_empirical_lcb_policy(
    races: list[dict[str, Any]],
    calibrator: Mapping[str, float],
    probability_blender: Callable[..., dict[str, float]],
    artifact: EmpiricalEVArtifact,
    daily_budget_yen: int,
    ranking_provider: RankingProvider | None = None,
    max_rank: int | None = None,
    *,
    allocation_mode: str = "kelly_floor",
    min_daily_exposure_fraction: float = 0.0,
    max_daily_exposure_fraction: float = 0.30,
    buy_threshold: float = 1.0,
    purchase_gate_enabled: bool = True,
    purchase_gate_denial_reason: str = "warmup_not_ready",
    max_tickets_per_race: int = MAX_TICKETS_PER_RACE,
) -> dict[str, Any]:
    """Use a pre-fitted prior-only artifact; current/future teachers are not accepted."""
    if daily_budget_yen <= 0:
        raise ValueError("daily_budget_yen must be positive")
    if not isfinite(float(buy_threshold)) or float(buy_threshold) < 1.0:
        raise ValueError("buy_threshold must be finite and at least gross ROI 1")
    if (
        isinstance(max_tickets_per_race, bool)
        or not isinstance(max_tickets_per_race, int)
        or max_tickets_per_race < 1
    ):
        raise ValueError("max_tickets_per_race must be a positive integer")
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
        candidate_decisions: list[dict[str, Any]] = []
        if artifact.ready:
            for race in day_races:
                race_candidates, race_decisions = _race_candidates(
                    race,
                    calibrator,
                    probability_blender,
                    artifact,
                    ranking_provider,
                    max_rank,
                    buy_threshold,
                    purchase_gate_enabled,
                    purchase_gate_denial_reason,
                    max_tickets_per_race,
                )
                candidates.extend(race_candidates)
                candidate_decisions.extend(race_decisions)
            candidates = sorted(
                candidates, key=_candidate_sort_key, reverse=True
            )[:MAX_DAILY_TICKETS]
        result = allocate_adaptive_day(
            race_date,
            candidates,
            {str(race["race_id"]) for race in day_races},
            daily_budget_yen=daily_budget_yen,
            fractional_kelly=0.25,
            max_daily_exposure_fraction=max_daily_exposure_fraction,
            min_daily_exposure_fraction=min_daily_exposure_fraction,
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_daily_tickets=MAX_DAILY_TICKETS,
            allocation_mode=allocation_mode,
            stake_granularity_yen=STAKE_YEN,
            min_stake_yen=STAKE_YEN,
        )
        result["eligible_candidate_audit"] = [
            _candidate_audit(candidate) for candidate in candidates
        ]
        result["candidate_decision_audit"] = candidate_decisions
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
    confidence = (
        bootstrap_daily_roi(daily)
        if daily
        else {
            "roi_ci95_lower": None,
            "probability_roi_above_one": None,
        }
    )
    return {
        "status": "ready" if artifact.ready else "calibration_not_ready",
        "allocation_policy": {
            "allocation_mode": allocation_mode,
            "min_daily_exposure_fraction": min_daily_exposure_fraction,
            "max_daily_exposure_fraction": max_daily_exposure_fraction,
        },
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
        "roi_ci95_lower": confidence.get("roi_ci95_lower"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }
