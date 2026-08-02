from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

from ..discrete_log_allocation import allocate_discrete_log_day
from ..chronological_bankroll import (
    simulate_chronological_bankroll_day,
    summarize_chronological_bankroll_days,
)
from ..fast_math import TRIFECTA_COMBINATIONS
from .four_head_nested_v22 import (
    FourHeadArtifact,
    LabeledRace,
    RacePrediction,
    artifact_fingerprint,
    predict_race,
    predict_purchase_hit_probabilities,
    predict_purchase_gross_payouts,
    prediction_fingerprint,
)


COMBINATIONS = tuple("-".join(map(str, value)) for value in TRIFECTA_COMBINATIONS)
INITIAL_BANKROLL_YEN = 10_000
STAKE_UNIT_YEN = 100
RESULT_AVAILABLE_AT_PROVENANCE = (
    "race_results.updated_at:max_complete_six_lane_result_conservative"
)


@dataclass(frozen=True)
class V22BankrollSettlement:
    """Post-decision official data plus the audit trail for the real T-5 input."""

    race_id: str
    decision_target_at: str
    odds_captured_at: str
    result_available_at: str
    official_winner_index: int
    official_closing_odds: tuple[float, ...]
    official_payout_yen: int
    result_available_at_source: str = RESULT_AVAILABLE_AT_PROVENANCE
    snapshot_id: int | None = None


def _timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_outer_boundary(
    artifact: FourHeadArtifact,
    races: Sequence[LabeledRace],
) -> None:
    if not races:
        raise ValueError("at least one outer race is required")
    training_ids = set(artifact.training_race_ids)
    seen: set[str] = set()
    previous: tuple[str, str] | None = None
    for race in races:
        decision = race.decision
        key = (decision.race_date, decision.race_id)
        if previous is not None and key <= previous:
            raise ValueError("outer races must be uniquely sorted by date and race_id")
        previous = key
        if decision.race_id in seen or decision.race_id in training_ids:
            raise ValueError("outer races overlap or contain duplicate race ids")
        seen.add(decision.race_id)
        if decision.race_date <= artifact.trained_through_date:
            raise ValueError("outer races must be strictly after artifact training")
        if len(decision.current_odds) != 120:
            raise ValueError("V22 bankroll evaluation requires 120 T-5 odds")
        odds = np.asarray(decision.current_odds, dtype=np.float64)
        if not np.isfinite(odds).all() or np.any(odds <= 1.0):
            raise ValueError("T-5 odds must be complete, finite, and above one")


def _settlement_map(
    races: Sequence[LabeledRace],
    settlements: Iterable[V22BankrollSettlement],
    *,
    max_t5_snapshot_age_seconds: float,
) -> tuple[dict[str, V22BankrollSettlement], dict[str, datetime]]:
    if max_t5_snapshot_age_seconds < 0 or not math.isfinite(
        max_t5_snapshot_age_seconds
    ):
        raise ValueError("max_t5_snapshot_age_seconds must be finite and non-negative")
    by_id: dict[str, V22BankrollSettlement] = {}
    decision_at: dict[str, datetime] = {}
    for item in settlements:
        if item.race_id in by_id:
            raise ValueError(f"duplicate settlement: {item.race_id}")
        target = _timestamp(item.decision_target_at, field="decision_target_at")
        captured = _timestamp(item.odds_captured_at, field="odds_captured_at")
        result_at = _timestamp(
            item.result_available_at, field="result_available_at"
        )
        age = (target - captured).total_seconds()
        if age < 0 or age > max_t5_snapshot_age_seconds:
            raise ValueError(f"unsafe T-5 odds snapshot: {item.race_id}")
        if result_at <= target:
            raise ValueError(f"settlement precedes the decision: {item.race_id}")
        if item.result_available_at_source != RESULT_AVAILABLE_AT_PROVENANCE:
            raise ValueError(f"unsafe result time provenance: {item.race_id}")
        if item.snapshot_id is not None and int(item.snapshot_id) < 1:
            raise ValueError("snapshot_id must be positive when provided")
        by_id[item.race_id] = item
        decision_at[item.race_id] = target

    race_ids = {race.decision.race_id for race in races}
    if set(by_id) != race_ids:
        missing = sorted(race_ids - set(by_id))
        extra = sorted(set(by_id) - race_ids)
        raise ValueError(
            f"settlement universe mismatch; missing={missing}, extra={extra}"
        )

    for race in races:
        item = by_id[race.decision.race_id]
        outcome = race.outcome
        if int(item.official_winner_index) != int(outcome.winner_index):
            raise ValueError(f"official winner mismatch: {item.race_id}")
        closing = np.asarray(item.official_closing_odds, dtype=np.float64)
        expected = np.asarray(outcome.closing_odds, dtype=np.float64)
        if (
            closing.shape != (120,)
            or not np.isfinite(closing).all()
            or np.any(closing <= 1.0)
            or not np.array_equal(closing, expected)
        ):
            raise ValueError(f"official closing odds mismatch: {item.race_id}")
        if isinstance(item.official_payout_yen, bool) or int(
            item.official_payout_yen
        ) < STAKE_UNIT_YEN:
            raise ValueError(f"invalid official payout: {item.race_id}")
    return by_id, decision_at


def _prediction_metrics(
    artifact: FourHeadArtifact,
    races: Sequence[LabeledRace],
    predictions: Sequence[RacePrediction],
) -> dict[str, Any]:
    trifecta_loss = winner_loss = 0.0
    trifecta_top1 = trifecta_top5 = winner_top1 = 0
    closing_errors: list[float] = []
    purchase_hit_loss = market_hit_loss = 0.0
    purchase_hit_top5 = purchase_probability_races = 0
    purchase_payout_log_errors: list[float] = []
    for race, prediction in zip(races, predictions, strict=True):
        probabilities = np.asarray(prediction.probabilities, dtype=np.float64)
        ranking = np.asarray(prediction.ranking_scores, dtype=np.float64)
        winner_index = int(race.outcome.winner_index)
        trifecta_loss -= math.log(max(float(probabilities[winner_index]), 1e-15))
        trifecta_top1 += int(int(np.argmax(probabilities)) == winner_index)
        trifecta_top5 += int(winner_index in np.argsort(-ranking)[:5])

        lane_probabilities = np.zeros(6, dtype=np.float64)
        for index, combination in enumerate(TRIFECTA_COMBINATIONS):
            lane_probabilities[combination[0] - 1] += probabilities[index]
        actual_lane = TRIFECTA_COMBINATIONS[winner_index][0] - 1
        winner_loss -= math.log(max(float(lane_probabilities[actual_lane]), 1e-15))
        winner_top1 += int(int(np.argmax(lane_probabilities)) == actual_lane)
        purchase_probabilities = predict_purchase_hit_probabilities(
            artifact, race.decision
        )
        if purchase_probabilities is not None:
            market_probabilities = 1.0 / np.asarray(
                race.decision.current_odds, dtype=np.float64
            )
            market_probabilities /= float(market_probabilities.sum())
            purchase_hit_loss -= math.log(
                max(float(purchase_probabilities[winner_index]), 1e-15)
            )
            market_hit_loss -= math.log(
                max(float(market_probabilities[winner_index]), 1e-15)
            )
            purchase_hit_top5 += int(
                winner_index in np.argsort(-purchase_probabilities)[:5]
            )
            purchase_probability_races += 1
        predicted_purchase_payouts = predict_purchase_gross_payouts(
            artifact, race.decision
        )
        if predicted_purchase_payouts is not None:
            purchase_payout_log_errors.extend(
                np.abs(
                    np.log(predicted_purchase_payouts)
                    - np.log(np.asarray(race.outcome.closing_odds, dtype=np.float64))
                ).tolist()
            )
        closing_errors.extend(
            np.abs(
                np.log(np.asarray(prediction.predicted_closing_odds))
                - np.log(np.asarray(race.outcome.closing_odds))
            ).tolist()
        )
    count = len(races)
    purchase_log_loss = (
        purchase_hit_loss / purchase_probability_races
        if purchase_probability_races
        else None
    )
    market_log_loss = (
        market_hit_loss / purchase_probability_races
        if purchase_probability_races
        else None
    )
    return {
        "races": count,
        "winner_log_loss": winner_loss / count,
        "winner_top1_accuracy": winner_top1 / count,
        "trifecta_log_loss": trifecta_loss / count,
        "trifecta_top1_accuracy": trifecta_top1 / count,
        "trifecta_top5_hit_rate": trifecta_top5 / count,
        "closing_odds_log_mae": float(np.mean(closing_errors)),
        "purchase_hit_log_loss": purchase_log_loss,
        "t5_market_log_loss": market_log_loss,
        "purchase_hit_log_loss_delta_vs_market": (
            purchase_log_loss - market_log_loss
            if purchase_log_loss is not None and market_log_loss is not None
            else None
        ),
        "purchase_hit_top5_rate": (
            purchase_hit_top5 / purchase_probability_races
            if purchase_probability_races
            else None
        ),
        "purchase_payout_log_mae": (
            float(np.mean(purchase_payout_log_errors))
            if purchase_payout_log_errors
            else None
        ),
    }


def _bankroll_stability(bankroll: dict[str, Any]) -> dict[str, Any]:
    stake = int(bankroll.get("stake_yen") or 0)
    returned = int(bankroll.get("return_yen") or 0)
    largest = int(bankroll.get("largest_hit_return_yen") or 0)
    square_sum = float(bankroll.get("hit_return_square_sum_yen2") or 0.0)
    evaluation_days = int(bankroll.get("race_days") or 0)
    winning_days = int(bankroll.get("winning_days") or 0)
    active_daily = [
        row
        for row in bankroll.get("daily") or []
        if int(row.get("stake_yen") or 0) > 0
    ]
    bootstrap_lower = None
    if active_daily:
        stakes = np.asarray(
            [float(row["stake_yen"]) for row in active_daily], dtype=np.float64
        )
        returns = np.asarray(
            [float(row["return_yen"]) for row in active_daily], dtype=np.float64
        )
        rng = np.random.default_rng(20260802)
        sampled = rng.integers(0, len(active_daily), size=(20_000, len(active_daily)))
        bootstrap_lower = float(
            np.quantile(returns[sampled].sum(axis=1) / stakes[sampled].sum(axis=1), 0.05)
        )
    return {
        "evaluation_days": evaluation_days,
        "tickets": int(bankroll.get("tickets") or 0),
        "hit_tickets": int(bankroll.get("hit_tickets") or 0),
        "winning_days": winning_days,
        "profitable_day_fraction": (
            winning_days / evaluation_days if evaluation_days else None
        ),
        "roi_without_largest_hit": (
            (returned - largest) / stake if stake else None
        ),
        "largest_hit_return_share": largest / returned if returned else None,
        "effective_hit_count": (
            returned * returned / square_sum if square_sum else 0.0
        ),
        "daily_cluster_bootstrap_roi_lower_95": bootstrap_lower,
    }


def _promotion_gate(
    bankroll: dict[str, Any], stability: dict[str, Any]
) -> dict[str, bool]:
    return {
        "minimum_30_evaluation_days": stability["evaluation_days"] >= 30,
        "minimum_1000_evaluated_races": int(
            bankroll.get("evaluated_races") or 0
        ) >= 1_000,
        "minimum_100_tickets": stability["tickets"] >= 100,
        "positive_profit": int(bankroll.get("profit_yen") or 0) > 0,
        "roi_above_one": float(bankroll.get("roi") or 0.0) > 1.0,
        "roi_without_largest_hit_above_one": float(
            stability["roi_without_largest_hit"] or 0.0
        ) > 1.0,
        "daily_bootstrap_lower_above_one": float(
            stability["daily_cluster_bootstrap_roi_lower_95"] or 0.0
        ) > 1.0,
        "minimum_20_effective_hits": float(
            stability["effective_hit_count"] or 0.0
        ) >= 20.0,
        "minimum_60_percent_profitable_days": float(
            stability["profitable_day_fraction"] or 0.0
        ) >= 0.60,
    }


def evaluate_four_head_v22_bankroll(
    artifact: FourHeadArtifact,
    outer_races: Iterable[LabeledRace],
    settlements: Iterable[V22BankrollSettlement],
    *,
    initial_bankroll_yen: int = INITIAL_BANKROLL_YEN,
    stake_unit_yen: int = STAKE_UNIT_YEN,
    max_t5_snapshot_age_seconds: float = 300.0,
) -> dict[str, Any]:
    """Evaluate frozen V22 outer predictions with the production bankroll engine."""

    if initial_bankroll_yen != INITIAL_BANKROLL_YEN:
        raise ValueError("formal V22 bankroll evaluation requires JPY10000 per day")
    if stake_unit_yen != STAKE_UNIT_YEN:
        raise ValueError("formal V22 bankroll evaluation requires JPY100 units")
    races = tuple(outer_races)
    _validate_outer_boundary(artifact, races)

    # Freeze every decision before settlement data is validated or made available.
    predictions = tuple(predict_race(artifact, race.decision) for race in races)
    frozen_sha256 = prediction_fingerprint(predictions)
    by_settlement, decision_times = _settlement_map(
        races,
        settlements,
        max_t5_snapshot_age_seconds=max_t5_snapshot_age_seconds,
    )

    races_by_date: dict[str, list[LabeledRace]] = defaultdict(list)
    predictions_by_id = {
        prediction.race_id: prediction for prediction in predictions
    }
    for race in races:
        races_by_date[race.decision.race_date].append(race)

    daily: list[dict[str, Any]] = []
    for race_date in sorted(races_by_date):
        day_races = races_by_date[race_date]
        candidates: list[dict[str, Any]] = []
        schedule: list[dict[str, Any]] = []
        settlement_events: list[dict[str, Any]] = []
        for race in day_races:
            race_id = race.decision.race_id
            prediction = predictions_by_id[race_id]
            settlement = by_settlement[race_id]
            decision_at = decision_times[race_id].isoformat()
            schedule.append({"race_id": race_id, "decision_at": decision_at})
            settlement_events.append(
                {
                    "race_id": race_id,
                    "result_available_at": settlement.result_available_at,
                    "result_available_at_source": settlement.result_available_at_source,
                    "payouts": {
                        COMBINATIONS[settlement.official_winner_index]: int(
                            settlement.official_payout_yen
                        )
                    },
                }
            )
            for index in prediction.selected_indices:
                t5_odds = float(race.decision.current_odds[index])
                learned_net_return = float(prediction.purchase_scores[index])
                learned_net_return = min(
                    max(learned_net_return, 0.0), t5_odds - 1.0
                )
                learned_ev = 1.0 + learned_net_return
                allocation_probability = learned_ev / t5_odds
                candidates.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "combination": COMBINATIONS[index],
                        "probability": allocation_probability,
                        "estimated_odds": t5_odds,
                        "estimated_ev": learned_ev,
                        "purchase_score": float(prediction.purchase_scores[index]),
                        "decision_at": decision_at,
                        "odds_source": "official_trifecta_t5_snapshot",
                        "real_odds_snapshot_id": settlement.snapshot_id,
                        "real_odds_captured_at": settlement.odds_captured_at,
                        # Legacy allocator field; this is the exact T-5 target.
                        "real_odds_deadline_at": settlement.decision_target_at,
                        "real_odds_decision_target_at": (
                            settlement.decision_target_at
                        ),
                        "real_odds_combinations": 120,
                    }
                )
        day_result = simulate_chronological_bankroll_day(
            race_date,
            candidates,
            {race.decision.race_id for race in day_races},
            settlement_events=settlement_events,
            initial_bankroll_yen=initial_bankroll_yen,
            daily_stake_limit_fraction=1.0,
            max_decision_exposure_fraction=0.30,
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_tickets_per_race=2,
            max_daily_tickets=None,
            schedule=schedule,
            stake_granularity_yen=stake_unit_yen,
            allocate_day=allocate_discrete_log_day,
            allocation_method="v22_four_head_chronological_discrete_log",
        )
        daily.append(day_result)

    bankroll = summarize_chronological_bankroll_days(daily)
    stability = _bankroll_stability(bankroll)
    promotion_gate = _promotion_gate(bankroll, stability)
    metrics = _prediction_metrics(artifact, races, predictions)
    return {
        "model_key": artifact.model_key,
        "artifact_sha256": artifact_fingerprint(artifact),
        "frozen_prediction_sha256": frozen_sha256,
        "evaluation_role": "formal_production_equivalent_bankroll_outer_only",
        "outer_outcomes_used_for_fit_selection_or_threshold": False,
        "prediction_metrics": metrics,
        **metrics,
        "bankroll": bankroll,
        "roi": bankroll["roi"],
        "stake_yen": bankroll["stake_yen"],
        "return_yen": bankroll["return_yen"],
        "profit_yen": bankroll["profit_yen"],
        "max_drawdown_yen": bankroll["max_drawdown_yen"],
        **stability,
        "promotion_gate": promotion_gate,
        "promotion_eligible": all(promotion_gate.values()),
        "daily": bankroll["daily"],
        "policy": {
            "initial_bankroll_yen_per_day": initial_bankroll_yen,
            "stake_granularity_yen": stake_unit_yen,
            "zero_or_more_tickets_per_race": True,
            "profit_reinvestment": True,
            "decision_odds": "complete_official_trifecta_snapshot_at_T-5",
            "official_closing_odds_role": "settlement_metrics_only",
            "official_payout_role": "post_allocation_settlement_only",
            "result_available_at_source": RESULT_AVAILABLE_AT_PROVENANCE,
            "allocation_api": (
                "simulate_chronological_bankroll_day+allocate_discrete_log_day"
            ),
            "allocation_signal": "learned_purchase_head_expected_unit_return",
        },
        "diagnostic_unit_roi_is_formal_roi": False,
        "real_betting_enabled": False,
    }
