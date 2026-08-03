from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, time
from typing import Any, Iterable, Mapping

from boatrace_ai.bankroll_bootstrap import bootstrap_daily_roi
from boatrace_ai.listwise.cluster_bootstrap import paired_cluster_mean_bootstrap
from boatrace_ai.listwise.closing_odds import decision_odds
from boatrace_ai.listwise.market_calibration import bankroll_reliability_metrics
from boatrace_ai.listwise.market_offset_calibration import (
    fit_market_offset_calibration,
)
from boatrace_ai.multinomial_kelly import (
    MultinomialKellyCandidate,
    allocate_multinomial_kelly,
)


MIN_TRAINING_DAYS = 7
MIN_TRAINING_RACES = 500
STARTING_BANKROLL_YEN = 10_000
DAILY_STAKE_CAP_YEN = 3_000
RACE_STAKE_CAP_YEN = 500
STAKE_UNIT_YEN = 100
EXPECTED_COMBINATIONS = 120
EPSILON = 1e-300


def evaluate_market_kelly_challenger(
    races: Iterable[Mapping[str, Any]],
    *,
    regularization: float = 1.0,
    evaluation_dates: Iterable[str] | None = None,
    select_regularization: bool = False,
    odds_safety_factor: float = 1.0,
    required_ticket_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate strict-prior market offsets with exact discrete Kelly stakes.

    Each date is an independent 10,000 yen bankroll simulation. Settled returns
    are available to later races on that date, while gross stake is capped at
    3,000 yen per date and 500 yen per race.
    """

    calibrated_races, calibration = attach_prequential_market_offsets(
        races,
        regularization=regularization,
        select_regularization=select_regularization,
    )
    return evaluate_attached_market_kelly_challenger(
        calibrated_races,
        calibration=calibration,
        evaluation_dates=evaluation_dates,
        odds_safety_factor=odds_safety_factor,
        required_ticket_count=required_ticket_count,
    )


def evaluate_attached_market_kelly_challenger(
    calibrated_races: Iterable[Mapping[str, Any]],
    *,
    calibration: Mapping[str, Any],
    evaluation_dates: Iterable[str] | None = None,
    odds_safety_factor: float = 1.0,
    required_ticket_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate already prequentially calibrated races without refitting."""
    odds_safety_factor = _odds_safety_factor(odds_safety_factor)
    required_ticket_count = _required_ticket_count(required_ticket_count)

    calibrated_races = [dict(race) for race in calibrated_races]
    requested_dates = (
        None
        if evaluation_dates is None
        else {_iso_day(value, "evaluation_date") for value in evaluation_dates}
    )
    evaluation_races = [
        race
        for race in calibrated_races
        if requested_dates is None or str(race["race_date"]) in requested_dates
    ]
    daily = _simulate_daily(
        evaluation_races,
        odds_safety_factor=odds_safety_factor,
        required_ticket_count=required_ticket_count,
    )
    evaluated_races = len(evaluation_races)
    stake_yen = sum(row["stake_yen"] for row in daily)
    return_yen = sum(row["return_yen"] for row in daily)
    reliability = bankroll_reliability_metrics(
        daily,
        evaluated_races=evaluated_races,
    )
    bootstrap = (
        bootstrap_daily_roi(daily)
        if daily
        else {
            "days": 0,
            "samples": 20_000,
            "valid_samples": 0,
            "roi_ci95_lower": None,
            "probability_roi_above_one": None,
        }
    )
    log_loss = _log_loss_comparison(evaluation_races)
    probability_calibration = _purchase_probability_calibration(daily)
    data_quality = _prospective_data_quality(evaluation_races)
    winning_days = sum(int(row["profit_yen"] > 0) for row in daily)
    purchase_days = sum(int(row["stake_yen"] > 0) for row in daily)
    return {
        "challenger": "market_offset_discrete_multinomial_kelly",
        "policy": {
            "starting_bankroll_yen": STARTING_BANKROLL_YEN,
            "daily_stake_cap_yen": DAILY_STAKE_CAP_YEN,
            "race_stake_cap_yen": RACE_STAKE_CAP_YEN,
            "stake_unit_yen": STAKE_UNIT_YEN,
            "outcomes_per_race": EXPECTED_COMBINATIONS,
            "zero_bet_allowed": True,
            "profit_reinvested_within_day": True,
            "odds_safety_factor": odds_safety_factor,
            "required_ticket_count": required_ticket_count,
        },
        "evaluation_days": len(daily),
        "evaluation_dates": sorted({row["race_date"] for row in daily}),
        "evaluated_races": evaluated_races,
        "tickets": sum(row["tickets"] for row in daily),
        "hit_tickets": sum(row["hit_tickets"] for row in daily),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": winning_days,
        "purchase_days": purchase_days,
        "profitable_day_fraction": (
            winning_days / len(daily) if daily else None
        ),
        "max_drawdown_yen": max(
            (row["max_drawdown_yen"] for row in daily), default=0
        ),
        "max_drawdown_rate": max(
            (row["max_drawdown_rate"] for row in daily), default=0.0
        ),
        "calibration": dict(calibration),
        "log_loss": log_loss,
        "purchase_probability_calibration": probability_calibration,
        "data_quality": data_quality,
        **reliability,
        "reliability": reliability,
        "edge_diagnostics": _edge_diagnostics(evaluation_races),
        "bootstrap": bootstrap,
        "promotion_gate": _promotion_gate(
            daily,
            reliability,
            bootstrap,
            market=log_loss,
            probability_calibration=probability_calibration,
            data_quality=data_quality,
            evaluated_races=evaluated_races,
        ),
        "daily": daily,
    }


def _promotion_gate(
    daily: list[dict[str, Any]],
    reliability: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    market: Mapping[str, Any],
    probability_calibration: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    evaluated_races: int,
) -> dict[str, Any]:
    tickets = sum(int(row["tickets"]) for row in daily)
    stake_yen = sum(int(row["stake_yen"]) for row in daily)
    return_yen = sum(int(row["return_yen"]) for row in daily)
    winning_days = sum(int(row["profit_yen"] > 0) for row in daily)
    profitable_day_fraction = winning_days / len(daily) if daily else 0.0
    effective_hit_count = float(
        reliability.get("effective_hit_count") or 0.0
    )
    log_loss_confidence = float(
        market.get("challenger_improvement_confidence") or 0.0
    )
    top5_confidence = float(
        market.get("challenger_top5_improvement_confidence") or 0.0
    )
    probability_calibration_pvalue = float(
        probability_calibration.get("probability_at_most_observed_hits") or 0.0
    )
    gates = {
        "minimum_clean_evaluation_days": 30,
        "minimum_evaluated_races": 1_000,
        "minimum_tickets": 200,
        "minimum_hits": 20,
        "minimum_effective_hits": 20.0,
        "minimum_profitable_day_fraction": 0.60,
        "minimum_market_confidence": 0.95,
        "minimum_selected_probability_calibration_pvalue": 0.05,
        "sample_size_pass": (
            len(daily) >= 30
            and tickets >= 200
            and sum(int(row["hit_tickets"]) for row in daily) >= 20
        ),
        "clean_evaluation_days_pass": len(daily) >= 30,
        "evaluated_races_pass": evaluated_races >= 1_000,
        "effective_hit_count_pass": effective_hit_count >= 20.0,
        "profitable_day_fraction_pass": profitable_day_fraction >= 0.60,
        "market_log_loss_confidence_pass": log_loss_confidence >= 0.95,
        "market_top5_confidence_pass": top5_confidence >= 0.95,
        "selected_probability_not_overconfident": (
            probability_calibration_pvalue >= 0.05
        ),
        "no_lookahead_pass": int(
            data_quality.get("lookahead_violations") or 0
        ) == 0,
        "operational_data_errors_zero": int(
            data_quality.get("operational_data_errors") or 0
        ) == 0,
        "roi_pass": bool(stake_yen and return_yen > stake_yen),
        "largest_hit_excluded_roi_pass": bool(
            float(reliability.get("roi_without_largest_hit") or 0.0) > 1.0
        ),
        "bootstrap_lower_95_pass": bool(
            float(bootstrap.get("roi_ci95_lower") or 0.0) > 1.0
        ),
        "bootstrap_probability_pass": bool(
            float(bootstrap.get("probability_roi_above_one") or 0.0) >= 0.95
        ),
        "bootstrap_status": "evaluated" if bootstrap.get("valid_samples") else "undefined_no_stake",
    }
    gates["pass"] = all(
        gates[key]
        for key in (
            "sample_size_pass",
            "clean_evaluation_days_pass",
            "evaluated_races_pass",
            "effective_hit_count_pass",
            "profitable_day_fraction_pass",
            "market_log_loss_confidence_pass",
            "market_top5_confidence_pass",
            "selected_probability_not_overconfident",
            "no_lookahead_pass",
            "operational_data_errors_zero",
            "roi_pass",
            "largest_hit_excluded_roi_pass",
            "bootstrap_lower_95_pass",
            "bootstrap_probability_pass",
        )
    )
    return gates


def _prospective_data_quality(
    races: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(races)
    duplicate_race_ids = len(rows) - len({
        str(row.get("race_id") or "") for row in rows
    })
    lookahead_violations = 0
    market_fallback_races = 0
    closing_policy_fallback_races = 0
    for race in rows:
        race_date = _iso_day(race.get("race_date"), "race_date")
        audit = race.get("_market_kelly_calibration") or {}
        if not bool(audit.get("ready")):
            market_fallback_races += 1
        trained_through = audit.get("trained_through_date")
        if trained_through is not None and _iso_day(
            trained_through, "trained_through_date"
        ) >= race_date:
            lookahead_violations += 1
        if bool(race.get("closing_odds_policy_fallback")):
            closing_policy_fallback_races += 1
    operational_data_errors = (
        max(0, duplicate_race_ids)
        + market_fallback_races
        + closing_policy_fallback_races
    )
    return {
        "evaluated_races": len(rows),
        "duplicate_race_ids": max(0, duplicate_race_ids),
        "market_calibration_fallback_races": market_fallback_races,
        "closing_policy_fallback_races": closing_policy_fallback_races,
        "lookahead_violations": lookahead_violations,
        "operational_data_errors": operational_data_errors,
        "pass": operational_data_errors == 0 and lookahead_violations == 0,
    }


def _purchase_probability_calibration(
    daily: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities: list[float] = []
    observed_hits = 0
    for day in daily:
        for decision in day.get("decisions") or []:
            allocations = decision.get("allocations") or []
            if not allocations:
                continue
            selected_probability = sum(
                float(row["probability"]) for row in allocations
            )
            if not math.isfinite(selected_probability) or not (
                0.0 <= selected_probability <= 1.0 + 1e-12
            ):
                raise ValueError("selected race probability must be in [0, 1]")
            probabilities.append(min(1.0, selected_probability))
            observed_hits += int(int(decision.get("actual_stake_yen") or 0) > 0)
    return {
        "selected_races": len(probabilities),
        "observed_hits": observed_hits,
        "expected_hits": sum(probabilities),
        "probability_at_most_observed_hits": _poisson_binomial_cdf(
            probabilities, observed_hits
        ),
        "method": (
            "exact_poisson_binomial_lower_tail_over_disjoint_race_selections"
        ),
    }


def _poisson_binomial_cdf(
    probabilities: Iterable[float], observed: int
) -> float:
    values = [float(value) for value in probabilities]
    if observed < 0:
        return 0.0
    distribution = [1.0] + [0.0] * len(values)
    for index, probability in enumerate(values, start=1):
        for hits in range(index, 0, -1):
            distribution[hits] = (
                distribution[hits] * (1.0 - probability)
                + distribution[hits - 1] * probability
            )
        distribution[0] *= 1.0 - probability
    return min(1.0, max(0.0, sum(distribution[: observed + 1])))


def attach_prequential_market_offsets(
    races: Iterable[Mapping[str, Any]],
    *,
    regularization: float = 1.0,
    select_regularization: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach one strictly-prior offset prediction or an explicit market fallback."""

    source = [dict(race) for race in races]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in source:
        grouped[_iso_day(race.get("race_date"), "race_date")].append(race)

    prior_records: list[dict[str, Any]] = []
    prior_dates: set[str] = set()
    annotated: list[dict[str, Any]] = []
    day_audits: list[dict[str, Any]] = []
    for prediction_day in sorted(grouped):
        training_days = len(prior_dates)
        training_races = len(prior_records)
        artifact = None
        regularization_selection = None
        fallback_reason = None
        if training_days < MIN_TRAINING_DAYS:
            fallback_reason = "insufficient_strictly_prior_days"
        elif training_races < MIN_TRAINING_RACES:
            fallback_reason = "insufficient_strictly_prior_races"
        else:
            selected_regularization = regularization
            if select_regularization:
                from .market_offset_selection import (
                    select_market_offset_regularization,
                )
                regularization_selection = select_market_offset_regularization(
                    prior_records,
                    prediction_date=prediction_day,
                )
                selected_regularization = float(
                    regularization_selection["selected_regularization"]
                )
            artifact = fit_market_offset_calibration(
                prior_records,
                prediction_date=prediction_day,
                regularization=selected_regularization,
                min_training_races=MIN_TRAINING_RACES,
            )
            _audit_artifact_boundary(artifact, prediction_day)
            artifact_days = len(getattr(artifact, "training_dates", ()))
            artifact_races = int(getattr(artifact, "training_races", 0))
            if not getattr(artifact, "fitted", False):
                fallback_reason = getattr(artifact, "fallback_reason", None) or "not_fitted"
            elif not getattr(artifact, "converged", False):
                fallback_reason = "not_converged"
            elif artifact_days < MIN_TRAINING_DAYS:
                fallback_reason = "artifact_has_insufficient_training_days"
            elif artifact_races < MIN_TRAINING_RACES:
                fallback_reason = "artifact_has_insufficient_training_races"

        ready = artifact is not None and fallback_reason is None
        trained_through_date = (
            getattr(artifact, "trained_through_date", None)
            if artifact is not None
            else max(prior_dates, default=None)
        )
        audit = {
            "prediction_date": prediction_day,
            "mode": "market_offset" if ready else "market_only_fallback",
            "ready": ready,
            "fallback_reason": fallback_reason,
            "trained_through_date": trained_through_date,
            "training_days": (
                len(getattr(artifact, "training_dates", ()))
                if artifact is not None
                else training_days
            ),
            "training_races": (
                int(getattr(artifact, "training_races", 0))
                if artifact is not None
                else training_races
            ),
            "fitted": bool(getattr(artifact, "fitted", False)),
            "converged": bool(getattr(artifact, "converged", False)),
            "artifact": (
                artifact.as_dict()
                if artifact is not None and callable(getattr(artifact, "as_dict", None))
                else None
            ),
            "regularization_selection": regularization_selection,
        }
        day_audits.append(audit)

        for race in grouped[prediction_day]:
            item = dict(race)
            if ready:
                prediction = artifact.predict(
                    item["model_probabilities"],
                    item["market_probabilities"],
                    decision_odds(item),
                    prediction_date=prediction_day,
                )
                probabilities = _normalized(
                    prediction.probabilities,
                    "calibrated probabilities",
                )
            else:
                probabilities = _normalized(
                    item["market_probabilities"],
                    "market probabilities",
                )
            item["_policy_calibrated_probabilities"] = probabilities
            item["_market_kelly_calibration"] = dict(audit)
            annotated.append(item)

        for race in grouped[prediction_day]:
            prior_records.append(_calibration_record(race, prediction_day))
        prior_dates.add(prediction_day)

    annotated.sort(key=_race_order_key)
    ready_days = sum(bool(row["ready"]) for row in day_audits)
    ready_races = sum(
        len(grouped[row["prediction_date"]])
        for row in day_audits
        if row["ready"]
    )
    return annotated, {
        "minimum_training_days": MIN_TRAINING_DAYS,
        "minimum_training_races": MIN_TRAINING_RACES,
        "ready_days": ready_days,
        "fallback_days": len(day_audits) - ready_days,
        "ready_races": ready_races,
        "fallback_races": len(annotated) - ready_races,
        "days": day_audits,
    }


def _calibration_record(race: Mapping[str, Any], race_day: str) -> dict[str, Any]:
    return {
        "race_id": race.get("race_id"),
        "race_date": race_day,
        "jcd": race.get("jcd"),
        "rno": race.get("rno", race.get("race_no")),
        "model_probabilities": race["model_probabilities"],
        "market_probabilities": race["market_probabilities"],
        "forecast_odds": decision_odds(dict(race)),
        "actual_combination": race["actual_combination"],
    }


def _audit_artifact_boundary(artifact: Any, prediction_day: str) -> None:
    trained_through = getattr(artifact, "trained_through_date", None)
    if trained_through is not None and _iso_day(
        trained_through, "trained_through_date"
    ) >= prediction_day:
        raise ValueError("market offset artifact is not strictly prior to holdout")
    for teacher_day in getattr(artifact, "training_dates", ()):
        if _iso_day(teacher_day, "training_date") >= prediction_day:
            raise ValueError("market offset artifact contains same-day or future teachers")


def _simulate_daily(
    races: list[dict[str, Any]],
    *,
    odds_safety_factor: float = 1.0,
    required_ticket_count: int | None = None,
) -> list[dict[str, Any]]:
    odds_safety_factor = _odds_safety_factor(odds_safety_factor)
    required_ticket_count = _required_ticket_count(required_ticket_count)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        grouped[_iso_day(race.get("race_date"), "race_date")].append(race)

    daily: list[dict[str, Any]] = []
    for race_day in sorted(grouped):
        bankroll = STARTING_BANKROLL_YEN
        peak = bankroll
        max_drawdown = 0
        max_drawdown_rate = 0.0
        staked = 0
        returned = 0
        tickets = 0
        hit_tickets = 0
        races_bet = 0
        hit_races = 0
        largest_hit = 0
        hit_return_square_sum = 0
        decisions: list[dict[str, Any]] = []
        ordered = sorted(grouped[race_day], key=_race_order_key)
        for race in ordered:
            probabilities = _normalized(
                race["_policy_calibrated_probabilities"],
                "policy probabilities",
            )
            odds = decision_odds(race)
            if set(probabilities) != set(odds):
                raise ValueError("probability and decision odds keys must match")
            if len(probabilities) != EXPECTED_COMBINATIONS:
                raise ValueError(
                    f"exactly {EXPECTED_COMBINATIONS} race outcomes are required"
                )
            candidates = [
                MultinomialKellyCandidate(
                    selection=selection,
                    probability=probabilities[selection],
                    final_odds=(
                        _positive_float(odds[selection], "decision odds")
                        / odds_safety_factor
                    ),
                )
                for selection in sorted(probabilities)
            ]
            bankroll_before = bankroll
            allocation = allocate_multinomial_kelly(
                bankroll,
                candidates,
                stake_unit_yen=STAKE_UNIT_YEN,
                race_cap_yen=RACE_STAKE_CAP_YEN,
                daily_cap_yen=DAILY_STAKE_CAP_YEN,
                daily_staked_yen=staked,
            )
            purchased = allocation.purchased
            if (
                required_ticket_count is not None
                and len(purchased) != required_ticket_count
            ):
                purchased = ()
            stake = sum(row.stake_yen for row in purchased)
            actual = str(race["actual_combination"])
            if actual not in probabilities:
                raise ValueError(f"actual combination {actual!r} is missing")
            actual_stake = next(
                (row.stake_yen for row in purchased if row.selection == actual),
                0,
            )
            payout_per_100 = _nonnegative_float(
                race["actual_payout_yen"],
                "actual_payout_yen",
            )
            race_return = int(round(actual_stake * payout_per_100 / STAKE_UNIT_YEN))
            bankroll = bankroll - stake + race_return
            staked += stake
            returned += race_return
            tickets += len(purchased)
            if purchased:
                races_bet += 1
            if actual_stake:
                hit_tickets += 1
                hit_races += 1
                largest_hit = max(largest_hit, race_return)
                hit_return_square_sum += race_return * race_return
            peak = max(peak, bankroll)
            drawdown = peak - bankroll
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_rate = max(
                max_drawdown_rate,
                drawdown / peak if peak else 0.0,
            )
            decisions.append({
                "race_id": race.get("race_id"),
                "jcd": race.get("jcd"),
                "rno": race.get("rno", race.get("race_no")),
                "race_time": _race_time_value(race),
                "bankroll_before_yen": bankroll_before,
                "stake_yen": stake,
                "return_yen": race_return,
                "bankroll_after_yen": bankroll,
                "tickets": len(purchased),
                "actual_combination": actual,
                "actual_stake_yen": actual_stake,
                "allocations": [
                    {
                        "selection": row.selection,
                        "units": row.units,
                        "stake_yen": row.stake_yen,
                        "probability": row.probability,
                        "forecast_odds": _positive_float(
                            odds[row.selection], "decision odds"
                        ),
                        "kelly_effective_odds": row.final_odds,
                    }
                    for row in purchased
                ],
            })

        daily_row = {
            "race_date": race_day,
            "evaluated_races": len(ordered),
            "starting_bankroll_yen": STARTING_BANKROLL_YEN,
            "ending_bankroll_yen": bankroll,
            "tickets": tickets,
            "hit_tickets": hit_tickets,
            "races_bet": races_bet,
            "hit_races": hit_races,
            "stake_yen": staked,
            "return_yen": returned,
            "profit_yen": returned - staked,
            "roi": returned / staked if staked else 0.0,
            "max_drawdown_yen": max_drawdown,
            "max_drawdown_rate": max_drawdown_rate,
            "largest_hit_return_yen": largest_hit,
            "hit_return_square_sum_yen2": hit_return_square_sum,
            "decisions": decisions,
        }
        daily_row["reliability"] = bankroll_reliability_metrics(
            [daily_row],
            evaluated_races=len(ordered),
        )
        daily.append(daily_row)
    return daily


def _log_loss_comparison(races: list[dict[str, Any]]) -> dict[str, Any]:
    losses = {"model": [], "market": [], "challenger": []}
    ready_losses = {"market": [], "challenger": []}
    race_dates: list[str] = []
    top5_differences: list[float] = []
    for race in races:
        actual = str(race["actual_combination"])
        vectors = {
            "model": _normalized(race["model_probabilities"], "model probabilities"),
            "market": _normalized(race["market_probabilities"], "market probabilities"),
            "challenger": _normalized(
                race["_policy_calibrated_probabilities"],
                "policy probabilities",
            ),
        }
        if any(actual not in values for values in vectors.values()):
            raise ValueError(f"actual combination {actual!r} is missing")
        for name, values in vectors.items():
            losses[name].append(-math.log(max(EPSILON, values[actual])))
        race_dates.append(str(race["race_date"]))
        market_top5 = set(sorted(
            vectors["market"],
            key=lambda key: (-vectors["market"][key], key),
        )[:5])
        challenger_top5 = set(sorted(
            vectors["challenger"],
            key=lambda key: (-vectors["challenger"][key], key),
        )[:5])
        top5_differences.append(float(
            int(actual in challenger_top5) - int(actual in market_top5)
        ))
        if race["_market_kelly_calibration"]["ready"]:
            ready_losses["market"].append(losses["market"][-1])
            ready_losses["challenger"].append(losses["challenger"][-1])

    model = _mean(losses["model"])
    market = _mean(losses["market"])
    challenger = _mean(losses["challenger"])
    ready_market = _mean(ready_losses["market"])
    ready_challenger = _mean(ready_losses["challenger"])
    loss_differences = [
        challenger_loss - market_loss
        for challenger_loss, market_loss in zip(
            losses["challenger"], losses["market"]
        )
    ]
    loss_bootstrap = (
        paired_cluster_mean_bootstrap(loss_differences, race_dates)
        if loss_differences else None
    )
    top5_bootstrap = (
        paired_cluster_mean_bootstrap(top5_differences, race_dates)
        if top5_differences else None
    )
    return {
        "races": len(races),
        "model": model,
        "market": market,
        "challenger": challenger,
        "challenger_delta_vs_market": (
            challenger - market
            if challenger is not None and market is not None
            else None
        ),
        "ready_races": len(ready_losses["challenger"]),
        "ready_market": ready_market,
        "ready_challenger": ready_challenger,
        "ready_challenger_delta_vs_market": (
            ready_challenger - ready_market
            if ready_challenger is not None and ready_market is not None
            else None
        ),
        "challenger_vs_market_day_bootstrap": loss_bootstrap,
        "challenger_top5_vs_market_day_bootstrap": top5_bootstrap,
        "challenger_improvement_confidence": (
            loss_bootstrap["probability_less_than_zero"]
            if loss_bootstrap else None
        ),
        "challenger_top5_improvement_confidence": (
            top5_bootstrap["probability_greater_than_zero"]
            if top5_bootstrap else None
        ),
    }


def _edge_diagnostics(races: list[dict[str, Any]]) -> dict[str, Any]:
    race_maxima: list[float] = []
    ready_maxima: list[float] = []
    positive_combinations = 0
    positive_races = 0
    ready_positive_races = 0
    for race in races:
        probabilities = _normalized(
            race["_policy_calibrated_probabilities"],
            "policy probabilities",
        )
        odds = decision_odds(race)
        if set(probabilities) != set(odds):
            raise ValueError("probability and decision odds keys must match")
        values = [
            probabilities[selection]
            * _positive_float(odds[selection], "decision odds")
            for selection in probabilities
        ]
        maximum = max(values)
        race_maxima.append(maximum)
        positive = sum(int(value > 1.0) for value in values)
        positive_combinations += positive
        positive_races += int(positive > 0)
        if race["_market_kelly_calibration"]["ready"]:
            ready_maxima.append(maximum)
            ready_positive_races += int(positive > 0)
    ordered = sorted(race_maxima)
    ready_ordered = sorted(ready_maxima)
    return {
        "races": len(races),
        "positive_ev_races": positive_races,
        "positive_ev_combinations": positive_combinations,
        "positive_ev_race_rate": positive_races / len(races) if races else None,
        "max_estimated_ev": max(ordered) if ordered else None,
        "race_max_ev_p50": _quantile(ordered, 0.50),
        "race_max_ev_p90": _quantile(ordered, 0.90),
        "race_max_ev_p95": _quantile(ordered, 0.95),
        "race_max_ev_p99": _quantile(ordered, 0.99),
        "ready_races": len(ready_ordered),
        "ready_positive_ev_races": ready_positive_races,
        "ready_max_estimated_ev": max(ready_ordered) if ready_ordered else None,
    }


def _quantile(ordered: list[float], probability: float) -> float | None:
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalized(values: Mapping[str, Any], name: str) -> dict[str, float]:
    if len(values) != EXPECTED_COMBINATIONS:
        raise ValueError(f"{name} must contain exactly {EXPECTED_COMBINATIONS} outcomes")
    prepared = {
        str(key): _nonnegative_float(value, name)
        for key, value in values.items()
    }
    total = math.fsum(prepared.values())
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return {key: value / total for key, value in prepared.items()}


def _odds_safety_factor(value: Any) -> float:
    factor = float(value)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("odds_safety_factor must be finite and at least 1.0")
    return factor


def _required_ticket_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("required_ticket_count must be an integer from 1 to 120")
    if not 1 <= value <= EXPECTED_COMBINATIONS:
        raise ValueError("required_ticket_count must be an integer from 1 to 120")
    return value


def _iso_day(value: Any, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _race_order_key(race: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _iso_day(race.get("race_date"), "race_date"),
        _time_sort_value(_race_time_value(race)),
        _integer_sort_value(race.get("jcd")),
        _integer_sort_value(race.get("rno", race.get("race_no"))),
        str(race.get("race_id", "")),
    )


def _race_time_value(race: Mapping[str, Any]) -> Any:
    for key in (
        "deadline_at",
        "deadline_time",
        "deadline",
        "cutoff_at",
        "start_at",
        "start_time",
        "race_start_at",
        "race_time",
    ):
        value = race.get(key)
        if value not in (None, ""):
            return value
    return None


def _time_sort_value(value: Any) -> tuple[int, float, str]:
    if value is None:
        return (1, math.inf, "")
    if isinstance(value, datetime):
        clock = value.timetz().replace(tzinfo=None)
        return (0, _clock_seconds(clock), "")
    if isinstance(value, time):
        return (0, _clock_seconds(value.replace(tzinfo=None)), "")
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (0, _clock_seconds(parsed.timetz().replace(tzinfo=None)), "")
    except ValueError:
        pass
    try:
        parsed_time = time.fromisoformat(text)
        return (0, _clock_seconds(parsed_time.replace(tzinfo=None)), "")
    except ValueError:
        return (0, math.inf, text)


def _clock_seconds(value: time) -> float:
    return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6


def _integer_sort_value(value: Any) -> tuple[int, int, str]:
    try:
        return (0, int(value), "")
    except (TypeError, ValueError, OverflowError):
        return (1, 0, str(value or ""))


def _positive_float(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None
