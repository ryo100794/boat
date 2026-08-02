from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
from math import fsum, log
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np

from .genetic_search import GeneticSearchSettings
from .joint_market_value import TRIFECTA_OUTCOMES, JointMarketScenario
from .joint_parameter_uncertainty import (
    bootstrap_joint_parameter_models,
    generate_parameter_path_draws,
)
from .joint_policy_ga import JointPolicySearchConfig, optimize_joint_portfolio
from .parimutuel_settlement import (
    ParimutuelSettlementRules,
    build_parimutuel_gross_payoff_model,
)
from .pool_scale_lower_bound import attach_pool_scale_lower_bound
from .terminal_probability_oof import (
    build_terminal_probability_oof_artifact,
    joint_observations_from_terminal_oof,
)


EVALUATION_VERSION = "joint_bankroll_strict_walk_forward_v2"
EPSILON = 1e-15
PURCHASE_UNIT_YEN = 100


def _load_scored_races(path: Path) -> list[dict[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("scored cache root must be a mapping")
    races = payload.get("races")
    if not isinstance(races, list) or not races or not all(
        isinstance(row, dict) for row in races
    ):
        raise ValueError("scored cache races must be a non-empty mapping list")
    return races


def _eligible_races(
    races: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = []
    reasons: Counter[str] = Counter()
    excluded_by_day: Counter[str] = Counter()
    for race in races:
        reason = None
        if not isinstance(race.get("official_closing_odds"), Mapping):
            reason = "missing_official_closing_odds"
        elif not isinstance(race.get("odds"), Mapping):
            reason = "missing_decision_odds"
        elif not isinstance(race.get("actual_payout_yen"), int):
            reason = "missing_actual_payout"
        elif not race.get("actual_combination"):
            reason = "missing_actual_combination"
        if reason is None:
            eligible.append(race)
        else:
            reasons[reason] += 1
            excluded_by_day[str(race.get("race_date") or "unknown")] += 1
    return eligible, {
        "input_races": len(races),
        "eligible_races": len(eligible),
        "excluded_races": len(races) - len(eligible),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "excluded_by_day": dict(sorted(excluded_by_day.items())),
    }


def _mean_generated_probability(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    outcomes: Sequence[str],
) -> dict[str, float]:
    result = {outcome: 0.0 for outcome in outcomes}
    outer_mass = 1.0 / len(parameter_draws)
    for draw in parameter_draws:
        total_weight = fsum(float(scenario.weight) for scenario in draw)
        if total_weight <= 0.0:
            raise ValueError("scenario draw weights must have positive mass")
        for scenario in draw:
            weight = outer_mass * float(scenario.weight) / total_weight
            for outcome in outcomes:
                result[outcome] += weight * float(
                    scenario.probabilities[outcome]
                )
    total = fsum(result.values())
    if total <= 0.0:
        raise ValueError("generated probability has no mass")
    return {outcome: value / total for outcome, value in result.items()}


def _rank_candidate_tickets(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    outcomes: Sequence[str],
    *,
    limit: int,
    payout_rate: float = 0.75,
) -> tuple[str, ...]:
    """Learned path preselection only; exact settlement remains the gate."""
    scores = {outcome: 0.0 for outcome in outcomes}
    outer_mass = 1.0 / len(parameter_draws)
    for draw in parameter_draws:
        total_weight = fsum(float(scenario.weight) for scenario in draw)
        if total_weight <= 0.0:
            raise ValueError("scenario draw weights must have positive mass")
        for scenario in draw:
            weight = outer_mass * float(scenario.weight) / total_weight
            shares = scenario.market_state.get("final_market_shares")
            if not isinstance(shares, Mapping):
                raise ValueError("scenario is missing final market shares")
            for outcome in outcomes:
                implied_multiplier = payout_rate / max(
                    EPSILON, float(shares[outcome])
                )
                scores[outcome] += (
                    weight
                    * float(scenario.probabilities[outcome])
                    * implied_multiplier
                )
    return tuple(sorted(outcomes, key=lambda key: (-scores[key], key))[:limit])


def _realized_receipt(
    bets_yen: Mapping[str, int],
    *,
    actual_combination: str,
    actual_payout_yen: int,
) -> int:
    if isinstance(actual_payout_yen, bool) or not isinstance(
        actual_payout_yen, int
    ) or actual_payout_yen < 0:
        raise ValueError("actual payout must be non-negative integer yen")
    hit_stake = int(bets_yen.get(actual_combination, 0))
    if hit_stake % PURCHASE_UNIT_YEN:
        raise ValueError("realized bets must use 100-yen units")
    return hit_stake // PURCHASE_UNIT_YEN * actual_payout_yen


def _instant(value: object, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _release_matured_receipts(
    pending: Sequence[tuple[datetime, int]],
    *,
    asof: datetime,
) -> tuple[list[tuple[datetime, int]], int]:
    matured = [row for row in pending if row[0] <= asof]
    remaining = [row for row in pending if row[0] > asof]
    return remaining, sum(receipt for _available_at, receipt in matured)


def _probability_metrics(
    predicted: Mapping[str, float],
    decision: Mapping[str, float],
    actual: str,
    outcomes: Sequence[str],
) -> dict[str, float]:
    one_hot = {outcome: float(outcome == actual) for outcome in outcomes}
    lanes = tuple(sorted({outcome.split("-")[0] for outcome in outcomes}))
    actual_lane = actual.split("-")[0]
    predicted_winner = {
        lane: fsum(
            float(predicted[outcome])
            for outcome in outcomes
            if outcome.split("-")[0] == lane
        )
        for lane in lanes
    }
    decision_winner = {
        lane: fsum(
            float(decision[outcome])
            for outcome in outcomes
            if outcome.split("-")[0] == lane
        )
        for lane in lanes
    }
    return {
        "generated_winner_log_loss": -log(
            max(EPSILON, predicted_winner[actual_lane])
        ),
        "decision_model_winner_log_loss": -log(
            max(EPSILON, decision_winner[actual_lane])
        ),
        "generated_winner_top1_accuracy": float(
            max(predicted_winner, key=predicted_winner.get) == actual_lane
        ),
        "decision_model_winner_top1_accuracy": float(
            max(decision_winner, key=decision_winner.get) == actual_lane
        ),
        "generated_log_loss": -log(max(EPSILON, float(predicted[actual]))),
        "decision_model_log_loss": -log(
            max(EPSILON, float(decision[actual]))
        ),
        "generated_brier": fsum(
            (float(predicted[key]) - one_hot[key]) ** 2 for key in outcomes
        ),
        "decision_model_brier": fsum(
            (float(decision[key]) - one_hot[key]) ** 2 for key in outcomes
        ),
        "generated_top5": float(
            actual
            in sorted(outcomes, key=predicted.get, reverse=True)[:5]
        ),
        "decision_model_top5": float(
            actual
            in sorted(outcomes, key=decision.get, reverse=True)[:5]
        ),
    }


def _average_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: fsum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def _day_block_roi_interval(
    days: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not days:
        return {
            "samples": samples,
            "roi_lower": None,
            "roi_upper": None,
            "probability_roi_above_one": None,
        }
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        selected = [rng.choice(days) for _ in days]
        stake = sum(int(day["stake_yen"]) for day in selected)
        if stake > 0:
            values.append(
                sum(int(day["return_yen"]) for day in selected) / stake
            )
    if not values:
        return {
            "samples": samples,
            "effective_samples": 0,
            "roi_lower": None,
            "roi_upper": None,
            "probability_roi_above_one": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": samples,
        "effective_samples": len(values),
        "block": "complete_operating_day",
        "quantile_method": "inverted_cdf",
        "roi_lower": float(np.quantile(array, alpha, method="inverted_cdf")),
        "roi_upper": float(
            np.quantile(array, 1.0 - alpha, method="inverted_cdf")
        ),
        "probability_roi_above_one": float(np.mean(array > 1.0)),
    }


def run_joint_bankroll_evaluation(
    scored_cache: Path,
    *,
    terminal_min_training_days: int = 5,
    joint_min_training_days: int = 3,
    outer_draws: int = 20,
    scenarios_per_draw: int = 64,
    rank: int = 8,
    pooling_strength: float = 20.0,
    learn_residual_scales: bool = True,
    candidate_ticket_count: int = 12,
    initial_daily_bankroll_yen: int = 10_000,
    maximum_portfolio_stake_yen: int = 10_000,
    maximum_ticket_stake_yen: int = 5_000,
    maximum_selected_tickets: int = 12,
    buy_margin: float = 0.0,
    inner_tail_fraction: float = 0.10,
    population_size: int = 8,
    generations: int = 3,
    bootstrap_samples: int = 2000,
    settlement_delay_seconds: int = 600,
    seed: int = 33041,
    expected_outcomes: Sequence[str] = TRIFECTA_OUTCOMES,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if outer_draws < 1 or scenarios_per_draw < 1:
        raise ValueError("outer_draws and scenarios_per_draw must be positive")
    if (
        candidate_ticket_count < 1
        or candidate_ticket_count > len(expected_outcomes)
    ):
        raise ValueError("candidate_ticket_count is outside the outcome set")
    if initial_daily_bankroll_yen < PURCHASE_UNIT_YEN:
        raise ValueError("initial daily bankroll must cover one purchase unit")
    if initial_daily_bankroll_yen % PURCHASE_UNIT_YEN:
        raise ValueError("initial daily bankroll must use 100-yen units")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if settlement_delay_seconds < 0:
        raise ValueError("settlement_delay_seconds must be non-negative")

    races, coverage = _eligible_races(_load_scored_races(scored_cache))
    terminal = build_terminal_probability_oof_artifact(
        races,
        minimum_training_days=terminal_min_training_days,
        expected_outcomes=expected_outcomes,
    )
    observations = joint_observations_from_terminal_oof(races, terminal)
    observations_by_day: dict[str, list[Any]] = {}
    for observation in observations:
        observations_by_day.setdefault(observation.race_date, []).append(observation)
    races_by_id = {str(race["race_id"]): race for race in races}
    outcomes = tuple(expected_outcomes)
    settlement_rules = ParimutuelSettlementRules()
    skip_reasons: Counter[str] = Counter()
    daily = []
    all_probability_rows = []

    ordered_dates = sorted(observations_by_day)
    minimum_prior_days = max(2, joint_min_training_days)
    total_evaluation_days = max(0, len(ordered_dates) - minimum_prior_days)
    for day_index, evaluation_date in enumerate(ordered_dates):
        prior = [row for row in observations if row.race_date < evaluation_date]
        prior_days = {row.race_date for row in prior}
        if len(prior_days) < minimum_prior_days:
            continue
        parameter_models = bootstrap_joint_parameter_models(
            prior,
            decision_date=evaluation_date,
            draws=outer_draws,
            seed=seed + day_index * 1000,
            expected_outcomes=outcomes,
            fit_options={
                "rank": rank,
                "pooling_strength": pooling_strength,
                "learn_residual_scales": learn_residual_scales,
            },
        )
        opening = initial_daily_bankroll_yen
        balance = opening
        peak = opening
        maximum_drawdown = 0
        day_stake = 0
        day_return = 0
        day_bet_races = 0
        day_ticket_orders = 0
        pending_receipts: list[tuple[datetime, int]] = []
        race_rows = []
        day_probability_rows = []
        current = sorted(
            observations_by_day[evaluation_date],
            key=lambda row: (
                str(races_by_id[row.race_id].get("captured_at") or ""),
                row.race_id,
            ),
        )
        for race_index, observation in enumerate(current):
            if progress_callback is not None and race_index % 25 == 0:
                progress_callback({
                    "event": "joint_bankroll_race_progress",
                    "model": EVALUATION_VERSION,
                    "evaluation_date": evaluation_date,
                    "completed_days": len(daily),
                    "total_evaluation_days": total_evaluation_days,
                    "race_index": race_index,
                    "races_on_day": len(current),
                })
            race = races_by_id[observation.race_id]
            purchase_at = _instant(race.get("captured_at"), "captured_at")
            pending_receipts, matured_receipt = _release_matured_receipts(
                pending_receipts, asof=purchase_at
            )
            if matured_receipt:
                balance += matured_receipt
                peak = max(peak, balance)
            path_seed = seed + day_index * 100_000 + race_index * 10
            paths = generate_parameter_path_draws(
                parameter_models,
                decision_probabilities=observation.decision_probabilities,
                decision_market_shares=observation.decision_market_shares,
                venue=observation.venue,
                decision_horizon_seconds=observation.decision_horizon_seconds,
                popularity_band=observation.popularity_band,
                scenarios_per_draw=scenarios_per_draw,
                seed=path_seed,
            )
            predicted = _mean_generated_probability(paths, outcomes)
            probability_row = _probability_metrics(
                predicted,
                observation.decision_probabilities,
                str(race["actual_combination"]),
                outcomes,
            )
            day_probability_rows.append(probability_row)
            all_probability_rows.append(probability_row)
            if balance < PURCHASE_UNIT_YEN:
                skip_reasons["daily_bankroll_exhausted"] += 1
                continue
            try:
                priced_paths, pool_bound = attach_pool_scale_lower_bound(
                    paths,
                    displayed_odds=race["odds"],
                    ordinary_outcomes=outcomes,
                    odds_asof=str(race.get("captured_at") or "decision_time"),
                    rules=settlement_rules,
                )
            except ValueError:
                skip_reasons["invalid_or_incomplete_decision_odds"] += 1
                continue
            candidates = _rank_candidate_tickets(
                priced_paths,
                outcomes,
                limit=candidate_ticket_count,
                payout_rate=(
                    settlement_rules.payout_rate_numerator
                    / settlement_rules.payout_rate_denominator
                ),
            )
            settle = build_parimutuel_gross_payoff_model(
                ordinary_outcomes=outcomes,
                rules=settlement_rules,
                cache_external_stakes=True,
            )
            portfolio_limit = min(
                maximum_portfolio_stake_yen,
                balance // PURCHASE_UNIT_YEN * PURCHASE_UNIT_YEN,
            )
            policy = JointPolicySearchConfig(
                maximum_portfolio_stake_yen=portfolio_limit,
                maximum_ticket_stake_yen=min(
                    maximum_ticket_stake_yen, portfolio_limit
                ),
                maximum_selected_tickets=min(
                    maximum_selected_tickets, len(candidates)
                ),
                buy_margin=buy_margin,
                inner_tail_fraction=inner_tail_fraction,
                minimum_outer_draws=outer_draws,
            )
            search = optimize_joint_portfolio(
                priced_paths,
                candidate_tickets=candidates,
                gross_payoff_model=settle,
                available_bankroll_yen=balance,
                expected_outcomes=outcomes,
                config=policy,
                genetic_settings=GeneticSearchSettings(
                    population_size=population_size,
                    generations=generations,
                    elite_count=min(3, population_size // 2),
                    mutation_rate=0.35,
                    random_injections=1,
                    max_workers=min(4, population_size),
                    seed=path_seed + 1,
                ),
            )
            selected = search["selected"]
            selected_bets = (
                dict(selected["bets_yen"])
                if search["purchase_authorized"]
                else {}
            )
            stake = sum(selected_bets.values())
            receipt = _realized_receipt(
                selected_bets,
                actual_combination=str(race["actual_combination"]),
                actual_payout_yen=int(race["actual_payout_yen"]),
            )
            balance -= stake
            maximum_drawdown = max(maximum_drawdown, peak - balance)
            settlement_at = _instant(
                race.get("odds_deadline_at"), "odds_deadline_at"
            ) + timedelta(seconds=settlement_delay_seconds)
            if settlement_at < purchase_at:
                raise ValueError("settlement availability precedes purchase")
            if receipt:
                pending_receipts.append((settlement_at, receipt))
            day_stake += stake
            day_return += receipt
            if stake:
                day_bet_races += 1
                day_ticket_orders += len(selected_bets)
            portfolio = selected["metrics"].get("portfolio") or {}
            growth = (
                selected["metrics"].get("bankroll_growth", {}).get("growth")
                or {}
            )
            race_rows.append({
                "race_id": observation.race_id,
                "venue": observation.venue,
                "stake_yen": stake,
                "return_yen": receipt,
                "profit_yen": receipt - stake,
                "available_cash_after_bet_yen": balance,
                "settlement_available_at": settlement_at.isoformat(),
                "selected_tickets": len(selected_bets),
                "purchase_authorized": bool(search["purchase_authorized"]),
                "portfolio_lower_quantile": portfolio.get("lower_quantile"),
                "bankroll_growth_lower_quantile": growth.get("lower_quantile"),
                "maximum_conditional_ruin_probability": growth.get(
                    "maximum_conditional_ruin_probability"
                ),
                "pool_scale_lower_bound_yen": pool_bound.total_sales_yen,
                "pool_scale_method": pool_bound.method,
            })
        for _available_at, final_receipt in sorted(pending_receipts):
            balance += final_receipt
            peak = max(peak, balance)
        day_metrics = _average_metrics(day_probability_rows)
        day_row = {
            "race_date": evaluation_date,
            "training_days": len(prior_days),
            "evaluated_races": len(day_probability_rows),
            "opening_bankroll_yen": opening,
            "closing_bankroll_yen": balance,
            "stake_yen": day_stake,
            "return_yen": day_return,
            "profit_yen": day_return - day_stake,
            "roi": day_return / day_stake if day_stake else None,
            "bet_races": day_bet_races,
            "ticket_orders": day_ticket_orders,
            "max_drawdown_yen": maximum_drawdown,
            "max_drawdown_ratio": maximum_drawdown / opening,
            "probability_metrics": day_metrics,
            "races": race_rows,
        }
        daily.append(day_row)
        if progress_callback is not None:
            progress_callback({
                "event": "joint_bankroll_day_completed",
                "model": EVALUATION_VERSION,
                "evaluation_date": evaluation_date,
                "completed_days": len(daily),
                "total_evaluation_days": total_evaluation_days,
                "evaluated_races": day_row["evaluated_races"],
                "stake_yen": day_row["stake_yen"],
                "return_yen": day_row["return_yen"],
                "closing_bankroll_yen": day_row["closing_bankroll_yen"],
            })

    total_stake = sum(day["stake_yen"] for day in daily)
    total_return = sum(day["return_yen"] for day in daily)
    realized_returns = [
        int(race["return_yen"])
        for day in daily
        for race in day["races"]
        if int(race["stake_yen"]) > 0
    ]
    largest_return = max(realized_returns, default=0)
    probability_metrics = _average_metrics(all_probability_rows)
    probability_metrics.update({
        "model_winner_log_loss": probability_metrics.get(
            "generated_winner_log_loss"
        ),
        "model_winner_top1_accuracy": probability_metrics.get(
            "generated_winner_top1_accuracy"
        ),
        "model_trifecta_log_loss": probability_metrics.get(
            "generated_log_loss"
        ),
        "model_trifecta_top5_hit_rate": probability_metrics.get(
            "generated_top5"
        ),
        "generated_log_loss_delta_vs_decision_model": (
            probability_metrics.get("generated_log_loss", 0.0)
            - probability_metrics.get("decision_model_log_loss", 0.0)
        ),
        "generated_brier_delta_vs_decision_model": (
            probability_metrics.get("generated_brier", 0.0)
            - probability_metrics.get("decision_model_brier", 0.0)
        ),
        "generated_top5_delta_vs_decision_model": (
            probability_metrics.get("generated_top5", 0.0)
            - probability_metrics.get("decision_model_top5", 0.0)
        ),
    })
    confidence = _day_block_roi_interval(
        daily, samples=bootstrap_samples, seed=seed + 9_000_000
    )
    primary_bankroll = {
        "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": total_return / total_stake if total_stake else None,
        "bet_count": sum(day["bet_races"] for day in daily),
        "ticket_orders": sum(day["ticket_orders"] for day in daily),
        "race_days": len(daily),
        "evaluated_races": len(all_probability_rows),
        "selected_races": sum(day["bet_races"] for day in daily),
        "tickets": sum(day["ticket_orders"] for day in daily),
        "winning_days": sum(day["profit_yen"] > 0 for day in daily),
        "profitable_day_fraction": (
            sum(day["profit_yen"] > 0 for day in daily) / len(daily)
            if daily else 0.0
        ),
        "largest_hit_return_share": (
            largest_return / total_return if total_return else None
        ),
        "roi_without_largest_hit": (
            (total_return - largest_return) / total_stake
            if total_stake else None
        ),
        "max_drawdown_yen": max(
            (day["max_drawdown_yen"] for day in daily), default=0
        ),
        "max_drawdown_ratio": max(
            (day["max_drawdown_ratio"] for day in daily), default=0.0
        ),
        "roi_ci95_lower": confidence["roi_lower"],
        "roi_ci95_upper": confidence["roi_upper"],
        "probability_roi_above_one": confidence[
            "probability_roi_above_one"
        ],
        "daily_cluster_bootstrap_roi_lower_95": confidence["roi_lower"],
        "bootstrap_probability_roi_above_one": confidence[
            "probability_roi_above_one"
        ],
    }
    gate = {
        "minimum_30_complete_days": len(daily) >= 30,
        "minimum_1000_ticket_orders": primary_bankroll["ticket_orders"] >= 1000,
        "positive_profit": primary_bankroll["profit_yen"] > 0,
        "roi_lower_bound_above_one": bool(
            confidence["roi_lower"] is not None
            and confidence["roi_lower"] > 1.0
        ),
        "maximum_drawdown_within_half_bankroll": (
            primary_bankroll["max_drawdown_ratio"] <= 0.50
        ),
        "generated_log_loss_not_worse": (
            probability_metrics.get(
                "generated_log_loss_delta_vs_decision_model", 1.0
            ) <= 0.0
        ),
    }
    promotion_eligible = all(gate.values())
    return {
        "model": EVALUATION_VERSION,
        "status": (
            "promotion_candidate" if promotion_eligible
            else "provisional_accumulate_sealed_days"
        ),
        "promotion_eligible": promotion_eligible,
        "deployment_eligible": False,
        "scored_cache": str(scored_cache),
        "coverage": coverage,
        "configuration": {
            "terminal_min_training_days": terminal_min_training_days,
            "joint_min_training_days": joint_min_training_days,
            "outer_draws": outer_draws,
            "scenarios_per_draw": scenarios_per_draw,
            "rank": rank,
            "pooling_strength": pooling_strength,
            "learn_residual_scales": learn_residual_scales,
            "candidate_ticket_count": candidate_ticket_count,
            "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
            "maximum_portfolio_stake_yen": maximum_portfolio_stake_yen,
            "maximum_ticket_stake_yen": maximum_ticket_stake_yen,
            "maximum_selected_tickets": maximum_selected_tickets,
            "buy_margin": buy_margin,
            "inner_tail_fraction": inner_tail_fraction,
            "population_size": population_size,
            "generations": generations,
            "bootstrap_samples": bootstrap_samples,
            "settlement_delay_seconds": settlement_delay_seconds,
            "seed": seed,
        },
        "terminal_probability_oof": {
            "version": terminal["version"],
            "predicted_races": terminal["predicted_races"],
            "prediction_dates": terminal["prediction_dates"],
        },
        "evaluated_days": len(daily),
        "evaluation_days": len(daily),
        "evaluated_races": len(all_probability_rows),
        "evaluation_from": daily[0]["race_date"] if daily else None,
        "evaluation_through": daily[-1]["race_date"] if daily else None,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "probability_metrics": probability_metrics,
        "primary_bankroll": primary_bankroll,
        "bankroll_confidence": confidence,
        "promotion_gate": gate,
        "promotion_gate_passed": sum(gate.values()),
        "promotion_gate_total": len(gate),
        "promotion_gate_failed": [
            key for key, passed in gate.items() if not passed
        ],
        "daily": daily,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-min-training-days", type=int, default=5)
    parser.add_argument("--joint-min-training-days", type=int, default=3)
    parser.add_argument("--outer-draws", type=int, default=20)
    parser.add_argument("--scenarios-per-draw", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pooling-strength", type=float, default=20.0)
    parser.add_argument("--no-learn-residual-scales", action="store_true")
    parser.add_argument("--candidate-ticket-count", type=int, default=12)
    parser.add_argument("--initial-daily-bankroll-yen", type=int, default=10_000)
    parser.add_argument("--maximum-portfolio-stake-yen", type=int, default=10_000)
    parser.add_argument("--maximum-ticket-stake-yen", type=int, default=5_000)
    parser.add_argument("--maximum-selected-tickets", type=int, default=12)
    parser.add_argument("--buy-margin", type=float, default=0.0)
    parser.add_argument("--inner-tail-fraction", type=float, default=0.10)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--settlement-delay-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=33041)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = vars(args).copy()
    output = options.pop("output")
    options["learn_residual_scales"] = not options.pop(
        "no_learn_residual_scales"
    )
    result = run_joint_bankroll_evaluation(
        **options,
        progress_callback=lambda row: print(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True),
            flush=True,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "model": result["model"],
        "evaluated_days": result["evaluated_days"],
        "evaluated_races": result["evaluated_races"],
        **result["primary_bankroll"],
        "promotion_eligible": result["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_VERSION",
    "_day_block_roi_interval",
    "_probability_metrics",
    "_rank_candidate_tickets",
    "_release_matured_receipts",
    "_realized_receipt",
    "run_joint_bankroll_evaluation",
]
