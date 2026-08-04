from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import hashlib
from math import ceil, fsum, log
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np

from .genetic_search import GeneticSearchSettings
from .joint_market_value import TRIFECTA_OUTCOMES, JointMarketScenario
from .joint_scenario_model import MODEL_VERSION as JOINT_SCENARIO_MODEL_VERSION
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


EVALUATION_VERSION = (
    "joint_bankroll_strict_walk_forward_v10_all_pregate_candidates"
)
EPSILON = 1e-15
PURCHASE_UNIT_YEN = 100
MINIMUM_OUTER_TAIL_OBSERVATIONS = 5
MINIMUM_INNER_TAIL_EFFECTIVE_SAMPLES = 5.0


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_protocol(
    *,
    scored_cache: Path,
    eligible_races: Sequence[Mapping[str, Any]],
    observations: Sequence[Any],
    evaluation_dates: Sequence[str],
    terminal: Mapping[str, Any],
    configuration: Mapping[str, Any],
    outcomes: Sequence[str],
    settlement_audit: Mapping[str, Any],
    bootstrap_condition_id: str,
) -> dict[str, Any]:
    """Freeze every input that changes the meaning of an evaluation."""
    evaluation_date_set = set(evaluation_dates)
    races_by_id = {str(row["race_id"]): row for row in eligible_races}
    population = []
    for observation in observations:
        if observation.race_date not in evaluation_date_set:
            continue
        race = races_by_id[observation.race_id]
        evaluation_time_t, snapshot_at, snapshot_age = _decision_context(race)
        population.append({
            "race_id": observation.race_id,
            "race_date": observation.race_date,
            "evaluation_time_t": evaluation_time_t.isoformat(),
            "evaluation_time_t_source": (
                "decision_at" if race.get("decision_at") else "odds_deadline_at"
            ),
            "odds_snapshot_captured_at": snapshot_at.isoformat(),
            "snapshot_age_seconds": snapshot_age,
            "odds_deadline_at": str(race.get("odds_deadline_at") or ""),
            "venue": observation.venue,
            "wager_type": "trifecta",
            "popularity_band_at_t": observation.popularity_band,
            "decision_horizon_seconds": observation.decision_horizon_seconds,
        })
    population.sort(key=lambda row: (row["evaluation_time_t"], row["race_id"]))
    evaluation_times = [row["evaluation_time_t"] for row in population]
    outcome_schema = tuple(str(value) for value in outcomes)
    protocol = {
        "version": "joint_evaluation_protocol_v2",
        "identity_scope": (
            "model_data_time_venue_wager_popularity_purchase_settlement_"
            "resampling"
        ),
        "model": {
            "evaluation_model": EVALUATION_VERSION,
            "terminal_probability_model": terminal.get("version"),
            "terminal_artifact_contract_sha256": terminal.get(
                "artifact_contract_sha256"
            ),
            "joint_scenario_model": JOINT_SCENARIO_MODEL_VERSION,
        },
        "input_data": {
            "scored_cache_sha256": _file_sha256(scored_cache),
            "eligible_races": len(eligible_races),
        },
        "evaluation_window": {
            "from": min(evaluation_dates) if evaluation_dates else None,
            "through": max(evaluation_dates) if evaluation_dates else None,
            "complete_days": len(evaluation_dates),
        },
        "evaluation_time_t": {
            "definition": "purchase_decision_timestamp",
            "source_field": "decision_at_else_odds_deadline_at",
            "earliest": min(evaluation_times) if evaluation_times else None,
            "latest": max(evaluation_times) if evaluation_times else None,
            "race_time_manifest_sha256": _canonical_sha256([
                [row["race_id"], row["evaluation_time_t"]]
                for row in population
            ]),
        },
        "odds_snapshot_age": {
            "definition": "evaluation_time_t_minus_odds_snapshot_captured_at",
            "unit": "seconds",
            "minimum": min(
                (row["snapshot_age_seconds"] for row in population),
                default=None,
            ),
            "maximum": max(
                (row["snapshot_age_seconds"] for row in population),
                default=None,
            ),
            "mean": (
                fsum(row["snapshot_age_seconds"] for row in population)
                / len(population)
                if population else None
            ),
            "manifest_sha256": _canonical_sha256([
                [
                    row["race_id"], row["evaluation_time_t"],
                    row["odds_snapshot_captured_at"],
                    row["snapshot_age_seconds"],
                ]
                for row in population
            ]),
        },
        "population": {
            "races": len(population),
            "venues": sorted({row["venue"] for row in population}),
            "wager_types": ["trifecta"],
            "popularity_bands_at_t": sorted({
                row["popularity_band_at_t"] for row in population
            }),
            "context_manifest_sha256": _canonical_sha256(population),
            "outcome_count": len(outcome_schema),
            "outcome_schema_sha256": _canonical_sha256(outcome_schema),
        },
        "training_and_joint_distribution": {
            **{
                key: configuration[key]
                for key in (
                    "terminal_min_training_days", "joint_min_training_days",
                    "outer_draws", "scenarios_per_draw", "rank",
                    "pooling_strength", "learn_residual_scales",
                )
            },
            "search_outer_draws": configuration.get(
                "search_outer_draws", configuration["outer_draws"]
            ),
            "validation_outer_draws": configuration["outer_draws"],
            "search_validation_draw_sets_disjoint": bool(configuration.get(
                "search_validation_draw_sets_disjoint", False
            )),
            "residual_scale_selection_scope": (
                "once_per_evaluation_date_on_all_strictly_prior_"
                "observations_fixed_across_outer_day_bootstrap_refits"
                if configuration.get("learn_residual_scales")
                else "fixed_full_residual"
            ),
            "outer_parameter_uncertainty": (
                "complete_day_bootstrap_refit_with_preselected_"
                "residual_scale_hyperparameters"
            ),
        },
        "purchase_rule": {
            "decision_rule": "V_buy_greater_than_buy_margin",
            "formal_value": "Q_alpha_outer_of_ES_beta_lower_portfolio_edge",
            "outer_alpha": 0.05,
            "minimum_inner_tail_effective_samples": float(configuration.get(
                "minimum_inner_tail_effective_samples",
                MINIMUM_INNER_TAIL_EFFECTIVE_SAMPLES,
            )),
            **{
                key: configuration[key]
                for key in (
                    "candidate_ticket_count", "initial_daily_bankroll_yen",
                    "maximum_portfolio_stake_yen",
                    "maximum_ticket_stake_yen",
                    "maximum_selected_tickets", "buy_margin",
                    "inner_tail_fraction",
                    "settlement_delay_seconds",
                )
            },
        },
        "optimizer": {
            "kind": "genetic_integer_stake_vector",
            "value_gate_prunes_growth_evaluation": True,
            "pruning_rule": (
                "skip_growth_only_when_portfolio_purchase_gate_is_false"
            ),
            "statistical_draw_counts_unchanged_by_pruning": True,
            "calibration_candidate": (
                "best_nonzero_search_portfolio_independently_validated_"
                "before_structural_and_calibration_purchase_gates"
            ),
            "calibration_population_includes_gate_rejections": True,
            "population_size": configuration["population_size"],
            "generations": configuration["generations"],
            "seed": configuration["seed"],
        },
        "settlement": dict(settlement_audit),
        "resampling_condition_id": bootstrap_condition_id,
        "strict_temporal_rule": "training_dates_strictly_before_evaluation_date",
    }
    return {"id": _canonical_sha256(protocol), "protocol": protocol}


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


def _decision_context(
    race: Mapping[str, Any],
) -> tuple[datetime, datetime, float]:
    source = "decision_at" if race.get("decision_at") else "odds_deadline_at"
    decision_at = _instant(race.get(source), source)
    captured_at = _instant(race.get("captured_at"), "captured_at")
    snapshot_age = (
        decision_at - captured_at.astimezone(decision_at.tzinfo)
    ).total_seconds()
    if snapshot_age < 0.0:
        raise ValueError("odds snapshot was captured after purchase decision")
    return decision_at, captured_at, snapshot_age


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


def _joint_value_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"recorded": False, "reason": "no_selected_portfolio"}
    moments = []
    for draw in value.get("moments_by_draw") or []:
        if not isinstance(draw, Mapping):
            continue
        for ticket in (draw.get("tickets") or {}).values():
            if isinstance(ticket, Mapping):
                moments.append(ticket)
    covariances = [
        float(row["probability_multiplier_covariance"])
        for row in moments
        if row.get("probability_multiplier_covariance") is not None
    ]
    expected_pi_d = [
        float(row["expected_probability_times_multiplier"])
        for row in moments
        if row.get("expected_probability_times_multiplier") is not None
    ]
    independent_pi_d = [
        float(row["independence_probability_times_multiplier"])
        for row in moments
        if row.get("independence_probability_times_multiplier") is not None
    ]
    joint_expected_edges = [
        float(row["joint_expected_edge"])
        for row in moments
        if row.get("joint_expected_edge") is not None
    ]
    product_identity_residuals = [
        float(row["expected_probability_times_multiplier"])
        - float(row["independence_probability_times_multiplier"])
        - float(row["probability_multiplier_covariance"])
        for row in moments
        if row.get("expected_probability_times_multiplier") is not None
        and row.get("independence_probability_times_multiplier") is not None
        and row.get("probability_multiplier_covariance") is not None
    ]
    independence_biases = []
    for row in moments:
        if row.get("independence_approximation_bias") is not None:
            independence_biases.append(
                float(row["independence_approximation_bias"])
            )
        elif (
            row.get("ordinary_hit_independence_approximation_edge") is not None
            and row.get("joint_expected_edge") is not None
        ):
            independence_biases.append(
                float(row["ordinary_hit_independence_approximation_edge"])
                - float(row["joint_expected_edge"])
            )
    outer_sample_count_r = int(value.get("parameter_draws") or 0)
    minimum_outer_draws = int(value.get("minimum_outer_draws") or 0)
    outer_alpha = float(value.get("outer_alpha") or 0.0)
    outer_tail_observations = (
        max(1, ceil(outer_alpha * outer_sample_count_r))
        if outer_sample_count_r > 0 and outer_alpha > 0.0 else 0
    )
    return {
        "recorded": True,
        "evaluator_version": value.get("version"),
        "shared_probability_price_scenarios": True,
        "outer_sample_count_r": outer_sample_count_r,
        "parameter_draws": outer_sample_count_r,
        "minimum_outer_draws": minimum_outer_draws,
        "inner_scenario_count_s_definition": value.get(
            "inner_scenario_count_s_definition"
        ),
        "inner_scenario_count_s_min": value.get("inner_scenario_count_s_min"),
        "inner_scenario_count_s_max": value.get("inner_scenario_count_s_max"),
        "inner_aggregation": value.get("inner_aggregation"),
        "inner_tail_fraction": value.get("inner_tail_fraction"),
        "inner_effective_samples_min": value.get(
            "inner_effective_samples_min"
        ),
        "inner_effective_samples_mean": value.get(
            "inner_effective_samples_mean"
        ),
        "inner_effective_samples_max": value.get(
            "inner_effective_samples_max"
        ),
        "inner_tail_effective_samples_min": value.get(
            "inner_tail_effective_samples_min"
        ),
        "inner_tail_effective_samples_mean": value.get(
            "inner_tail_effective_samples_mean"
        ),
        "inner_tail_effective_samples_max": value.get(
            "inner_tail_effective_samples_max"
        ),
        "minimum_inner_tail_effective_samples": value.get(
            "minimum_inner_tail_effective_samples"
        ),
        "inner_tail_support_for_purchase": value.get(
            "inner_tail_support_for_purchase"
        ),
        "outer_alpha": outer_alpha,
        "outer_tail_observations": outer_tail_observations,
        "minimum_outer_tail_observations_for_promotion": (
            MINIMUM_OUTER_TAIL_OBSERVATIONS
        ),
        "outer_tail_support_for_promotion": (
            outer_tail_observations >= MINIMUM_OUTER_TAIL_OBSERVATIONS
        ),
        "outer_quantile_method": value.get("outer_quantile_method"),
        "portfolio_path_aggregation": (
            value.get("inner_aggregation")
            == "portfolio_path_weighted_lower_tail_mean"
        ),
        "complete_vector_repricing": bool(
            value.get("marginal_contributions_computed")
        ),
        "moment_observations": len(moments),
        "expected_probability_times_multiplier_definition": (
            "weighted_E_pi_D_on_shared_joint_market_paths"
        ),
        "expected_probability_times_multiplier_mean": (
            fsum(expected_pi_d) / len(expected_pi_d)
            if expected_pi_d else None
        ),
        "expected_probability_times_multiplier_min": (
            min(expected_pi_d) if expected_pi_d else None
        ),
        "expected_probability_times_multiplier_max": (
            max(expected_pi_d) if expected_pi_d else None
        ),
        "independence_probability_times_multiplier_mean": (
            fsum(independent_pi_d) / len(independent_pi_d)
            if independent_pi_d else None
        ),
        "joint_expected_edge_mean": (
            fsum(joint_expected_edges) / len(joint_expected_edges)
            if joint_expected_edges else None
        ),
        "product_identity_residual_mean": (
            fsum(product_identity_residuals) / len(product_identity_residuals)
            if product_identity_residuals else None
        ),
        "product_identity_residual_max_abs": (
            max(abs(item) for item in product_identity_residuals)
            if product_identity_residuals else None
        ),
        "product_identity_consistent": (
            max(abs(item) for item in product_identity_residuals) <= 1e-12
            if product_identity_residuals else None
        ),
        "probability_multiplier_covariance_mean": (
            fsum(covariances) / len(covariances) if covariances else None
        ),
        "probability_multiplier_covariance_min": (
            min(covariances) if covariances else None
        ),
        "probability_multiplier_covariance_max": (
            max(covariances) if covariances else None
        ),
        "negative_covariance_fraction": (
            sum(value < 0.0 for value in covariances) / len(covariances)
            if covariances else None
        ),
        "independence_approximation_bias_definition": (
            "E_pi_times_E_D_minus_E_pi_D_equals_negative_covariance"
        ),
        "independence_approximation_bias_mean": (
            fsum(independence_biases) / len(independence_biases)
            if independence_biases else None
        ),
        "independence_approximation_bias_min": (
            min(independence_biases) if independence_biases else None
        ),
        "independence_approximation_bias_max": (
            max(independence_biases) if independence_biases else None
        ),
        "positive_independence_bias_fraction": (
            sum(value > 0.0 for value in independence_biases)
            / len(independence_biases)
            if independence_biases else None
        ),
        "independence_approximation_overstatement_mean": (
            fsum(independence_biases) / len(independence_biases)
            if independence_biases else None
        ),
        "independence_approximation_overstatement_max": (
            max(independence_biases) if independence_biases else None
        ),
    }


def _aggregate_joint_value_audits(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recorded = [row for row in audits if row.get("recorded")]
    if not recorded:
        return {
            "recorded": False,
            "audited_portfolios": 0,
            "reason": "no_authorized_portfolio_with_audit",
        }
    weights = [max(1, int(row.get("moment_observations") or 0)) for row in recorded]
    total_weight = sum(weights)

    def weighted(field: str) -> float | None:
        pairs = [
            (float(row[field]), weight)
            for row, weight in zip(recorded, weights)
            if row.get(field) is not None
        ]
        denominator = sum(weight for _value, weight in pairs)
        return (
            fsum(value * weight for value, weight in pairs) / denominator
            if denominator else None
        )

    ess_values = [
        float(row["inner_effective_samples_min"])
        for row in recorded
        if row.get("inner_effective_samples_min") is not None
    ]
    ess_max_values = [
        float(row["inner_effective_samples_max"])
        for row in recorded
        if row.get("inner_effective_samples_max") is not None
    ]
    tail_ess_values = [
        float(row["inner_tail_effective_samples_min"])
        for row in recorded
        if row.get("inner_tail_effective_samples_min") is not None
    ]
    tail_ess_max_values = [
        float(row["inner_tail_effective_samples_max"])
        for row in recorded
        if row.get("inner_tail_effective_samples_max") is not None
    ]
    minimum_tail_ess_values = [
        float(row["minimum_inner_tail_effective_samples"])
        for row in recorded
        if row.get("minimum_inner_tail_effective_samples") is not None
    ]
    inner_tail_fractions = [
        float(row["inner_tail_fraction"])
        for row in recorded
        if row.get("inner_tail_fraction") is not None
    ]
    inner_scenario_min_values = [
        int(row["inner_scenario_count_s_min"])
        for row in recorded
        if row.get("inner_scenario_count_s_min") is not None
    ]
    inner_scenario_max_values = [
        int(row["inner_scenario_count_s_max"])
        for row in recorded
        if row.get("inner_scenario_count_s_max") is not None
    ]
    outer_sample_counts = [
        int(row.get("outer_sample_count_r") or row.get("parameter_draws") or 0)
        for row in recorded
    ]
    minimum_outer_draws = [
        int(row.get("minimum_outer_draws") or 0) for row in recorded
    ]
    outer_tail_observations = [
        int(row.get("outer_tail_observations") or 0) for row in recorded
    ]
    outer_alphas = [
        float(row.get("outer_alpha") or 0.0) for row in recorded
    ]
    identity_residual_max_abs_values = [
        float(row["product_identity_residual_max_abs"])
        for row in recorded
        if row.get("product_identity_residual_max_abs") is not None
    ]
    identity_consistency_values = [
        row.get("product_identity_consistent")
        for row in recorded
        if row.get("product_identity_consistent") is not None
    ]
    independence_bias_min_values = [
        float(row["independence_approximation_bias_min"])
        for row in recorded
        if row.get("independence_approximation_bias_min") is not None
    ]
    independence_bias_max_values = [
        float(row["independence_approximation_bias_max"])
        for row in recorded
        if row.get("independence_approximation_bias_max") is not None
    ]
    return {
        "recorded": True,
        "audited_portfolios": len(recorded),
        "moment_observations": sum(
            int(row.get("moment_observations") or 0) for row in recorded
        ),
        "shared_probability_price_scenarios": all(
            row.get("shared_probability_price_scenarios") is True
            for row in recorded
        ),
        "portfolio_path_aggregation": all(
            row.get("portfolio_path_aggregation") is True for row in recorded
        ),
        "complete_vector_repricing": all(
            row.get("complete_vector_repricing") is True for row in recorded
        ),
        "outer_sample_count_r_definition": (
            "number_of_outer_model_or_parameter_uncertainty_draws"
        ),
        "outer_sample_count_r_min": min(outer_sample_counts),
        "outer_sample_count_r_max": max(outer_sample_counts),
        "parameter_draws_min": min(outer_sample_counts),
        "minimum_outer_draws_max": max(minimum_outer_draws),
        "outer_alpha_min": min(outer_alphas),
        "outer_alpha_max": max(outer_alphas),
        "outer_tail_observations_min": min(outer_tail_observations),
        "outer_tail_observations_max": max(outer_tail_observations),
        "minimum_outer_tail_observations_for_promotion": (
            MINIMUM_OUTER_TAIL_OBSERVATIONS
        ),
        "outer_tail_support_for_promotion": (
            min(outer_tail_observations) >= MINIMUM_OUTER_TAIL_OBSERVATIONS
        ),
        "inner_scenario_count_s_definition": (
            "future_joint_market_paths_per_outer_parameter_draw"
        ),
        "inner_scenario_count_s_min": (
            min(inner_scenario_min_values) if inner_scenario_min_values else None
        ),
        "inner_scenario_count_s_max": (
            max(inner_scenario_max_values) if inner_scenario_max_values else None
        ),
        "inner_effective_samples_min": min(ess_values) if ess_values else None,
        "inner_effective_samples_mean": weighted(
            "inner_effective_samples_mean"
        ),
        "inner_effective_samples_max": (
            max(ess_max_values) if ess_max_values else None
        ),
        "inner_tail_effective_samples_min": (
            min(tail_ess_values) if tail_ess_values else None
        ),
        "inner_tail_effective_samples_mean": weighted(
            "inner_tail_effective_samples_mean"
        ),
        "inner_tail_effective_samples_max": (
            max(tail_ess_max_values) if tail_ess_max_values else None
        ),
        "inner_tail_fraction_min": (
            min(inner_tail_fractions) if inner_tail_fractions else None
        ),
        "inner_tail_fraction_max": (
            max(inner_tail_fractions) if inner_tail_fractions else None
        ),
        "minimum_inner_tail_effective_samples_max": (
            max(minimum_tail_ess_values) if minimum_tail_ess_values else None
        ),
        "inner_tail_support_for_promotion": bool(
            tail_ess_values
            and minimum_tail_ess_values
            and min(tail_ess_values) >= max(minimum_tail_ess_values)
        ),
        "probability_multiplier_covariance_mean": weighted(
            "probability_multiplier_covariance_mean"
        ),
        "expected_probability_times_multiplier_definition": (
            "weighted_E_pi_D_on_shared_joint_market_paths"
        ),
        "expected_probability_times_multiplier_mean": weighted(
            "expected_probability_times_multiplier_mean"
        ),
        "independence_probability_times_multiplier_mean": weighted(
            "independence_probability_times_multiplier_mean"
        ),
        "joint_expected_edge_mean": weighted("joint_expected_edge_mean"),
        "product_identity_residual_mean": weighted(
            "product_identity_residual_mean"
        ),
        "product_identity_residual_max_abs": (
            max(identity_residual_max_abs_values)
            if identity_residual_max_abs_values else None
        ),
        "product_identity_consistent": (
            all(value is True for value in identity_consistency_values)
            if len(identity_consistency_values) == len(recorded) else None
        ),
        "negative_covariance_fraction": weighted(
            "negative_covariance_fraction"
        ),
        "independence_approximation_bias_definition": (
            "E_pi_times_E_D_minus_E_pi_D_equals_negative_covariance"
        ),
        "independence_approximation_bias_mean": weighted(
            "independence_approximation_bias_mean"
        ),
        "independence_approximation_bias_min": (
            min(independence_bias_min_values)
            if independence_bias_min_values else None
        ),
        "independence_approximation_bias_max": (
            max(independence_bias_max_values)
            if independence_bias_max_values else None
        ),
        "positive_independence_bias_fraction": weighted(
            "positive_independence_bias_fraction"
        ),
        "independence_approximation_overstatement_mean": weighted(
            "independence_approximation_overstatement_mean"
        ),
        "outer_quantile_method": "inverted_cdf",
        "weighting": "ticket_draw_moment_count",
        "weight_total": total_weight,
    }


def _block_roi_interval(
    blocks: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    block: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not blocks:
        return {
            "samples": samples,
            "block": block,
            "quantile_method": "inverted_cdf",
            "roi_lower": None,
            "roi_upper": None,
            "probability_roi_above_one": None,
        }
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        selected = [rng.choice(blocks) for _ in blocks]
        stake = sum(int(row["stake_yen"]) for row in selected)
        if stake > 0:
            values.append(
                sum(int(row["return_yen"]) for row in selected) / stake
            )
    if not values:
        return {
            "samples": samples,
            "effective_samples": 0,
            "block": block,
            "quantile_method": "inverted_cdf",
            "roi_lower": None,
            "roi_upper": None,
            "probability_roi_above_one": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": samples,
        "effective_samples": len(values),
        "block": block,
        "quantile_method": "inverted_cdf",
        "roi_lower": float(np.quantile(array, alpha, method="inverted_cdf")),
        "roi_upper": float(
            np.quantile(array, 1.0 - alpha, method="inverted_cdf")
        ),
        "probability_roi_above_one": float(np.mean(array > 1.0)),
    }


def _day_block_roi_interval(
    days: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    return _block_roi_interval(
        days,
        samples=samples,
        seed=seed,
        block="complete_operating_day",
        alpha=alpha,
    )


def _day_venue_blocks(
    days: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks = []
    for day in days:
        by_venue: dict[str, dict[str, int]] = {}
        for race in day.get("races") or []:
            venue = str(race.get("venue") or "unknown")
            row = by_venue.setdefault(
                venue, {"stake_yen": 0, "return_yen": 0}
            )
            row["stake_yen"] += int(race.get("stake_yen") or 0)
            row["return_yen"] += int(race.get("return_yen") or 0)
        blocks.extend(by_venue.values())
    return blocks


def _meeting_blocks(
    days: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_venue_day: dict[str, dict[str, dict[str, int]]] = {}
    for day in days:
        race_date = str(day.get("race_date") or "")
        for race in day.get("races") or []:
            venue = str(race.get("venue") or "unknown")
            row = by_venue_day.setdefault(venue, {}).setdefault(
                race_date, {"stake_yen": 0, "return_yen": 0}
            )
            row["stake_yen"] += int(race.get("stake_yen") or 0)
            row["return_yen"] += int(race.get("return_yen") or 0)
    result = []
    for rows in by_venue_day.values():
        current = None
        previous = None
        for race_date, values in sorted(rows.items()):
            try:
                parsed = date.fromisoformat(race_date)
            except ValueError:
                result.append(dict(values))
                current = None
                previous = None
                continue

            if previous is None or (parsed - previous).days > 1:
                current = {"stake_yen": 0, "return_yen": 0}
                result.append(current)
            current["stake_yen"] += values["stake_yen"]
            current["return_yen"] += values["return_yen"]
            previous = parsed
    return result


def build_block_bootstrap_evidence(
    days: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Build reproducible primary and sensitivity ROI evidence."""
    normalized_days = []
    for day in days:
        row = dict(day)
        races = row.get("races") or []
        row.setdefault(
            "stake_yen",
            sum(int(race.get("stake_yen") or 0) for race in races),
        )
        row.setdefault(
            "return_yen",
            sum(int(race.get("return_yen") or 0) for race in races),
        )
        normalized_days.append(row)
    confidence = _day_block_roi_interval(
        normalized_days, samples=samples, seed=seed + 9_000_000
    )
    day_venue_confidence = _block_roi_interval(
        _day_venue_blocks(normalized_days),
        samples=samples,
        seed=seed + 9_000_001,
        block="independent_day_venue_sensitivity",
    )
    meeting_confidence = _block_roi_interval(
        _meeting_blocks(normalized_days),
        samples=samples,
        seed=seed + 9_000_002,
        block="consecutive_venue_meeting_sensitivity",
    )
    condition = {
        "version": "joint_bankroll_block_bootstrap_v1",
        "formal_gate": "Q0.05_ROI_greater_than_1",
        "alpha": 0.05,
        "quantile_method": "inverted_cdf",
        "samples": samples,
        "primary_block": "complete_operating_day",
        "sensitivity_blocks": [
            "independent_day_venue_sensitivity",
            "consecutive_venue_meeting_sensitivity",
        ],
        "seed": seed + 9_000_000,
    }
    condition_id = hashlib.sha256(
        json.dumps(
            condition,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **confidence,
        "condition_id": condition_id,
        "condition": condition,
        "formal_gate_passed": bool(
            confidence["roi_lower"] is not None
            and confidence["roi_lower"] > 1.0
        ),
        "probability_roi_above_one_is_diagnostic_only": True,
        "sensitivity": {
            "day_venue": day_venue_confidence,
            "venue_meeting": meeting_confidence,
        },
    }


def _purchase_value_realization_calibration(
    days: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare validated portfolio value with the same portfolio's realization."""
    rows = []
    mismatched_portfolios = 0
    for day in days:
        race_date = str(day.get("race_date") or "")
        for race in day.get("races") or []:
            value = race.get("purchase_value")
            stake = int(race.get("best_search_stake_yen") or 0)
            if value is None or stake <= 0:
                continue
            if race.get("purchase_value_bets_match_best_search") is not True:
                mismatched_portfolios += 1
                continue
            rows.append({
                "race_date": race_date,
                "purchase_value": float(value),
                "predicted_roi": 1.0 + float(value),
                "stake_yen": stake,
                "return_yen": int(
                    race.get("best_search_hypothetical_return_yen") or 0
                ),
            })
    rows.sort(key=lambda row: (row["purchase_value"], row["race_date"]))
    bins = []
    for decile_index, indices in enumerate(
        np.array_split(np.arange(len(rows)), min(10, len(rows)))
        if rows else [],
        start=1,
    ):
        selected = [rows[int(index)] for index in indices]
        stake_yen = sum(row["stake_yen"] for row in selected)
        return_yen = sum(row["return_yen"] for row in selected)
        predicted_return_yen = fsum(
            row["predicted_roi"] * row["stake_yen"] for row in selected
        )
        by_day: dict[str, dict[str, Any]] = {}
        for row in selected:
            daily = by_day.setdefault(
                row["race_date"],
                {
                    "race_date": row["race_date"],
                    "stake_yen": 0,
                    "return_yen": 0,
                },
            )
            daily["stake_yen"] += row["stake_yen"]
            daily["return_yen"] += row["return_yen"]
        confidence = _day_block_roi_interval(
            list(by_day.values()),
            samples=samples,
            seed=seed + 9_100_000 + decile_index,
        )
        bins.append({
            "decile": decile_index,
            "candidate_portfolios": len(selected),
            "evaluation_days": len(by_day),
            "minimum_purchase_value": min(
                row["purchase_value"] for row in selected
            ),
            "maximum_purchase_value": max(
                row["purchase_value"] for row in selected
            ),
            "mean_purchase_value": fsum(
                row["purchase_value"] for row in selected
            ) / len(selected),
            "predicted_roi": (
                predicted_return_yen / stake_yen if stake_yen else None
            ),
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": return_yen - stake_yen,
            "realized_roi": return_yen / stake_yen if stake_yen else None,
            "daily_block_roi_lower_95": confidence["roi_lower"],
            "daily_block_roi_upper_95": confidence["roi_upper"],
            "bootstrap_samples": confidence["samples"],
        })
    realized = [
        float(row["realized_roi"])
        for row in bins
        if row["realized_roi"] is not None
    ]
    return {
        "version": "joint_purchase_value_realization_deciles_v1",
        "population": (
            "validated_best_search_portfolios_with_identical_realized_bets"
        ),
        "predicted_roi_definition": "1_plus_validated_V_buy",
        "realized_roi_definition": (
            "same_portfolio_integer_settlement_return_divided_by_stake"
        ),
        "candidate_portfolios": len(rows),
        "excluded_mismatched_portfolios": mismatched_portfolios,
        "quantile_bins": len(bins),
        "monotone_realized_roi": bool(
            realized
            and all(left <= right for left, right in zip(realized, realized[1:]))
        ),
        "deciles": bins,
    }


def run_joint_bankroll_evaluation(
    scored_cache: Path,
    *,
    terminal_min_training_days: int = 5,
    joint_min_training_days: int = 3,
    outer_draws: int = 20,
    search_outer_draws: int | None = None,
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
    if search_outer_draws is not None and search_outer_draws < 1:
        raise ValueError("search_outer_draws must be positive when provided")
    effective_search_outer_draws = (
        outer_draws if search_outer_draws is None else search_outer_draws
    )
    separate_validation_draws = search_outer_draws is not None
    total_parameter_draws = (
        outer_draws + effective_search_outer_draws
        if separate_validation_draws else outer_draws
    )
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
            draws=total_parameter_draws,
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
                _decision_context(races_by_id[row.race_id])[0],
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
            purchase_at, snapshot_at, snapshot_age = _decision_context(race)
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
            if separate_validation_draws:
                search_paths = paths[:effective_search_outer_draws]
                validation_paths = paths[effective_search_outer_draws:]
            else:
                search_paths = paths
                validation_paths = paths
            if (
                len(search_paths) != effective_search_outer_draws
                or len(validation_paths) != outer_draws
            ):
                raise RuntimeError("joint parameter draw split is inconsistent")
            predicted = _mean_generated_probability(validation_paths, outcomes)
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
            if separate_validation_draws:
                search_priced_paths = priced_paths[:effective_search_outer_draws]
                validation_priced_paths = priced_paths[effective_search_outer_draws:]
            else:
                search_priced_paths = priced_paths
                validation_priced_paths = priced_paths
            candidates = _rank_candidate_tickets(
                search_priced_paths,
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
                minimum_outer_draws=effective_search_outer_draws,
                minimum_inner_tail_effective_samples=(
                    MINIMUM_INNER_TAIL_EFFECTIVE_SAMPLES
                ),
            )
            search = optimize_joint_portfolio(
                search_priced_paths,
                validation_parameter_draws=(
                    validation_priced_paths if separate_validation_draws else None
                ),
                validation_minimum_outer_draws=outer_draws,
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
                    # A 120-outcome path set is expensive to pickle for every
                    # GA candidate. Real-race profiling is faster with the
                    # shared in-process scenario graph and sparse settlement.
                    execution_backend="thread",
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
            validated_joint_value = selected.get("joint_value") or {}
            validated_joint_growth = selected.get("bankroll_growth") or {}
            portfolio = validated_joint_value.get("portfolio") or {}
            growth = validated_joint_growth.get("growth") or {}
            joint_value_audit = _joint_value_audit(validated_joint_value)
            best_search = search.get("best_search_candidate") or {}
            best_search_metrics = best_search.get("metrics") or {}
            best_search_bets = dict(best_search.get("bets_yen") or {})
            best_validation_value = (
                best_search.get("validation_joint_value") or {}
            )
            best_validation_growth = (
                best_search.get("validation_bankroll_growth") or {}
            )
            best_validation_portfolio = (
                best_validation_value.get("portfolio") or {}
            )
            best_validation_growth_summary = (
                best_validation_growth.get("growth") or {}
            )
            validated_bets = dict(selected.get("bets_yen") or {})
            best_search_stake = sum(best_search_bets.values())
            best_search_return = _realized_receipt(
                best_search_bets,
                actual_combination=str(race["actual_combination"]),
                actual_payout_yen=int(race["actual_payout_yen"]),
            ) if best_search_stake else 0
            race_rows.append({
                "race_id": observation.race_id,
                "race_date": evaluation_date,
                "venue": observation.venue,
                "wager_type": "trifecta",
                "popularity_band_at_t": observation.popularity_band,
                "evaluation_time_t": purchase_at.isoformat(),
                "evaluation_time_t_source": (
                    "decision_at" if race.get("decision_at") else "odds_deadline_at"
                ),
                "odds_snapshot_captured_at": snapshot_at.isoformat(),
                "snapshot_age_seconds": snapshot_age,
                "decision_horizon_seconds": observation.decision_horizon_seconds,
                "actual_combination": str(race["actual_combination"]),
                "actual_payout_yen": int(race["actual_payout_yen"]),
                "stake_yen": stake,
                "return_yen": receipt,
                "profit_yen": receipt - stake,
                "available_cash_after_bet_yen": balance,
                "settlement_available_at": settlement_at.isoformat(),
                "selected_tickets": len(selected_bets),
                "selected_bets_yen": selected_bets,
                "purchase_authorized": bool(search["purchase_authorized"]),
                "feasible_candidates_found": int(
                    search.get("feasible_candidates_found") or 0
                ),
                "optimizer_unique_candidate_evaluations": int(
                    search.get("unique_candidate_evaluations") or 0
                ),
                "optimizer_growth_evaluations": int(
                    search.get("growth_evaluations") or 0
                ),
                "optimizer_growth_evaluations_skipped": int(
                    search.get("growth_evaluations_skipped") or 0
                ),
                "search_outer_sample_count_r": int(
                    search.get("search_parameter_draws") or 0
                ),
                "validation_outer_sample_count_r": int(
                    search.get("validation_parameter_draws") or 0
                ),
                "validation_uses_separate_draw_set": bool(
                    search.get("validation_uses_separate_draw_set")
                ),
                "best_search_fitness": best_search.get("fitness"),
                "best_search_constraint_violation": best_search_metrics.get(
                    "constraint_violation"
                ),
                "best_search_edge_excess": best_search_metrics.get(
                    "edge_excess"
                ),
                "best_search_growth_excess": best_search_metrics.get(
                    "growth_excess"
                ),
                "best_search_bets_yen": best_search_bets,
                "purchase_value_bets_match_best_search": (
                    validated_bets == best_search_bets
                ),
                "best_search_stake_yen": best_search_stake,
                "best_search_hypothetical_return_yen": best_search_return,
                "best_search_hypothetical_profit_yen": (
                    best_search_return - best_search_stake
                ),
                "best_search_hypothetical_roi": (
                    best_search_return / best_search_stake
                    if best_search_stake else None
                ),
                "pregate_candidate_generated": best_search_stake > 0,
                "best_search_validation_portfolio_lower_quantile": (
                    best_validation_portfolio.get("lower_quantile")
                ),
                "best_search_validation_purchase_value_gate_passed": bool(
                    best_validation_portfolio.get(
                        "passes_purchase_gate"
                    )
                ),
                "best_search_validation_purchase_gate_evaluable": bool(
                    best_validation_portfolio.get(
                        "purchase_gate_evaluable"
                    )
                ),
                "best_search_validation_bankroll_growth_lower_quantile": (
                    best_validation_growth_summary.get("lower_quantile")
                ),
                "best_search_validation_growth_gate_passed": bool(
                    best_validation_growth_summary.get(
                        "passes_growth_gate"
                    )
                ),
                "best_search_validation_growth_evaluation_skipped": (
                    bool(best_search_bets)
                    and not bool(best_validation_growth)
                ),
                "purchase_value": portfolio.get("lower_quantile"),
                "predicted_roi_lower_bound": (
                    1.0 + float(portfolio["lower_quantile"])
                    if portfolio.get("lower_quantile") is not None
                    else None
                ),
                "purchase_safety_margin": buy_margin,
                "purchase_value_excess": (
                    float(portfolio["lower_quantile"]) - buy_margin
                    if portfolio.get("lower_quantile") is not None
                    else None
                ),
                "purchase_value_gate_passed": bool(
                    portfolio.get("passes_purchase_gate")
                ),
                "portfolio_lower_quantile": portfolio.get("lower_quantile"),
                "bankroll_growth_lower_quantile": growth.get("lower_quantile"),
                "maximum_conditional_ruin_probability": growth.get(
                    "maximum_conditional_ruin_probability"
                ),
                "pool_scale_lower_bound_yen": pool_bound.total_sales_yen,
                "pool_scale_method": pool_bound.method,
                "joint_value_audit": joint_value_audit,
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
    confidence = build_block_bootstrap_evidence(
        daily, samples=bootstrap_samples, seed=seed
    )
    purchased_races = [
        race
        for day in daily
        for race in day["races"]
        if int(race.get("stake_yen") or 0) > 0
    ]
    purchase_values = [
        float(race["portfolio_lower_quantile"])
        for race in purchased_races
        if race.get("portfolio_lower_quantile") is not None
    ]
    best_search_rows = [
        race
        for day in daily
        for race in day["races"]
        if int(race.get("best_search_stake_yen") or 0) > 0
    ]
    best_search_total_stake = sum(
        int(race["best_search_stake_yen"]) for race in best_search_rows
    )
    best_search_total_return = sum(
        int(race.get("best_search_hypothetical_return_yen") or 0)
        for race in best_search_rows
    )
    calibration_ledger = {
        "version": "joint_edge_calibration_ledger_v1",
        "role": "evaluation_only_never_used_by_same_period_purchase_gate",
        "candidate_portfolios": len(best_search_rows),
        "stake_yen": best_search_total_stake,
        "return_yen": best_search_total_return,
        "profit_yen": best_search_total_return - best_search_total_stake,
        "roi": (
            best_search_total_return / best_search_total_stake
            if best_search_total_stake else None
        ),
        "authorized_portfolios": sum(
            bool(race.get("purchase_authorized")) for race in best_search_rows
        ),
        "teacher_available_for_strictly_later_days": True,
    }
    purchase_value_realization_calibration = (
        _purchase_value_realization_calibration(
            daily,
            samples=bootstrap_samples,
            seed=seed,
        )
    )
    joint_value_audit = _aggregate_joint_value_audits([
        race.get("joint_value_audit") or {}
        for day in daily
        for race in day["races"]
    ])
    settlement_audit = {
        "version": "parimutuel_integer_settlement_v1",
        "integer_yen_accounting": True,
        "self_impact_repricing": True,
        "payout_rate_numerator": settlement_rules.payout_rate_numerator,
        "payout_rate_denominator": settlement_rules.payout_rate_denominator,
        "face_unit_yen": settlement_rules.face_unit_yen,
        "purchase_unit_yen": settlement_rules.purchase_unit_yen,
        "full_refund_terminal_states": list(
            settlement_rules.refund_terminal_states
        ),
        "partial_refund_supported": True,
        "special_payout_addition_supported": True,
        "rounding": "integer_pool_floor_per_face_unit",
    }
    joint_purchase_value = {
        "definition": (
            "Q_alpha over outer parameter draws of the lower-tail mean "
            "portfolio expected edge"
        ),
        "outer_alpha": 0.05,
        "outer_sample_count_r_requested": outer_draws,
        "search_outer_sample_count_r_requested": (
            effective_search_outer_draws
        ),
        "search_validation_draw_sets_disjoint": separate_validation_draws,
        "outer_tail_observations_requested": max(1, ceil(0.05 * outer_draws)),
        "minimum_outer_tail_observations_for_promotion": (
            MINIMUM_OUTER_TAIL_OBSERVATIONS
        ),
        "inner_scenario_count_s_requested": scenarios_per_draw,
        "minimum_inner_tail_effective_samples": (
            MINIMUM_INNER_TAIL_EFFECTIVE_SAMPLES
        ),
        "inner_tail_fraction": inner_tail_fraction,
        "outer_quantile_method": "inverted_cdf",
        "selected_portfolios": len(purchased_races),
        "evaluated_values": len(purchase_values),
        "safety_margin": buy_margin,
        "minimum": min(purchase_values) if purchase_values else None,
        "mean": (
            fsum(purchase_values) / len(purchase_values)
            if purchase_values else None
        ),
        "maximum": max(purchase_values) if purchase_values else None,
        "minimum_excess": (
            min(purchase_values) - buy_margin if purchase_values else None
        ),
        "all_above_safety_margin": bool(
            purchased_races
            and len(purchase_values) == len(purchased_races)
            and all(value > buy_margin for value in purchase_values)
        ),
    }
    configuration = {
        "terminal_min_training_days": terminal_min_training_days,
        "joint_min_training_days": joint_min_training_days,
        "outer_draws": outer_draws,
        "search_outer_draws": effective_search_outer_draws,
        "search_validation_draw_sets_disjoint": separate_validation_draws,
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
        "minimum_inner_tail_effective_samples": (
            MINIMUM_INNER_TAIL_EFFECTIVE_SAMPLES
        ),
        "population_size": population_size,
        "generations": generations,
        "bootstrap_samples": bootstrap_samples,
        "settlement_delay_seconds": settlement_delay_seconds,
        "seed": seed,
    }
    evaluation_protocol = _evaluation_protocol(
        scored_cache=scored_cache,
        eligible_races=races,
        observations=observations,
        evaluation_dates=[day["race_date"] for day in daily],
        terminal=terminal,
        configuration=configuration,
        outcomes=outcomes,
        settlement_audit=settlement_audit,
        bootstrap_condition_id=confidence["condition_id"],
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
        "joint_purchase_value_above_safety_margin": bool(
            joint_purchase_value["all_above_safety_margin"]
        ),
        "minimum_outer_tail_support": bool(
            joint_value_audit.get("outer_tail_support_for_promotion")
        ),
        "minimum_inner_tail_support": bool(
            joint_value_audit.get("inner_tail_support_for_promotion")
        ),
        "independent_search_validation_draw_sets": (
            separate_validation_draws
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
        "configuration": configuration,
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
        "joint_purchase_value": joint_purchase_value,
        "joint_value_audit": joint_value_audit,
        "settlement_audit": settlement_audit,
        "evaluation_protocol_id": evaluation_protocol["id"],
        "evaluation_protocol": evaluation_protocol["protocol"],
        "calibration_ledger": calibration_ledger,
        "purchase_value_realization_calibration": (
            purchase_value_realization_calibration
        ),
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
    parser.add_argument("--search-outer-draws", type=int)
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
