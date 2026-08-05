from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from ..archive_closing_odds import (
    OFFICIAL_SOURCE_KEY,
    SOURCE_KEY,
    ensure_archive_schema,
)
from ..bankroll_bootstrap import bootstrap_daily_roi
from ..db import connection, init_db
from .contextual_market_residual_v24 import (
    contextual_probabilities,
    fit_temporal_contextual_residual,
)
from .direct_context_empirical_v26 import (
    evaluate_temporal_direct_context_empirical,
)
from .direct_context_market_residual_v25 import (
    direct_context_probabilities,
    extract_lane_context,
    fit_temporal_direct_context_residual,
)
from .flat_policy import simulate_chronological_flat_policy
from .market_calibration import (
    _validate_artifact_before_period,
    blend_probabilities,
    iter_scored_artifact_feature_rows,
    normalized_market_probabilities,
    probability_metrics,
    simulate_policy,
    write_json_atomic,
)
from .market_residual import (
    fit_log_pool_newton,
    residual_probability_metrics,
)
from .course_interaction_residual import (
    evaluate_temporal_course_interaction,
)
from .pruned_direct_context_evaluation_v27 import (
    evaluate_temporal_pruned_residual,
)
from .payout_weighted_ranking import (
    evaluate_temporal_payout_weighted_roles,
)
from .conditional_ticket_residual_v30 import (
    evaluate_temporal_conditional_ticket_residual,
)
from .ticket_utility_ranking_v31 import (
    evaluate_temporal_ticket_utility_roles,
)
from .ticket_utility_ranking_v33 import (
    evaluate_calibration_aligned_ticket_utility,
)
from .nonlinear_market_residual_v38 import (
    fit_temporal_nonlinear_market_residual,
    nonlinear_residual_probabilities,
)
from .nested_nonlinear_value_v40 import (
    evaluate_nested_nonlinear_value_v40,
)
from .nonlinear_context_search_v41 import (
    fit_temporal_nonlinear_context_search,
)
from .stacked_market_residual_v42 import (
    fit_temporal_stacked_market_residual,
    stacked_probabilities,
)
from .nested_stacked_value_v43 import (
    evaluate_nested_stacked_value_v43,
)
from .mature_stacked_value import evaluate_mature_stacked_value


MODEL_NAME = "archive_closing_market_oracle_v1"
EVALUATION_VERSION = 23
TARGETED_TEMPORAL_COMPONENTS = (
    "mature_stacked_contextual_value",
    "mature_stacked_contextual_value_daily_refit",
    "mature_stacked_contextual_value_daily_refit_bandwise",
)
PRIMARY_CALIBRATOR = {"model_weight": 0.75, "temperature": 1.0}
PRIMARY_POLICY: dict[str, Any] = {
    "name": "preregistered_closing_oracle_ev105_120_odds80_r3_ratio105_kelly025",
    "ev_threshold": 1.05,
    "max_estimated_ev": 1.20,
    "max_odds": 80.0,
    "max_tickets_per_race": 3,
    "min_model_market_ratio": 1.05,
    "staking_mode": "kelly_025",
}
V23_TOP5_ORACLE_POLICY: dict[str, Any] = {
    "name": "observed_closing_oracle_top5_ev100_105_flat100_v1",
    "max_model_rank": 5,
    "min_odds": None,
    "max_odds": None,
    "ev_threshold": 1.0,
    "max_estimated_ev": 1.05,
    "min_model_market_ratio": 0.0,
    "stake_per_ticket_yen": 100,
}
TEMPORAL_RESIDUAL_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "name": "residual_top5_ev100_120_odds80_flat100_v1",
        "max_model_rank": 5,
        "min_odds": None,
        "max_odds": 80.0,
        "ev_threshold": 1.0,
        "max_estimated_ev": 1.20,
        "min_model_market_ratio": 0.0,
        "stake_per_ticket_yen": 100,
    },
    {
        "name": "residual_top10_ev105_150_odds80_flat100_v1",
        "max_model_rank": 10,
        "min_odds": None,
        "max_odds": 80.0,
        "ev_threshold": 1.05,
        "max_estimated_ev": 1.50,
        "min_model_market_ratio": 0.0,
        "stake_per_ticket_yen": 100,
    },
    {
        "name": "residual_top20_ev110_200_odds120_flat100_v1",
        "max_model_rank": 20,
        "min_odds": None,
        "max_odds": 120.0,
        "ev_threshold": 1.10,
        "max_estimated_ev": 2.0,
        "min_model_market_ratio": 0.0,
        "stake_per_ticket_yen": 100,
    },
    {
        "name": "residual_tail_ev105_150_odds100_500_flat100_v1",
        "max_model_rank": 120,
        "min_odds": 100.0,
        "max_odds": 500.0,
        "ev_threshold": 1.05,
        "max_estimated_ev": 1.50,
        "min_model_market_ratio": 0.0,
        "stake_per_ticket_yen": 100,
    },
)


def narrow_ev_diagnostic_policies() -> tuple[dict[str, Any], ...]:
    """Build the fixed research grid around break-even estimated EV.

    These policies are evaluated only on the untouched temporal split. Their
    results are diagnostic and cannot be used as promotion evidence.
    """
    policies: list[dict[str, Any]] = []
    for max_rank in (1, 3, 5):
        for lower, upper in (
            (0.95, 1.00),
            (1.00, 1.025),
            (1.025, 1.05),
            (1.05, 1.10),
            (1.10, 1.20),
        ):
            policies.append(
                {
                    "name": (
                        f"diagnostic_v25_top{max_rank}_"
                        f"ev{lower:.3f}_{upper:.3f}_odds80_flat100"
                    ),
                    "max_model_rank": max_rank,
                    "min_odds": None,
                    "max_odds": 80.0,
                    "ev_threshold": lower,
                    "max_estimated_ev": upper,
                    "min_model_market_ratio": 0.0,
                    "stake_per_ticket_yen": 100,
                }
            )
    return tuple(policies)


NARROW_EV_DIAGNOSTIC_POLICIES = narrow_ev_diagnostic_policies()
DIAGNOSTIC_CONFIGS: tuple[tuple[str, dict[str, float], dict[str, Any]], ...] = (
    ("model_only_conservative", {"model_weight": 1.0, "temperature": 1.0}, {
        **PRIMARY_POLICY, "name": "model_only_ev120_odds80_r3", "ev_threshold": 1.20,
        "max_estimated_ev": None,
    }),
    ("blend_075_primary", PRIMARY_CALIBRATOR, PRIMARY_POLICY),
    ("blend_050_conservative", {"model_weight": 0.5, "temperature": 1.0}, {
        **PRIMARY_POLICY, "name": "blend050_ev102_odds40_r1", "ev_threshold": 1.02,
        "max_odds": 40.0, "max_tickets_per_race": 1,
    }),
)


def restrict_probabilities_to_available(
    probabilities: Mapping[str, float], available: Iterable[str]
) -> dict[str, float]:
    keys = tuple(sorted(set(available)))
    if not keys or any(key not in probabilities for key in keys):
        raise ValueError("model probabilities do not cover archive market")
    values = {key: float(probabilities[key]) for key in keys}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("available model probability mass must be positive")
    return {key: value / total for key, value in values.items()}


def load_archive_markets(
    conn: Any,
    *,
    from_date: str,
    through_date: str,
    source_key: str = SOURCE_KEY,
) -> dict[str, dict[str, Any]]:
    ensure_archive_schema(conn)
    verification_statuses = (
        (
            "official_primary_winner_payout_match",
            "official_primary_special_settlement",
        )
        if source_key == OFFICIAL_SOURCE_KEY
        else (
            "winner_only_match_unverified_market",
            "winner_only_match_unverified_market",
        )
    )
    markets: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        """
        SELECT a.race_id, r.race_date, r.jcd, r.rno, r.deadline_at, a.odds_count,
               a.verification_status,
               p.combination AS actual_combination,
               p.payout_yen AS actual_payout_yen,
               p.popularity AS actual_popularity
        FROM archive_closing_odds_snapshots a
        JOIN races r ON r.race_id = a.race_id
        JOIN payouts p ON p.race_id = a.race_id AND p.bet_type = '3連単'
        WHERE a.source_key = ?
          AND a.verification_status IN (?, ?)
          AND r.race_date BETWEEN ? AND ?
        ORDER BY r.race_date, r.jcd, r.rno, p.combination, p.payout_yen
        """,
        (
            source_key,
            verification_statuses[0],
            verification_statuses[1],
            from_date,
            through_date,
        ),
    ):
        race_id = str(row["race_id"])
        settlement = {
            "race_id": race_id,
            "combination": str(row["actual_combination"]),
            "payout_yen": int(row["actual_payout_yen"]),
            "popularity": row["actual_popularity"],
        }
        if race_id not in markets:
            markets[race_id] = {
                "race_id": race_id,
                "race_date": str(row["race_date"]),
                "jcd": str(row["jcd"]),
                "rno": int(row["rno"]),
                "archive_odds_count": int(row["odds_count"]),
                "archive_verification_status": str(row["verification_status"]),
                "captured_at": (
                    str(row["deadline_at"]) if row["deadline_at"] else None
                ),
                "odds_deadline_at": (
                    str(row["deadline_at"]) if row["deadline_at"] else None
                ),
                "actual_combination": settlement["combination"],
                "actual_payout_yen": settlement["payout_yen"],
                "settlements": [],
                "odds": {},
            }
        markets[race_id]["settlements"].append(settlement)
    for row in conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM archive_closing_odds o
        JOIN races r ON r.race_id = o.race_id
        WHERE o.source_key = ? AND r.race_date BETWEEN ? AND ?
        ORDER BY o.race_id, o.combination
        """,
        (source_key, from_date, through_date),
    ):
        market = markets.get(str(row["race_id"]))
        if market is not None:
            market["odds"][str(row["combination"])] = float(row["odds"])
    for market in markets.values():
        market["settlements"] = tuple(market["settlements"])
    return markets


OFFICIAL_MINIMUM_COVERAGE_RATIO = 0.995


def official_archive_coverage(
    conn: Any,
    *,
    from_date: str,
    through_date: str,
) -> dict[str, Any]:
    ensure_archive_schema(conn)
    monthly: list[dict[str, Any]] = []
    for row in conn.execute(
        """
        WITH payout_targets AS (
          SELECT race_id
          FROM payouts
          WHERE bet_type = '3連単' AND payout_yen IS NOT NULL
          GROUP BY race_id
        )
        SELECT SUBSTR(CAST(r.race_date AS TEXT), 1, 7) AS month,
               COUNT(*) AS eligible_targets,
               SUM(CASE WHEN s.race_id IS NOT NULL THEN 1 ELSE 0 END)
                 AS stored_snapshots,
               SUM(CASE WHEN a.status = 'excluded_non_six_boat'
                        THEN 1 ELSE 0 END) AS excluded_non_six_boat,
               SUM(CASE WHEN a.status = 'invalid' THEN 1 ELSE 0 END)
                 AS invalid_attempts,
               SUM(CASE WHEN a.status IN (
                     'fetch_error', 'http_error', 'not_found'
                   ) THEN 1 ELSE 0 END) AS fetch_failure_attempts
        FROM races r
        JOIN payout_targets p ON p.race_id = r.race_id
        LEFT JOIN archive_closing_odds_snapshots s
          ON s.race_id = r.race_id AND s.source_key = ?
        LEFT JOIN archive_closing_odds_attempts a
          ON a.race_id = r.race_id AND a.source_key = ?
        WHERE r.race_date BETWEEN ? AND ?
        GROUP BY SUBSTR(CAST(r.race_date AS TEXT), 1, 7)
        ORDER BY month
        """,
        (
            OFFICIAL_SOURCE_KEY,
            OFFICIAL_SOURCE_KEY,
            from_date,
            through_date,
        ),
    ):
        eligible = int(row["eligible_targets"] or 0)
        stored = int(row["stored_snapshots"] or 0)
        excluded = int(row["excluded_non_six_boat"] or 0)
        expected = max(0, eligible - excluded)
        unresolved = max(0, expected - stored)
        coverage_ratio = stored / expected if expected else None
        monthly.append({
            "month": str(row["month"]),
            "eligible_target_races": eligible,
            "excluded_non_six_boat_races": excluded,
            "expected_six_boat_races": expected,
            "official_snapshot_races": stored,
            "unresolved_races": unresolved,
            "invalid_attempt_races": int(row["invalid_attempts"] or 0),
            "fetch_failure_attempt_races": int(
                row["fetch_failure_attempts"] or 0
            ),
            "coverage_ratio": coverage_ratio,
            "coverage_ready": (
                coverage_ratio is not None
                and coverage_ratio >= OFFICIAL_MINIMUM_COVERAGE_RATIO
            ),
        })

    eligible = sum(row["eligible_target_races"] for row in monthly)
    excluded = sum(row["excluded_non_six_boat_races"] for row in monthly)
    expected = sum(row["expected_six_boat_races"] for row in monthly)
    stored = sum(row["official_snapshot_races"] for row in monthly)
    unresolved = sum(row["unresolved_races"] for row in monthly)
    invalid = sum(row["invalid_attempt_races"] for row in monthly)
    fetch_failed = sum(
        row["fetch_failure_attempt_races"] for row in monthly
    )
    coverage_ratio = stored / expected if expected else None
    return {
        "official_eligible_target_races": eligible,
        "official_excluded_non_six_boat_races": excluded,
        "official_expected_six_boat_races": expected,
        "official_snapshot_races": stored,
        "official_unresolved_races": unresolved,
        "official_invalid_attempt_races": invalid,
        "official_fetch_failure_attempt_races": fetch_failed,
        "official_coverage_ratio": coverage_ratio,
        "official_minimum_required_coverage": OFFICIAL_MINIMUM_COVERAGE_RATIO,
        "official_coverage_ready": (
            coverage_ratio is not None
            and coverage_ratio >= OFFICIAL_MINIMUM_COVERAGE_RATIO
        ),
        "official_monthly_coverage": monthly,
    }


def score_archive_markets(
    conn: Any,
    *,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_artifact_before_period(artifact, from_date=from_date)
    coverage = official_archive_coverage(
        conn,
        from_date=from_date,
        through_date=through_date,
    )
    markets = load_archive_markets(
        conn, from_date=from_date, through_date=through_date,
        source_key=OFFICIAL_SOURCE_KEY,
    )
    target_ids = set(markets)
    races: list[dict[str, Any]] = []
    skipped_incomplete = skipped_probability = 0
    feature_scored_ids: set[str] = set()
    for feature_rows, probabilities in iter_scored_artifact_feature_rows(
        conn, target_ids=target_ids, artifact=artifact
    ):
        race_id = str(feature_rows[0]["meta"]["race_id"])
        feature_scored_ids.add(race_id)
        market = markets.get(race_id)
        if market is None:
            continue
        odds = market["odds"]
        settlement_combinations = {
            str(row["combination"])
            for row in market.get("settlements") or ()
        }
        if (
            len(odds) != int(market["archive_odds_count"])
            or not 1 <= len(odds) <= 120
            or not settlement_combinations
            or not settlement_combinations.issubset(odds)
        ):
            skipped_incomplete += 1
            continue
        try:
            available_model = restrict_probabilities_to_available(
                probabilities, odds
            )
        except ValueError:
            skipped_probability += 1
            continue
        market_probabilities = normalized_market_probabilities(odds)
        if set(market_probabilities) != set(odds):
            skipped_incomplete += 1
            continue
        races.append({
            **market,
            "model_probabilities": available_model,
            "market_probabilities": market_probabilities,
            "lane_context": extract_lane_context(feature_rows),
            "archive_source_key": OFFICIAL_SOURCE_KEY,
            "archive_market_role": "closing_oracle_research_only",
        })
    races.sort(key=lambda row: (row["race_date"], row["jcd"], row["rno"]))
    return races, {
        **coverage,
        "archive_target_races": len(markets),
        "evaluated_races": len(races),
        "partial_market_races": sum(
            int(len(row["odds"]) < 120) for row in races
        ),
        "skipped_no_complete_features": len(target_ids - feature_scored_ids),
        "skipped_incomplete_archive": skipped_incomplete,
        "skipped_probability_mismatch": skipped_probability,
        "fully_official_verified_races": sum(
            int(row["archive_verification_status"] == "all_market_official_match")
            for row in races
        ),
        "winner_verified_secondary_races": sum(
            int(
                row["archive_verification_status"]
                == "winner_only_match_unverified_market"
            )
            for row in races
        ),
    }


def temporal_residual_diagnostic(
    races: list[dict[str, Any]],
    *,
    calibration_fraction: float = 0.75,
    regularization: float = 0.01,
    calibration_through: str | None = None,
    daily_budget_yen: int = 10_000,
    temporal_component: str | None = None,
) -> dict[str, Any]:
    """Fit a market residual on prior days and score untouched later days."""
    if (
        temporal_component is not None
        and temporal_component not in TARGETED_TEMPORAL_COMPONENTS
    ):
        raise ValueError(f"unknown temporal component: {temporal_component}")
    dates = sorted({str(race["race_date"]) for race in races})
    if len(dates) < 4:
        return {
            "status": "insufficient_days",
            "dates": len(dates),
            "calibration_days": 0,
            "evaluation_days": 0,
        }
    if calibration_through is None:
        split_index = max(
            1,
            min(len(dates) - 1, int(len(dates) * calibration_fraction)),
        )
    else:
        split_index = sum(race_date <= calibration_through for race_date in dates)
        if split_index < 1 or split_index >= len(dates):
            raise ValueError(
                "calibration-through must leave calibration and evaluation days"
            )
    calibration_dates = set(dates[:split_index])
    evaluation_dates = set(dates[split_index:])
    calibration = [
        race for race in races if str(race["race_date"]) in calibration_dates
    ]
    evaluation = [
        race for race in races if str(race["race_date"]) in evaluation_dates
    ]
    if temporal_component in TARGETED_TEMPORAL_COMPONENTS:
        calibration_update_mode = (
            "daily_strict_prior_refit"
            if temporal_component in {
                "mature_stacked_contextual_value_daily_refit",
                "mature_stacked_contextual_value_daily_refit_bandwise",
            }
            else "fixed"
        )
        value_shape_constraint = (
            "bandwise"
            if temporal_component
            == "mature_stacked_contextual_value_daily_refit_bandwise"
            else "isotonic"
        )
        mature = evaluate_mature_stacked_value(
            calibration,
            evaluation,
            daily_budget_yen=daily_budget_yen,
            calibration_update_mode=calibration_update_mode,
            value_shape_constraint=value_shape_constraint,
        )
        probability_metrics = mature.get("evaluation_probability_metrics")
        probability_metrics = (
            probability_metrics
            if isinstance(probability_metrics, Mapping)
            else {"evaluated_races": len(evaluation)}
        )
        probability_artifact = mature.get("probability_artifact")
        probability_artifact = (
            probability_artifact
            if isinstance(probability_artifact, Mapping)
            else {}
        )
        probability_selection = mature.get("probability_selection")
        probability_selection = (
            probability_selection
            if isinstance(probability_selection, Mapping)
            else {}
        )
        return {
            "status": "completed",
            "validation_design": (
                "Targeted execution of the preregistered mature stacked value "
                "component on the same prior calibration and untouched outer "
                "period; unrelated residual families are not refit."
            ),
            "targeted_temporal_component": temporal_component,
            "calibration_from": dates[0],
            "calibration_through": dates[split_index - 1],
            "evaluation_from": dates[split_index],
            "evaluation_through": dates[-1],
            "calibration_days": split_index,
            "evaluation_days": len(dates) - split_index,
            "calibration_races": len(calibration),
            "evaluation_races": len(evaluation),
            "stacked_market_residual_v42": {
                "status": mature.get("status"),
                "metrics": dict(probability_metrics),
                "artifact": dict(probability_artifact),
                "selected_stack": probability_selection.get("selected_stack"),
                "selected_weights": probability_selection.get("selected_weights"),
                "outer_period_used_for_selection": False,
            },
            "mature_stacked_contextual_value": mature,
        }

    calibrator = fit_log_pool_newton(
        calibration,
        regularization=regularization,
    )
    metrics = residual_probability_metrics(
        evaluation,
        calibrator,
        include_raw_model=True,
    )
    purchase_diagnostics = []
    for policy in TEMPORAL_RESIDUAL_POLICIES:
        simulation = simulate_chronological_flat_policy(
            evaluation,
            calibrator={
                "model_weight": float(calibrator["model_weight"]),
                "temperature": float(calibrator["temperature"]),
            },
            policy=policy,
            probability_blender=blend_probabilities,
            initial_bankroll_yen=daily_budget_yen,
        )
        bootstrap = (
            bootstrap_daily_roi(simulation["daily"])
            if simulation["daily"]
            else {
                "days": 0,
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        purchase_diagnostics.append(
            {
                "policy": dict(policy),
                "simulation": simulation,
                "bootstrap": bootstrap,
            }
        )
    contextual = fit_temporal_contextual_residual(calibration, evaluation)
    contextual_evaluation = [
        {
            **race,
            "model_probabilities": contextual_probabilities(
                race,
                contextual["artifact"],
            ),
        }
        for race in evaluation
    ]
    contextual_purchase_diagnostics = []
    for policy in TEMPORAL_RESIDUAL_POLICIES:
        simulation = simulate_chronological_flat_policy(
            contextual_evaluation,
            calibrator={"model_weight": 1.0, "temperature": 1.0},
            policy=policy,
            probability_blender=blend_probabilities,
            initial_bankroll_yen=daily_budget_yen,
        )
        bootstrap = (
            bootstrap_daily_roi(simulation["daily"])
            if simulation["daily"]
            else {
                "days": 0,
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        contextual_purchase_diagnostics.append(
            {
                "policy": dict(policy),
                "simulation": simulation,
                "bootstrap": bootstrap,
            }
        )
    contextual["purchase_diagnostics"] = contextual_purchase_diagnostics
    direct_context = fit_temporal_direct_context_residual(calibration, evaluation)
    direct_context_evaluation = [
        {
            **race,
            "model_probabilities": direct_context_probabilities(
                race,
                direct_context["artifact"],
            ),
        }
        for race in evaluation
    ]
    direct_context_purchase_diagnostics = []
    for policy in (*TEMPORAL_RESIDUAL_POLICIES, *NARROW_EV_DIAGNOSTIC_POLICIES):
        simulation = simulate_chronological_flat_policy(
            direct_context_evaluation,
            calibrator={"model_weight": 1.0, "temperature": 1.0},
            policy=policy,
            probability_blender=blend_probabilities,
            initial_bankroll_yen=daily_budget_yen,
        )
        bootstrap = (
            bootstrap_daily_roi(simulation["daily"])
            if simulation["daily"]
            else {
                "days": 0,
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        direct_context_purchase_diagnostics.append(
            {
                "policy": dict(policy),
                "simulation": simulation,
                "bootstrap": bootstrap,
            }
        )
    direct_context["purchase_diagnostics"] = direct_context_purchase_diagnostics
    direct_context["purchase_diagnostic_role"] = (
        "retrospective untouched-split research only; selected rows require "
        "registration and later prospective evaluation"
    )
    direct_context_empirical = evaluate_temporal_direct_context_empirical(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    pruned_direct_context = evaluate_temporal_pruned_residual(
        calibration,
        evaluation,
        policies=TEMPORAL_RESIDUAL_POLICIES,
        daily_budget_yen=daily_budget_yen,
    )
    course_interaction = evaluate_temporal_course_interaction(
        calibration,
        evaluation,
        policies=TEMPORAL_RESIDUAL_POLICIES,
        daily_budget_yen=daily_budget_yen,
    )
    payout_weighted_roles = evaluate_temporal_payout_weighted_roles(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    conditional_ticket_residual = evaluate_temporal_conditional_ticket_residual(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    ticket_utility_roles = evaluate_temporal_ticket_utility_roles(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
        probability_artifact=(
            payout_weighted_roles.get("probability_artifact")
            if isinstance(payout_weighted_roles, Mapping)
            else None
        ),
    )
    calibration_aligned_ticket_utility = (
        evaluate_calibration_aligned_ticket_utility(
            calibration,
            evaluation,
            daily_budget_yen=daily_budget_yen,
        )
    )
    nonlinear_purchase_diagnostics = []
    if len({str(race["race_date"]) for race in calibration}) >= 5:
        nonlinear_market_residual = fit_temporal_nonlinear_market_residual(
            calibration,
            evaluation,
        )
        for shrinkage, role in (
            (
                float(nonlinear_market_residual["selected_shrinkage"]),
                "inner_log_loss_selected",
            ),
            (1.0, "fixed_full_residual_research_control"),
        ):
            nonlinear_evaluation = [
                {
                    **race,
                    "model_probabilities": nonlinear_residual_probabilities(
                        race,
                        nonlinear_market_residual["artifact"],
                        shrinkage=shrinkage,
                    ),
                }
                for race in evaluation
            ]
            for policy in TEMPORAL_RESIDUAL_POLICIES:
                simulation = simulate_chronological_flat_policy(
                    nonlinear_evaluation,
                    calibrator={"model_weight": 1.0, "temperature": 1.0},
                    policy=policy,
                    probability_blender=blend_probabilities,
                    initial_bankroll_yen=daily_budget_yen,
                )
                bootstrap = (
                    bootstrap_daily_roi(simulation["daily"])
                    if simulation["daily"]
                    else {
                        "days": 0,
                        "roi": None,
                        "roi_ci95_lower": None,
                        "probability_roi_above_one": None,
                    }
                )
                nonlinear_purchase_diagnostics.append({
                    "role": role,
                    "shrinkage": shrinkage,
                    "policy": dict(policy),
                    "simulation": simulation,
                    "bootstrap": bootstrap,
                })
    else:
        nonlinear_market_residual = {
            "model": "nonlinear_market_offset_residual_v38",
            "status": "insufficient_calibration_days",
            "calibration_days": len(
                {str(race["race_date"]) for race in calibration}
            ),
            "required_calibration_days": 5,
        }
    nonlinear_market_residual["purchase_diagnostics"] = (
        nonlinear_purchase_diagnostics
    )
    nonlinear_market_residual["purchase_diagnostic_role"] = (
        "untouched outer-split research only; full residual strength is a "
        "fixed sensitivity control and cannot be selected from outer ROI"
    )
    if len({str(race["race_date"]) for race in calibration}) >= 5:
        nonlinear_context_search = fit_temporal_nonlinear_context_search(
            calibration,
            evaluation,
        )
    else:
        nonlinear_context_search = {
            "model": "nonlinear_market_offset_context_search_v41",
            "status": "insufficient_calibration_days",
            "calibration_days": len(
                {str(race["race_date"]) for race in calibration}
            ),
            "required_calibration_days": 5,
        }
    nonlinear_context_purchase_diagnostics = []
    nonlinear_context_roles = (
        (
            (
                float(nonlinear_context_search["selected_shrinkage"]),
                "inner_log_loss_selected",
            ),
            (1.0, "fixed_full_residual_research_control"),
        )
        if "selected_shrinkage" in nonlinear_context_search
        else ()
    )
    for shrinkage, role in nonlinear_context_roles:
        nonlinear_context_evaluation = [
            {
                **race,
                "model_probabilities": nonlinear_residual_probabilities(
                    race,
                    nonlinear_context_search["artifact"],
                    shrinkage=shrinkage,
                ),
            }
            for race in evaluation
        ]
        for policy in TEMPORAL_RESIDUAL_POLICIES:
            simulation = simulate_chronological_flat_policy(
                nonlinear_context_evaluation,
                calibrator={"model_weight": 1.0, "temperature": 1.0},
                policy=policy,
                probability_blender=blend_probabilities,
                initial_bankroll_yen=daily_budget_yen,
            )
            bootstrap = (
                bootstrap_daily_roi(simulation["daily"])
                if simulation["daily"]
                else {
                    "days": 0,
                    "roi": None,
                    "roi_ci95_lower": None,
                    "probability_roi_above_one": None,
                }
            )
            nonlinear_context_purchase_diagnostics.append({
                "role": role,
                "shrinkage": shrinkage,
                "policy": dict(policy),
                "simulation": simulation,
                "bootstrap": bootstrap,
            })
    nonlinear_context_search["purchase_diagnostics"] = (
        nonlinear_context_purchase_diagnostics
    )
    nonlinear_context_search["purchase_diagnostic_role"] = (
        "untouched outer-split research only; context breadth is selected by "
        "inner log loss and never by outer ROI"
    )
    if len({str(race["race_date"]) for race in calibration}) >= 10:
        stacked_market_residual = fit_temporal_stacked_market_residual(
            calibration,
            evaluation,
        )
    else:
        stacked_market_residual = {
            "model": "stacked_market_residual_v42",
            "status": "insufficient_calibration_days",
            "calibration_days": len(
                {str(race["race_date"]) for race in calibration}
            ),
            "required_calibration_days": 10,
        }
    stacked_purchase_diagnostics = []
    if "artifact" in stacked_market_residual:
        stacked_evaluation = [
            {
                **race,
                "model_probabilities": stacked_probabilities(
                    race, stacked_market_residual["artifact"]
                ),
            }
            for race in evaluation
        ]
        for policy in TEMPORAL_RESIDUAL_POLICIES:
            simulation = simulate_chronological_flat_policy(
                stacked_evaluation,
                calibrator={"model_weight": 1.0, "temperature": 1.0},
                policy=policy,
                probability_blender=blend_probabilities,
                initial_bankroll_yen=daily_budget_yen,
            )
            bootstrap = (
                bootstrap_daily_roi(simulation["daily"])
                if simulation["daily"]
                else {
                    "days": 0,
                    "roi": None,
                    "roi_ci95_lower": None,
                    "probability_roi_above_one": None,
                }
            )
            stacked_purchase_diagnostics.append({
                "role": "inner_log_loss_selected_stack",
                "policy": dict(policy),
                "simulation": simulation,
                "bootstrap": bootstrap,
            })
    stacked_market_residual["purchase_diagnostics"] = (
        stacked_purchase_diagnostics
    )
    stacked_market_residual["purchase_diagnostic_role"] = (
        "untouched outer-split research only; stack membership and weights are "
        "selected exclusively on prior-day validation"
    )
    nested_nonlinear_value = evaluate_nested_nonlinear_value_v40(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    nested_stacked_value = evaluate_nested_stacked_value_v43(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    mature_stacked_value = evaluate_mature_stacked_value(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
    )
    return {
        "status": "completed",
        "validation_design": (
            "Residual coefficients are fit on the earliest complete days and "
            "scored once on untouched later days"
        ),
        "calibration_from": dates[0],
        "calibration_through": dates[split_index - 1],
        "evaluation_from": dates[split_index],
        "evaluation_through": dates[-1],
        "calibration_days": split_index,
        "evaluation_days": len(dates) - split_index,
        "calibration_races": len(calibration),
        "evaluation_races": len(evaluation),
        "calibrator": calibrator,
        "metrics": metrics,
        "purchase_diagnostics": purchase_diagnostics,
        "contextual_market_residual_v24": contextual,
        "direct_context_market_residual_v25": direct_context,
        "direct_context_empirical_lcb_v26": direct_context_empirical,
        "pruned_direct_context_market_residual_v27": pruned_direct_context,
        "course_interaction_market_residual_v28": course_interaction,
        "payout_weighted_role_model_v29": payout_weighted_roles,
        "conditional_ticket_residual_v30": conditional_ticket_residual,
        "ticket_utility_robust_temporal_ranking_v32": ticket_utility_roles,
        "ticket_utility_calibration_aligned_v33": calibration_aligned_ticket_utility,
        "nonlinear_market_offset_residual_v38": nonlinear_market_residual,
        "nested_nonlinear_value_calibration_v40": nested_nonlinear_value,
        "nonlinear_market_offset_context_search_v41": nonlinear_context_search,
        "stacked_market_residual_v42": stacked_market_residual,
        "nested_stacked_value_calibration_v43": nested_stacked_value,
        "mature_stacked_contextual_value": mature_stacked_value,
    }


def evaluate_archive_oracle(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    temporal_calibration_through: str | None = None,
    temporal_component: str | None = None,
) -> dict[str, Any]:
    diagnostics = []
    primary = None
    for name, calibrator, policy in DIAGNOSTIC_CONFIGS:
        result = simulate_policy(
            races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
        )
        item = {
            "name": name,
            "calibrator": dict(calibrator),
            "policy": dict(policy),
            **result,
        }
        diagnostics.append(item)
        if name == "blend_075_primary":
            primary = item
    if primary is None:
        raise AssertionError("primary oracle policy is missing")
    v23_top5_oracle = simulate_chronological_flat_policy(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy=V23_TOP5_ORACLE_POLICY,
        probability_blender=blend_probabilities,
        initial_bankroll_yen=daily_budget_yen,
    )
    v23_top5_oracle_bootstrap = (
        bootstrap_daily_roi(v23_top5_oracle["daily"])
        if v23_top5_oracle["daily"]
        else {"days": 0, "roi": None, "roi_ci95_lower": None,
              "probability_roi_above_one": None}
    )
    bootstrap = (
        bootstrap_daily_roi(primary["daily"])
        if primary["daily"]
        else {
            "days": 0, "roi": None, "roi_ci95_lower": None,
            "probability_roi_above_one": None,
        }
    )
    research_gate = {
        "minimum_days": int(primary["race_days"]) >= 300,
        "minimum_races": len(races) >= 40_000,
        "minimum_tickets": int(primary["tickets"]) >= 1_000,
        "positive_profit": int(primary["profit_yen"]) > 0,
        "roi_above_one": float(primary["roi"]) > 1.0,
        "roi_lower95_above_one": float(bootstrap.get("roi_ci95_lower") or 0.0) > 1.0,
        "largest_hit_excluded_roi_above_one": float(
            primary.get("roi_without_largest_hit") or 0.0
        ) > 1.0,
        "effective_hit_count": float(primary.get("effective_hit_count") or 0.0) >= 30.0,
    }
    temporal_residual = temporal_residual_diagnostic(
        races,
        calibration_through=temporal_calibration_through,
        daily_budget_yen=daily_budget_yen,
        temporal_component=temporal_component,
    )
    prediction = probability_metrics(races, calibrator=PRIMARY_CALIBRATOR)
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "comparison_role": "unavailable_at_decision_closing_oracle_research_only",
        "market_source_scope": (
            "boatrace official historical closing trifecta odds"
        ),
        "production_transfer_required": True,
        "promotion_eligible": False,
        "research_gate_pass": all(research_gate.values()),
        "research_gate": research_gate,
        "probability_metrics": prediction,
        "trifecta_log_loss": prediction["calibrated_trifecta_log_loss"],
        "trifecta_top5_hit_rate": prediction[
            "calibrated_trifecta_top5_hit_rate"
        ],
        "winner_log_loss": prediction["calibrated_winner_log_loss"],
        "winner_top1_accuracy": prediction[
            "calibrated_winner_top1_accuracy"
        ],
        "temporal_residual_diagnostic": temporal_residual,
        "v23_top5_observed_closing_oracle": v23_top5_oracle,
        "v23_top5_observed_closing_oracle_bootstrap": v23_top5_oracle_bootstrap,
        "primary": primary,
        "primary_bootstrap": bootstrap,
        "diagnostics": diagnostics,
        "evaluated_races": len(races),
        "roi": primary["roi"],
        "profit_yen": primary["profit_yen"],
        "stake_yen": primary["stake_yen"],
        "return_yen": primary["return_yen"],
        "max_drawdown_yen": primary["max_drawdown_yen"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-bounded research upper bound using archived closing odds"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--temporal-calibration-through")
    parser.add_argument(
        "--temporal-component",
        choices=TARGETED_TEMPORAL_COMPONENTS,
    )
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def externalize_targeted_mature_evidence(
    result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    temporal = result.get("temporal_residual_diagnostic")
    if not isinstance(temporal, Mapping):
        return result
    mature = temporal.get("mature_stacked_contextual_value")
    stacked = temporal.get("stacked_market_residual_v42")
    if not isinstance(mature, Mapping) or not isinstance(stacked, Mapping):
        return result
    sidecar_path = output_path.with_suffix(".research.joblib")
    temporary = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    sidecar_payload = {
        "schema_version": 1,
        "model": mature.get("model"),
        "mature_stacked_contextual_value": dict(mature),
        "stacked_market_residual_v42": dict(stacked),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sidecar_payload, temporary)
    temporary.replace(sidecar_path)
    sidecar_sha256 = _file_sha256(sidecar_path)

    bankroll = dict(mature.get("bankroll") or {})
    full_daily = [
        dict(row) for row in bankroll.get("daily") or ()
        if isinstance(row, Mapping)
    ]
    audit_fields = {
        "candidate_decision_audit",
        "eligible_candidate_audit",
        "selected_sample",
    }
    compact_daily = [
        {key: value for key, value in row.items() if key not in audit_fields}
        for row in full_daily
    ]
    decision_count = sum(
        len(row.get("candidate_decision_audit") or ())
        for row in full_daily
    )
    eligible_count = sum(
        len(row.get("eligible_candidate_audit") or ())
        for row in full_daily
    )
    probability_artifact = mature.get("probability_artifact")
    artifact_sha256 = (
        probability_artifact.get("artifact_sha256")
        if isinstance(probability_artifact, Mapping)
        else None
    )
    sidecar = {
        "path": str(sidecar_path),
        "sha256": sidecar_sha256,
        "bytes": sidecar_path.stat().st_size,
        "format": "joblib",
        "schema_version": 1,
        "contains_full_probability_artifact": True,
        "contains_full_candidate_decision_audit": True,
        "candidate_decision_count": decision_count,
        "eligible_candidate_count": eligible_count,
    }
    compact_mature = {
        **dict(mature),
        "probability_artifact": {
            "externalized": True,
            "artifact_sha256": artifact_sha256,
            "sidecar_sha256": sidecar_sha256,
        },
        "bankroll": {**bankroll, "daily": compact_daily},
        "research_sidecar": sidecar,
    }
    compact_stacked = {
        **dict(stacked),
        "artifact": {
            "externalized": True,
            "artifact_sha256": artifact_sha256,
            "sidecar_sha256": sidecar_sha256,
        },
        "research_sidecar": sidecar,
    }
    compact_temporal = {
        **dict(temporal),
        "stacked_market_residual_v42": compact_stacked,
        "mature_stacked_contextual_value": compact_mature,
        "research_sidecar": sidecar,
    }
    return {
        **result,
        "temporal_residual_diagnostic": compact_temporal,
        "research_sidecar": sidecar,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_date > args.through_date:
        raise ValueError("from-date must not be after through-date")
    if args.daily_budget_yen < 100:
        raise ValueError("daily-budget-yen must be at least 100")
    init_db(args.db)
    artifact = joblib.load(args.model)
    with connection(args.db) as conn:
        races, dataset = score_archive_markets(
            conn,
            artifact=artifact,
            from_date=args.from_date,
            through_date=args.through_date,
        )
    result = evaluate_archive_oracle(
        races,
        daily_budget_yen=args.daily_budget_yen,
        temporal_calibration_through=args.temporal_calibration_through,
        temporal_component=args.temporal_component,
    )
    result.update({
        "evaluation_version": EVALUATION_VERSION,
        "targeted_temporal_component": args.temporal_component,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "from_date": args.from_date,
        "through_date": args.through_date,
        "source_model": str(args.model),
        "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "dataset": dataset,
    })
    if args.temporal_component in TARGETED_TEMPORAL_COMPONENTS:
        result = externalize_targeted_mature_evidence(result, args.output)
    write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"diagnostics", "primary"}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
