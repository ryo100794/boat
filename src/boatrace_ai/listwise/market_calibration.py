from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from statistics import NormalDist
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np

from ..adaptive_allocation import allocate_adaptive_day
from ..archive_closing_odds import OFFICIAL_SOURCE_KEY, SOURCE_KEY
from ..bankroll_bootstrap import (
    DEFAULT_CHUNK_SIZE as BANKROLL_BOOTSTRAP_CHUNK_SIZE,
    DEFAULT_SEED as BANKROLL_BOOTSTRAP_SEED,
    bootstrap_daily_roi,
)
from ..bankroll_backtest import _load_trifecta_payouts
from ..chronological_bankroll import (
    settlement_events_from_races,
    simulate_chronological_bankroll_day,
    summarize_chronological_bankroll_days,
)
from ..db import connection, init_db
from ..feature_tuning import (
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from ..features import (
    MODEL_DECISION_LEAD_MINUTES,
    MODEL_FEATURE_CUTOFF_FROM_START_MINUTES,
    latest_trifecta_odds_before_deadline,
    stored_jst_timestamp_sql,
)
from ..modeling import trifecta_predictions
from ..odds_quality import TRIFECTA_PARSER_VERSION, plausible_trifecta_odds
from .bankroll_diagnostics import sequential_top5_ev_kelly_diagnostic
from .flat_policy import (
    select_flat_policy,
    simulate_chronological_flat_policy,
    simulate_flat_policy,
    summarize_flat_candidates,
)
from .dual_head_conformal_policy_v32 import (
    POLICY as V32_DUAL_HEAD_CONFORMAL_POLICY,
    REGISTERED_AFTER as V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER,
    simulate_dual_head_conformal_policy_v32,
)
from .market_edge_diagnostics import edge_records, summarize_edge_records
from .direct_context_market_residual_v25 import (
    FEATURE_DIMENSION,
    extract_lane_context,
)
from .v25_top1_narrow_policy_v33 import (
    POLICY as V33_V25_TOP1_NARROW_POLICY,
    REGISTERED_AFTER as V33_V25_TOP1_NARROW_REGISTERED_AFTER,
    simulate_v25_top1_narrow_v33,
)
from .odds_path_operational import (
    attach_odds_path_model,
    fit_odds_path_model,
    fit_performance_priors,
)
from .cluster_bootstrap import paired_cluster_mean_bootstrap
from .closing_odds import decision_odds
from .closing_odds_momentum import (
    attach_selected_closing_odds,
    select_closing_odds_model,
    selected_closing_odds_metrics,
)
from .closing_odds_quantile import (
    walk_forward_closing_odds_quantiles,
)
from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .empirical_lcb_policy import (
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .conditional_order import ConditionalOrderModel, conditional_probabilities
from .conditional_stagewise import (
    ConditionalStagewiseModel,
    conditional_position_utilities,
)
from .model import ListwiseLinearModel, stable_softmax
from .paired_bootstrap import paired_mean_bootstrap
from .stagewise_blend import (
    StagewiseBlendModel,
    blend_probabilities as blend_architecture_probabilities,
)
from .stagewise_mlp import classifier_position_scores, stagewise_trifecta_probabilities
from ..fast_math import TRIFECTA_COMBINATIONS


CLOSING_ODDS_SOURCE_PRIORITY = (OFFICIAL_SOURCE_KEY, SOURCE_KEY)


MODEL_NAME = "listwise_newton_market_calibrated_v1"
MARKET_EVALUATION_VERSION = 33
MARKET_FORMAL_EVALUATION_FROM = "2026-07-22"
EV_BAND_HYPOTHESIS_REGISTERED_AFTER = "2026-07-25"
CONSERVATIVE_MARKET_KELLY_REGISTERED_AFTER = "2026-07-28"
CONFORMAL_LOWER_KELLY_REGISTERED_AFTER = "2026-07-28"
TREND_POINT_KELLY_REGISTERED_AFTER = "2026-07-28"
# exp(0.177 observed closing-odds log-MAE) is about 1.19. Register the
# rounded 1.20 haircut before evaluating any later unseen day.
CONSERVATIVE_MARKET_KELLY_ODDS_SAFETY_FACTOR = 1.20
MARKET_MAX_SNAPSHOT_AGE_SECONDS = 65.0
CLEAN_DAY_CALIBRATOR_STRATEGIES = frozenset({
    "odds_path_role_integrated_registered_band_lcb_v14",
    "odds_path_role_integrated_selection_free_envelope_v15",
})
ODDS_CHECKPOINT_SCHEMA_VERSION = 1
ODDS_CHECKPOINT_OFFSETS_SECONDS = (300, 120, 60, 30, 10)
PREFETCH_CHECKPOINTS_KEY = "odds_checkpoints"
SCORED_CACHE_VERSION = 14
MIN_CLOSING_ODDS_TRAINING_DAYS = 7
MIN_CLOSING_ODDS_TRAINING_RACES = 500
STAKE_YEN = 100
MIN_EMPIRICAL_LCB_EVALUATION_DAYS = 30
MIN_EMPIRICAL_LCB_TICKETS = 300
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (0.75, 1.0, 1.25)
EV_THRESHOLDS = (1.05, 1.10, 1.15, 1.20, 1.30, 1.50)
MAX_ODDS = (20.0, 40.0, 80.0, None)
MAX_ESTIMATED_EV = (1.10, 1.20)
MAX_TICKETS_PER_RACE = (1, 2, 3, 5)
MIN_MODEL_MARKET_RATIOS = (1.0, 1.05, 1.10, 1.20)
STAKING_MODES = {
    "kelly_025": {
        "fractional_kelly": 0.25,
        "allocation_mode": "kelly_floor",
        "min_daily_exposure_fraction": 0.0,
    },
    "kelly_100": {
        "fractional_kelly": 1.0,
        "allocation_mode": "kelly_floor",
        "min_daily_exposure_fraction": 0.0,
    },
    "normalized_010": {
        "fractional_kelly": 0.25,
        "allocation_mode": "normalized_kelly",
        "min_daily_exposure_fraction": 0.10,
    },
}
EPSILON = 1e-12
REGISTERED_EV_BAND_POLICY: dict[str, Any] = {
    "name": "registered_ev1.00_to1.10_r3_kelly100",
    "ev_threshold": 1.0,
    "max_estimated_ev": 1.10,
    "max_odds": None,
    "max_tickets_per_race": 3,
    "min_model_market_ratio": 1.0,
    "staking_mode": "kelly_100",
}
PROSPECTIVE_NORMALIZED_EV_REGISTERED_AFTER = "2026-07-27"
PROSPECTIVE_NORMALIZED_EV_POLICY: dict[str, Any] = {
    "name": "registered_ev1.00_to1.10_r3_normalized010_v2",
    "ev_threshold": 1.0,
    "max_estimated_ev": 1.10,
    "max_odds": None,
    "max_tickets_per_race": 3,
    "min_model_market_ratio": 1.0,
    "staking_mode": "normalized_010",
}
PROSPECTIVE_TOP5_NARROW_EV_REGISTERED_AFTER = "2026-07-28"
OBSERVED_CLOSING_RETURN_V4_REGISTERED_AFTER = "2026-07-29"
PREQUENTIAL_SHRINKAGE_RETURN_V6_REGISTERED_AFTER = "2026-07-29"
V17_STRATEGY_NAME = "odds_path_observed_closing_return_robust_policy_v17"
V17_MODEL_NAME = V17_STRATEGY_NAME
V17_COMPARISON_ROLE = (
    "strict_prior_observed_closing_robust_policy_chronological_shadow"
)
V18_STRATEGY_NAME = (
    "odds_path_observed_closing_return_schedule_quota_v18"
)
V18_MODEL_NAME = V18_STRATEGY_NAME
V18_COMPARISON_ROLE = (
    "strict_prior_observed_closing_schedule_quota_chronological_shadow"
)
V19_STRATEGY_NAME = (
    "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19"
)
V19_MODEL_NAME = V19_STRATEGY_NAME
V19_COMPARISON_ROLE = (
    "strict_prior_observed_closing_schedule_quota_raw_nonregression_challenger"
)
V20_STRATEGY_NAME = (
    "odds_path_observed_closing_return_schedule_quota_dual_head_v20"
)
V20_MODEL_NAME = V20_STRATEGY_NAME
V20_COMPARISON_ROLE = (
    "strict_prior_dual_head_probability_v19_purchase_v18_evaluation_only"
)
V21_STRATEGY_NAME = (
    "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
)
V21_MODEL_NAME = V21_STRATEGY_NAME
V21_COMPARISON_ROLE = (
    "strict_prior_triple_head_probability_v19_ranking_v18_purchase_v18_evaluation_only"
)
V35_STRATEGY_NAME = (
    "odds_path_observed_closing_return_stable_policy_triple_head_v35"
)
V35_MODEL_NAME = V35_STRATEGY_NAME
V35_COMPARISON_ROLE = (
    "strict_prior_triple_head_v21_stable_paired_policy_v35_evaluation_only"
)
V35_MIN_ADAPTIVE_POLICY_DAYS = 7
V35_MIN_ADAPTIVE_QUOTA_DAYS = 30
V35_FIXED_DAILY_TICKET_LIMIT = 10
V35_MIN_DAILY_TICKET_LIMIT = 5
V35_MAX_DAILY_TICKET_LIMIT = 20
V35_FAMILY_WISE_ALPHA = 0.05
V18_TICKET_LIMIT_QUANTILE = 0.25
V17_POLICY_BOOTSTRAP_SAMPLES = 2_000
MIN_PROSPECTIVE_ARCHITECTURE_DAYS = 30
SCHEDULE_QUOTA_STRATEGIES = frozenset({
    V18_STRATEGY_NAME,
    V19_STRATEGY_NAME,
    V20_STRATEGY_NAME,
    V21_STRATEGY_NAME,
    V35_STRATEGY_NAME,
})
ROBUST_POLICY_STRATEGIES = frozenset({
    V17_STRATEGY_NAME,
    V18_STRATEGY_NAME,
    V19_STRATEGY_NAME,
})
EVALUATION_ONLY_STRATEGIES = frozenset({
    V20_STRATEGY_NAME,
    V21_STRATEGY_NAME,
    V35_STRATEGY_NAME,
})
CHRONOLOGICAL_BANKROLL_STRATEGIES = frozenset({
    *ROBUST_POLICY_STRATEGIES,
    *EVALUATION_ONLY_STRATEGIES,
})
TRIPLE_HEAD_STRATEGIES = frozenset({
    V21_STRATEGY_NAME,
    V35_STRATEGY_NAME,
})
MIN_PROSPECTIVE_ARCHITECTURE_TICKETS = 300
V6_RETURN_HIT_PRIORS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
V6_RETURN_MULTIPLIER_BOUNDS = (
    (0.50, 1.50),
    (0.75, 1.25),
    (0.90, 1.10),
)
V6_FALLBACK_RETURN_PARAMETERS = {
    "return_hit_prior": 20.0,
    "min_return_multiplier": 0.75,
    "max_return_multiplier": 1.25,
}
V6_MIN_HISTORY_DAYS = 4
V6_MIN_INNER_EVALUATION_DAYS = 2
V6_MAX_INNER_EVALUATION_DAYS = 7
PROSPECTIVE_TOP5_NARROW_EV_POLICY: dict[str, Any] = {
    "name": "registered_top5_ev1.00_to1.05_flat100_v1",
    "max_model_rank": 5,
    "min_odds": None,
    "max_odds": None,
    "ev_threshold": 1.0,
    "max_estimated_ev": 1.05,
    "min_model_market_ratio": 0.0,
    "stake_per_ticket_yen": STAKE_YEN,
}


def robust_policy_comparison_role(calibrator_strategy: str) -> str:
    if calibrator_strategy == V35_STRATEGY_NAME:
        return V35_COMPARISON_ROLE
    if calibrator_strategy == V21_STRATEGY_NAME:
        return V21_COMPARISON_ROLE
    if calibrator_strategy == V20_STRATEGY_NAME:
        return V20_COMPARISON_ROLE
    if calibrator_strategy == V19_STRATEGY_NAME:
        return V19_COMPARISON_ROLE
    if calibrator_strategy in SCHEDULE_QUOTA_STRATEGIES:
        return V18_COMPARISON_ROLE
    return V17_COMPARISON_ROLE


def fit_market_residual_calibrator(
    races: list[dict[str, Any]],
    *,
    calibrator_strategy: str,
) -> dict[str, Any]:
    from .market_residual import (
        fit_fixed_regularization,
        select_regularization_prequential,
    )

    enforce_raw_nonregression = calibrator_strategy == V19_STRATEGY_NAME
    if len({str(race["race_date"]) for race in races}) >= 2:
        return select_regularization_prequential(
            races,
            enforce_raw_nonregression=enforce_raw_nonregression,
        )
    return fit_fixed_regularization(
        races,
        enforce_raw_nonregression=enforce_raw_nonregression,
    )


def fit_v20_dual_head_calibrators(
    races: list[dict[str, Any]],
) -> dict[str, Any]:
    training_dates = sorted({str(race["race_date"]) for race in races})
    probability_selection = fit_market_residual_calibrator(
        races,
        calibrator_strategy=V19_STRATEGY_NAME,
    )
    purchase_selection = fit_market_residual_calibrator(
        races,
        calibrator_strategy=V18_STRATEGY_NAME,
    )
    return {
        "architecture": "strict_prior_dual_calibrator_heads_v20",
        "selection_data": "strict_prior_training_and_inner_prequential_folds_only",
        "outer_holdout_used": False,
        "training_dates": training_dates,
        "trained_through_date": training_dates[-1] if training_dates else None,
        "probability_head": {
            "role": "probability_reporting_and_promotion_calibration",
            "calibrator_strategy": V19_STRATEGY_NAME,
            "raw_nonregression_enforced": True,
            "calibrator": dict(probability_selection["final_calibrator"]),
            "selection": probability_selection,
        },
        "purchase_head": {
            "role": "purchase_policy_and_chronological_bankroll",
            "calibrator_strategy": V18_STRATEGY_NAME,
            "raw_nonregression_enforced": False,
            "policy_strategy": V18_STRATEGY_NAME,
            "calibrator": dict(purchase_selection["final_calibrator"]),
            "selection": purchase_selection,
        },
    }


def fit_v21_triple_head_calibrators(
    races: list[dict[str, Any]],
) -> dict[str, Any]:
    training_dates = sorted({str(race["race_date"]) for race in races})
    probability_selection = fit_market_residual_calibrator(
        races,
        calibrator_strategy=V19_STRATEGY_NAME,
    )
    v18_selection = fit_market_residual_calibrator(
        races,
        calibrator_strategy=V18_STRATEGY_NAME,
    )
    probability_calibrator = dict(probability_selection["final_calibrator"])
    v18_calibrator = dict(v18_selection["final_calibrator"])
    return {
        "architecture": "strict_prior_triple_calibrator_heads_v21",
        "selection_data": "strict_prior_training_and_inner_prequential_folds_only",
        "outer_holdout_used": False,
        "training_dates": training_dates,
        "trained_through_date": training_dates[-1] if training_dates else None,
        "probability_head": {
            "role": "winner_and_trifecta_logloss",
            "calibrator_strategy": V19_STRATEGY_NAME,
            "raw_nonregression_enforced": True,
            "calibrator": probability_calibrator,
            "selection": probability_selection,
        },
        "ranking_head": {
            "role": "trifecta_top5_ranking",
            "calibrator_strategy": V18_STRATEGY_NAME,
            "raw_nonregression_enforced": False,
            "calibrator": v18_calibrator,
            "selection": v18_selection,
        },
        "purchase_head": {
            "role": "purchase_policy_and_chronological_bankroll",
            "calibrator_strategy": V18_STRATEGY_NAME,
            "raw_nonregression_enforced": False,
            "policy_strategy": V18_STRATEGY_NAME,
            "calibrator": dict(v18_calibrator),
            "selection": v18_selection,
        },
        "ranking_purchase_share_v18_selection": True,
    }


def odds_path_model_name(calibrator_strategy: str) -> str:
    if calibrator_strategy == "odds_path_return":
        return "odds_path_operational_v1"
    if calibrator_strategy == "odds_path_probability":
        return "odds_path_probability_only_v2"
    if calibrator_strategy == "odds_path_closing_return":
        return "odds_path_closing_return_v3"
    if calibrator_strategy == "odds_path_observed_closing_return":
        return "odds_path_observed_closing_return_v4"
    if calibrator_strategy == V35_STRATEGY_NAME:
        return V35_MODEL_NAME
    if calibrator_strategy == V21_STRATEGY_NAME:
        return V21_MODEL_NAME
    if calibrator_strategy == V20_STRATEGY_NAME:
        return V20_MODEL_NAME
    if calibrator_strategy == V19_STRATEGY_NAME:
        return V19_MODEL_NAME
    if calibrator_strategy in SCHEDULE_QUOTA_STRATEGIES:
        return V18_MODEL_NAME
    if calibrator_strategy == V17_STRATEGY_NAME:
        return V17_MODEL_NAME
    if calibrator_strategy == "odds_path_hit_shrunk_return":
        return "odds_path_hit_shrunk_closing_return_v5"
    if calibrator_strategy == "odds_path_prequential_shrinkage_return":
        return "odds_path_prequential_shrinkage_return_v6"
    if calibrator_strategy == "odds_path_crossfit_conservative_ev":
        return "odds_path_crossfit_conservative_ev_v7"
    if calibrator_strategy == "odds_path_market_offset_crossfit_conservative_ev":
        return "odds_path_market_offset_crossfit_conservative_ev_v8"
    if calibrator_strategy == "odds_path_market_offset_discrete_log_ev_v9":
        return "odds_path_market_offset_discrete_log_ev_v9"
    if (
        calibrator_strategy
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    ):
        return "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    if calibrator_strategy == "odds_path_role_integrated_multihorizon_v11":
        return "odds_path_role_integrated_multihorizon_v11"
    if calibrator_strategy == "odds_path_role_integrated_t300_nonlinear_v12":
        return "odds_path_role_integrated_t300_nonlinear_v12"
    if calibrator_strategy == "odds_path_role_integrated_edge_conditional_lcb_v13":
        return "odds_path_role_integrated_edge_conditional_lcb_v13"
    if calibrator_strategy == "odds_path_role_integrated_registered_band_lcb_v14":
        return "odds_path_role_integrated_registered_band_lcb_v14"
    if calibrator_strategy == "odds_path_role_integrated_selection_free_envelope_v15":
        return "odds_path_role_integrated_selection_free_envelope_v15"
    if calibrator_strategy == "odds_path_role_integrated_fixed_band_passthrough_v16":
        return "odds_path_role_integrated_fixed_band_passthrough_v16"
    return MODEL_NAME


def select_calibrator_evaluation_races(
    calibrator_strategy: str,
    *,
    races: list[dict[str, Any]],
    clean_races: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if calibrator_strategy in CLEAN_DAY_CALIBRATOR_STRATEGIES:
        return clean_races
    return races


def attach_forecast_closing_return_prices(
    races: list[dict[str, Any]],
    closing_policy_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    adjusted = apply_prequential_closing_odds_policy_inputs(
        races,
        closing_policy_inputs,
    )
    prices_by_race_id = {
        str(race["race_id"]): decision_odds(race) for race in adjusted
    }
    result = []
    for race in races:
        item = dict(race)
        item["performance_return_odds"] = prices_by_race_id[str(race["race_id"])]
        result.append(item)
    return result


def attach_observed_closing_return_prices(
    races: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for race in races:
        observed = race.get("closing_odds") or {}
        prices = observed if len(observed) == 120 else decision_odds(race)
        item = dict(race)
        item["performance_return_odds"] = {
            str(combination): float(odds)
            for combination, odds in prices.items()
        }
        result.append(item)
    return result


def _v6_fallback_selection(
    *,
    training_dates: list[str],
    reason: str,
    candidates: list[dict[str, Any]] | None = None,
    inner_evaluation_dates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "method": "inner_prequential_robust_return_selection",
        "status": "conservative_fallback",
        "fallback_reason": reason,
        "leakage_guard": "every inner validation date uses strictly earlier dates",
        "training_dates": list(training_dates),
        "inner_evaluation_dates": list(inner_evaluation_dates or []),
        "selected": dict(V6_FALLBACK_RETURN_PARAMETERS),
        "minimum_history_days": V6_MIN_HISTORY_DAYS,
        "minimum_inner_evaluation_days": V6_MIN_INNER_EVALUATION_DAYS,
        "candidates": list(candidates or []),
    }


def _v6_selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["roi_without_largest_hit"]),
        float(row["profitable_day_fraction"]),
        float(row["median_profit_per_day_yen"]),
        float(row["tickets"]),
        float(row["return_hit_prior"]),
        -(
            float(row["max_return_multiplier"])
            - float(row["min_return_multiplier"])
        ),
        float(row["min_return_multiplier"]),
    )


def select_return_shrinkage_prequential(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
) -> dict[str, Any]:
    """Select v6 return shrinkage without observing the outer holdout date."""
    prepared_races = attach_observed_closing_return_prices(races)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in prepared_races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < V6_MIN_HISTORY_DAYS:
        return _v6_fallback_selection(
            training_dates=dates,
            reason="insufficient_history_days",
        )

    inner_dates = dates[2:][-V6_MAX_INNER_EVALUATION_DAYS:]
    if len(inner_dates) < V6_MIN_INNER_EVALUATION_DAYS:
        return _v6_fallback_selection(
            training_dates=dates,
            inner_evaluation_dates=inner_dates,
            reason="insufficient_inner_evaluation_days",
        )

    closing_policy_inputs = prequential_closing_odds_policy_inputs(prepared_races)
    from .market_residual import select_regularization_prequential

    inner_contexts = []
    for evaluation_date in inner_dates:
        training = [
            race
            for date in dates
            if date < evaluation_date
            for race in by_day[date]
        ]
        base_model = fit_odds_path_model(
            training,
            return_price_basis="observed_closing",
        )
        transformed_training = attach_odds_path_model(training, base_model)
        calibrator = dict(
            select_regularization_prequential(transformed_training)[
                "final_calibrator"
            ]
        )
        inner_contexts.append(
            {
                "training": training,
                "holdout": by_day[evaluation_date],
                "base_model": base_model,
                "calibrator": calibrator,
            }
        )

    candidates: list[dict[str, Any]] = []
    for return_hit_prior in V6_RETURN_HIT_PRIORS:
        for min_multiplier, max_multiplier in V6_RETURN_MULTIPLIER_BOUNDS:
            daily_rows: list[dict[str, Any]] = []
            evaluated_races = 0
            for context in inner_contexts:
                candidate_model = dict(context["base_model"])
                candidate_model["performance_priors"] = fit_performance_priors(
                    context["training"],
                    return_hit_prior=return_hit_prior,
                    min_return_multiplier=min_multiplier,
                    max_return_multiplier=max_multiplier,
                )
                transformed_holdout = attach_odds_path_model(
                    context["holdout"],
                    candidate_model,
                )
                policy_holdout = apply_prequential_closing_odds_policy_inputs(
                    transformed_holdout,
                    closing_policy_inputs,
                )
                result = simulate_policy(
                    policy_holdout,
                    calibrator=context["calibrator"],
                    policy=REGISTERED_EV_BAND_POLICY,
                    daily_budget_yen=daily_budget_yen,
                )
                daily_rows.extend(result["daily"])
                evaluated_races += len(policy_holdout)

            reliability = bankroll_reliability_metrics(
                daily_rows,
                evaluated_races=evaluated_races,
            )
            tickets = sum(int(row.get("tickets") or 0) for row in daily_rows)
            stake_yen = sum(int(row.get("stake_yen") or 0) for row in daily_rows)
            return_yen = sum(int(row.get("return_yen") or 0) for row in daily_rows)
            winning_days = sum(int(row.get("profit_yen") or 0) > 0 for row in daily_rows)
            minimum_tickets = max(10, 2 * len(inner_dates))
            candidates.append(
                {
                    "return_hit_prior": float(return_hit_prior),
                    "min_return_multiplier": float(min_multiplier),
                    "max_return_multiplier": float(max_multiplier),
                    "inner_evaluation_days": len(daily_rows),
                    "inner_evaluated_races": evaluated_races,
                    "tickets": tickets,
                    "minimum_tickets": minimum_tickets,
                    "ticket_sufficiency_pass": tickets >= minimum_tickets,
                    "stake_yen": stake_yen,
                    "return_yen": return_yen,
                    "profit_yen": return_yen - stake_yen,
                    "roi": return_yen / stake_yen if stake_yen else 0.0,
                    "roi_without_largest_hit": float(
                        reliability.get("roi_without_largest_hit") or 0.0
                    ),
                    "winning_days": winning_days,
                    "profitable_day_fraction": (
                        winning_days / len(daily_rows) if daily_rows else 0.0
                    ),
                    "median_profit_per_day_yen": float(
                        np.median(
                            [int(row.get("profit_yen") or 0) for row in daily_rows]
                        )
                        if daily_rows
                        else 0.0
                    ),
                }
            )

    eligible = [row for row in candidates if row["ticket_sufficiency_pass"]]
    if not eligible:
        return _v6_fallback_selection(
            training_dates=dates,
            inner_evaluation_dates=inner_dates,
            candidates=candidates,
            reason="insufficient_inner_tickets",
        )
    selected = max(
        eligible,
        key=_v6_selection_key,
    )
    return {
        "version": 1,
        "method": "inner_prequential_robust_return_selection",
        "status": "selected",
        "fallback_reason": None,
        "leakage_guard": "every inner validation date uses strictly earlier dates",
        "training_dates": dates,
        "inner_evaluation_dates": inner_dates,
        "selected": {
            key: selected[key]
            for key in (
                "return_hit_prior",
                "min_return_multiplier",
                "max_return_multiplier",
            )
        },
        "selection_order": [
            "roi_without_largest_hit",
            "profitable_day_fraction",
            "median_profit_per_day_yen",
            "tickets",
            "conservative_shrinkage_tiebreak",
        ],
        "minimum_history_days": V6_MIN_HISTORY_DAYS,
        "minimum_inner_evaluation_days": V6_MIN_INNER_EVALUATION_DAYS,
        "candidates": candidates,
    }


def fit_v6_odds_path_model(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = attach_observed_closing_return_prices(races)
    selection = select_return_shrinkage_prequential(
        prepared,
        daily_budget_yen=daily_budget_yen,
    )
    parameters = selection["selected"]
    model = fit_odds_path_model(
        prepared,
        return_price_basis="observed_closing",
        return_hit_prior=float(parameters["return_hit_prior"]),
        min_return_multiplier=float(parameters["min_return_multiplier"]),
        max_return_multiplier=float(parameters["max_return_multiplier"]),
        adaptive_return_selection=selection,
    )
    return model, prepared


def artifact_drop_feature_groups(artifact: dict[str, Any]) -> tuple[str, ...]:
    return normalize_drop_feature_groups(
        artifact.get("drop_feature_groups") or (),
    )


def iter_artifact_feature_rows(
    conn,
    *,
    target_ids: set[str],
    artifact: dict[str, Any],
):
    return iter_race_feature_rows(
        conn,
        include_races=target_ids,
        drop_feature_groups=artifact_drop_feature_groups(artifact),
        feature_schema_version=artifact.get("feature_schema_version"),
    )


def artifact_model_probabilities(
    artifact: dict[str, Any],
    feature_rows: list[dict[str, Any]],
) -> dict[str, float]:
    model = artifact.get("model")
    hasher = artifact.get("hasher")
    if hasher is None:
        raise ValueError("model artifact lacks a feature hasher")
    classifier = artifact.get("classifier")
    model_kind = str(artifact.get("model_kind") or "").strip().lower()
    if model is None and classifier is not None:
        return artifact_classifier_probabilities_batch(
            artifact, [feature_rows]
        )[0]
    matrix = _ensure_sparse_index32(
        hasher.transform([to_hashable(item["features"]) for item in feature_rows])
    )
    if isinstance(model, ListwiseLinearModel):
        scores = np.asarray(model.scaler.transform(matrix).dot(model.weights)).reshape(6)
        lane_probabilities = stable_softmax(scores)
        return {
            row["combination"]: float(row["probability"])
            for row in trifecta_predictions(
                {lane: float(lane_probabilities[lane - 1]) for lane in range(1, 7)}
            )
        }
    if isinstance(model, StagewiseBlendModel):
        listwise = model.listwise_model
        lane_probabilities = stable_softmax(
            np.asarray(listwise.scaler.transform(matrix).dot(listwise.weights)).reshape(1, 6)
        )
        listwise_probabilities = stagewise_trifecta_probabilities(
            np.repeat(lane_probabilities[:, :, None], 3, axis=2)
        )
        _classes, position_scores = classifier_position_scores(
            model.stagewise_model, matrix
        )
        stagewise_probabilities = stagewise_trifecta_probabilities(
            position_scores.reshape(1, 6, 3)
        )
        probabilities = blend_architecture_probabilities(
            listwise_probabilities,
            stagewise_probabilities,
            stagewise_weight=model.stagewise_weight,
        )[0]
        return {
            "-".join(str(lane) for lane in combination): float(probability)
            for combination, probability in zip(TRIFECTA_COMBINATIONS, probabilities)
        }
    if isinstance(model, ConditionalStagewiseModel):
        probabilities = stagewise_trifecta_probabilities(
            conditional_position_utilities(model, matrix)
        )[0]
        return {
            "-".join(str(lane) for lane in combination): float(probability)
            for combination, probability in zip(TRIFECTA_COMBINATIONS, probabilities)
        }
    raise ValueError("unsupported model artifact type for market scoring")


def artifact_classifier_probabilities_batch(
    artifact: dict[str, Any],
    feature_races: list[list[dict[str, Any]]],
) -> list[dict[str, float]]:
    if not feature_races:
        return []
    classifier = artifact.get("classifier")
    hasher = artifact.get("hasher")
    model_kind = str(artifact.get("model_kind") or "").strip().lower()
    if (
        classifier is None
        or hasher is None
        or model_kind not in {"linear", "mlp", "lightgbm"}
    ):
        raise ValueError("unsupported classifier model kind for market scoring")
    if any(len(feature_rows) != 6 for feature_rows in feature_races):
        raise ValueError("classifier market scoring requires six lanes per race")
    flattened = [row for feature_rows in feature_races for row in feature_rows]
    matrix = _ensure_sparse_index32(
        hasher.transform([to_hashable(item["features"]) for item in flattened])
    )
    scaler = artifact.get("scaler")
    transformed = matrix if scaler is None else scaler.transform(matrix)
    raw = np.asarray(classifier.predict_proba(transformed), dtype=np.float64)
    classes = np.asarray(getattr(classifier, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if (
        raw.ndim != 2
        or raw.shape[0] != len(feature_races) * 6
        or len(positive) != 1
    ):
        raise ValueError(
            "classifier artifact must score six lanes per race for binary winner probability"
        )
    lane_scores = raw[:, int(positive[0])].reshape(len(feature_races), 6)
    if not np.all(np.isfinite(lane_scores)) or np.any(lane_scores < 0.0):
        raise ValueError("classifier artifact returned invalid winner probabilities")
    totals = lane_scores.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("classifier artifact returned zero winner probability mass")
    lane_probabilities = lane_scores / totals
    order_model = artifact.get("conditional_order_model")
    trifecta_matrix = None
    if order_model is not None:
        if not isinstance(order_model, ConditionalOrderModel):
            raise ValueError("classifier artifact has an invalid conditional order model")
        trifecta_matrix = conditional_probabilities(
            np.log(np.clip(lane_probabilities, 1e-15, 1.0)),
            order_model,
        )
    result = []
    for race_index, probabilities in enumerate(lane_probabilities):
        trifecta_values = (
            trifecta_matrix[race_index] if trifecta_matrix is not None else None
        )
        result.append(
            {
                row["combination"]: float(row["probability"])
                for row in trifecta_predictions(
                    {
                        lane: float(probabilities[lane - 1])
                        for lane in range(1, 7)
                    },
                    trifecta_probabilities=trifecta_values,
                )
            }
        )
    return result


def iter_scored_artifact_feature_rows(
    conn,
    *,
    target_ids: set[str],
    artifact: dict[str, Any],
    batch_races: int = 128,
):
    iterator = iter_artifact_feature_rows(
        conn,
        target_ids=target_ids,
        artifact=artifact,
    )
    classifier_batch = bool(
        artifact.get("classifier") is not None
        and str(artifact.get("model_kind") or "").strip().lower()
        in {"linear", "mlp", "lightgbm"}
    )
    while True:
        batch = list(islice(iterator, batch_races if classifier_batch else 1))
        if not batch:
            return
        probabilities = (
            artifact_classifier_probabilities_batch(artifact, batch)
            if classifier_batch
            else [artifact_model_probabilities(artifact, batch[0])]
        )
        yield from zip(batch, probabilities)


def normalized_market_probabilities(odds: dict[str, float]) -> dict[str, float]:
    inverse = {
        combination: 1.0 / float(value)
        for combination, value in odds.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    total = sum(inverse.values())
    if not inverse or total <= 0.0:
        return {}
    return {combination: value / total for combination, value in inverse.items()}


def blend_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between zero and one")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive")
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    logits = np.asarray(
        [
            (
                model_weight * math.log(max(EPSILON, float(model[combination])))
                + (1.0 - model_weight)
                * math.log(max(EPSILON, float(market[combination])))
            )
            / temperature
            for combination in combinations
        ],
        dtype=np.float64,
    )
    probabilities = stable_softmax(logits)
    return {
        combination: float(probability)
        for combination, probability in zip(combinations, probabilities)
    }


def select_calibrator(races: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not races:
        raise ValueError("calibration requires at least one race")
    rows: list[dict[str, float]] = []
    for model_weight in BLEND_WEIGHTS:
        for temperature in TEMPERATURES:
            losses = []
            top5_hits = 0
            for race in races:
                probabilities = blend_probabilities(
                    race["model_probabilities"],
                    race["market_probabilities"],
                    model_weight=model_weight,
                    temperature=temperature,
                )
                actual = str(race["actual_combination"])
                losses.append(-math.log(max(EPSILON, probabilities.get(actual, 0.0))))
                top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
                top5_hits += int(actual in top5)
            rows.append(
                {
                    "model_weight": model_weight,
                    "temperature": temperature,
                    "trifecta_log_loss": sum(losses) / len(losses),
                    "trifecta_top5_hit_rate": top5_hits / len(losses),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            row["trifecta_log_loss"],
            -row["trifecta_top5_hit_rate"],
            row["model_weight"],
        ),
    )
    return {
        "model_weight": float(selected["model_weight"]),
        "temperature": float(selected["temperature"]),
    }, rows


def default_policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [{"name": "no_bet", "no_bet": True}]
    for ev_threshold in EV_THRESHOLDS:
        ev_caps = (None,) + tuple(
            cap for cap in MAX_ESTIMATED_EV if cap > ev_threshold
        )
        for max_ev in ev_caps:
            for max_odds in MAX_ODDS:
                for max_tickets in MAX_TICKETS_PER_RACE:
                    for min_ratio in MIN_MODEL_MARKET_RATIOS:
                        for staking_mode in STAKING_MODES:
                            odds_name = "none" if max_odds is None else str(int(max_odds))
                            cap_name = "" if max_ev is None else f"_evcap{int(max_ev * 100)}"
                            policies.append(
                                {
                                    "name": (
                                        f"ev{ev_threshold:.2f}_odds{odds_name}_"
                                        f"r{max_tickets}_ratio{min_ratio:.2f}"
                                        f"{cap_name}_{staking_mode}"
                                    ),
                                    "ev_threshold": ev_threshold,
                                    "max_estimated_ev": max_ev,
                                    "max_odds": max_odds,
                                    "max_tickets_per_race": max_tickets,
                                    "min_model_market_ratio": min_ratio,
                                    "staking_mode": staking_mode,
                                }
                            )
    return policies


def simulate_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    policy: dict[str, Any],
    daily_budget_yen: int,
    prepared_policy_matrix: dict[str, Any] | None = None,
    include_chronological: bool = False,
    include_robust_metrics: bool = False,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    races_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        races_by_day[str(race["race_date"])].append(race)
    if not policy.get("no_bet") and prepared_policy_matrix is not None:
        for race in races:
            evaluated_by_day[str(race["race_date"])].add(str(race["race_id"]))
        estimated_ev = prepared_policy_matrix["estimated_ev"]
        odds_matrix = prepared_policy_matrix["odds"]
        ratio_matrix = prepared_policy_matrix["model_market_ratio"]
        mask = estimated_ev >= float(policy["ev_threshold"])
        if policy.get("max_odds") is not None:
            mask &= odds_matrix <= float(policy["max_odds"])
        mask &= ratio_matrix >= float(policy["min_model_market_ratio"])
        if policy.get("max_estimated_ev") is not None:
            mask &= estimated_ev <= float(policy["max_estimated_ev"])
        order = prepared_policy_matrix["order"]
        ordered_mask = np.take_along_axis(mask, order, axis=1)
        selected_mask = ordered_mask & (
            np.cumsum(ordered_mask, axis=1)
            <= int(policy["max_tickets_per_race"])
        )
        combinations = prepared_policy_matrix["combinations"]
        probability_matrix = prepared_policy_matrix["probabilities"]
        market_matrix = prepared_policy_matrix["market_probabilities"]
        model_matrix = prepared_policy_matrix["model_probabilities"]
        multiplier_matrix = prepared_policy_matrix["return_multipliers"]
        for race_index, race in enumerate(races):
            for ordered_index in np.flatnonzero(selected_mask[race_index]):
                combination_index = int(order[race_index, ordered_index])
                combination = combinations[combination_index]
                odds = float(odds_matrix[race_index, combination_index])
                probability = float(
                    probability_matrix[race_index, combination_index]
                )
                market_probability = float(
                    market_matrix[race_index, combination_index]
                )
                return_multiplier = float(
                    multiplier_matrix[race_index, combination_index]
                )
                by_day[str(race["race_date"])].append(
                    policy_candidate(
                        race,
                        combination=combination,
                        probability=probability,
                        market_probability=market_probability,
                        model_probability=float(
                            model_matrix[race_index, combination_index]
                        ),
                        odds=odds,
                        estimated_ev=float(
                            estimated_ev[race_index, combination_index]
                        ),
                        return_multiplier=return_multiplier,
                    )
                )
    elif not policy.get("no_bet"):
        for race in races:
            race_id = str(race["race_id"])
            race_date = str(race["race_date"])
            evaluated_by_day[race_date].add(race_id)
            calibrated = race.get("_policy_calibrated_probabilities")
            if calibrated is None:
                calibrated = blend_probabilities(
                    race["model_probabilities"],
                    race["market_probabilities"],
                    model_weight=float(calibrator["model_weight"]),
                    temperature=float(calibrator["temperature"]),
                )
            candidates = []
            for combination, probability in calibrated.items():
                odds = float(decision_odds(race)[combination])
                market_probability = float(race["market_probabilities"][combination])
                return_multiplier = float(
                    (race.get("historical_return_multipliers") or {}).get(
                        combination, 1.0
                    )
                )
                estimated_ev = probability * odds * return_multiplier
                if estimated_ev < float(policy["ev_threshold"]):
                    continue
                if policy.get("max_odds") is not None and odds > float(policy["max_odds"]):
                    continue
                ratio = probability / max(EPSILON, market_probability)
                if ratio < float(policy["min_model_market_ratio"]):
                    continue
                if policy.get("max_estimated_ev") is not None and estimated_ev > float(
                    policy["max_estimated_ev"]
                ):
                    continue
                candidates.append(
                    policy_candidate(
                        race,
                        combination=combination,
                        probability=probability,
                        market_probability=market_probability,
                        model_probability=float(
                            race["model_probabilities"][combination]
                        ),
                        odds=odds,
                        estimated_ev=estimated_ev,
                        return_multiplier=return_multiplier,
                    )
                )
            candidates.sort(
                key=lambda item: (item["estimated_ev"], item["probability"]),
                reverse=True,
            )
            by_day[race_date].extend(
                candidates[: int(policy["max_tickets_per_race"])]
            )
    else:
        for race in races:
            evaluated_by_day[str(race["race_date"])].add(str(race["race_id"]))

    daily = []
    chronological_daily = []
    stake_yen = return_yen = tickets = hit_tickets = 0
    cumulative_profit = peak_profit = max_drawdown_yen = 0
    for race_date in sorted(evaluated_by_day):
        staking = STAKING_MODES.get(
            str(policy.get("staking_mode") or "kelly_025"),
            STAKING_MODES["kelly_025"],
        )
        result = allocate_adaptive_day(
            race_date,
            by_day.get(race_date, []),
            evaluated_by_day[race_date],
            daily_budget_yen=daily_budget_yen,
            fractional_kelly=float(staking["fractional_kelly"]),
            max_daily_exposure_fraction=0.30,
            min_daily_exposure_fraction=float(staking["min_daily_exposure_fraction"]),
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_daily_tickets=30,
            allocation_mode=str(staking["allocation_mode"]),
            stake_granularity_yen=STAKE_YEN,
            min_stake_yen=STAKE_YEN,
        )
        if include_chronological:
            ticket_control = policy.get("v18_ticket_control")
            schedule = None
            max_daily_tickets = None
            if ticket_control is not None:
                max_daily_tickets = int(
                    ticket_control["learned_daily_ticket_limit"]
                )
                schedule = [
                    {
                        "race_id": str(race["race_id"]),
                        "race_date": str(race["race_date"]),
                        "rno": int(race["rno"]),
                        "odds_deadline_at": race.get("odds_deadline_at"),
                    }
                    for race in races_by_day[race_date]
                ]
            chronological = simulate_chronological_bankroll_day(
                race_date,
                by_day.get(race_date, []),
                evaluated_by_day[race_date],
                settlement_events=settlement_events_from_races(
                    races_by_day[race_date]
                ),
                initial_bankroll_yen=daily_budget_yen,
                daily_stake_limit_fraction=1.0,
                max_daily_tickets=max_daily_tickets,
                schedule=schedule,
                schedule_quota_rounding=str(
                    (ticket_control or {}).get("schedule_quota_rounding")
                    or "floor"
                ),
                schedule_quota_opportunity=(
                    (ticket_control or {}).get("schedule_quota_opportunity")
                ),
                max_decision_exposure_fraction=0.30,
                race_cap_fraction=0.05,
                ticket_cap_fraction=0.02,
                stake_granularity_yen=STAKE_YEN,
                allocate_day=allocate_adaptive_day,
                allocator_kwargs={
                    "fractional_kelly": float(staking["fractional_kelly"]),
                    "min_daily_exposure_fraction": float(
                        staking["min_daily_exposure_fraction"]
                    ),
                    "allocation_mode": str(staking["allocation_mode"]),
                },
                allocation_method=(
                    (
                        "chronological_v18_schedule_quota_"
                        + str(
                            ticket_control.get("schedule_quota_rounding")
                            or "floor"
                        )
                        + "_"
                        if ticket_control is not None
                        else "chronological_adaptive_"
                    ) + str(staking["allocation_mode"])
                ),
            )
            result["chronological_bankroll"] = chronological
            chronological_daily.append(chronological)
        cumulative_profit += int(result["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown_yen = max(max_drawdown_yen, peak_profit - cumulative_profit)
        result["cumulative_profit_yen"] = cumulative_profit
        daily.append(result)
        stake_yen += int(result["stake_yen"])
        return_yen += int(result["return_yen"])
        tickets += int(result["tickets"])
        hit_tickets += int(result["hit_tickets"])
    reliability = bankroll_reliability_metrics(daily, evaluated_races=len(races))
    robust_metrics: dict[str, Any] = {}
    if include_robust_metrics and daily:
        bootstrap = bootstrap_daily_roi(
            daily, samples=V17_POLICY_BOOTSTRAP_SAMPLES
        )
        robust_metrics["daily_cluster_bootstrap_roi_lower_95"] = bootstrap[
            "roi_ci95_lower"
        ]
    result = {
        "evaluated_races": len(races),
        "race_days": len(daily),
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": max_drawdown_yen,
        "winning_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "profitable_day_fraction": (
            sum(int(row["profit_yen"] > 0) for row in daily) / len(daily)
            if daily else None
        ),
        "normalized_drawdown": (
            max_drawdown_yen / stake_yen if stake_yen else None
        ),
        **reliability,
        **robust_metrics,
        "daily": daily,
    }
    if include_chronological:
        result["chronological_bankroll"] = summarize_chronological_bankroll_days(
            chronological_daily
        )
    return result


def policy_candidate(
    race: dict[str, Any],
    *,
    combination: str,
    probability: float,
    market_probability: float,
    model_probability: float,
    odds: float,
    estimated_ev: float,
    return_multiplier: float,
) -> dict[str, Any]:
    return {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "jcd": race["jcd"],
        "rno": int(race["rno"]),
        "combination": combination,
        "probability": probability,
        "market_probability": market_probability,
        "model_probability": model_probability,
        "estimated_odds": odds,
        "estimated_ev": estimated_ev,
        "historical_return_multiplier": return_multiplier,
        "estimated_payout_yen": odds * STAKE_YEN,
        "payout_history_count": 0,
        "odds_source": (
            "forecast_final_from_real_t5"
            if race.get("estimated_final_odds")
            else "real_t5"
        ),
        "actual_combination": race["actual_combination"],
        "actual_payout_yen": int(race["actual_payout_yen"]),
        "hit": combination == race["actual_combination"],
        "real_odds_snapshot_id": race.get("snapshot_id"),
        "real_odds_captured_at": race.get("captured_at"),
        "real_odds_deadline_at": race.get("odds_deadline_at"),
        "real_odds_combinations": len(race["odds"]),
    }


def prepare_policy_matrix(
    races: list[dict[str, Any]], calibrator: dict[str, float]
) -> dict[str, Any]:
    if not races:
        return {}
    combinations = sorted(
        set(races[0]["model_probabilities"])
        & set(races[0]["market_probabilities"])
        & set(decision_odds(races[0]))
    )
    probabilities = []
    market_probabilities = []
    model_probabilities = []
    odds_rows = []
    multiplier_rows = []
    for race in races:
        calibrated = race.get("_policy_calibrated_probabilities")
        if calibrated is None:
            calibrated = blend_probabilities(
                race["model_probabilities"],
                race["market_probabilities"],
                model_weight=float(calibrator["model_weight"]),
                temperature=float(calibrator["temperature"]),
            )
        if set(calibrated) != set(combinations):
            raise ValueError("policy races must share the same combinations")
        decision = decision_odds(race)
        multipliers = race.get("historical_return_multipliers") or {}
        probabilities.append([float(calibrated[key]) for key in combinations])
        market_probabilities.append(
            [float(race["market_probabilities"][key]) for key in combinations]
        )
        model_probabilities.append(
            [float(race["model_probabilities"][key]) for key in combinations]
        )
        odds_rows.append([float(decision[key]) for key in combinations])
        multiplier_rows.append(
            [float(multipliers.get(key, 1.0)) for key in combinations]
        )
    probability_matrix = np.asarray(probabilities, dtype=np.float64)
    market_matrix = np.asarray(market_probabilities, dtype=np.float64)
    model_matrix = np.asarray(model_probabilities, dtype=np.float64)
    odds_matrix = np.asarray(odds_rows, dtype=np.float64)
    multiplier_matrix = np.asarray(multiplier_rows, dtype=np.float64)
    estimated_ev = probability_matrix * odds_matrix * multiplier_matrix
    order = np.lexsort((-probability_matrix, -estimated_ev), axis=1)
    return {
        "combinations": combinations,
        "probabilities": probability_matrix,
        "market_probabilities": market_matrix,
        "model_probabilities": model_matrix,
        "odds": odds_matrix,
        "return_multipliers": multiplier_matrix,
        "estimated_ev": estimated_ev,
        "model_market_ratio": probability_matrix
        / np.maximum(EPSILON, market_matrix),
        "order": order,
    }


def _wilson_interval(hits: int, trials: int, *, z: float = 1.96) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    probability = hits / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bankroll_reliability_metrics(
    daily: list[dict[str, Any]], *, evaluated_races: int
) -> dict[str, Any]:
    tickets = sum(int(row.get("tickets") or 0) for row in daily)
    hit_tickets = sum(int(row.get("hit_tickets") or 0) for row in daily)
    selected_races = sum(int(row.get("races_bet") or 0) for row in daily)
    hit_races = sum(int(row.get("hit_races") or 0) for row in daily)
    stake_yen = sum(int(row.get("stake_yen") or 0) for row in daily)
    return_yen = sum(int(row.get("return_yen") or 0) for row in daily)
    largest_hit = max(
        (int(row.get("largest_hit_return_yen") or 0) for row in daily),
        default=0,
    )
    return_square_sum = sum(
        int(row.get("hit_return_square_sum_yen2") or 0) for row in daily
    )
    ticket_lower, ticket_upper = _wilson_interval(hit_tickets, tickets)
    race_lower, race_upper = _wilson_interval(hit_races, selected_races)
    return_without_largest = max(0, return_yen - largest_hit)
    concentration = (
        largest_hit / return_yen if return_yen > 0 else None
    )
    hhi = (
        return_square_sum / (return_yen * return_yen)
        if return_yen > 0
        else None
    )
    return {
        "selected_races": selected_races,
        "hit_races": hit_races,
        "race_selection_rate": (
            selected_races / evaluated_races if evaluated_races else None
        ),
        "avg_tickets_per_selected_race": (
            tickets / selected_races if selected_races else None
        ),
        "ticket_hit_rate": hit_tickets / tickets if tickets else None,
        "ticket_hit_rate_ci95_lower": ticket_lower,
        "ticket_hit_rate_ci95_upper": ticket_upper,
        "race_hit_rate": hit_races / selected_races if selected_races else None,
        "race_hit_rate_ci95_lower": race_lower,
        "race_hit_rate_ci95_upper": race_upper,
        "evaluated_race_hit_rate": (
            hit_races / evaluated_races if evaluated_races else None
        ),
        "largest_hit_return_yen": largest_hit,
        "largest_hit_return_share": concentration,
        "hit_return_hhi": hhi,
        "effective_hit_count": 1.0 / hhi if hhi else None,
        "return_without_largest_hit_yen": return_without_largest,
        "profit_without_largest_hit_yen": return_without_largest - stake_yen,
        "roi_without_largest_hit": (
            return_without_largest / stake_yen if stake_yen else None
        ),
    }


def summarize_registered_policy_daily(
    daily: list[dict[str, Any]],
    *,
    evaluated_races: int,
    policy: dict[str, Any] | None = None,
    registered_after: str = EV_BAND_HYPOTHESIS_REGISTERED_AFTER,
) -> dict[str, Any]:
    selected_policy = REGISTERED_EV_BAND_POLICY if policy is None else policy
    stake_yen = sum(int(row.get("stake_yen") or 0) for row in daily)
    return_yen = sum(int(row.get("return_yen") or 0) for row in daily)
    winning_days = sum(int((row.get("profit_yen") or 0) > 0) for row in daily)
    bootstrap = bootstrap_daily_roi(daily) if daily else None
    return {
        "status": "evaluating" if daily else "waiting_for_first_unseen_day",
        "comparison_role": "prospective_only_pre_registered_policy_chronological_shadow",
        "allocation_time_basis": "decision_time_order_with_settlement_only_reinvestment",
        "registered_after": registered_after,
        "policy": dict(selected_policy),
        "evaluation_days": len(daily),
        "evaluated_races": evaluated_races,
        "tickets": sum(int(row.get("tickets") or 0) for row in daily),
        "hit_tickets": sum(int(row.get("hit_tickets") or 0) for row in daily),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": winning_days,
        "profitable_day_fraction": winning_days / len(daily) if daily else None,
        "daily_cluster_bootstrap_roi_lower_95": (
            bootstrap["roi_ci95_lower"] if bootstrap is not None else None
        ),
        "probability_roi_above_one": (
            bootstrap["probability_roi_above_one"]
            if bootstrap is not None
            else None
        ),
        **bankroll_reliability_metrics(daily, evaluated_races=evaluated_races),
        "daily": daily,
    }


def _fit_prior_empirical_ev_artifact(
    records: list[dict[str, Any]], evaluation_date: str
):
    teacher_dates = sorted({str(row["race_date"]) for row in records})
    future_dates = [date for date in teacher_dates if date >= evaluation_date]
    if future_dates:
        raise ValueError(
            "empirical EV teachers must precede evaluation_date: "
            f"{future_dates[0]} >= {evaluation_date}"
        )
    artifact = fit_contextual_empirical_ev_calibration(
        records,
        prediction_date=evaluation_date,
    )
    if (
        artifact.trained_through_date is not None
        and artifact.trained_through_date >= evaluation_date
    ):
        raise AssertionError("empirical EV artifact crossed evaluation boundary")
    return artifact


def _summarize_empirical_lcb_walk_forward(
    daily: list[dict[str, Any]],
    *,
    evaluated_races: int,
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    stake_yen = sum(int(row.get("stake_yen") or 0) for row in daily)
    return_yen = sum(int(row.get("return_yen") or 0) for row in daily)
    ready_folds = sum(int(bool(fold.get("calibration_ready"))) for fold in folds)
    tickets = sum(int(row.get("tickets") or 0) for row in daily)
    sample_size_pass = (
        ready_folds >= MIN_EMPIRICAL_LCB_EVALUATION_DAYS
        and tickets >= MIN_EMPIRICAL_LCB_TICKETS
    )
    return {
        "status": "evaluating" if ready_folds else "calibration_not_ready",
        "comparison_role": "prior_only_empirical_ev_lcb95_production_candidate",
        "validation_design": (
            "Each fold fits empirical EV only from prior evaluated folds; the "
            "current realized payout is appended after its purchase simulation"
        ),
        "evaluation_days": len(daily),
        "evaluated_races": evaluated_races,
        "calibration_ready_folds": ready_folds,
        "minimum_ready_evaluation_days": MIN_EMPIRICAL_LCB_EVALUATION_DAYS,
        "minimum_tickets": MIN_EMPIRICAL_LCB_TICKETS,
        "sample_size_pass": sample_size_pass,
        "eligible_days": sum(
            int(bool(row.get("eligible_candidate_audit"))) for row in daily
        ),
        "no_bet_days": sum(int((row.get("tickets") or 0) == 0) for row in daily),
        "profitable_days": sum(int((row.get("profit_yen") or 0) > 0) for row in daily),
        "tickets": tickets,
        "hit_tickets": sum(int(row.get("hit_tickets") or 0) for row in daily),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        **bankroll_reliability_metrics(daily, evaluated_races=evaluated_races),
        "folds": folds,
        "daily": daily,
    }


def select_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    daily_budget_yen: int,
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    prepared_races = []
    for race in races:
        item = dict(race)
        item["_policy_calibrated_probabilities"] = blend_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        prepared_races.append(item)
    prepared_policy_matrix = prepare_policy_matrix(prepared_races, calibrator)
    minimum_tickets = max(10, math.ceil(len(races) * 0.05))
    minimum_stake = minimum_tickets * STAKE_YEN
    for policy in policies or default_policy_grid():
        result = simulate_policy(
            prepared_races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
            prepared_policy_matrix=prepared_policy_matrix,
        )
        eligible = bool(
            policy.get("no_bet")
            or policy_calibration_eligible(
                result,
                minimum_tickets=minimum_tickets,
                minimum_stake_yen=minimum_stake,
            )
        )
        rows.append(
            {
                "policy": dict(policy),
                "eligible": eligible,
                **{key: value for key, value in result.items() if key != "daily"},
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(
        eligible_rows,
        key=lambda row: (
            int(row["profit_yen"]) - 0.25 * int(row["max_drawdown_yen"]),
            float(row["roi"]),
            -int(row["tickets"]),
        ),
    )
    return dict(selected["policy"]), rows


V17_POLICY_RANKING_METRICS = (
    "daily_cluster_bootstrap_roi_lower_95",
    "roi_without_largest_hit",
    "profitable_day_fraction",
    "effective_hit_count",
    "largest_hit_return_share",
    "normalized_drawdown",
    "roi",
)


def _v17_metric_sort_value(value: Any, *, minimize: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(parsed):
        return math.inf
    return parsed if minimize else -parsed


def v17_policy_ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _v17_metric_sort_value(
            row.get("daily_cluster_bootstrap_roi_lower_95")
        ),
        _v17_metric_sort_value(row.get("roi_without_largest_hit")),
        _v17_metric_sort_value(row.get("profitable_day_fraction")),
        _v17_metric_sort_value(row.get("effective_hit_count")),
        _v17_metric_sort_value(
            row.get("largest_hit_return_share"), minimize=True
        ),
        _v17_metric_sort_value(row.get("normalized_drawdown"), minimize=True),
        _v17_metric_sort_value(row.get("roi")),
        str((row.get("policy") or {}).get("name") or ""),
    )


def _v17_batch_bootstrap_roi_lowers(
    policy_daily_rows: list[list[dict[str, Any]]],
    *,
    samples: int = V17_POLICY_BOOTSTRAP_SAMPLES,
    seed: int = BANKROLL_BOOTSTRAP_SEED,
    chunk_size: int = BANKROLL_BOOTSTRAP_CHUNK_SIZE,
    policy_block_size: int = 256,
) -> list[float | None]:
    """Compute the unchanged daily bootstrap statistic with shared resamples."""

    if not policy_daily_rows:
        return []
    dates = tuple(str(row["race_date"]) for row in policy_daily_rows[0])
    if not dates:
        return [None] * len(policy_daily_rows)
    if any(
        tuple(str(row["race_date"]) for row in daily) != dates
        for daily in policy_daily_rows[1:]
    ):
        raise ValueError("V17 policy daily rows must share the same date boundary")

    policy_stakes = np.asarray(
        [
            [int(row["stake_yen"]) for row in daily]
            for daily in policy_daily_rows
        ],
        dtype=np.int64,
    )
    policy_returns = np.asarray(
        [
            [int(row["return_yen"]) for row in daily]
            for daily in policy_daily_rows
        ],
        dtype=np.int64,
    )
    signatures = np.concatenate((policy_stakes, policy_returns), axis=1)
    unique_signatures, inverse = np.unique(
        signatures,
        axis=0,
        return_inverse=True,
    )
    day_count = len(dates)
    stakes = unique_signatures[:, :day_count]
    returns = unique_signatures[:, day_count:]
    rng = np.random.default_rng(seed)
    sample_counts = np.zeros((samples, day_count), dtype=np.int16)
    for start in range(0, samples, chunk_size):
        current_size = min(chunk_size, samples - start)
        sampled_days = rng.integers(
            0,
            day_count,
            size=(current_size, day_count),
        )
        rows = np.repeat(np.arange(current_size), day_count)
        np.add.at(
            sample_counts[start : start + current_size],
            (rows, sampled_days.ravel()),
            1,
        )

    unique_lowers: list[float | None] = []
    for start in range(0, len(unique_signatures), policy_block_size):
        stop = min(start + policy_block_size, len(unique_signatures))
        sampled_stakes = sample_counts @ stakes[start:stop].T
        sampled_returns = sample_counts @ returns[start:stop].T
        for column in range(stop - start):
            valid = sampled_stakes[:, column] > 0
            if not np.any(valid):
                unique_lowers.append(None)
                continue
            roi = (
                sampled_returns[valid, column]
                / sampled_stakes[valid, column]
            )
            unique_lowers.append(float(np.quantile(roi, 0.05)))
    return [unique_lowers[int(index)] for index in inverse]


def select_policy_v17(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    daily_budget_yen: int,
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_races = []
    for race in races:
        item = dict(race)
        item["_policy_calibrated_probabilities"] = blend_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        prepared_races.append(item)
    prepared_matrix = prepare_policy_matrix(prepared_races, calibrator)
    minimum_tickets = max(10, math.ceil(len(races) * 0.05))
    minimum_stake = minimum_tickets * STAKE_YEN
    rows = []
    policy_daily_rows = []
    for policy in policies or default_policy_grid():
        policy_result = simulate_policy(
            prepared_races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
            prepared_policy_matrix=prepared_matrix,
            include_chronological=False,
            include_robust_metrics=False,
        )
        policy_daily_rows.append(policy_result["daily"])
        eligible = bool(
            policy.get("no_bet")
            or policy_calibration_eligible(
                policy_result,
                minimum_tickets=minimum_tickets,
                minimum_stake_yen=minimum_stake,
            )
        )
        rows.append({
            "policy": dict(policy),
            "eligible": eligible,
            **{
                key: value
                for key, value in policy_result.items()
                if key != "daily"
            },
        })
    bootstrap_lowers = _v17_batch_bootstrap_roi_lowers(policy_daily_rows)
    for row, lower in zip(rows, bootstrap_lowers, strict=True):
        row["daily_cluster_bootstrap_roi_lower_95"] = lower
    eligible_candidates = [
        row
        for row in rows
        if row["eligible"] and not row["policy"].get("no_bet")
    ]
    if eligible_candidates:
        selected = min(eligible_candidates, key=v17_policy_ranking_key)
    else:
        no_bet_rows = [row for row in rows if row["policy"].get("no_bet")]
        selected = min(
            no_bet_rows,
            key=lambda row: str(row["policy"].get("name") or ""),
        ) if no_bet_rows else {"policy": {"name": "no_bet", "no_bet": True}}
    return dict(selected["policy"]), rows


def learn_v18_daily_ticket_control(
    daily_rows: Iterable[dict[str, Any]],
    *,
    quantile: float = V18_TICKET_LIMIT_QUANTILE,
) -> dict[str, Any]:
    """Learn a conservative ticket cap from strict-prior daily selections."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    counts = sorted(max(0, int(row.get("tickets") or 0)) for row in daily_rows)
    if counts:
        rank = math.floor((len(counts) - 1) * quantile)
        learned_limit = counts[rank]
    else:
        learned_limit = 0
    return {
        "method": "strict_prior_daily_ticket_lower_quantile",
        "quantile": quantile,
        "prior_days": len(counts),
        "prior_daily_ticket_counts": counts,
        "learned_daily_ticket_limit": learned_limit,
        "schedule_quota_rounding": "floor",
        "stake_granularity_yen": STAKE_YEN,
        "result_or_payout_fields_used": False,
    }


def select_v18_schedule_quota_rounding(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    policy: dict[str, Any],
    ticket_control: dict[str, Any],
    daily_budget_yen: int,
) -> tuple[str, list[dict[str, Any]]]:
    diagnostics = []
    for rounding in ("floor", "ceil"):
        candidate_control = {
            **ticket_control,
            "schedule_quota_rounding": rounding,
        }
        candidate_policy = {
            **policy,
            "v18_ticket_control": candidate_control,
        }
        result = simulate_policy(
            races,
            calibrator=calibrator,
            policy=candidate_policy,
            daily_budget_yen=daily_budget_yen,
            include_chronological=True,
            include_robust_metrics=False,
        )
        bankroll = result["chronological_bankroll"]
        confidence = bootstrap_daily_roi(
            bankroll["daily"], samples=V17_POLICY_BOOTSTRAP_SAMPLES
        )
        race_days = int(bankroll["race_days"])
        winning_days = int(bankroll["winning_days"])
        diagnostics.append({
            "rounding": rounding,
            "race_days": race_days,
            "tickets": int(bankroll["tickets"]),
            "hit_tickets": int(bankroll["hit_tickets"]),
            "stake_yen": int(bankroll["stake_yen"]),
            "return_yen": int(bankroll["return_yen"]),
            "profit_yen": int(bankroll["profit_yen"]),
            "roi": float(bankroll["roi"]),
            "winning_days": winning_days,
            "profitable_day_fraction": (
                winning_days / race_days if race_days else None
            ),
            "roi_ci95_lower": confidence.get("roi_ci95_lower"),
            "probability_roi_above_one": confidence.get(
                "probability_roi_above_one"
            ),
        })

    selected = max(
        diagnostics,
        key=lambda row: (
            float(row.get("roi_ci95_lower") or 0.0),
            float(row.get("probability_roi_above_one") or 0.0),
            float(row.get("profitable_day_fraction") or 0.0),
            float(row.get("roi") or 0.0),
            int(row.get("profit_yen") or 0),
            int(row["rounding"] == "floor"),
        ),
    )
    return str(selected["rounding"]), diagnostics


def _leave_one_day_out_min_roi(
    daily: list[dict[str, Any]],
) -> float | None:
    """Return the ROI after removing whichever single day helps it most."""
    if len(daily) < 2:
        return None
    total_stake = sum(float(row.get("stake_yen") or 0.0) for row in daily)
    total_return = sum(float(row.get("return_yen") or 0.0) for row in daily)
    leave_one_out = []
    for row in daily:
        remaining_stake = total_stake - float(row.get("stake_yen") or 0.0)
        if remaining_stake <= 0.0:
            continue
        remaining_return = total_return - float(row.get("return_yen") or 0.0)
        leave_one_out.append(remaining_return / remaining_stake)
    return min(leave_one_out) if leave_one_out else None


def select_v18_schedule_quota_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    policy: dict[str, Any],
    ticket_control: dict[str, Any],
    daily_budget_yen: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a quota release rule using strict-prior days only."""
    minimum_score = float(policy.get("ev_threshold") or 0.0)
    learned_limit = int(ticket_control["learned_daily_ticket_limit"])
    prior_counts = sorted(
        max(0, int(value))
        for value in ticket_control.get("prior_daily_ticket_counts", [])
    )
    median_limit = (
        prior_counts[math.floor((len(prior_counts) - 1) * 0.50)]
        if prior_counts else learned_limit
    )
    prior_max_limit = min(
        max(prior_counts, default=learned_limit),
        daily_budget_yen // STAKE_YEN,
    )
    candidates = [
        {
            "name": "floor", "schedule_quota_rounding": "floor",
            "learned_daily_ticket_limit": learned_limit,
        },
        {
            "name": "ceil", "schedule_quota_rounding": "ceil",
            "learned_daily_ticket_limit": learned_limit,
        },
    ]
    for after_fraction in (0.90, 0.95):
        for score_quantile in (0.75, 0.90):
            candidates.append({
                "name": (
                    f"opportunity_f{int(after_fraction * 100)}_"
                    f"q{int(score_quantile * 100)}"
                ),
                "schedule_quota_rounding": "floor",
                "learned_daily_ticket_limit": learned_limit,
                "schedule_quota_opportunity": {
                    "after_fraction": after_fraction,
                    "score_quantile": score_quantile,
                    "reserve_slots": 1,
                    "minimum_score": minimum_score,
                },
            })
    for reserve_fraction in (0.25, 0.50):
        online_reserve_slots = max(1, math.ceil(learned_limit * reserve_fraction))
        for after_fraction in (0.25, 0.50):
            for score_quantile in (0.75, 0.90):
                reserve_percent = int(reserve_fraction * 100)
                candidates.append({
                "name": (
                    f"online_reserve_f{int(after_fraction * 100)}_"
                    f"q{int(score_quantile * 100)}_r{reserve_percent}"
                ),
                "schedule_quota_rounding": "floor",
                "learned_daily_ticket_limit": learned_limit,
                "schedule_quota_opportunity": {
                    "quota_mode": "online_reserve",
                    "after_fraction": after_fraction,
                    "score_quantile": score_quantile,
                    "reserve_slots": online_reserve_slots,
                    "minimum_score": minimum_score,
                },
            })
    if median_limit != learned_limit:
        candidates.extend([
            {
                "name": "median_floor",
                "schedule_quota_rounding": "floor",
                "learned_daily_ticket_limit": median_limit,
            },
            {
                "name": "median_ceil",
                "schedule_quota_rounding": "ceil",
                "learned_daily_ticket_limit": median_limit,
            },
        ])
    if prior_max_limit not in {learned_limit, median_limit}:
        candidates.append({
            "name": "prior_max_ceil",
            "schedule_quota_rounding": "ceil",
            "learned_daily_ticket_limit": prior_max_limit,
        })

    prepared_policy_matrix = (
        prepare_policy_matrix(races, calibrator) if races else None
    )
    diagnostics = []
    for candidate in candidates:
        candidate_control = {
            **ticket_control,
            "learned_daily_ticket_limit": candidate[
                "learned_daily_ticket_limit"
            ],
            "schedule_quota_rounding": candidate["schedule_quota_rounding"],
            "schedule_quota_opportunity": candidate.get(
                "schedule_quota_opportunity"
            ),
        }
        result = simulate_policy(
            races,
            calibrator=calibrator,
            policy={**policy, "v18_ticket_control": candidate_control},
            daily_budget_yen=daily_budget_yen,
            prepared_policy_matrix=prepared_policy_matrix,
            include_chronological=True,
            include_robust_metrics=False,
        )
        bankroll = result["chronological_bankroll"]
        confidence = bootstrap_daily_roi(
            bankroll["daily"], samples=V17_POLICY_BOOTSTRAP_SAMPLES
        )
        race_days = int(bankroll["race_days"])
        winning_days = int(bankroll["winning_days"])
        diagnostics.append({
            **candidate,
            "race_days": race_days,
            "tickets": int(bankroll["tickets"]),
            "hit_tickets": int(bankroll["hit_tickets"]),
            "stake_yen": int(bankroll["stake_yen"]),
            "return_yen": int(bankroll["return_yen"]),
            "profit_yen": int(bankroll["profit_yen"]),
            "roi": float(bankroll["roi"]),
            "winning_days": winning_days,
            "leave_one_day_out_min_roi": _leave_one_day_out_min_roi(
                bankroll["daily"]
            ),
            "profitable_day_fraction": (
                winning_days / race_days if race_days else None
            ),
            "roi_ci95_lower": confidence.get("roi_ci95_lower"),
            "probability_roi_above_one": confidence.get(
                "probability_roi_above_one"
            ),
        })

    selected = max(
        diagnostics,
        key=lambda row: (
            float(row.get("leave_one_day_out_min_roi") or 0.0),
            float(row.get("roi_ci95_lower") or 0.0),
            float(row.get("probability_roi_above_one") or 0.0),
            float(row.get("profitable_day_fraction") or 0.0),
            float(row.get("roi") or 0.0),
            int(row.get("profit_yen") or 0),
            int(row["name"] == "floor"),
        ),
    )
    selected_control = {
        "learned_daily_ticket_limit": int(
            selected["learned_daily_ticket_limit"]
        ),
        "schedule_quota_rounding": str(
            selected["schedule_quota_rounding"]
        ),
        "schedule_quota_opportunity": selected.get(
            "schedule_quota_opportunity"
        ),
    }
    return selected_control, diagnostics


def select_policy_v18(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    daily_budget_yen: int,
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep V17 policy ranking and learn its operational daily ticket cap."""
    selected, rows = select_policy_v17(
        races,
        calibrator=calibrator,
        daily_budget_yen=daily_budget_yen,
        policies=policies,
    )
    prepared_races = []
    for race in races:
        item = dict(race)
        item["_policy_calibrated_probabilities"] = blend_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        prepared_races.append(item)
    selected_result = simulate_policy(
        prepared_races,
        calibrator=calibrator,
        policy=selected,
        daily_budget_yen=daily_budget_yen,
        include_chronological=False,
        include_robust_metrics=False,
    )
    ticket_control = learn_v18_daily_ticket_control(
        selected_result["daily"]
    )
    selected_control, quota_diagnostics = select_v18_schedule_quota_policy(
        prepared_races,
        calibrator=calibrator,
        policy=selected,
        ticket_control=ticket_control,
        daily_budget_yen=daily_budget_yen,
    )
    rounding_diagnostics = [
        row for row in quota_diagnostics if row["name"] in {"floor", "ceil"}
    ]
    selected_rounding = max(
        rounding_diagnostics,
        key=lambda row: (
            float(row.get("roi_ci95_lower") or 0.0),
            float(row.get("probability_roi_above_one") or 0.0),
            float(row.get("profitable_day_fraction") or 0.0),
            float(row.get("roi") or 0.0),
            int(row.get("profit_yen") or 0),
            int(row["name"] == "floor"),
        ),
    )["name"]
    ticket_control.update({
        **selected_control,
        "schedule_quota_rounding_selection": {
            "source": "strict_prior_calibration_days_only",
            "selected": selected_rounding,
            "candidates": rounding_diagnostics,
            "evaluation_or_future_fields_used": False,
        },
        "schedule_quota_policy_selection": {
            "source": "strict_prior_calibration_days_only",
            "selected": next(
                row["name"]
                for row in quota_diagnostics
                if row["schedule_quota_rounding"]
                == selected_control["schedule_quota_rounding"]
                and row["learned_daily_ticket_limit"]
                == selected_control["learned_daily_ticket_limit"]
                and row.get("schedule_quota_opportunity")
                == selected_control["schedule_quota_opportunity"]
            ),
            "candidates": quota_diagnostics,
            "evaluation_or_future_fields_used": False,
        },
    })
    v18_policy = dict(selected)
    v18_policy["v18_ticket_control"] = ticket_control
    return v18_policy, rows


def v35_registered_policy_candidates() -> list[dict[str, Any]]:
    """Return the pre-registered anchor and one-axis V35 challengers."""
    anchor = dict(PROSPECTIVE_NORMALIZED_EV_POLICY)
    specifications = (
        ("ev_threshold", 1.05, "ev_threshold_105"),
        ("max_estimated_ev", 1.20, "max_estimated_ev_120"),
        ("max_odds", 40.0, "max_odds_40"),
        ("max_tickets_per_race", 1, "max_tickets_per_race_1"),
        ("min_model_market_ratio", 1.05, "min_model_market_ratio_105"),
        ("staking_mode", "kelly_025", "staking_kelly_025"),
    )
    candidates = [anchor]
    for field, value, suffix in specifications:
        candidate = dict(anchor)
        candidate[field] = value
        candidate["name"] = f"v35_{suffix}"
        candidates.append(candidate)
    if len(candidates) > 8:
        raise AssertionError("V35 registered candidate set exceeds eight policies")
    return candidates


def _v35_ticket_control(
    anchor_daily: list[dict[str, Any]],
    *,
    prior_days: int,
) -> dict[str, Any]:
    learned = learn_v18_daily_ticket_control(anchor_daily)
    empirical_limit = min(
        V35_MAX_DAILY_TICKET_LIMIT,
        max(
            V35_MIN_DAILY_TICKET_LIMIT,
            int(learned["learned_daily_ticket_limit"]),
        ),
    )
    alpha = min(
        1.0,
        max(
            0.0,
            (prior_days - V35_MIN_ADAPTIVE_POLICY_DAYS)
            / (
                V35_MIN_ADAPTIVE_QUOTA_DAYS
                - V35_MIN_ADAPTIVE_POLICY_DAYS
            ),
        ),
    )
    if prior_days < V35_MIN_ADAPTIVE_QUOTA_DAYS:
        daily_limit = V35_FIXED_DAILY_TICKET_LIMIT
        quota_regime = "fixed_10_floor_before_30_prior_days"
    else:
        daily_limit = math.floor(
            (1.0 - alpha) * V35_FIXED_DAILY_TICKET_LIMIT
            + alpha * empirical_limit
        )
        daily_limit = min(
            V35_MAX_DAILY_TICKET_LIMIT,
            max(V35_MIN_DAILY_TICKET_LIMIT, daily_limit),
        )
        quota_regime = "strict_prior_q25_shrunk_and_bounded_floor"
    return {
        **learned,
        "method": "v35_strict_prior_stable_daily_ticket_limit",
        "learned_daily_ticket_limit": daily_limit,
        "empirical_q25_daily_ticket_limit": empirical_limit,
        "shrinkage_alpha": alpha,
        "quota_regime": quota_regime,
        "schedule_quota_rounding": "floor",
        "schedule_quota_opportunity": None,
        "result_or_payout_fields_used": False,
    }


def _v35_daily_profits(
    result: dict[str, Any],
) -> dict[str, float]:
    chronological = result["chronological_bankroll"]
    return {
        str(row["race_date"]): float(row.get("profit_yen") or 0.0)
        for row in chronological["daily"]
    }


def _v35_paired_profit_lcb(
    anchor_daily: dict[str, float],
    candidate_daily: dict[str, float],
    *,
    comparison_count: int,
) -> tuple[float | None, float | None, float | None]:
    dates = sorted(set(anchor_daily) | set(candidate_daily))
    differences = np.asarray(
        [
            candidate_daily.get(date, 0.0) - anchor_daily.get(date, 0.0)
            for date in dates
        ],
        dtype=np.float64,
    )
    if differences.size < 2 or comparison_count < 1:
        return None, None, None
    mean_difference = float(np.mean(differences))
    standard_error = float(
        np.std(differences, ddof=1) / math.sqrt(differences.size)
    )
    critical_value = NormalDist().inv_cdf(
        1.0 - V35_FAMILY_WISE_ALPHA / comparison_count
    )
    stability_penalty = critical_value * standard_error
    return (
        mean_difference - stability_penalty,
        stability_penalty,
        mean_difference,
    )


def select_policy_v35(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    daily_budget_yen: int,
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a V21-path policy using strict-prior paired daily stability."""
    registered = [
        dict(policy)
        for policy in (
            policies if policies is not None else v35_registered_policy_candidates()
        )
    ]
    if not registered:
        raise ValueError("V35 requires its registered anchor policy")
    if len(registered) > 8:
        raise ValueError("V35 allows at most eight pre-registered policies")
    anchor = dict(registered[0])
    prior_days = len({str(race["race_date"]) for race in races})
    prepared_races = []
    for race in races:
        item = dict(race)
        item["_policy_calibrated_probabilities"] = blend_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        prepared_races.append(item)
    prepared_matrix = (
        prepare_policy_matrix(prepared_races, calibrator)
        if prepared_races
        else None
    )
    anchor_unconstrained = simulate_policy(
        prepared_races,
        calibrator=calibrator,
        policy=anchor,
        daily_budget_yen=daily_budget_yen,
        prepared_policy_matrix=prepared_matrix,
        include_chronological=False,
        include_robust_metrics=False,
    )
    ticket_control = _v35_ticket_control(
        anchor_unconstrained["daily"],
        prior_days=prior_days,
    )
    evaluated_policies = (
        registered
        if prior_days >= V35_MIN_ADAPTIVE_POLICY_DAYS
        else [anchor]
    )
    rows = []
    results = []
    for candidate in evaluated_policies:
        controlled = {
            **candidate,
            "v18_ticket_control": dict(ticket_control),
        }
        result = simulate_policy(
            prepared_races,
            calibrator=calibrator,
            policy=controlled,
            daily_budget_yen=daily_budget_yen,
            prepared_policy_matrix=prepared_matrix,
            include_chronological=True,
            include_robust_metrics=False,
        )
        bankroll = result["chronological_bankroll"]
        results.append(result)
        rows.append({
            "policy": dict(candidate),
            "eligible": True,
            "tickets": int(bankroll["tickets"]),
            "hit_tickets": int(bankroll["hit_tickets"]),
            "stake_yen": int(bankroll["stake_yen"]),
            "return_yen": int(bankroll["return_yen"]),
            "profit_yen": int(bankroll["profit_yen"]),
            "roi": float(bankroll["roi"]),
            "max_drawdown_yen": int(bankroll["max_drawdown_yen"]),
            "winning_days": int(bankroll["winning_days"]),
            "race_days": int(bankroll["race_days"]),
            "paired_profit_lcb_yen": 0.0 if not rows else None,
            "stability_penalty_yen": 0.0 if not rows else None,
            "paired_profit_mean_difference_yen": 0.0 if not rows else None,
        })

    selected_index = 0
    fallback_reason = None
    selection_regime = "fixed_anchor_warmup"
    comparison_count = max(0, len(evaluated_policies) - 1)
    if prior_days < V35_MIN_ADAPTIVE_POLICY_DAYS:
        fallback_reason = "strict_prior_days_below_7"
    else:
        selection_regime = "paired_stability_selection"
        anchor_daily = _v35_daily_profits(results[0])
        qualified = []
        for index in range(1, len(results)):
            lower, penalty, mean_difference = _v35_paired_profit_lcb(
                anchor_daily,
                _v35_daily_profits(results[index]),
                comparison_count=comparison_count,
            )
            rows[index].update({
                "paired_profit_lcb_yen": lower,
                "stability_penalty_yen": penalty,
                "paired_profit_mean_difference_yen": mean_difference,
            })
            if lower is not None and lower > 0.0:
                qualified.append(index)
        if qualified:
            selected_index = min(
                qualified,
                key=lambda index: (
                    -float(rows[index]["paired_profit_lcb_yen"]),
                    -float(rows[index]["paired_profit_mean_difference_yen"]),
                    str(rows[index]["policy"]["name"]),
                ),
            )
        else:
            selection_regime = "paired_stability_fallback"
            fallback_reason = "no_candidate_positive_bonferroni_lcb"

    selected_row = rows[selected_index]
    diagnostics = {
        "selection_regime": selection_regime,
        "prior_days": prior_days,
        "fixed_anchor_policy": anchor,
        "adaptive_candidate_count": comparison_count,
        "paired_profit_lcb_yen": selected_row["paired_profit_lcb_yen"],
        "stability_penalty_yen": selected_row["stability_penalty_yen"],
        "fallback_reason": fallback_reason,
    }
    selected_policy = {
        **evaluated_policies[selected_index],
        "v18_ticket_control": ticket_control,
        "v35_selection_diagnostics": diagnostics,
    }
    return selected_policy, rows


def policy_calibration_eligible(
    result: dict[str, Any],
    *,
    minimum_tickets: int,
    minimum_stake_yen: int,
) -> bool:
    race_days = int(result["race_days"])
    minimum_winning_days = min(
        race_days,
        max(1, math.ceil(race_days * 0.60)),
    )
    return bool(
        int(result["tickets"]) >= minimum_tickets
        and int(result["stake_yen"]) >= minimum_stake_yen
        and int(result["profit_yen"]) > 0
        and float(result["roi"]) >= 1.05
        and int(result["winning_days"]) >= minimum_winning_days
        and int(result["max_drawdown_yen"]) <= int(result["stake_yen"]) * 0.75
    )


def summarize_policy_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if not row["policy"].get("no_bet")]
    funded = [row for row in candidates if int(row["stake_yen"]) > 0]

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "policy": row["policy"],
            "eligible": bool(row["eligible"]),
            "tickets": int(row["tickets"]),
            "hit_tickets": int(row["hit_tickets"]),
            "stake_yen": int(row["stake_yen"]),
            "return_yen": int(row["return_yen"]),
            "profit_yen": int(row["profit_yen"]),
            "roi": float(row["roi"]),
            "max_drawdown_yen": int(row["max_drawdown_yen"]),
            "winning_days": int(row["winning_days"]),
            "race_days": int(row["race_days"]),
        }

    return {
        "candidate_count": len(candidates),
        "funded_candidate_count": len(funded),
        "profitable_candidate_count": sum(int(row["profit_yen"] > 0) for row in funded),
        "eligible_candidate_count": sum(bool(row["eligible"]) for row in candidates),
        "best_profit": compact(
            max(funded, key=lambda row: (row["profit_yen"], row["roi"]), default=None)
        ),
        "best_roi": compact(
            max(funded, key=lambda row: (row["roi"], row["profit_yen"]), default=None)
        ),
    }


def probability_metrics(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float] | None = None,
) -> dict[str, float | int | None]:
    losses = {"model": [], "market": [], "calibrated": []}
    top5_hits = {key: 0 for key in losses}
    winner_losses = {key: [] for key in losses}
    winner_top1_hits = {key: 0 for key in losses}
    for race in races:
        sources = {
            "model": race["model_probabilities"],
            "market": race["market_probabilities"],
        }
        if calibrator is not None:
            sources["calibrated"] = blend_probabilities(
                sources["model"],
                sources["market"],
                model_weight=float(calibrator["model_weight"]),
                temperature=float(calibrator["temperature"]),
            )
        actual = str(race["actual_combination"])
        actual_winner = actual.split("-", 1)[0]
        for name, probabilities in sources.items():
            losses[name].append(-math.log(max(EPSILON, probabilities.get(actual, 0.0))))
            top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
            top5_hits[name] += int(actual in top5)
            winner_probabilities: dict[str, float] = defaultdict(float)
            for combination, probability in probabilities.items():
                winner_probabilities[str(combination).split("-", 1)[0]] += float(
                    probability
                )
            winner_losses[name].append(
                -math.log(
                    max(EPSILON, winner_probabilities.get(actual_winner, 0.0))
                )
            )
            predicted_winner = max(
                winner_probabilities,
                key=winner_probabilities.get,
            )
            winner_top1_hits[name] += int(predicted_winner == actual_winner)
    result: dict[str, float | int | None] = {"evaluated_races": len(races)}
    for name in ("model", "market", "calibrated"):
        values = losses[name]
        result[f"{name}_trifecta_log_loss"] = sum(values) / len(values) if values else None
        result[f"{name}_trifecta_top5_hit_rate"] = (
            top5_hits[name] / len(values) if values else None
        )
        winner_values = winner_losses[name]
        result[f"{name}_winner_log_loss"] = (
            sum(winner_values) / len(winner_values) if winner_values else None
        )
        result[f"{name}_winner_top1_accuracy"] = (
            winner_top1_hits[name] / len(winner_values)
            if winner_values
            else None
        )
    return result


def split_head_probability_metrics(
    races: list[dict[str, Any]],
    *,
    probability_calibrator: dict[str, float],
    ranking_calibrator: dict[str, float],
) -> dict[str, float | int | None]:
    probability_head = probability_metrics(
        races,
        calibrator=probability_calibrator,
    )
    ranking_head = probability_metrics(
        races,
        calibrator=ranking_calibrator,
    )
    result = dict(probability_head)
    result["calibrated_trifecta_top5_hit_rate"] = ranking_head[
        "calibrated_trifecta_top5_hit_rate"
    ]
    return result


def paired_market_differences(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
) -> tuple[list[float], list[float]]:
    loss_differences = []
    top5_differences = []
    for race in races:
        market = race["market_probabilities"]
        calibrated = blend_probabilities(
            race["model_probabilities"],
            market,
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        actual = str(race["actual_combination"])
        market_loss = -math.log(max(EPSILON, float(market.get(actual, 0.0))))
        calibrated_loss = -math.log(
            max(EPSILON, float(calibrated.get(actual, 0.0)))
        )
        loss_differences.append(calibrated_loss - market_loss)
        market_top5 = sorted(market, key=market.get, reverse=True)[:5]
        calibrated_top5 = sorted(
            calibrated, key=calibrated.get, reverse=True
        )[:5]
        top5_differences.append(
            float(actual in calibrated_top5) - float(actual in market_top5)
        )
    return loss_differences, top5_differences


def split_head_paired_market_differences(
    races: list[dict[str, Any]],
    *,
    probability_calibrator: dict[str, float],
    ranking_calibrator: dict[str, float],
) -> tuple[list[float], list[float]]:
    loss_differences, _ = paired_market_differences(
        races,
        calibrator=probability_calibrator,
    )
    _, top5_differences = paired_market_differences(
        races,
        calibrator=ranking_calibrator,
    )
    return loss_differences, top5_differences


def market_comparison_confidence(
    loss_differences: list[float],
    top5_differences: list[float],
    *,
    cluster_labels: list[str],
    minimum_cluster_days: int = 5,
) -> dict[str, Any]:
    loss = paired_mean_bootstrap(
        loss_differences,
        samples=20_000,
        seed=20260722,
    )
    top5 = paired_mean_bootstrap(
        top5_differences,
        samples=20_000,
        seed=20260723,
    )
    clustered_loss = paired_cluster_mean_bootstrap(
        loss_differences,
        cluster_labels,
        samples=20_000,
        seed=20260726,
    )
    clustered_top5 = paired_cluster_mean_bootstrap(
        top5_differences,
        cluster_labels,
        samples=20_000,
        seed=20260727,
    )
    race_level_pass = bool(
        float(loss["ci95_upper"]) <= 0.0
        and float(top5["ci95_lower"]) >= 0.0
    )
    cluster_count = int(clustered_loss["clusters"])
    day_cluster_pass = bool(
        cluster_count >= minimum_cluster_days
        and float(clustered_loss["ci95_upper"]) <= 0.0
        and float(clustered_top5["ci95_lower"]) >= 0.0
    )
    return {
        "comparison_role": (
            "paired race-level and whole-day cluster bootstrap; negative "
            "LogLoss difference is better"
        ),
        "evaluation_races": len(loss_differences),
        "evaluation_days": cluster_count,
        "minimum_cluster_days": minimum_cluster_days,
        "log_loss_difference_calibrated_minus_market": loss,
        "top5_hit_difference_calibrated_minus_market": top5,
        "day_cluster_log_loss_difference_calibrated_minus_market": clustered_loss,
        "day_cluster_top5_hit_difference_calibrated_minus_market": clustered_top5,
        "race_level_confidence_pass": race_level_pass,
        "day_cluster_confidence_pass": day_cluster_pass,
        "confidence_pass": race_level_pass and day_cluster_pass,
    }


def predefined_ticket_diagnostics(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int = 10_000,
) -> dict[str, Any]:
    descriptions = {
        "top5_flat": "モデルTop5を全点100円購入",
        "top5_odds_gte_5": "モデルTop5のうちT-5オッズ5倍以上を100円購入",
        "top5_ev_gte_1": "モデルTop5のうちモデル確率×T-5オッズ1.0以上を100円購入",
    }
    totals = {
        name: {
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "hit_tickets": 0,
            "hit_races": 0,
        }
        for name in descriptions
    }
    for race in races:
        probabilities = race["model_probabilities"]
        odds = decision_odds(race)
        actual = str(race["actual_combination"])
        top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        selections = {
            "top5_flat": top5,
            "top5_odds_gte_5": [
                combination
                for combination in top5
                if float(odds[combination]) >= 5.0
            ],
            "top5_ev_gte_1": [
                combination
                for combination in top5
                if float(probabilities[combination]) * float(odds[combination]) >= 1.0
            ],
        }
        for name, combinations in selections.items():
            row = totals[name]
            row["tickets"] += len(combinations)
            row["stake_yen"] += len(combinations) * STAKE_YEN
            if actual in combinations:
                row["return_yen"] += int(race["actual_payout_yen"])
                row["hit_tickets"] += 1
                row["hit_races"] += 1

    strategies = {}
    for name, values in totals.items():
        stake = int(values["stake_yen"])
        returned = int(values["return_yen"])
        strategies[name] = {
            "description": descriptions[name],
            "evaluated_races": len(races),
            **values,
            "profit_yen": returned - stake,
            "roi": returned / stake if stake else None,
            "race_hit_rate": values["hit_races"] / len(races) if races else None,
        }
    return {
        "comparison_role": "fixed_ticket_diagnostic_not_policy_selection_or_promotion",
        "uses_only_evaluation_folds": True,
        "daily_budget_applied": False,
        "stake_per_ticket_yen": STAKE_YEN,
        "strategies": strategies,
        "adaptive_bankroll": sequential_top5_ev_kelly_diagnostic(
            races,
            daily_budget_yen=daily_budget_yen,
        ),
    }


def waiting_walk_forward_result(
    races: list[dict[str, Any]],
    *,
    dates: list[str],
    evaluation_dates: list[str] | None = None,
    daily_budget_yen: int,
    min_calibration_days: int,
    calibrator_strategy: str = "grid",
) -> dict[str, Any]:
    candidate_dates = dates if evaluation_dates is None else evaluation_dates
    if candidate_dates:
        required_additional_days = min(
            max(
                0,
                min_calibration_days
                - sum(candidate_date > date for date in dates),
            )
            for candidate_date in candidate_dates
        )
        if required_additional_days == 0:
            required_additional_days = 1
    else:
        required_additional_days = 1
    probability_metrics = {
        "evaluated_races": 0,
        "model_trifecta_log_loss": None,
        "model_trifecta_top5_hit_rate": None,
        "market_trifecta_log_loss": None,
        "market_trifecta_top5_hit_rate": None,
        "calibrated_trifecta_log_loss": None,
        "winner_log_loss": None,
        "winner_top1_accuracy": None,
        "calibrated_trifecta_top5_hit_rate": None,
        "model_winner_log_loss": None,
        "model_winner_top1_accuracy": None,
        "market_winner_log_loss": None,
        "market_winner_top1_accuracy": None,
        "calibrated_winner_log_loss": None,
        "calibrated_winner_top1_accuracy": None,
    }
    promotion_gate = {
        "minimum_evaluation_races": 1000,
        "minimum_evaluation_days": 30,
        "minimum_profitable_fold_fraction": 0.60,
        "sample_size_pass": False,
        "positive_profit_pass": False,
        "roi_pass": False,
        "fold_stability_pass": False,
        "calibration_pass": False,
        "market_confidence_pass": False,
        "no_lookahead_pass": True,
    }
    return {
        "model": odds_path_model_name(calibrator_strategy),
        "status": "waiting_for_clean_evaluation_day",
        "calibrator_strategy": calibrator_strategy,
        "comparison_role": (
            robust_policy_comparison_role(calibrator_strategy)
            if calibrator_strategy in ROBUST_POLICY_STRATEGIES
            else "real_t5_odds_nested_daily_walk_forward_shadow"
        ),
        "deployment_mode": (
            "shadow_only" if calibrator_strategy in ROBUST_POLICY_STRATEGIES else "evaluation"
        ),
        "real_betting_enabled": False,
        "validation_design": (
            "Each evaluation day has complete T-5 coverage and is untouched; calibration "
            "and policy selection use only earlier eligible T-5 races"
        ),
        "daily_budget_yen": daily_budget_yen,
        "available_races": len(races),
        "available_days": len(dates),
        "evaluation_candidate_days": len(candidate_dates),
        "evaluation_candidate_dates": candidate_dates,
        "minimum_calibration_days": min_calibration_days,
        "required_additional_days": required_additional_days,
        "evaluated_races": 0,
        "evaluation_races": 0,
        "evaluation_days": 0,
        "probability_metrics": probability_metrics,
        "market_comparison": {
            "comparison_role": "waiting for the first untouched evaluation day",
            "evaluation_races": 0,
            "evaluation_days": 0,
            "minimum_cluster_days": 5,
            "race_level_confidence_pass": False,
            "day_cluster_confidence_pass": False,
            "confidence_pass": False,
        },
        "ticket_diagnostics": predefined_ticket_diagnostics(
            [], daily_budget_yen=daily_budget_yen
        ),
        "calibrated_trifecta_log_loss": None,
        "winner_log_loss": None,
        "winner_top1_accuracy": None,
        "trifecta_top5_hit_rate": None,
        "tickets": 0,
        "hit_tickets": 0,
        "stake_yen": 0,
        "return_yen": 0,
        "profit_yen": 0,
        "roi": 0.0,
        "max_drawdown_yen": 0,
        "profitable_folds": 0,
        "folds": [],
        "daily": [],
        "chronological_bankroll": {
            **summarize_chronological_bankroll_days([]),
            "daily_cluster_bootstrap_roi_lower_95": None,
            "profitable_day_fraction": None,
            "normalized_drawdown": None,
            "primary_promotion_bankroll": (
                calibrator_strategy in ROBUST_POLICY_STRATEGIES
            ),
        },
        "flat_policy_walk_forward": {
            "comparison_role": "preselected_on_prior_days_fixed_100_yen_shadow",
            "evaluation_races": 0,
            "evaluation_days": 0,
            "tickets": 0,
            "hit_tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "winning_days": 0,
            "daily": [],
        },
        "empirical_lcb_walk_forward": _summarize_empirical_lcb_walk_forward(
            [],
            evaluated_races=0,
            folds=[],
        ),
        "registered_ev_band_walk_forward": summarize_registered_policy_daily(
            [],
            evaluated_races=0,
        ),
        "prospective_normalized_ev_walk_forward": summarize_registered_policy_daily(
            [],
            evaluated_races=0,
            policy=PROSPECTIVE_NORMALIZED_EV_POLICY,
            registered_after=PROSPECTIVE_NORMALIZED_EV_REGISTERED_AFTER,
        ),
        "prospective_top5_narrow_ev_walk_forward": summarize_registered_policy_daily(
            [],
            evaluated_races=0,
            policy=PROSPECTIVE_TOP5_NARROW_EV_POLICY,
            registered_after=PROSPECTIVE_TOP5_NARROW_EV_REGISTERED_AFTER,
        ),
        "v32_dual_head_conformal_retrospective_diagnostic": {
            **summarize_registered_policy_daily(
                [],
                evaluated_races=0,
                policy=V32_DUAL_HEAD_CONFORMAL_POLICY,
                registered_after=V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "promotion_evidence": False,
        },
        "v32_dual_head_conformal_prospective_walk_forward": (
            summarize_registered_policy_daily(
                [],
                evaluated_races=0,
                policy=V32_DUAL_HEAD_CONFORMAL_POLICY,
                registered_after=V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER,
            )
        ),
        "v33_v25_top1_narrow_retrospective_diagnostic": {
            **summarize_registered_policy_daily(
                [],
                evaluated_races=0,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "promotion_evidence": False,
        },
        "v33_v25_top1_narrow_forecast_only_diagnostic": {
            **summarize_registered_policy_daily(
                [],
                evaluated_races=0,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "promotion_evidence": False,
        },
        "v33_v25_top1_narrow_prospective_walk_forward": (
            summarize_registered_policy_daily(
                [],
                evaluated_races=0,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            )
        ),
        "market_offset_registered_policy_walk_forward": {
            "comparison_role": "market_offset_with_registered_policy_challenger",
            "status": "waiting_for_clean_evaluation_day",
            "evaluated_races": 0,
            "evaluation_days": 0,
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "promotion_eligible": False,
            "daily": [],
        },
        "market_offset_multinomial_kelly_walk_forward": {
            "challenger": "market_offset_discrete_multinomial_kelly",
            "status": "waiting_for_clean_evaluation_day",
            "promotion_gate": {"pass": False},
            "daily": [],
        },
        "conservative_market_offset_kelly_walk_forward": {
            "challenger": "conservative_market_offset_discrete_multinomial_kelly",
            "comparison_role": (
                "pure prospective market-offset Kelly with a registered odds-error haircut"
            ),
            "status": "waiting_for_first_unseen_day",
            "registered_after": CONSERVATIVE_MARKET_KELLY_REGISTERED_AFTER,
            "policy": {
                "odds_safety_factor": (
                    CONSERVATIVE_MARKET_KELLY_ODDS_SAFETY_FACTOR
                ),
                "zero_bet_allowed": True,
            },
            "evaluated_races": 0,
            "evaluation_days": 0,
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "promotion_gate": {"pass": False},
            "daily": [],
        },
        "promotion_gate": promotion_gate,
        "promotion_eligible": False,
    }


def verifiable_closing_odds_races(
    races: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep teachers whose official update or 120-price vector actually changed."""
    return [
        race
        for race in races
        if len(race.get("closing_odds") or {}) == 120
        and (
            race.get("closing_source_changed") is True
            or race.get("closing_odds_changed") is True
        )
    ]


def closing_odds_training_ready(
    races: Iterable[dict[str, Any]],
    *,
    min_training_days: int = MIN_CLOSING_ODDS_TRAINING_DAYS,
    min_training_races: int = MIN_CLOSING_ODDS_TRAINING_RACES,
) -> bool:
    eligible = list(races)
    return bool(
        len(eligible) >= min_training_races
        and len({str(race["race_date"]) for race in eligible})
        >= min_training_days
    )


CLOSING_ODDS_FORECAST_POLICY_INPUT = "oof_forecast_final_from_real_t5"
CLOSING_ODDS_FALLBACK_POLICY_INPUT = "observed_t5_fallback"
_CLOSING_ODDS_POLICY_FIELDS = (
    "estimated_final_odds",
    "closing_odds_forecast_source",
    "closing_odds_forecast_target",
    "closing_odds_policy_input",
    "closing_odds_policy_fallback",
    "closing_odds_policy_fallback_reason",
    "closing_odds_model_training_days",
    "closing_odds_model_training_races",
    "closing_odds_model_trained_through_date",
)


def _attach_t5_policy_fallback(
    races: Iterable[dict[str, Any]], *, reason: str
) -> list[dict[str, Any]]:
    result = []
    for race in races:
        item = dict(race)
        item.pop("estimated_final_odds", None)
        item.pop("closing_odds_forecast_source", None)
        item.pop("closing_odds_forecast_target", None)
        item["closing_odds_policy_input"] = CLOSING_ODDS_FALLBACK_POLICY_INPUT
        item["closing_odds_policy_fallback"] = True
        item["closing_odds_policy_fallback_reason"] = reason
        result.append(item)
    return result


def _attach_oof_closing_odds_forecast(
    races: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    training_dates: list[str],
    training_races: int,
) -> list[dict[str, Any]]:
    augmented = attach_selected_closing_odds(races, selection)
    result = []
    for race in augmented:
        forecast = race.get("estimated_final_odds") or {}
        observed = race.get("odds") or {}
        if (
            len(observed) != 120
            or set(forecast) != set(observed)
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in forecast.values()
            )
        ):
            result.extend(
                _attach_t5_policy_fallback(
                    [race], reason="incomplete_closing_odds_forecast"
                )
            )
            continue
        item = dict(race)
        item["closing_odds_policy_input"] = CLOSING_ODDS_FORECAST_POLICY_INPUT
        item["closing_odds_policy_fallback"] = False
        item.pop("closing_odds_policy_fallback_reason", None)
        item["closing_odds_model_training_days"] = len(training_dates)
        item["closing_odds_model_training_races"] = int(training_races)
        item["closing_odds_model_trained_through_date"] = training_dates[-1]
        result.append(item)
    return result


def prequential_closing_odds_policy_inputs(
    races: list[dict[str, Any]],
    *,
    min_training_days: int = MIN_CLOSING_ODDS_TRAINING_DAYS,
    min_training_races: int = MIN_CLOSING_ODDS_TRAINING_RACES,
) -> dict[str, Any]:
    """Build date-OOF policy prices using strictly earlier closing teachers."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    attached_by_race_id: dict[str, dict[str, Any]] = {}
    folds: dict[str, dict[str, Any]] = {}
    for index, prediction_date in enumerate(dates):
        training_dates = dates[:index]
        if any(date >= prediction_date for date in training_dates):
            raise AssertionError("closing odds training crossed prediction date")
        prior = [race for date in training_dates for race in by_day[date]]
        teachers = verifiable_closing_odds_races(prior)
        teacher_dates = sorted({str(race["race_date"]) for race in teachers})
        if any(date >= prediction_date for date in teacher_dates):
            raise AssertionError("closing odds teacher crossed prediction date")

        selection = None
        evaluation = None
        model = None
        fallback_reason = None
        if not teachers:
            fallback_reason = "no_strictly_prior_closing_odds_teachers"
        elif not closing_odds_training_ready(
            teachers,
            min_training_days=min_training_days,
            min_training_races=min_training_races,
        ):
            fallback_reason = "insufficient_strictly_prior_closing_odds_teachers"
        else:
            try:
                selection = select_closing_odds_model(teachers)
                attached = _attach_oof_closing_odds_forecast(
                    by_day[prediction_date],
                    selection,
                    training_dates=teacher_dates,
                    training_races=len(teachers),
                )
            except (KeyError, ValueError, np.linalg.LinAlgError):
                selection = None
                fallback_reason = "closing_odds_model_fit_failed"
            else:
                selected_model = str(selection["selected"])
                model = dict(selection[f"{selected_model}_model"])
                model["model_type"] = selected_model
                evaluation = selected_closing_odds_metrics(
                    verifiable_closing_odds_races(by_day[prediction_date]),
                    selection,
                )
        if selection is None:
            attached = _attach_t5_policy_fallback(
                by_day[prediction_date],
                reason=str(fallback_reason),
            )
        for race in attached:
            attached_by_race_id[str(race["race_id"])] = race
        folds[prediction_date] = {
            "prediction_date": prediction_date,
            "training_dates": teacher_dates,
            "trained_through_date": teacher_dates[-1] if teacher_dates else None,
            "training_races": len(teachers),
            "selection": selection,
            "model": model,
            "evaluation": evaluation,
            "policy_input": (
                CLOSING_ODDS_FORECAST_POLICY_INPUT
                if selection is not None
                else CLOSING_ODDS_FALLBACK_POLICY_INPUT
            ),
            "fallback_reason": fallback_reason,
        }
    return {"races_by_id": attached_by_race_id, "folds": folds}


def apply_prequential_closing_odds_policy_inputs(
    races: list[dict[str, Any]], inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    attached_by_race_id = inputs["races_by_id"]
    result = []
    for race in races:
        source = attached_by_race_id[str(race["race_id"])]
        item = dict(race)
        for key in _CLOSING_ODDS_POLICY_FIELDS:
            if key in source:
                item[key] = source[key]
            else:
                item.pop(key, None)
        result.append(item)
    return result


def apply_prequential_conformal_lower_odds_policy_inputs(
    races: list[dict[str, Any]], inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach strictly-prior conformal lower closing odds for a challenger."""
    return _apply_prequential_quantile_odds_policy_inputs(
        races,
        inputs,
        forecasts_key="policy_forecasts_by_race_id",
        policy_input="oof_adaptive_conformal_lower_from_real_t5",
        missing_reason="insufficient_conformal_closing_odds_teachers",
        incomplete_reason="incomplete_conformal_closing_odds_forecast",
    )


def apply_prequential_trend_point_odds_policy_inputs(
    races: list[dict[str, Any]], inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach strictly-prior trend-model median closing odds for a challenger."""
    return _apply_prequential_quantile_odds_policy_inputs(
        races,
        inputs,
        forecasts_key="point_policy_forecasts_by_race_id",
        policy_input="oof_trend_conditional_median_from_real_t5",
        missing_reason="insufficient_trend_closing_odds_teachers",
        incomplete_reason="incomplete_trend_closing_odds_forecast",
    )


def _apply_prequential_quantile_odds_policy_inputs(
    races: list[dict[str, Any]],
    inputs: dict[str, Any],
    *,
    forecasts_key: str,
    policy_input: str,
    missing_reason: str,
    incomplete_reason: str,
) -> list[dict[str, Any]]:
    forecasts = inputs.get(forecasts_key) or {}
    result = []
    for race in races:
        item = dict(race)
        forecast = forecasts.get(str(race["race_id"]))
        if not isinstance(forecast, dict):
            result.extend(
                _attach_t5_policy_fallback(
                    [item], reason=missing_reason
                )
            )
            continue
        estimated = forecast.get("estimated_final_odds") or {}
        if len(estimated) != 120 or set(estimated) != set(race.get("odds") or {}):
            result.extend(
                _attach_t5_policy_fallback(
                    [item], reason=incomplete_reason
                )
            )
            continue
        item.update(forecast)
        item["closing_odds_policy_input"] = policy_input
        item["closing_odds_policy_fallback"] = False
        item.pop("closing_odds_policy_fallback_reason", None)
        result.append(item)
    return result


def evaluate_closing_odds_quantiles(
    races: Iterable[dict[str, Any]], *, include_policy_forecasts: bool = False
) -> dict[str, Any]:
    eligible = verifiable_closing_odds_races(races)
    days = sorted({str(race["race_date"]) for race in eligible})
    if len(days) < 2:
        return {
            "status": "insufficient_independent_snapshots",
            "teacher": "last_preclose_odds_with_verified_source_or_value_change",
            "eligible_races": len(eligible),
            "eligible_days": len(days),
            "minimum_evaluation_days": 2,
        }
    result = walk_forward_closing_odds_quantiles(
        eligible,
        minimum_training_days=1,
        include_policy_forecasts=include_policy_forecasts,
    )
    result["status"] = "evaluated"
    result["teacher"] = "last_preclose_odds_with_verified_source_or_value_change"
    return result


def prequential_conformal_lower_odds_policy_inputs(
    races: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build lower-bound policy prices and their shared forecast report."""
    eligible = verifiable_closing_odds_races(races)
    days = sorted({str(race["race_date"]) for race in eligible})
    if len(days) < 2:
        return {
            "status": "insufficient_independent_snapshots",
            "teacher": "last_preclose_odds_with_verified_source_or_value_change",
            "eligible_races": len(eligible),
            "eligible_days": len(days),
            "minimum_evaluation_days": 2,
            "policy_forecasts_by_race_id": {},
        }
    result = walk_forward_closing_odds_quantiles(
        eligible,
        minimum_training_days=1,
        include_policy_forecasts=True,
    )
    result["status"] = "evaluated"
    result["teacher"] = "last_preclose_odds_with_verified_source_or_value_change"
    return result


def fit_deployment_configuration(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    calibrator_strategy: str,
) -> dict[str, Any]:
    """Refit a next-day configuration on all completed evaluation data."""
    if not races:
        raise ValueError("deployment configuration requires completed races")
    dates = sorted({str(race["race_date"]) for race in races})
    calibrator_selection: dict[str, Any] | None = None
    calibrator_candidates = 0
    operational_model = None
    dual_head_calibration = None
    probability_calibrator = None
    ranking_calibrator = None
    purchase_calibrator = None
    probability_calibrator_selection = None
    ranking_calibrator_selection = None
    purchase_calibrator_selection = None
    if calibrator_strategy in {
        "odds_path_return",
        "odds_path_probability",
        "odds_path_closing_return",
        "odds_path_observed_closing_return",
        V17_STRATEGY_NAME,
        V18_STRATEGY_NAME,
        V19_STRATEGY_NAME,
        V20_STRATEGY_NAME,
        V21_STRATEGY_NAME,
        V35_STRATEGY_NAME,
        "odds_path_hit_shrunk_return",
        "odds_path_prequential_shrinkage_return",
    }:
        if calibrator_strategy == "odds_path_prequential_shrinkage_return":
            operational_model, races = fit_v6_odds_path_model(
                races,
                daily_budget_yen=daily_budget_yen,
            )
        else:
            operational_model = fit_odds_path_model(
                races,
                use_return_multipliers=calibrator_strategy != "odds_path_probability",
                return_price_basis=(
                    "forecast_closing"
                    if calibrator_strategy == "odds_path_closing_return"
                    else "observed_closing"
                    if calibrator_strategy in {
                        "odds_path_observed_closing_return",
                        V17_STRATEGY_NAME,
                        V18_STRATEGY_NAME,
                        V19_STRATEGY_NAME,
                        V20_STRATEGY_NAME,
                        V21_STRATEGY_NAME,
                        V35_STRATEGY_NAME,
                        "odds_path_hit_shrunk_return",
                    }
                    else "decision_t5"
                ),
                return_hit_prior=(
                    20.0
                    if calibrator_strategy == "odds_path_hit_shrunk_return"
                    else 0.0
                ),
                min_return_multiplier=(
                    0.5
                    if calibrator_strategy == "odds_path_hit_shrunk_return"
                    else 0.25
                ),
                max_return_multiplier=(
                    1.5
                    if calibrator_strategy == "odds_path_hit_shrunk_return"
                    else 2.0
                ),
            )
        races = attach_odds_path_model(races, operational_model)
        if calibrator_strategy in {
            V20_STRATEGY_NAME,
            V21_STRATEGY_NAME,
            V35_STRATEGY_NAME,
        }:
            dual_head_calibration = (
                fit_v21_triple_head_calibrators(races)
                if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                else fit_v20_dual_head_calibrators(races)
            )
            probability_calibrator_selection = dual_head_calibration[
                "probability_head"
            ]["selection"]
            ranking_calibrator_selection = dual_head_calibration.get(
                "ranking_head", dual_head_calibration["probability_head"]
            )["selection"]
            purchase_calibrator_selection = dual_head_calibration[
                "purchase_head"
            ]["selection"]
            probability_calibrator = dict(
                dual_head_calibration["probability_head"]["calibrator"]
            )
            ranking_calibrator = dict(
                dual_head_calibration.get(
                    "ranking_head", dual_head_calibration["probability_head"]
                )["calibrator"]
            )
            purchase_calibrator = dict(
                dual_head_calibration["purchase_head"]["calibrator"]
            )
            calibrator_selection = probability_calibrator_selection
            calibrator = probability_calibrator
        else:
            calibrator_selection = fit_market_residual_calibrator(
                races,
                calibrator_strategy=calibrator_strategy,
            )
            calibrator = dict(calibrator_selection["final_calibrator"])
    elif calibrator_strategy == "newton_residual":
        from .market_residual import (
            fit_fixed_regularization,
            select_regularization_prequential,
        )

        calibrator_selection = (
            select_regularization_prequential(races)
            if len(dates) >= 2
            else fit_fixed_regularization(races)
        )
        calibrator = dict(calibrator_selection["final_calibrator"])
    elif calibrator_strategy == "orthogonal_residual":
        from .market_orthogonal_residual import (
            fit_fixed_regularization,
            select_regularization_prequential,
        )

        calibrator_selection = (
            select_regularization_prequential(races)
            if len(dates) >= 2
            else fit_fixed_regularization(races)
        )
        calibrator = dict(calibrator_selection["final_calibrator"])
    elif calibrator_strategy == "grid":
        calibrator, candidates = select_calibrator(races)
        calibrator_candidates = len(candidates)
    else:
        raise ValueError(f"unsupported calibrator strategy: {calibrator_strategy}")

    if probability_calibrator is None:
        probability_calibrator = calibrator
    if ranking_calibrator is None:
        ranking_calibrator = probability_calibrator
    if purchase_calibrator is None:
        purchase_calibrator = calibrator
    if probability_calibrator_selection is None:
        probability_calibrator_selection = calibrator_selection
    if ranking_calibrator_selection is None:
        ranking_calibrator_selection = probability_calibrator_selection
    if purchase_calibrator_selection is None:
        purchase_calibrator_selection = calibrator_selection

    closing_odds_selection = None
    closing_training_races = verifiable_closing_odds_races(races)
    if closing_odds_training_ready(closing_training_races):
        try:
            closing_odds_selection = select_closing_odds_model(
                closing_training_races
            )
        except ValueError:
            pass
    closing_policy_inputs = prequential_closing_odds_policy_inputs(races)
    policy_races = apply_prequential_closing_odds_policy_inputs(
        races, closing_policy_inputs
    )
    forecast_policy_races = sum(
        int(race.get("closing_odds_policy_fallback") is False)
        for race in policy_races
    )
    policy_selector = (
        select_policy_v35
        if calibrator_strategy == V35_STRATEGY_NAME
        else select_policy_v18
        if calibrator_strategy in SCHEDULE_QUOTA_STRATEGIES
        else select_policy_v17
        if calibrator_strategy == V17_STRATEGY_NAME
        else select_policy
    )
    selected_policy, policy_grid = policy_selector(
        policy_races,
        calibrator=purchase_calibrator,
        daily_budget_yen=daily_budget_yen,
    )
    return {
        "role": (
            "evaluation_only_triple_head_refit"
            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
            else "evaluation_only_dual_head_refit"
            if calibrator_strategy == V20_STRATEGY_NAME
            else "next_day_refit_not_evaluation"
        ),
        "validation_design": (
            "Refit only after all listed dates are complete; valid strictly after "
            "trained_through_date"
        ),
        "calibrator_strategy": calibrator_strategy,
        "comparison_role": (
            robust_policy_comparison_role(calibrator_strategy)
            if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES
            else "next_day_refit_not_evaluation"
        ),
        "deployment_mode": (
            "evaluation_only"
            if calibrator_strategy in EVALUATION_ONLY_STRATEGIES
            else "shadow_only"
            if calibrator_strategy in ROBUST_POLICY_STRATEGIES
            else "evaluation"
        ),
        "real_betting_enabled": False,
        "daily_stake_limit_fraction": (
            1.0
            if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES
            else None
        ),
        "trained_dates": dates,
        "trained_through_date": dates[-1],
        "training_races": len(races),
        "calibrator": probability_calibrator,
        "operational_model": operational_model,
        "calibrator_selection": probability_calibrator_selection,
        **(
            {
                (
                    "triple_head_calibration"
                    if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                    else "dual_head_calibration"
                ): dual_head_calibration,
                "probability_calibrator": probability_calibrator,
                "probability_calibrator_selection": probability_calibrator_selection,
                **(
                    {
                        "ranking_calibrator": ranking_calibrator,
                        "ranking_calibrator_selection": ranking_calibrator_selection,
                    }
                    if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                    else {}
                ),
                "purchase_calibrator": purchase_calibrator,
                "purchase_calibrator_selection": purchase_calibrator_selection,
            }
            if dual_head_calibration is not None
            else {}
        ),
        "calibrator_candidates": calibrator_candidates,
        "closing_odds_selection": closing_odds_selection,
        "closing_odds_training_races": len(closing_training_races),
        "closing_odds_training_days": len(
            {str(race["race_date"]) for race in closing_training_races}
        ),
        "closing_odds_policy_input": (
            "date_oof_forecast_with_explicit_observed_t5_fallback"
        ),
        "closing_odds_policy_enabled": forecast_policy_races > 0,
        "closing_odds_forecast_policy_races": forecast_policy_races,
        "closing_odds_fallback_policy_races": (
            len(policy_races) - forecast_policy_races
        ),
        "selected_policy": selected_policy,
        "policy_diagnostics": summarize_policy_candidates(policy_grid),
    }


def walk_forward_evaluate(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int = 10_000,
    min_calibration_days: int = 2,
    calibrator_strategy: str = "grid",
    evaluation_dates: Iterable[str] | None = None,
    v12_closing_fallback_policy: str = "v11",
    v25_probability_artifact: dict[str, Any] | None = None,
    closing_odds_min_training_days: int = MIN_CLOSING_ODDS_TRAINING_DAYS,
    closing_odds_min_training_races: int = MIN_CLOSING_ODDS_TRAINING_RACES,
) -> dict[str, Any]:
    if calibrator_strategy == "odds_path_crossfit_conservative_ev":
        from .odds_path_conservative_v7 import walk_forward_evaluate_v7

        return walk_forward_evaluate_v7(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
        )
    if calibrator_strategy == "odds_path_market_offset_crossfit_conservative_ev":
        from .odds_path_conservative_v8 import walk_forward_evaluate_v8

        return walk_forward_evaluate_v8(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
        )
    if calibrator_strategy == "odds_path_market_offset_discrete_log_ev_v9":
        from .odds_path_discrete_v9 import walk_forward_evaluate_v9

        return walk_forward_evaluate_v9(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
        )
    if (
        calibrator_strategy
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    ):
        from .odds_path_selection_conformal_v10 import walk_forward_evaluate_v10

        return walk_forward_evaluate_v10(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
        )
    if calibrator_strategy == "odds_path_role_integrated_multihorizon_v11":
        from .odds_path_role_integrated_v11 import walk_forward_evaluate_v11

        return walk_forward_evaluate_v11(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
        )
    if calibrator_strategy == "odds_path_role_integrated_t300_nonlinear_v12":
        from .odds_path_role_integrated_v12 import walk_forward_evaluate_v12

        return walk_forward_evaluate_v12(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
            closing_fallback_policy=v12_closing_fallback_policy,
        )
    if calibrator_strategy == "odds_path_role_integrated_edge_conditional_lcb_v13":
        from .odds_path_role_integrated_v13 import walk_forward_evaluate_v13

        return walk_forward_evaluate_v13(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
            closing_fallback_policy=v12_closing_fallback_policy,
        )
    if calibrator_strategy == "odds_path_role_integrated_registered_band_lcb_v14":
        from .odds_path_role_integrated_v14 import walk_forward_evaluate_v14

        return walk_forward_evaluate_v14(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
            closing_fallback_policy=v12_closing_fallback_policy,
        )
    if calibrator_strategy == "odds_path_role_integrated_selection_free_envelope_v15":
        from .odds_path_role_integrated_v15 import walk_forward_evaluate_v15

        result = walk_forward_evaluate_v15(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
            closing_fallback_policy=v12_closing_fallback_policy,
        )
        result["model"] = odds_path_model_name(calibrator_strategy)
        result["calibrator_strategy"] = calibrator_strategy
        deployment = result.get("deployment_configuration")
        if isinstance(deployment, dict):
            deployment["calibrator_strategy"] = calibrator_strategy
        return result
    if calibrator_strategy == "odds_path_role_integrated_fixed_band_passthrough_v16":
        from .odds_path_role_integrated_v16 import walk_forward_evaluate_v16

        result = walk_forward_evaluate_v16(
            races,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            evaluation_dates=evaluation_dates,
            closing_fallback_policy=v12_closing_fallback_policy,
        )
        result["model"] = odds_path_model_name(calibrator_strategy)
        result["calibrator_strategy"] = calibrator_strategy
        deployment = result.get("deployment_configuration")
        if isinstance(deployment, dict):
            deployment["calibrator_strategy"] = calibrator_strategy
        return result
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    candidate_dates = (
        dates
        if evaluation_dates is None
        else sorted({str(date) for date in evaluation_dates if str(date) in by_day})
    )
    fold_dates = []
    for evaluation_date in candidate_dates:
        calibration_dates = [date for date in dates if date < evaluation_date]
        if len(calibration_dates) >= min_calibration_days:
            fold_dates.append((calibration_dates, evaluation_date))
    if not fold_dates:
        return waiting_walk_forward_result(
            races,
            dates=dates,
            evaluation_dates=candidate_dates,
            daily_budget_yen=daily_budget_yen,
            min_calibration_days=min_calibration_days,
            calibrator_strategy=calibrator_strategy,
        )

    closing_policy_inputs = prequential_closing_odds_policy_inputs(
        races,
        min_training_days=closing_odds_min_training_days,
        min_training_races=closing_odds_min_training_races,
    )
    conformal_lower_inputs = prequential_conformal_lower_odds_policy_inputs(races)
    closing_odds_forecast = {
        key: value
        for key, value in conformal_lower_inputs.items()
        if key
        not in {
            "policy_forecasts_by_race_id",
            "point_policy_forecasts_by_race_id",
        }
    }
    conformal_lower_policy_races = (
        apply_prequential_conformal_lower_odds_policy_inputs(
            races, conformal_lower_inputs
        )
    )
    conformal_lower_policy_races_by_id = {
        str(race["race_id"]): race for race in conformal_lower_policy_races
    }
    trend_point_policy_races = apply_prequential_trend_point_odds_policy_inputs(
        races, conformal_lower_inputs
    )
    from .market_kelly_challenger import (
        attach_prequential_market_offsets,
    )

    all_closing_policy_races = apply_prequential_closing_odds_policy_inputs(
        races,
        closing_policy_inputs,
    )
    market_offset_input_ready = bool(all_closing_policy_races) and all(
        len(race.get("model_probabilities") or {}) == 120
        and len(race.get("market_probabilities") or {}) == 120
        and len(decision_odds(race)) == 120
        for race in all_closing_policy_races
    )
    if market_offset_input_ready:
        market_offset_policy_races, market_offset_calibration = (
            attach_prequential_market_offsets(
                all_closing_policy_races,
                select_regularization=True,
            )
        )
    else:
        market_offset_policy_races = []
        market_offset_calibration = {
            "status": "unavailable_incomplete_120_outcome_input",
            "ready_days": 0,
            "fallback_days": len(dates),
            "ready_races": 0,
            "fallback_races": len(all_closing_policy_races),
            "days": [],
        }
    conformal_lower_input_ready = bool(conformal_lower_policy_races) and all(
        len(race.get("model_probabilities") or {}) == 120
        and len(race.get("market_probabilities") or {}) == 120
        and len(decision_odds(race)) == 120
        for race in conformal_lower_policy_races
    )
    if conformal_lower_input_ready:
        conformal_lower_market_races, conformal_lower_market_calibration = (
            attach_prequential_market_offsets(
                conformal_lower_policy_races,
                select_regularization=True,
            )
        )
    else:
        conformal_lower_market_races = []
        conformal_lower_market_calibration = {
            "status": "unavailable_incomplete_120_outcome_input",
            "ready_days": 0,
            "fallback_days": len(dates),
            "ready_races": 0,
            "fallback_races": len(conformal_lower_policy_races),
            "days": [],
        }
    trend_point_input_ready = bool(trend_point_policy_races) and all(
        len(race.get("model_probabilities") or {}) == 120
        and len(race.get("market_probabilities") or {}) == 120
        and len(decision_odds(race)) == 120
        for race in trend_point_policy_races
    )
    if trend_point_input_ready:
        trend_point_market_races, trend_point_market_calibration = (
            attach_prequential_market_offsets(
                trend_point_policy_races,
                select_regularization=True,
            )
        )
    else:
        trend_point_market_races = []
        trend_point_market_calibration = {
            "status": "unavailable_incomplete_120_outcome_input",
            "ready_days": 0,
            "fallback_days": len(dates),
            "ready_races": 0,
            "fallback_races": len(trend_point_policy_races),
            "days": [],
        }
    folds = []
    evaluation_races: list[dict[str, Any]] = []
    evaluation_policy_races: list[dict[str, Any]] = []
    market_loss_differences: list[float] = []
    market_top5_differences: list[float] = []
    market_cluster_labels: list[str] = []
    daily_rows = []
    chronological_daily_rows: list[dict[str, Any]] = []
    flat_daily_rows = []
    registered_daily_rows = []
    registered_evaluated_races = 0
    prospective_daily_rows = []
    prospective_evaluated_races = 0
    prospective_top5_daily_rows = []
    prospective_top5_evaluated_races = 0
    top5_narrow_retrospective_daily_rows = []
    top5_narrow_retrospective_evaluated_races = 0
    v32_retrospective_daily_rows = []
    v32_retrospective_evaluated_races = 0
    v32_prospective_daily_rows = []
    v32_prospective_evaluated_races = 0
    v33_retrospective_daily_rows = []
    v33_retrospective_evaluated_races = 0
    v33_prospective_daily_rows = []
    v33_prospective_evaluated_races = 0
    v33_forecast_daily_rows = []
    v33_forecast_evaluated_races = 0
    prospective_architecture_daily_rows = []
    prospective_architecture_evaluated_races = 0
    prospective_architecture_config = {
        "odds_path_observed_closing_return": {
            "registered_after": OBSERVED_CLOSING_RETURN_V4_REGISTERED_AFTER,
            "output_key": "prospective_observed_closing_return_v4_walk_forward",
            "architecture": "odds_path_observed_closing_return_v4",
        },
        "odds_path_prequential_shrinkage_return": {
            "registered_after": PREQUENTIAL_SHRINKAGE_RETURN_V6_REGISTERED_AFTER,
            "output_key": "prospective_prequential_shrinkage_return_v6_walk_forward",
            "architecture": "odds_path_prequential_shrinkage_return_v6",
        },
    }.get(calibrator_strategy)
    edge_diagnostic_records = []
    empirical_history_records: list[dict[str, Any]] = []
    empirical_daily_rows: list[dict[str, Any]] = []
    empirical_fold_rows: list[dict[str, Any]] = []
    empirical_evaluated_races = 0
    for calibration_dates, evaluation_date in fold_dates:
        calibration_races = [race for date in calibration_dates for race in by_day[date]]
        holdout = by_day[evaluation_date]
        calibrator_selection = None
        operational_model = None
        dual_head_calibration = None
        probability_calibrator = None
        ranking_calibrator = None
        purchase_calibrator = None
        probability_calibrator_selection = None
        ranking_calibrator_selection = None
        purchase_calibrator_selection = None
        if calibrator_strategy in {
            "odds_path_return",
            "odds_path_probability",
            "odds_path_closing_return",
            "odds_path_observed_closing_return",
            V17_STRATEGY_NAME,
            V18_STRATEGY_NAME,
            V19_STRATEGY_NAME,
            V20_STRATEGY_NAME,
            V21_STRATEGY_NAME,
            V35_STRATEGY_NAME,
            "odds_path_hit_shrunk_return",
            "odds_path_prequential_shrinkage_return",
        }:
            if calibrator_strategy == "odds_path_prequential_shrinkage_return":
                operational_model, calibration_races = fit_v6_odds_path_model(
                    calibration_races,
                    daily_budget_yen=daily_budget_yen,
                )
            elif calibrator_strategy == "odds_path_closing_return":
                calibration_races = attach_forecast_closing_return_prices(
                    calibration_races,
                    closing_policy_inputs,
                )
            elif calibrator_strategy in {
                "odds_path_observed_closing_return",
                V17_STRATEGY_NAME,
                V18_STRATEGY_NAME,
                V19_STRATEGY_NAME,
                V20_STRATEGY_NAME,
                V21_STRATEGY_NAME,
                V35_STRATEGY_NAME,
                "odds_path_hit_shrunk_return",
            }:
                calibration_races = attach_observed_closing_return_prices(
                    calibration_races
                )
            if calibrator_strategy != "odds_path_prequential_shrinkage_return":
                operational_model = fit_odds_path_model(
                    calibration_races,
                    use_return_multipliers=(
                        calibrator_strategy != "odds_path_probability"
                    ),
                    return_price_basis=(
                        "forecast_closing"
                        if calibrator_strategy == "odds_path_closing_return"
                        else "observed_closing"
                        if calibrator_strategy in {
                            "odds_path_observed_closing_return",
                            V17_STRATEGY_NAME,
                            V18_STRATEGY_NAME,
                            V19_STRATEGY_NAME,
                            V20_STRATEGY_NAME,
                            V21_STRATEGY_NAME,
                            V35_STRATEGY_NAME,
                            "odds_path_hit_shrunk_return",
                        }
                        else "decision_t5"
                    ),
                    return_hit_prior=(
                        20.0
                        if calibrator_strategy == "odds_path_hit_shrunk_return"
                        else 0.0
                    ),
                    min_return_multiplier=(
                        0.5
                        if calibrator_strategy == "odds_path_hit_shrunk_return"
                        else 0.25
                    ),
                    max_return_multiplier=(
                        1.5
                        if calibrator_strategy == "odds_path_hit_shrunk_return"
                        else 2.0
                    ),
                )
            calibration_races = attach_odds_path_model(
                calibration_races, operational_model
            )
            holdout = attach_odds_path_model(holdout, operational_model)
            if calibrator_strategy in {
                V20_STRATEGY_NAME,
                V21_STRATEGY_NAME,
                V35_STRATEGY_NAME,
            }:
                dual_head_calibration = (
                    fit_v21_triple_head_calibrators(calibration_races)
                    if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                    else fit_v20_dual_head_calibrators(calibration_races)
                )
                probability_calibrator_selection = dual_head_calibration[
                    "probability_head"
                ]["selection"]
                ranking_calibrator_selection = dual_head_calibration.get(
                    "ranking_head", dual_head_calibration["probability_head"]
                )["selection"]
                purchase_calibrator_selection = dual_head_calibration[
                    "purchase_head"
                ]["selection"]
                probability_calibrator = dict(
                    dual_head_calibration["probability_head"]["calibrator"]
                )
                ranking_calibrator = dict(
                    dual_head_calibration.get(
                        "ranking_head", dual_head_calibration["probability_head"]
                    )["calibrator"]
                )
                purchase_calibrator = dict(
                    dual_head_calibration["purchase_head"]["calibrator"]
                )
                calibrator_selection = probability_calibrator_selection
                calibrator = probability_calibrator
            else:
                calibrator_selection = fit_market_residual_calibrator(
                    calibration_races,
                    calibrator_strategy=calibrator_strategy,
                )
                calibrator = dict(calibrator_selection["final_calibrator"])
            calibrator_grid = []
        elif calibrator_strategy == "newton_residual":
            from .market_residual import (
                fit_fixed_regularization,
                select_regularization_prequential,
            )

            calibration_day_count = len(
                {str(race["race_date"]) for race in calibration_races}
            )
            calibrator_selection = (
                select_regularization_prequential(calibration_races)
                if calibration_day_count >= 2
                else fit_fixed_regularization(calibration_races)
            )
            calibrator = dict(calibrator_selection["final_calibrator"])
            calibrator_grid = []
        elif calibrator_strategy == "orthogonal_residual":
            from .market_orthogonal_residual import (
                fit_fixed_regularization,
                select_regularization_prequential,
            )

            calibration_day_count = len(
                {str(race["race_date"]) for race in calibration_races}
            )
            calibrator_selection = (
                select_regularization_prequential(calibration_races)
                if calibration_day_count >= 2
                else fit_fixed_regularization(calibration_races)
            )
            calibrator = dict(calibrator_selection["final_calibrator"])
            calibrator_grid = []
        elif calibrator_strategy == "grid":
            calibrator, calibrator_grid = select_calibrator(calibration_races)
        else:
            raise ValueError(f"unsupported calibrator strategy: {calibrator_strategy}")
        if probability_calibrator is None:
            probability_calibrator = calibrator
        if ranking_calibrator is None:
            ranking_calibrator = probability_calibrator
        if purchase_calibrator is None:
            purchase_calibrator = calibrator
        if probability_calibrator_selection is None:
            probability_calibrator_selection = calibrator_selection
        if ranking_calibrator_selection is None:
            ranking_calibrator_selection = probability_calibrator_selection
        if purchase_calibrator_selection is None:
            purchase_calibrator_selection = calibrator_selection

        calibration_policy_races = apply_prequential_closing_odds_policy_inputs(
            calibration_races, closing_policy_inputs
        )
        holdout_policy_races = apply_prequential_closing_odds_policy_inputs(
            holdout, closing_policy_inputs
        )
        holdout_conformal_lower_races = [
            conformal_lower_policy_races_by_id[str(race["race_id"])]
            for race in holdout
            if str(race["race_id"]) in conformal_lower_policy_races_by_id
        ]
        if len(holdout_conformal_lower_races) != len(holdout):
            raise ValueError(
                "V32 diagnostic requires a strict-prior conformal lower forecast "
                "for every holdout race"
            )
        closing_training_races = verifiable_closing_odds_races(calibration_races)
        closing_holdout_races = verifiable_closing_odds_races(holdout)
        closing_policy_fold = closing_policy_inputs["folds"][evaluation_date]
        closing_odds_selection = closing_policy_fold["selection"]
        closing_odds_model = closing_policy_fold["model"]
        closing_odds_evaluation = closing_policy_fold["evaluation"]
        policy_selector = (
            select_policy_v35
            if calibrator_strategy == V35_STRATEGY_NAME
            else select_policy_v18
            if calibrator_strategy in SCHEDULE_QUOTA_STRATEGIES
            else select_policy_v17
            if calibrator_strategy == V17_STRATEGY_NAME
            else select_policy
        )
        policy, policy_grid = policy_selector(
            calibration_policy_races,
            calibrator=purchase_calibrator,
            daily_budget_yen=daily_budget_yen,
        )
        bankroll = simulate_policy(
            holdout_policy_races,
            calibrator=purchase_calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
            include_chronological=True,
        )
        registered_bankroll = None
        if evaluation_date > EV_BAND_HYPOTHESIS_REGISTERED_AFTER:
            registered_bankroll = simulate_policy(
                holdout_policy_races,
                calibrator=purchase_calibrator,
                policy=REGISTERED_EV_BAND_POLICY,
                daily_budget_yen=daily_budget_yen,
                include_chronological=True,
            )
        prospective_bankroll = None
        if evaluation_date > PROSPECTIVE_NORMALIZED_EV_REGISTERED_AFTER:
            prospective_bankroll = simulate_policy(
                holdout_policy_races,
                calibrator=purchase_calibrator,
                policy=PROSPECTIVE_NORMALIZED_EV_POLICY,
                daily_budget_yen=daily_budget_yen,
                include_chronological=True,
            )
        top5_narrow_simulator = (
            simulate_chronological_flat_policy
            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
            else simulate_flat_policy
        )
        top5_narrow_retrospective_bankroll = top5_narrow_simulator(
            holdout_policy_races,
            calibrator=purchase_calibrator,
            policy=PROSPECTIVE_TOP5_NARROW_EV_POLICY,
            probability_blender=blend_probabilities,
        )
        prospective_top5_bankroll = (
            top5_narrow_retrospective_bankroll
            if evaluation_date > PROSPECTIVE_TOP5_NARROW_EV_REGISTERED_AFTER
            else None
        )
        v32_retrospective_bankroll = None
        v32_prospective_bankroll = None
        v33_retrospective_bankroll = None
        v33_prospective_bankroll = None
        if calibrator_strategy in TRIPLE_HEAD_STRATEGIES:
            v32_retrospective_bankroll = simulate_dual_head_conformal_policy_v32(
                holdout_conformal_lower_races,
                probability_calibrator=probability_calibrator,
                ranking_calibrator=ranking_calibrator,
                probability_blender=blend_probabilities,
                initial_bankroll_yen=daily_budget_yen,
            )
            if evaluation_date > V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER:
                v32_prospective_bankroll = v32_retrospective_bankroll
            if v25_probability_artifact is not None:
                v33_retrospective_bankroll = simulate_v25_top1_narrow_v33(
                    holdout_policy_races,
                    probability_artifact=v25_probability_artifact,
                    initial_bankroll_yen=daily_budget_yen,
                )
                if evaluation_date > V33_V25_TOP1_NARROW_REGISTERED_AFTER:
                    v33_prospective_bankroll = v33_retrospective_bankroll
        flat_policy, flat_policy_grid = select_flat_policy(
            calibration_policy_races,
            calibrator=purchase_calibrator,
            probability_blender=blend_probabilities,
        )
        empirical_artifact = _fit_prior_empirical_ev_artifact(
            empirical_history_records, evaluation_date
        )
        empirical_bankroll = simulate_empirical_lcb_policy(
            holdout_policy_races,
            purchase_calibrator,
            blend_probabilities,
            empirical_artifact,
            daily_budget_yen,
        )
        empirical_fold_rows.append(
            {
                "fold": len(folds) + 1,
                "evaluation_date": evaluation_date,
                "calibration_ready": empirical_artifact.ready,
                "trained_through_date": empirical_artifact.trained_through_date,
                "training_days": empirical_artifact.training_days,
                "training_tickets": empirical_artifact.tickets,
                "candidate_days": empirical_artifact.candidate_days,
                "ready_reasons": list(empirical_artifact.ready_reasons),
            }
        )
        flat_bankroll = simulate_flat_policy(
            holdout_policy_races,
            calibrator=purchase_calibrator,
            policy=flat_policy,
            probability_blender=blend_probabilities,
        )
        metrics = (
            split_head_probability_metrics(
                holdout,
                probability_calibrator=probability_calibrator,
                ranking_calibrator=ranking_calibrator,
            )
            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
            else probability_metrics(
                holdout, calibrator=probability_calibrator
            )
        )
        edge_diagnostic_records.extend(
            edge_records(
                holdout_policy_races,
                calibrator=purchase_calibrator,
                probability_blender=blend_probabilities,
            )
        )
        if calibrator_strategy in TRIPLE_HEAD_STRATEGIES:
            (
                fold_loss_differences,
                fold_top5_differences,
            ) = split_head_paired_market_differences(
                holdout,
                probability_calibrator=probability_calibrator,
                ranking_calibrator=ranking_calibrator,
            )
        else:
            (
                fold_loss_differences,
                fold_top5_differences,
            ) = paired_market_differences(
                holdout,
                calibrator=probability_calibrator,
            )
        market_loss_differences.extend(fold_loss_differences)
        market_top5_differences.extend(fold_top5_differences)
        market_cluster_labels.extend(
            [evaluation_date] * len(fold_loss_differences)
        )
        folds.append(
            {
                "fold": len(folds) + 1,
                "calibration_dates": calibration_dates,
                "evaluation_date": evaluation_date,
                "calibration_races": len(calibration_races),
                "evaluation_races": len(holdout),
                "calibrator": probability_calibrator,
                "calibrator_strategy": calibrator_strategy,
                "calibrator_selection": probability_calibrator_selection,
                **(
                    {
                        (
                            "triple_head_calibration"
                            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                            else "dual_head_calibration"
                        ): dual_head_calibration,
                        "probability_calibrator": probability_calibrator,
                        "probability_calibrator_selection": (
                            probability_calibrator_selection
                        ),
                        **(
                            {
                                "ranking_calibrator": ranking_calibrator,
                                "ranking_calibrator_selection": (
                                    ranking_calibrator_selection
                                ),
                            }
                            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                            else {}
                        ),
                        "purchase_calibrator": purchase_calibrator,
                        "purchase_calibrator_selection": (
                            purchase_calibrator_selection
                        ),
                        "probability_metrics_head": "probability_head",
                        **(
                            {
                                "trifecta_top5_head": "ranking_head",
                                "market_logloss_comparison_head": (
                                    "probability_head"
                                ),
                                "market_top5_comparison_head": "ranking_head",
                            }
                            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
                            else {}
                        ),
                        "chronological_bankroll_head": "purchase_head",
                    }
                    if dual_head_calibration is not None
                    else {}
                ),
                "operational_model": operational_model,
                "closing_odds_model": closing_odds_model,
                "closing_odds_selection": closing_odds_selection,
                "closing_odds_evaluation": closing_odds_evaluation,
                "closing_odds_training_races": len(closing_training_races),
                "closing_odds_holdout_races": len(closing_holdout_races),
                "closing_odds_model_trained_through_date": closing_policy_fold[
                    "trained_through_date"
                ],
                "closing_odds_policy_input": closing_policy_fold["policy_input"],
                "closing_odds_policy_fallback_reason": closing_policy_fold[
                    "fallback_reason"
                ],
                "selected_policy": policy,
                "calibrator_candidates": len(calibrator_grid),
                "policy_candidates": len(policy_grid),
                "calibrator_top5": (
                    sorted(
                        calibrator_grid,
                        key=lambda row: (
                            row["trifecta_log_loss"],
                            -row["trifecta_top5_hit_rate"],
                        ),
                    )[:5]
                    if calibrator_strategy == "grid"
                    else []
                ),
                "policy_diagnostics": summarize_policy_candidates(policy_grid),
                "selected_flat_policy": flat_policy,
                "flat_policy_diagnostics": summarize_flat_candidates(flat_policy_grid),
                "flat_bankroll": {
                    key: value for key, value in flat_bankroll.items() if key != "daily"
                },
                "probability_metrics": metrics,
                "market_comparison": {
                    "log_loss_mean_difference": (
                        sum(fold_loss_differences) / len(fold_loss_differences)
                    ),
                    "top5_mean_difference": (
                        sum(fold_top5_differences) / len(fold_top5_differences)
                    ),
                },
                "bankroll": {key: value for key, value in bankroll.items() if key != "daily"},
                "registered_ev_band_bankroll": (
                    {
                        key: value
                        for key, value in registered_bankroll.items()
                        if key != "daily"
                    }
                    if registered_bankroll is not None
                    else None
                ),
                "prospective_normalized_ev_bankroll": (
                    {
                        key: value
                        for key, value in prospective_bankroll.items()
                        if key != "daily"
                    }
                    if prospective_bankroll is not None
                    else None
                ),
                "prospective_top5_narrow_ev_bankroll": (
                    {
                        key: value
                        for key, value in prospective_top5_bankroll.items()
                        if key != "daily"
                    }
                    if prospective_top5_bankroll is not None
                    else None
                ),
                "top5_narrow_retrospective_bankroll": {
                    key: value
                    for key, value in top5_narrow_retrospective_bankroll.items()
                    if key != "daily"
                },
                "v32_dual_head_conformal_retrospective_bankroll": (
                    {
                        key: value
                        for key, value in v32_retrospective_bankroll.items()
                        if key != "daily"
                    }
                    if v32_retrospective_bankroll is not None
                    else None
                ),
                "v32_dual_head_conformal_prospective_bankroll": (
                    {
                        key: value
                        for key, value in v32_prospective_bankroll.items()
                        if key != "daily"
                    }
                    if v32_prospective_bankroll is not None
                    else None
                ),
                "v33_v25_top1_narrow_retrospective_bankroll": (
                    {
                        key: value
                        for key, value in v33_retrospective_bankroll.items()
                        if key != "daily"
                    }
                    if v33_retrospective_bankroll is not None
                    else None
                ),
                "v33_v25_top1_narrow_prospective_bankroll": (
                    {
                        key: value
                        for key, value in v33_prospective_bankroll.items()
                        if key != "daily"
                    }
                    if v33_prospective_bankroll is not None
                    else None
                ),
                "empirical_lcb_bankroll": {
                    key: value
                    for key, value in empirical_bankroll.items()
                    if key != "daily"
                },
            }
        )
        daily_rows.extend(bankroll["daily"])
        chronological_daily_rows.extend(
            bankroll["chronological_bankroll"]["daily"]
        )
        if (
            prospective_architecture_config is not None
            and evaluation_date
            > str(prospective_architecture_config["registered_after"])
        ):
            prospective_architecture_daily_rows.extend(bankroll["daily"])
            prospective_architecture_evaluated_races += len(
                holdout_policy_races
            )
        flat_daily_rows.extend(flat_bankroll["daily"])
        if registered_bankroll is not None:
            registered_daily_rows.extend(
                registered_bankroll["chronological_bankroll"]["daily"]
            )
            registered_evaluated_races += len(holdout_policy_races)
        if prospective_bankroll is not None:
            prospective_daily_rows.extend(
                prospective_bankroll["chronological_bankroll"]["daily"]
            )
            prospective_evaluated_races += len(holdout_policy_races)
        if prospective_top5_bankroll is not None:
            prospective_top5_daily_rows.extend(
                prospective_top5_bankroll["daily"]
            )
            prospective_top5_evaluated_races += len(holdout_policy_races)
        top5_narrow_retrospective_daily_rows.extend(
            top5_narrow_retrospective_bankroll["daily"]
        )
        top5_narrow_retrospective_evaluated_races += len(holdout_policy_races)
        if v32_retrospective_bankroll is not None:
            v32_retrospective_daily_rows.extend(
                v32_retrospective_bankroll["daily"]
            )
            v32_retrospective_evaluated_races += len(
                holdout_conformal_lower_races
            )
        if v32_prospective_bankroll is not None:
            v32_prospective_daily_rows.extend(v32_prospective_bankroll["daily"])
            v32_prospective_evaluated_races += len(
                holdout_conformal_lower_races
            )
        if v33_retrospective_bankroll is not None:
            v33_retrospective_daily_rows.extend(
                v33_retrospective_bankroll["daily"]
            )
            v33_retrospective_evaluated_races += len(holdout_policy_races)
            if (
                closing_policy_fold["policy_input"]
                == "oof_forecast_final_from_real_t5"
            ):
                v33_forecast_daily_rows.extend(v33_retrospective_bankroll["daily"])
                v33_forecast_evaluated_races += len(holdout_policy_races)
        if v33_prospective_bankroll is not None:
            v33_prospective_daily_rows.extend(v33_prospective_bankroll["daily"])
            v33_prospective_evaluated_races += len(holdout_policy_races)
        evaluation_races.extend(holdout)
        evaluation_policy_races.extend(holdout_policy_races)
        empirical_daily_rows.extend(empirical_bankroll["daily"])
        empirical_evaluated_races += len(holdout_policy_races)
        current_empirical_records = policy_edge_records(
            holdout_policy_races,
            purchase_calibrator,
            blend_probabilities,
        )
        if any(
            str(row["race_date"]) != evaluation_date
            for row in current_empirical_records
        ):
            raise AssertionError("empirical EV fold produced a mismatched teacher date")
        empirical_history_records.extend(current_empirical_records)

    evaluation_date_set = {date for _calibration, date in fold_dates}
    market_offset_evaluation_races = [
        race
        for race in market_offset_policy_races
        if str(race["race_date"]) in evaluation_date_set
    ]
    if not market_offset_input_ready:
        # Keep legacy small-vector fixtures and explicit incomplete inputs
        # evaluable while the 120-outcome challenger remains unavailable.
        market_offset_evaluation_races = list(evaluation_policy_races)
    market_offset_registered = simulate_policy(
        market_offset_evaluation_races,
        calibrator={"model_weight": 0.0, "temperature": 1.0},
        policy=REGISTERED_EV_BAND_POLICY,
        daily_budget_yen=daily_budget_yen,
    )
    market_offset_registered.update(
        {
            "comparison_role": "market_offset_with_registered_policy_challenger",
            "calibration": market_offset_calibration,
            "promotion_eligible": False,
        }
    )
    market_offset_bootstrap = bootstrap_daily_roi(market_offset_registered["daily"])
    market_offset_registered["bootstrap"] = market_offset_bootstrap
    market_offset_registered["promotion_gate"] = {
        "minimum_purchase_days": 30,
        "minimum_tickets": 300,
        "sample_size_pass": (
            sum(
                int((row.get("stake_yen") or 0) > 0)
                for row in market_offset_registered["daily"]
            )
            >= 30
            and int(market_offset_registered["tickets"]) >= 300
        ),
        "roi_pass": float(market_offset_registered["roi"] or 0.0) > 1.0,
        "largest_hit_excluded_roi_pass": (
            float(market_offset_registered.get("roi_without_largest_hit") or 0.0)
            > 1.0
        ),
        "bootstrap_lower_95_pass": (
            float(market_offset_bootstrap.get("roi_ci95_lower") or 0.0) > 1.0
        ),
        "bootstrap_probability_pass": (
            float(market_offset_bootstrap.get("probability_roi_above_one") or 0.0)
            >= 0.95
        ),
    }
    market_offset_registered["promotion_gate"]["pass"] = all(
        market_offset_registered["promotion_gate"][key]
        for key in (
            "sample_size_pass",
            "roi_pass",
            "largest_hit_excluded_roi_pass",
            "bootstrap_lower_95_pass",
            "bootstrap_probability_pass",
        )
    )
    from .market_kelly_challenger import (
        evaluate_attached_market_kelly_challenger,
    )
    market_offset_multinomial_kelly = evaluate_attached_market_kelly_challenger(
        market_offset_policy_races,
        calibration=market_offset_calibration,
        evaluation_dates=evaluation_date_set,
    )
    conservative_evaluation_dates = sorted(
        race_date
        for race_date in evaluation_date_set
        if race_date > CONSERVATIVE_MARKET_KELLY_REGISTERED_AFTER
    )
    conservative_market_offset_kelly = evaluate_attached_market_kelly_challenger(
        market_offset_policy_races,
        calibration=market_offset_calibration,
        evaluation_dates=conservative_evaluation_dates,
        odds_safety_factor=CONSERVATIVE_MARKET_KELLY_ODDS_SAFETY_FACTOR,
    )
    conservative_market_offset_kelly.update({
        "challenger": "conservative_market_offset_discrete_multinomial_kelly",
        "comparison_role": (
            "pure prospective market-offset Kelly with a registered odds-error haircut"
        ),
        "registered_after": CONSERVATIVE_MARKET_KELLY_REGISTERED_AFTER,
        "status": (
            "evaluating"
            if conservative_evaluation_dates
            else "waiting_for_first_unseen_day"
        ),
        "promotion_eligible": bool(
            conservative_market_offset_kelly["promotion_gate"]["pass"]
        ),
    })
    conformal_lower_diagnostic = evaluate_attached_market_kelly_challenger(
        conformal_lower_market_races,
        calibration=conformal_lower_market_calibration,
        evaluation_dates=evaluation_date_set,
    )
    conformal_lower_diagnostic.update({
        "challenger": "conformal_lower_market_offset_discrete_multinomial_kelly",
        "comparison_role": (
            "retrospective research diagnostic using a strictly-prior adaptive "
            "conformal lower closing-odds bound"
        ),
        "promotion_eligible": False,
    })
    conformal_lower_evaluation_dates = sorted(
        race_date
        for race_date in evaluation_date_set
        if race_date > CONFORMAL_LOWER_KELLY_REGISTERED_AFTER
    )
    conformal_lower_prospective = evaluate_attached_market_kelly_challenger(
        conformal_lower_market_races,
        calibration=conformal_lower_market_calibration,
        evaluation_dates=conformal_lower_evaluation_dates,
    )
    conformal_lower_prospective.update({
        "challenger": "conformal_lower_market_offset_discrete_multinomial_kelly",
        "comparison_role": (
            "pure prospective Kelly using a registered adaptive conformal lower "
            "closing-odds bound"
        ),
        "registered_after": CONFORMAL_LOWER_KELLY_REGISTERED_AFTER,
        "status": (
            "evaluating"
            if conformal_lower_evaluation_dates
            else "waiting_for_first_unseen_day"
        ),
        "promotion_eligible": bool(
            conformal_lower_prospective["promotion_gate"]["pass"]
        ),
    })
    trend_point_diagnostic = evaluate_attached_market_kelly_challenger(
        trend_point_market_races,
        calibration=trend_point_market_calibration,
        evaluation_dates=evaluation_date_set,
    )
    trend_point_diagnostic.update({
        "challenger": "trend_point_market_offset_discrete_multinomial_kelly",
        "comparison_role": (
            "retrospective research diagnostic using a strictly-prior trend-model "
            "conditional median closing-odds forecast"
        ),
        "promotion_eligible": False,
    })
    trend_point_evaluation_dates = sorted(
        race_date
        for race_date in evaluation_date_set
        if race_date > TREND_POINT_KELLY_REGISTERED_AFTER
    )
    trend_point_prospective = evaluate_attached_market_kelly_challenger(
        trend_point_market_races,
        calibration=trend_point_market_calibration,
        evaluation_dates=trend_point_evaluation_dates,
    )
    trend_point_prospective.update({
        "challenger": "trend_point_market_offset_discrete_multinomial_kelly",
        "comparison_role": (
            "pure prospective Kelly using a registered trend-model conditional "
            "median closing-odds forecast"
        ),
        "registered_after": TREND_POINT_KELLY_REGISTERED_AFTER,
        "status": (
            "evaluating"
            if trend_point_evaluation_dates
            else "waiting_for_first_unseen_day"
        ),
        "promotion_eligible": bool(
            trend_point_prospective["promotion_gate"]["pass"]
        ),
    })

    stake_yen = sum(int(row["stake_yen"]) for row in daily_rows)
    return_yen = sum(int(row["return_yen"]) for row in daily_rows)
    cumulative_profit = peak_profit = max_drawdown_yen = 0
    for row in daily_rows:
        cumulative_profit += int(row["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown_yen = max(max_drawdown_yen, peak_profit - cumulative_profit)
        row["cumulative_profit_yen"] = cumulative_profit
    profitable_folds = sum(int(fold["bankroll"]["profit_yen"] > 0) for fold in folds)
    flat_stake_yen = sum(int(row["stake_yen"]) for row in flat_daily_rows)
    flat_return_yen = sum(int(row["return_yen"]) for row in flat_daily_rows)
    chronological_bankroll = summarize_chronological_bankroll_days(
        chronological_daily_rows
    )
    chronological_reliability = bankroll_reliability_metrics(
        chronological_daily_rows, evaluated_races=len(evaluation_races)
    )
    chronological_bootstrap = (
        bootstrap_daily_roi(chronological_daily_rows)
        if chronological_daily_rows else {}
    )
    chronological_bankroll.update({
        **chronological_reliability,
        "profitable_day_fraction": (
            chronological_bankroll["winning_days"]
            / chronological_bankroll["race_days"]
            if chronological_bankroll["race_days"] else None
        ),
        "normalized_drawdown": (
            chronological_bankroll["max_drawdown_yen"]
            / chronological_bankroll["stake_yen"]
            if chronological_bankroll["stake_yen"] else None
        ),
        "daily_cluster_bootstrap_roi_lower_95": (
            chronological_bootstrap.get("roi_ci95_lower")
        ),
        "bootstrap_probability_roi_above_one": (
            chronological_bootstrap.get("probability_roi_above_one")
        ),
        "primary_promotion_bankroll": (
            calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES
        ),
        "daily_stake_limit_fraction": 1.0,
        "gross_stake_allowance_rule": (
            "initial_allowance_plus_positive_part_of_cumulative_net_realized_profit"
        ),
        "legacy_cash_recycling_only_is_primary": False,
    })
    aggregate_metrics = _aggregate_fold_probability_metrics(folds)
    market_comparison = market_comparison_confidence(
        market_loss_differences,
        market_top5_differences,
        cluster_labels=market_cluster_labels,
    )
    if calibrator_strategy in TRIPLE_HEAD_STRATEGIES:
        market_comparison.update({
            "logloss_difference_source": "probability_head",
            "top5_difference_source": "ranking_head",
            "probability_calibrator_strategy": V19_STRATEGY_NAME,
            "ranking_calibrator_strategy": V18_STRATEGY_NAME,
            "selection_data": (
                "strict_prior_training_and_inner_prequential_folds_only"
            ),
            "outer_holdout_used_for_selection": False,
        })
    empirical_lcb_walk_forward = _summarize_empirical_lcb_walk_forward(
        empirical_daily_rows,
        evaluated_races=empirical_evaluated_races,
        folds=empirical_fold_rows,
    )
    prospective_architecture = summarize_registered_policy_daily(
        prospective_architecture_daily_rows,
        evaluated_races=prospective_architecture_evaluated_races,
        policy={
            "name": "architecture_daily_prior_selected_policy",
            "architecture": (
                prospective_architecture_config["architecture"]
                if prospective_architecture_config is not None
                else None
            ),
            "minimum_days": MIN_PROSPECTIVE_ARCHITECTURE_DAYS,
            "minimum_tickets": MIN_PROSPECTIVE_ARCHITECTURE_TICKETS,
        },
        registered_after=(
            str(prospective_architecture_config["registered_after"])
            if prospective_architecture_config is not None
            else OBSERVED_CLOSING_RETURN_V4_REGISTERED_AFTER
        ),
    )
    prospective_architecture_pass = bool(
        prospective_architecture_config is not None
        and prospective_architecture["evaluation_days"]
        >= MIN_PROSPECTIVE_ARCHITECTURE_DAYS
        and prospective_architecture["tickets"]
        >= MIN_PROSPECTIVE_ARCHITECTURE_TICKETS
        and prospective_architecture["profit_yen"] > 0
        and float(prospective_architecture["roi"] or 0.0) > 1.0
        and float(
            prospective_architecture["roi_without_largest_hit"] or 0.0
        )
        > 1.0
    )
    promotion_gate = {
        "minimum_evaluation_races": 1000,
        "minimum_evaluation_days": 30,
        "minimum_profitable_fold_fraction": 0.60,
        "sample_size_pass": len(evaluation_races) >= 1000 and len(daily_rows) >= 30,
        "positive_profit_pass": return_yen > stake_yen and stake_yen > 0,
        "roi_pass": return_yen / stake_yen > 1.0 if stake_yen else False,
        "fold_stability_pass": profitable_folds >= math.ceil(len(folds) * 0.60),
        "calibration_pass": _calibration_gate_pass(aggregate_metrics),
        "market_confidence_pass": bool(market_comparison["confidence_pass"]),
        "empirical_lcb_policy_pass": bool(
            empirical_lcb_walk_forward["sample_size_pass"]
            and empirical_lcb_walk_forward["profit_yen"] > 0
            and float(empirical_lcb_walk_forward["roi"] or 0.0) > 1.0
            and float(
                empirical_lcb_walk_forward["roi_without_largest_hit"] or 0.0
            ) > 1.0
        ),
        "no_lookahead_pass": True,
        "prospective_architecture_pass": (
            prospective_architecture_pass
            if prospective_architecture_config is not None
            else True
        ),
    }
    if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES:
        promotion_gate = {
            "primary_bankroll": "chronological_bankroll",
            "minimum_evaluation_races": 1000,
            "minimum_evaluation_days": 30,
            "minimum_profitable_day_fraction": 0.60,
            "minimum_effective_hit_count": 20,
            "sample_size_pass": (
                len(evaluation_races) >= 1000
                and chronological_bankroll["race_days"] >= 30
            ),
            "positive_profit_pass": (
                chronological_bankroll["profit_yen"] > 0
                and chronological_bankroll["stake_yen"] > 0
            ),
            "roi_pass": float(chronological_bankroll["roi"] or 0.0) > 1.0,
            "profitable_day_fraction_pass": float(
                chronological_bankroll.get("profitable_day_fraction") or 0.0
            ) >= 0.60,
            "largest_hit_excluded_roi_pass": float(
                chronological_bankroll.get("roi_without_largest_hit") or 0.0
            ) > 1.0,
            "bootstrap_lower_95_pass": float(
                chronological_bankroll.get(
                    "daily_cluster_bootstrap_roi_lower_95"
                ) or 0.0
            ) > 1.0,
            "effective_hit_count_pass": float(
                chronological_bankroll.get("effective_hit_count") or 0.0
            ) >= 20.0,
            "calibration_pass": _calibration_gate_pass(aggregate_metrics),
            "market_confidence_pass": bool(market_comparison["confidence_pass"]),
            "no_lookahead_pass": True,
            "real_betting_disabled_pass": True,
        }
    deployment_races = (
        attach_forecast_closing_return_prices(races, closing_policy_inputs)
        if calibrator_strategy == "odds_path_closing_return"
        else attach_observed_closing_return_prices(races)
        if calibrator_strategy in {
            "odds_path_observed_closing_return",
            V17_STRATEGY_NAME,
            V18_STRATEGY_NAME,
            V19_STRATEGY_NAME,
            V20_STRATEGY_NAME,
            V21_STRATEGY_NAME,
            V35_STRATEGY_NAME,
            "odds_path_hit_shrunk_return",
            "odds_path_prequential_shrinkage_return",
        }
        else races
    )
    deployment_configuration = fit_deployment_configuration(
        deployment_races,
        daily_budget_yen=daily_budget_yen,
        calibrator_strategy=calibrator_strategy,
    )
    deployment_gate = {
        "minimum_evaluation_days": 7,
        "evaluation_days": len(daily_rows),
        "evaluation_races": len(evaluation_races),
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "profitable_folds": profitable_folds,
        "required_profitable_folds": math.ceil(len(folds) * 0.60),
        "days_pass": len(daily_rows) >= 7,
        "roi_pass": return_yen > stake_yen and stake_yen > 0,
        "fold_stability_pass": profitable_folds >= math.ceil(len(folds) * 0.60),
        "prospective_architecture_pass": (
            prospective_architecture_pass
            if prospective_architecture_config is not None
            else True
        ),
    }
    if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES:
        deployment_gate = {
            "primary_bankroll": "chronological_bankroll",
            "minimum_evaluation_days": 7,
            "evaluation_days": chronological_bankroll["race_days"],
            "evaluation_races": len(evaluation_races),
            "roi": chronological_bankroll["roi"],
            "days_pass": chronological_bankroll["race_days"] >= 7,
            "roi_pass": float(chronological_bankroll["roi"] or 0.0) > 1.0,
            "profitable_day_fraction_pass": float(
                chronological_bankroll.get("profitable_day_fraction") or 0.0
            ) >= 0.60,
            "largest_hit_excluded_roi_pass": float(
                chronological_bankroll.get("roi_without_largest_hit") or 0.0
            ) > 1.0,
            "bootstrap_lower_95_pass": float(
                chronological_bankroll.get(
                    "daily_cluster_bootstrap_roi_lower_95"
                ) or 0.0
            ) > 1.0,
            "effective_hit_count_pass": float(
                chronological_bankroll.get("effective_hit_count") or 0.0
            ) >= 20.0,
            "real_betting_disabled_pass": True,
        }
    deployment_gate["pass"] = (
        all(
            bool(value)
            for key, value in deployment_gate.items()
            if key.endswith("_pass")
        )
        if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES
        else all(
            deployment_gate[key]
            for key in (
                "days_pass",
                "roi_pass",
                "fold_stability_pass",
                "prospective_architecture_pass",
            )
        )
    )
    deployment_configuration["walk_forward_gate"] = deployment_gate
    if not deployment_gate["pass"]:
        deployment_configuration["candidate_policy"] = deployment_configuration[
            "selected_policy"
        ]
        deployment_configuration["selected_policy"] = {"name": "no_bet", "no_bet": True}
        deployment_configuration["operational_status"] = "shadow_only_insufficient_evidence"
    else:
        deployment_configuration["operational_status"] = "eligible_for_shadow_promotion"
    if calibrator_strategy in ROBUST_POLICY_STRATEGIES:
        deployment_configuration.update({
            "comparison_role": robust_policy_comparison_role(calibrator_strategy),
            "deployment_mode": "shadow_only",
            "real_betting_enabled": False,
            "daily_stake_limit_fraction": 1.0,
            "primary_promotion_bankroll": "chronological_bankroll",
            "policy_selection": (
                "strict_prior_lexicographic_robust_v17_schedule_quota_v18"
                if calibrator_strategy in SCHEDULE_QUOTA_STRATEGIES
                else "strict_prior_lexicographic_robust_v17"
            ),
        })
        deployment_configuration["operational_status"] = (
            "shadow_only_evidence_passed"
            if deployment_gate["pass"]
            else "shadow_only_insufficient_evidence"
        )
    if calibrator_strategy == V20_STRATEGY_NAME:
        candidate_policy = deployment_configuration.get(
            "candidate_policy",
            deployment_configuration["selected_policy"],
        )
        deployment_configuration.update({
            "comparison_role": V20_COMPARISON_ROLE,
            "deployment_mode": "evaluation_only",
            "real_betting_enabled": False,
            "daily_stake_limit_fraction": 1.0,
            "primary_promotion_bankroll": "chronological_bankroll",
            "probability_metrics_head": "probability_head",
            "chronological_bankroll_head": "purchase_head",
            "policy_selection": (
                "v18_strict_prior_residual_schedule_quota_purchase_head"
            ),
            "candidate_policy": candidate_policy,
            "selected_policy": {"name": "no_bet", "no_bet": True},
            "operational_status": "evaluation_only_challenger",
        })
    if calibrator_strategy in TRIPLE_HEAD_STRATEGIES:
        candidate_policy = deployment_configuration.get(
            "candidate_policy",
            deployment_configuration["selected_policy"],
        )
        deployment_configuration.update({
            "comparison_role": V21_COMPARISON_ROLE,
            "deployment_mode": "evaluation_only",
            "real_betting_enabled": False,
            "daily_stake_limit_fraction": 1.0,
            "primary_promotion_bankroll": "chronological_bankroll",
            "winner_and_logloss_head": "probability_head",
            "trifecta_top5_head": "ranking_head",
            "market_logloss_comparison_head": "probability_head",
            "market_top5_comparison_head": "ranking_head",
            "chronological_bankroll_head": "purchase_head",
            "policy_selection": (
                "v18_strict_prior_residual_schedule_quota_purchase_head"
            ),
            "candidate_policy": candidate_policy,
            "selected_policy": {"name": "no_bet", "no_bet": True},
            "operational_status": "evaluation_only_challenger",
        })
    reliability = bankroll_reliability_metrics(
        daily_rows,
        evaluated_races=len(evaluation_races),
    )
    return {
        "model": odds_path_model_name(calibrator_strategy),
        "comparison_role": (
            robust_policy_comparison_role(calibrator_strategy)
            if calibrator_strategy in CHRONOLOGICAL_BANKROLL_STRATEGIES
            else "real_t5_odds_nested_daily_walk_forward_shadow"
        ),
        "calibrator_strategy": calibrator_strategy,
        "deployment_mode": (
            "evaluation_only"
            if calibrator_strategy in EVALUATION_ONLY_STRATEGIES
            else "shadow_only"
            if calibrator_strategy in ROBUST_POLICY_STRATEGIES
            else "evaluation"
        ),
        "real_betting_enabled": False,
        "validation_design": (
            "Each evaluation day has complete T-5 coverage and is untouched; calibration "
            "and policy selection use only earlier eligible T-5 races"
        ),
        "daily_budget_yen": daily_budget_yen,
        "available_races": len(races),
        "available_days": len(dates),
        "evaluation_candidate_days": len(candidate_dates),
        "evaluation_candidate_dates": candidate_dates,
        "evaluated_races": len(evaluation_races),
        "evaluation_races": len(evaluation_races),
        "evaluation_days": len(daily_rows),
        "probability_metrics": aggregate_metrics,
        "closing_odds_forecast": closing_odds_forecast,
        "market_comparison": market_comparison,
        "ticket_diagnostics": predefined_ticket_diagnostics(
            evaluation_policy_races,
            daily_budget_yen=daily_budget_yen,
        ),
        "edge_diagnostics": summarize_edge_records(edge_diagnostic_records),
        "calibrated_trifecta_log_loss": aggregate_metrics.get(
            "calibrated_trifecta_log_loss"
        ),
        "winner_log_loss": aggregate_metrics.get("calibrated_winner_log_loss"),
        "winner_top1_accuracy": aggregate_metrics.get(
            "calibrated_winner_top1_accuracy"
        ),
        "trifecta_top5_hit_rate": aggregate_metrics.get(
            "calibrated_trifecta_top5_hit_rate"
        ),
        "tickets": sum(int(row["tickets"]) for row in daily_rows),
        "hit_tickets": sum(int(row["hit_tickets"]) for row in daily_rows),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": max_drawdown_yen,
        "profitable_folds": profitable_folds,
        **reliability,
        "folds": folds,
        "daily": daily_rows,
        "chronological_bankroll": chronological_bankroll,
        "flat_policy_walk_forward": {
            "comparison_role": "preselected_on_prior_days_fixed_100_yen_shadow",
            "evaluation_races": len(evaluation_races),
            "evaluation_days": len(flat_daily_rows),
            "tickets": sum(int(row["tickets"]) for row in flat_daily_rows),
            "hit_tickets": sum(int(row["hits"]) for row in flat_daily_rows),
            "stake_yen": flat_stake_yen,
            "return_yen": flat_return_yen,
            "profit_yen": flat_return_yen - flat_stake_yen,
            "roi": flat_return_yen / flat_stake_yen if flat_stake_yen else 0.0,
            "winning_days": sum(int(row["profit_yen"] > 0) for row in flat_daily_rows),
            "daily": flat_daily_rows,
        },
        "empirical_lcb_walk_forward": empirical_lcb_walk_forward,
        "registered_ev_band_walk_forward": summarize_registered_policy_daily(
            registered_daily_rows,
            evaluated_races=registered_evaluated_races,
        ),
        "prospective_normalized_ev_walk_forward": summarize_registered_policy_daily(
            prospective_daily_rows,
            evaluated_races=prospective_evaluated_races,
            policy=PROSPECTIVE_NORMALIZED_EV_POLICY,
            registered_after=PROSPECTIVE_NORMALIZED_EV_REGISTERED_AFTER,
        ),
        "prospective_top5_narrow_ev_walk_forward": summarize_registered_policy_daily(
            prospective_top5_daily_rows,
            evaluated_races=prospective_top5_evaluated_races,
            policy=PROSPECTIVE_TOP5_NARROW_EV_POLICY,
            registered_after=PROSPECTIVE_TOP5_NARROW_EV_REGISTERED_AFTER,
        ),
        "top5_narrow_retrospective_diagnostic": {
            **summarize_registered_policy_daily(
                top5_narrow_retrospective_daily_rows,
                evaluated_races=top5_narrow_retrospective_evaluated_races,
                policy=PROSPECTIVE_TOP5_NARROW_EV_POLICY,
                registered_after=PROSPECTIVE_TOP5_NARROW_EV_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "comparison_role": (
                "strict-prior fold predictions across all evaluation days; "
                "pre-registration days may have informed policy selection"
            ),
            "promotion_evidence": False,
        },
        "v32_dual_head_conformal_retrospective_diagnostic": {
            **summarize_registered_policy_daily(
                v32_retrospective_daily_rows,
                evaluated_races=v32_retrospective_evaluated_races,
                policy=V32_DUAL_HEAD_CONFORMAL_POLICY,
                registered_after=V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "comparison_role": (
                "strict-prior fold probability and ranking heads with date-OOF "
                "conformal lower closing odds across all evaluation days"
            ),
            "promotion_evidence": False,
        },
        "v32_dual_head_conformal_prospective_walk_forward": (
            summarize_registered_policy_daily(
                v32_prospective_daily_rows,
                evaluated_races=v32_prospective_evaluated_races,
                policy=V32_DUAL_HEAD_CONFORMAL_POLICY,
                registered_after=V32_DUAL_HEAD_CONFORMAL_REGISTERED_AFTER,
            )
        ),
        "v33_v25_top1_narrow_retrospective_diagnostic": {
            **summarize_registered_policy_daily(
                v33_retrospective_daily_rows,
                evaluated_races=v33_retrospective_evaluated_races,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "comparison_role": (
                "fixed V25 probability artifact with strict-prior T-5 closing-odds "
                "forecasts across all evaluation days"
            ),
            "promotion_evidence": False,
        },
        "v33_v25_top1_narrow_forecast_only_diagnostic": {
            **summarize_registered_policy_daily(
                v33_forecast_daily_rows,
                evaluated_races=v33_forecast_evaluated_races,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            ),
            "status": "diagnostic_only_not_promotion_evidence",
            "comparison_role": (
                "only folds using strictly-prior final-odds forecasts from real T-5"
            ),
            "promotion_evidence": False,
        },
        "v33_v25_top1_narrow_prospective_walk_forward": (
            summarize_registered_policy_daily(
                v33_prospective_daily_rows,
                evaluated_races=v33_prospective_evaluated_races,
                policy=V33_V25_TOP1_NARROW_POLICY,
                registered_after=V33_V25_TOP1_NARROW_REGISTERED_AFTER,
            )
        ),
        **(
            {
                str(prospective_architecture_config["output_key"]): (
                    prospective_architecture
                )
            }
            if prospective_architecture_config is not None
            else {}
        ),
        "market_offset_registered_policy_walk_forward": (
            market_offset_registered
        ),
        "market_offset_multinomial_kelly_walk_forward": market_offset_multinomial_kelly,
        "conservative_market_offset_kelly_walk_forward": (
            conservative_market_offset_kelly
        ),
        "conformal_lower_market_offset_kelly_diagnostic": (
            conformal_lower_diagnostic
        ),
        "conformal_lower_market_offset_kelly_walk_forward": (
            conformal_lower_prospective
        ),
        "trend_point_market_offset_kelly_diagnostic": trend_point_diagnostic,
        "trend_point_market_offset_kelly_walk_forward": trend_point_prospective,
        "deployment_configuration": deployment_configuration,
        **(
            {
                "dual_head_architecture": {
                    "architecture": "strict_prior_dual_calibrator_heads_v20",
                    "probability_head_role": (
                        "probability_reporting_and_promotion_calibration"
                    ),
                    "purchase_head_role": (
                        "purchase_policy_and_chronological_bankroll"
                    ),
                    "probability_calibrator_strategy": V19_STRATEGY_NAME,
                    "purchase_calibrator_strategy": V18_STRATEGY_NAME,
                    "selection_data": (
                        "strict_prior_training_and_inner_prequential_folds_only"
                    ),
                    "outer_holdout_used": False,
                    "probability_metrics_source": "probability_head",
                    "promotion_calibration_source": "probability_head",
                    "chronological_bankroll_source": "purchase_head",
                }
            }
            if calibrator_strategy == V20_STRATEGY_NAME
            else {}
        ),
        **(
            {
                "triple_head_architecture": {
                    "architecture": "strict_prior_triple_calibrator_heads_v21",
                    "probability_head_role": "winner_and_trifecta_logloss",
                    "ranking_head_role": "trifecta_top5_ranking",
                    "purchase_head_role": (
                        "purchase_policy_and_chronological_bankroll"
                    ),
                    "probability_calibrator_strategy": V19_STRATEGY_NAME,
                    "ranking_calibrator_strategy": V18_STRATEGY_NAME,
                    "purchase_calibrator_strategy": V18_STRATEGY_NAME,
                    "ranking_purchase_share_v18_selection": True,
                    "selection_data": (
                        "strict_prior_training_and_inner_prequential_folds_only"
                    ),
                    "outer_holdout_used": False,
                    "winner_and_logloss_source": "probability_head",
                    "trifecta_top5_source": "ranking_head",
                    "market_logloss_difference_source": "probability_head",
                    "market_top5_difference_source": "ranking_head",
                    "chronological_bankroll_source": "purchase_head",
                }
            }
            if calibrator_strategy in TRIPLE_HEAD_STRATEGIES
            else {}
        ),
        "promotion_gate": promotion_gate,
        "promotion_eligible": (
            False
            if calibrator_strategy in EVALUATION_ONLY_STRATEGIES
            else all(
                value
                for key, value in promotion_gate.items()
                if key.endswith("_pass")
            )
        ),
    }


def _aggregate_fold_probability_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(fold["evaluation_races"]) for fold in folds)
    result: dict[str, Any] = {"evaluated_races": total}
    for source in ("model", "market", "calibrated"):
        for metric in (
            "winner_log_loss",
            "winner_top1_accuracy",
            "trifecta_log_loss",
            "trifecta_top5_hit_rate",
        ):
            key = f"{source}_{metric}"
            result[key] = (
                sum(
                    float(fold["probability_metrics"][key])
                    * int(fold["evaluation_races"])
                    for fold in folds
                )
                / total
                if total
                else None
            )
    return result


def _calibration_gate_pass(metrics: dict[str, Any]) -> bool:
    calibrated = metrics.get("calibrated_trifecta_log_loss")
    model = metrics.get("model_trifecta_log_loss")
    market = metrics.get("market_trifecta_log_loss")
    return bool(
        calibrated is not None
        and model is not None
        and market is not None
        and float(calibrated) <= min(float(model), float(market))
    )


def score_real_odds_races(
    conn,
    *,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str | None = None,
    max_snapshot_age_seconds: float = MARKET_MAX_SNAPSHOT_AGE_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _validate_artifact_before_period(artifact, from_date=from_date)
    model = artifact.get("model")
    hasher = artifact.get("hasher")
    model_kind = str(artifact.get("model_kind") or "").strip().lower()
    supported_classifier = (
        artifact.get("classifier") is not None
        and model_kind in {"linear", "mlp", "lightgbm"}
    )
    supported_listwise = isinstance(
        model,
        (ListwiseLinearModel, StagewiseBlendModel, ConditionalStagewiseModel),
    )
    if (not supported_listwise and not supported_classifier) or hasher is None:
        raise ValueError("model artifact must contain a supported model and hasher")
    race_keys = load_complete_race_ids(conn)
    target_ids = {
        str(race_id)
        for race_id, race_date, _jcd, _rno in race_keys
        if str(race_date) >= from_date
        and (through_date is None or str(race_date) <= through_date)
    }
    checkpoint_diagnostics: dict[str, int] = defaultdict(int)
    prefetched_snapshots = prefetch_trifecta_snapshots(
        conn,
        target_ids=target_ids,
        max_snapshot_age_seconds=max_snapshot_age_seconds,
        checkpoint_diagnostics=checkpoint_diagnostics,
    )
    official_closing_by_race = prefetch_official_closing_odds(
        conn,
        target_ids=target_ids,
    )
    payouts = _load_trifecta_payouts(conn)
    races = []
    skipped_no_odds = skipped_stale_odds = skipped_no_payout = 0
    closing_odds_races = skipped_no_closing_odds = skipped_stale_closing_odds = 0
    momentum_races = 0
    momentum_skipped: dict[str, int] = defaultdict(int)
    for feature_rows, model_probabilities in iter_scored_artifact_feature_rows(
        conn,
        target_ids=target_ids,
        artifact=artifact,
    ):
        meta_rows = [item["meta"] for item in feature_rows]
        race_id = str(meta_rows[0]["race_id"])
        payout = payouts.get(race_id)
        if payout is None:
            skipped_no_payout += 1
            continue
        snapshot = (
            (prefetched_snapshots.get(race_id) or {}).get(
                MODEL_DECISION_LEAD_MINUTES
            )
            if prefetched_snapshots is not None
            else latest_trifecta_odds_before_deadline(
                conn,
                race_id,
                min_combinations=120,
                decision_lead_minutes=MODEL_DECISION_LEAD_MINUTES,
            )
        )
        if snapshot is None or len(snapshot.get("odds") or {}) != 120:
            skipped_no_odds += 1
            continue
        snapshot_age = snapshot_age_seconds(snapshot)
        if (
            snapshot_age is None
            or snapshot_age < 0.0
            or snapshot_age > max_snapshot_age_seconds
        ):
            skipped_stale_odds += 1
            continue
        if prefetched_snapshots is not None:
            momentum_fields, momentum_reason = earlier_market_fields_from_snapshot(
                (prefetched_snapshots.get(race_id) or {}).get(10),
                current_snapshot=snapshot,
                max_snapshot_age_seconds=max_snapshot_age_seconds,
            )
        else:
            momentum_fields, momentum_reason = earlier_market_fields(
                conn,
                race_id,
                current_snapshot=snapshot,
                max_snapshot_age_seconds=max_snapshot_age_seconds,
            )
        if momentum_fields is None:
            momentum_skipped[momentum_reason] += 1
        else:
            momentum_races += 1
        closing_snapshot = (
            (prefetched_snapshots.get(race_id) or {}).get(0)
            if prefetched_snapshots is not None
            else latest_trifecta_odds_before_deadline(
                conn,
                race_id,
                min_combinations=120,
                decision_lead_minutes=0,
            )
        )
        closing_odds = None
        closing_age = None
        closing_source_changed = None
        closing_odds_changed = None
        if closing_snapshot is None or len(closing_snapshot.get("odds") or {}) != 120:
            skipped_no_closing_odds += 1
        else:
            closing_age = snapshot_age_seconds(closing_snapshot)
            if (
                closing_age is None
                or closing_age < 0.0
                or closing_age > max_snapshot_age_seconds
                or not snapshot_captured_after(closing_snapshot, snapshot)
            ):
                skipped_stale_closing_odds += 1
            else:
                closing_odds = {
                    key: float(value)
                    for key, value in closing_snapshot["odds"].items()
                }
                closing_source_changed = (
                    snapshot.get("source_update_time")
                    != closing_snapshot.get("source_update_time")
                )
                closing_odds_changed = any(
                    not math.isclose(
                        float(snapshot["odds"][key]),
                        float(closing_odds[key]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for key in closing_odds
                )
                closing_odds_races += 1
        if prefetched_snapshots is not None:
            odds_checkpoints = dict(
                (prefetched_snapshots.get(race_id) or {}).get(
                    PREFETCH_CHECKPOINTS_KEY
                )
                or {}
            )
        else:
            odds_checkpoints = load_odds_checkpoints(
                conn,
                race_id,
                max_snapshot_age_seconds=max_snapshot_age_seconds,
                diagnostics=checkpoint_diagnostics,
            )
        odds = {key: float(value) for key, value in snapshot["odds"].items()}
        market_probabilities = normalized_market_probabilities(odds)
        if set(model_probabilities) != set(odds) or set(market_probabilities) != set(odds):
            skipped_no_odds += 1
            continue
        races.append(
            {
                "race_id": race_id,
                "race_date": str(meta_rows[0]["race_date"]),
                "jcd": str(meta_rows[0]["jcd"]),
                "rno": int(meta_rows[0]["rno"]),
                "actual_combination": str(payout["combination"]),
                "actual_payout_yen": int(payout["payout_yen"]),
                "lane_context": extract_lane_context(feature_rows),
                "model_probabilities": model_probabilities,
                "market_probabilities": market_probabilities,
                "odds": odds,
                "closing_odds": closing_odds,
                "closing_snapshot_id": (
                    closing_snapshot.get("snapshot_id")
                    if closing_odds is not None
                    else None
                ),
                "closing_captured_at": (
                    closing_snapshot.get("captured_at")
                    if closing_odds is not None
                    else None
                ),
                "closing_source_update_time": (
                    closing_snapshot.get("source_update_time")
                    if closing_odds is not None
                    else None
                ),
                "closing_snapshot_age_seconds": closing_age,
                "closing_source_changed": closing_source_changed,
                "closing_odds_changed": closing_odds_changed,
                "snapshot_id": snapshot.get("snapshot_id"),
                "captured_at": snapshot.get("captured_at"),
                "source_update_time": snapshot.get("source_update_time"),
                "input_snapshot_age_seconds": snapshot_age,
                "odds_deadline_at": snapshot.get("odds_deadline_at"),
                "odds_checkpoints": odds_checkpoints,
                **official_closing_by_race.get(race_id, {}),
                **(
                    odds_path_fields_from_snapshots(
                        prefetched_snapshots.get(race_id) or {},
                        current_snapshot=snapshot,
                    )
                    if prefetched_snapshots is not None
                    else odds_path_fields(
                        conn,
                        race_id,
                        current_snapshot=snapshot,
                        max_snapshot_age_seconds=max_snapshot_age_seconds,
                    )
                ),
                **(momentum_fields or {}),
            }
        )
    return races, {
        "target_complete_races": len(target_ids),
        "eligible_real_odds_races": len(races),
        "skipped_no_real_odds": skipped_no_odds,
        "skipped_stale_real_odds": skipped_stale_odds,
        "closing_odds_races": closing_odds_races,
        "skipped_no_closing_odds": skipped_no_closing_odds,
        "skipped_stale_closing_odds": skipped_stale_closing_odds,
        "momentum_races": momentum_races,
        "skipped_no_earlier_odds": momentum_skipped.get("missing", 0),
        "skipped_stale_earlier_odds": momentum_skipped.get("stale", 0),
        "skipped_invalid_momentum_interval": momentum_skipped.get("interval", 0),
        "skipped_momentum_combination_mismatch": momentum_skipped.get(
            "mismatch", 0
        ),
        "odds_checkpoint_races": sum(
            int(bool(race.get("odds_checkpoints"))) for race in races
        ),
        **{
            f"odds_checkpoint_{offset}_races": sum(
                int(str(offset) in (race.get("odds_checkpoints") or {}))
                for race in races
            )
            for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS
        },
        "official_closing_odds_races": sum(
            int(len(race.get("official_closing_odds") or {}) == 120)
            for race in races
        ),
        "primary_official_closing_odds_races": sum(
            int(
                race.get("official_closing_source_key")
                == OFFICIAL_SOURCE_KEY
            )
            for race in races
        ),
        "fallback_mirror_closing_odds_races": sum(
            int(race.get("official_closing_source_key") == SOURCE_KEY)
            for race in races
        ),
        "odds_checkpoint_metadata_conflicts": int(
            checkpoint_diagnostics.get("metadata_conflict", 0)
        ),
        "skipped_no_payout": skipped_no_payout,
        "odds_path_two_point_races": sum(
            int(int(race.get("odds_path_points") or 0) >= 2) for race in races
        ),
        "odds_path_four_point_races": sum(
            int(int(race.get("odds_path_points") or 0) >= 4) for race in races
        ),
    }


def earlier_market_fields(
    conn,
    race_id: str,
    *,
    current_snapshot: dict[str, Any],
    max_snapshot_age_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    earlier = latest_trifecta_odds_before_deadline(
        conn,
        race_id,
        min_combinations=120,
        decision_lead_minutes=10,
    )
    return earlier_market_fields_from_snapshot(
        earlier,
        current_snapshot=current_snapshot,
        max_snapshot_age_seconds=max_snapshot_age_seconds,
    )


def earlier_market_fields_from_snapshot(
    earlier: dict[str, Any] | None,
    *,
    current_snapshot: dict[str, Any],
    max_snapshot_age_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    if earlier is None or len(earlier.get("odds") or {}) != 120:
        return None, "missing"
    age = snapshot_age_seconds(earlier)
    if age is None or age < 0.0 or age > max_snapshot_age_seconds:
        return None, "stale"
    if not snapshot_captured_after(current_snapshot, earlier):
        return None, "interval"
    try:
        current_at = datetime.fromisoformat(str(current_snapshot["captured_at"]))
        earlier_at = datetime.fromisoformat(str(earlier["captured_at"]))
    except (KeyError, TypeError, ValueError):
        return None, "interval"
    if current_at.tzinfo is None:
        current_at = current_at.replace(tzinfo=timezone.utc)
    if earlier_at.tzinfo is None:
        earlier_at = earlier_at.replace(tzinfo=timezone.utc)
    gap_seconds = (
        current_at - earlier_at.astimezone(current_at.tzinfo)
    ).total_seconds()
    if gap_seconds <= 0.0 or gap_seconds > 900.0:
        return None, "interval"
    earlier_odds = {key: float(value) for key, value in earlier["odds"].items()}
    earlier_market = normalized_market_probabilities(earlier_odds)
    current_odds = current_snapshot.get("odds") or {}
    if set(earlier_market) != set(current_odds):
        return None, "mismatch"
    return {
        "earlier_market_probabilities": earlier_market,
        "earlier_snapshot_id": earlier.get("snapshot_id"),
        "earlier_captured_at": earlier.get("captured_at"),
        "earlier_snapshot_age_seconds": age,
        "momentum_interval_seconds": gap_seconds,
        "momentum_scale": 300.0 / gap_seconds,
    }, "ok"


def odds_path_fields(
    conn,
    race_id: str,
    *,
    current_snapshot: dict[str, Any],
    max_snapshot_age_seconds: float,
) -> dict[str, Any]:
    points = []
    seen = set()
    for lead in (30, 20, 10, 7):
        snapshot = latest_trifecta_odds_before_deadline(
            conn,
            race_id,
            min_combinations=120,
            decision_lead_minutes=lead,
        )
        if snapshot is None or snapshot.get("snapshot_id") in seen:
            continue
        age = snapshot_age_seconds(snapshot)
        if age is None or age < 0.0 or age > max_snapshot_age_seconds:
            continue
        odds = {
            key: float(value)
            for key, value in (snapshot.get("odds") or {}).items()
        }
        if len(odds) != 120:
            continue
        seen.add(snapshot.get("snapshot_id"))
        points.append(
            {
                "minutes_before_decision": float(
                    lead - MODEL_DECISION_LEAD_MINUTES
                ),
                "snapshot_id": snapshot.get("snapshot_id"),
                "captured_at": snapshot.get("captured_at"),
                "market_probabilities": normalized_market_probabilities(odds),
            }
        )
    return odds_path_fields_from_snapshots(
        {},
        current_snapshot=current_snapshot,
        earlier_snapshots=points,
    )


def odds_path_fields_from_snapshots(
    snapshots: dict[int, dict[str, Any]],
    *,
    current_snapshot: dict[str, Any],
    earlier_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    points = list(earlier_snapshots or [])
    if earlier_snapshots is None:
        seen = set()
        for lead in (30, 20, 10, 7):
            snapshot = snapshots.get(lead)
            if snapshot is None or snapshot.get("snapshot_id") in seen:
                continue
            odds = {
                key: float(value)
                for key, value in (snapshot.get("odds") or {}).items()
            }
            if len(odds) != 120:
                continue
            seen.add(snapshot.get("snapshot_id"))
            points.append(
                {
                    "minutes_before_decision": float(
                        lead - MODEL_DECISION_LEAD_MINUTES
                    ),
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "captured_at": snapshot.get("captured_at"),
                    "market_probabilities": normalized_market_probabilities(odds),
                }
            )
    current_odds = {
        key: float(value)
        for key, value in (current_snapshot.get("odds") or {}).items()
    }
    points.append(
        {
            "minutes_before_decision": 0.0,
            "snapshot_id": current_snapshot.get("snapshot_id"),
            "captured_at": current_snapshot.get("captured_at"),
            "market_probabilities": normalized_market_probabilities(current_odds),
        }
    )
    return {"odds_path": points, "odds_path_points": len(points)}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _nonnegative_finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0.0 else None


def _timestamp(value: Any, *, default_tz=timezone.utc) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    return parsed


def _source_update_staleness_seconds(
    source_update_time: Any, *, captured_at: datetime
) -> float | None:
    if source_update_time in (None, ""):
        return None
    source_text = str(source_update_time).strip()
    parsed = _timestamp(
        source_text, default_tz=captured_at.tzinfo or timezone.utc
    )
    if parsed is not None and "T" in source_text:
        return max(
            0.0,
            (
                captured_at
                - parsed.astimezone(captured_at.tzinfo or timezone.utc)
            ).total_seconds(),
        )
    try:
        clock = [int(part) for part in source_text.split(":")]
    except ValueError:
        return None
    if len(clock) not in (2, 3):
        return None
    hour, minute = clock[:2]
    second = clock[2] if len(clock) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    jst = timezone(timedelta(hours=9))
    captured_jst = captured_at.astimezone(jst)
    source_at = captured_jst.replace(
        hour=hour, minute=minute, second=second, microsecond=0
    )
    if source_at - captured_jst > timedelta(minutes=1):
        source_at -= timedelta(days=1)
    return max(0.0, (captured_jst - source_at).total_seconds())


def normalize_odds_checkpoint(
    snapshot: dict[str, Any],
    *,
    target_offset_seconds: int,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    offset = int(target_offset_seconds)
    if offset not in ODDS_CHECKPOINT_OFFSETS_SECONDS:
        return None
    odds = {
        str(key): float(value)
        for key, value in (snapshot.get("odds") or {}).items()
    }
    if len(odds) != 120 or not plausible_trifecta_odds(odds):
        return None
    captured = _timestamp(snapshot.get("captured_at"))
    betting_deadline = _timestamp(
        snapshot.get("betting_deadline_at"),
        default_tz=(captured.tzinfo if captured is not None else timezone.utc),
    )
    if captured is None or betting_deadline is None:
        return None
    captured = captured.astimezone(betting_deadline.tzinfo or timezone.utc)
    captured_age = (betting_deadline - captured).total_seconds()
    if captured_age < float(offset):
        return None

    raw = _json_object(snapshot.get("raw_json"))
    collection = _json_object(raw.get("_collection"))
    collection_offset = collection.get("target_offset_seconds")
    try:
        collection_offset = (
            int(collection_offset) if collection_offset is not None else None
        )
    except (TypeError, ValueError):
        collection_offset = None
    explicit_checkpoint = collection_offset == offset
    measured_age = _nonnegative_finite(collection.get("captured_age_seconds"))
    if explicit_checkpoint and measured_age is not None:
        if (
            measured_age < float(offset)
            or not math.isclose(
                measured_age, captured_age, rel_tol=0.0, abs_tol=1.0
            )
        ):
            if diagnostics is not None:
                diagnostics["metadata_conflict"] = (
                    int(diagnostics.get("metadata_conflict", 0)) + 1
                )
            return None
        captured_age = measured_age
    source_update_time = (
        collection.get("source_update_time")
        if collection.get("source_update_time") not in (None, "")
        else snapshot.get("source_update_time")
    )
    source_staleness = _nonnegative_finite(
        collection.get("source_update_staleness_seconds")
    )
    if source_staleness is None:
        source_staleness = _source_update_staleness_seconds(
            source_update_time,
            captured_at=captured,
        )
    provenance = {
        "mode": (
            "explicit_checkpoint"
            if explicit_checkpoint
            else "timestamp_reconstructed"
        ),
        "observation_label": collection.get("observation_label"),
        "event_id": collection.get("event_id"),
        "collection_target_offset_seconds": collection_offset,
    }
    return {
        "odds": odds,
        "market_probabilities": normalized_market_probabilities(odds),
        "snapshot_id": snapshot.get("snapshot_id"),
        "captured_at": str(snapshot.get("captured_at")),
        "target_offset_seconds": offset,
        "captured_age_seconds": float(captured_age),
        "source_update_time": source_update_time,
        "source_update_staleness_seconds": source_staleness,
        "provenance": provenance,
    }


def odds_checkpoints_from_snapshots(
    snapshots: dict[int, dict[str, Any]],
    *,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS:
        snapshot = snapshots.get(offset)
        if snapshot is None:
            continue
        point = normalize_odds_checkpoint(
            snapshot,
            target_offset_seconds=offset,
            diagnostics=diagnostics,
        )
        if point is not None:
            checkpoints[str(offset)] = point
    return checkpoints


def load_odds_checkpoints(
    conn,
    race_id: str,
    *,
    max_snapshot_age_seconds: float,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS:
        snapshot = latest_trifecta_odds_before_deadline(
            conn,
            race_id,
            min_combinations=120,
            target_offset_seconds=offset,
            max_snapshot_age_seconds=max_snapshot_age_seconds,
        )
        if snapshot is not None and len(snapshot.get("odds") or {}) == 120:
            snapshots[offset] = snapshot
    return odds_checkpoints_from_snapshots(
        snapshots, diagnostics=diagnostics
    )


def available_odds_checkpoints(
    checkpoints: dict[str, dict[str, Any]] | None,
    *,
    as_of_offset_seconds: int,
) -> dict[str, dict[str, Any]]:
    as_of = max(0, int(as_of_offset_seconds))
    available: dict[str, dict[str, Any]] = {}
    source = checkpoints or {}
    for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS:
        if offset < as_of:
            continue
        point = source.get(str(offset))
        if not isinstance(point, dict):
            continue
        if int(point.get("target_offset_seconds", -1)) != offset:
            continue
        copied = dict(point)
        copied["odds"] = dict(point.get("odds") or {})
        copied["market_probabilities"] = dict(
            point.get("market_probabilities") or {}
        )
        copied["provenance"] = dict(point.get("provenance") or {})
        available[str(offset)] = copied
    return available


def prefetch_trifecta_snapshots(
    conn,
    *,
    target_ids: set[str],
    max_snapshot_age_seconds: float,
    checkpoint_diagnostics: dict[str, int] | None = None,
) -> dict[str, dict[Any, Any]] | None:
    if getattr(conn, "dialect", "sqlite") != "postgresql":
        return None
    if not target_ids:
        return {}
    captured_at = stored_jst_timestamp_sql(conn, "os.captured_at")
    start_at = stored_jst_timestamp_sql(conn, "r.deadline_at")
    legacy_leads = {
        0: 0, 300: 5, 420: 7, 600: 10, 1200: 20, 1800: 30
    }
    target_offsets = sorted(
        set(legacy_leads) | set(ODDS_CHECKPOINT_OFFSETS_SECONDS)
    )
    target_values = ", ".join(f"({offset})" for offset in target_offsets)
    snapshots_by_race: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    race_ids = sorted(target_ids)
    for chunk_start in range(0, len(race_ids), 500):
        chunk = race_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _race_id in chunk)
        rows = conn.execute(
            f"""
            WITH targets(target_offset_seconds) AS (
              VALUES {target_values}
            )
            SELECT
              r.race_id,
              targets.target_offset_seconds,
              selected.snapshot_id,
              selected.captured_at,
              selected.source_update_time,
              selected.raw_json,
              selected.odds_deadline_at,
              selected.betting_deadline_at,
              ot.combination,
              ot.odds
            FROM races r
            CROSS JOIN targets
            JOIN LATERAL (
              SELECT
                os.snapshot_id,
                os.captured_at,
                os.source_update_time,
                os.raw_json,
                {start_at} - INTERVAL '5 minutes'
                  - (targets.target_offset_seconds * INTERVAL '1 second')
                  AS odds_deadline_at,
                {start_at} - INTERVAL '5 minutes' AS betting_deadline_at
              FROM odds_snapshots os
              WHERE os.race_id = r.race_id
                AND os.bet_type = 'trifecta'
                AND os.parser_version = ?
                AND {captured_at} <= {start_at} - INTERVAL '5 minutes'
                  - (targets.target_offset_seconds * INTERVAL '1 second')
                AND {captured_at} >= {start_at} - INTERVAL '5 minutes'
                  - (targets.target_offset_seconds * INTERVAL '1 second')
                  - (? * INTERVAL '1 second')
                AND (
                  SELECT COUNT(*)
                  FROM odds_trifecta complete_odds
                  WHERE complete_odds.snapshot_id = os.snapshot_id
                    AND complete_odds.odds IS NOT NULL
                    AND complete_odds.odds > 0
                ) = 120
              ORDER BY {captured_at} DESC, os.snapshot_id DESC
              LIMIT 1
            ) selected ON TRUE
            JOIN odds_trifecta ot ON ot.snapshot_id = selected.snapshot_id
            WHERE r.race_id IN ({placeholders})
              AND ot.odds IS NOT NULL
              AND ot.odds > 0
            ORDER BY r.race_id, targets.target_offset_seconds, ot.combination
            """,
            [
                TRIFECTA_PARSER_VERSION,
                float(max_snapshot_age_seconds),
                *chunk,
            ],
        ).fetchall()
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["race_id"]), int(row["target_offset_seconds"]))
            snapshot = grouped.setdefault(
                key,
                {
                    "snapshot_id": int(row["snapshot_id"]),
                    "captured_at": str(row["captured_at"]),
                    "source_update_time": row["source_update_time"],
                    "raw_json": row["raw_json"],
                    "odds_deadline_at": str(row["odds_deadline_at"]),
                    "betting_deadline_at": str(row["betting_deadline_at"]),
                    "target_offset_seconds": key[1],
                    "decision_lead_minutes": (
                        legacy_leads.get(key[1], key[1] / 60.0)
                    ),
                    "odds": {},
                },
            )
            snapshot["odds"][str(row["combination"])] = float(row["odds"])
        for (race_id, offset), snapshot in grouped.items():
            if plausible_trifecta_odds(snapshot["odds"]):
                snapshot["odds_count"] = 120
                snapshots_by_race[race_id][offset] = snapshot

    result: dict[str, dict[Any, Any]] = defaultdict(dict)
    for race_id, snapshots in snapshots_by_race.items():
        for offset, lead in legacy_leads.items():
            snapshot = snapshots.get(offset)
            if snapshot is not None:
                result[race_id][lead] = snapshot
        result[race_id][PREFETCH_CHECKPOINTS_KEY] = (
            odds_checkpoints_from_snapshots(
                snapshots, diagnostics=checkpoint_diagnostics
            )
        )
    return dict(result)


def _relation_exists(conn, relation_name: str) -> bool:
    if getattr(conn, "dialect", "sqlite") == "postgresql":
        row = conn.execute(
            "SELECT to_regclass(?) AS relation_name",
            (relation_name,),
        ).fetchone()
        return bool(row and row["relation_name"])
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (relation_name,),
    ).fetchone()
    return bool(row)


def prefetch_official_closing_odds(
    conn,
    *,
    target_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not target_ids:
        return {}
    if not _relation_exists(
        conn, "archive_closing_odds_snapshots"
    ) or not _relation_exists(conn, "archive_closing_odds"):
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    race_ids = sorted(target_ids)
    for chunk_start in range(0, len(race_ids), 500):
        chunk = race_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _race_id in chunk)
        rows = conn.execute(
            f"""
            SELECT
              s.race_id,
              s.source_key,
              s.fetched_at,
              s.source_url,
              s.payload_sha256,
              s.parser_version,
              s.odds_count,
              s.verification_status,
              s.raw_json,
              o.combination,
              o.odds
            FROM archive_closing_odds_snapshots s
            JOIN archive_closing_odds o
              ON o.race_id = s.race_id
             AND o.source_key = s.source_key
            WHERE s.source_key IN (?, ?)
              AND s.odds_count = 120
              AND s.race_id IN ({placeholders})
              AND (
                SELECT COUNT(*)
                FROM archive_closing_odds complete_odds
                WHERE complete_odds.race_id = s.race_id
                  AND complete_odds.source_key = s.source_key
                  AND complete_odds.odds IS NOT NULL
                  AND complete_odds.odds > 0
              ) = 120
              AND o.odds IS NOT NULL
              AND o.odds > 0
            ORDER BY s.race_id,
              CASE WHEN s.source_key = ? THEN 0 ELSE 1 END,
              o.combination
            """,
            [*CLOSING_ODDS_SOURCE_PRIORITY, *chunk, OFFICIAL_SOURCE_KEY],
        ).fetchall()
        for row in rows:
            race_id = str(row["race_id"])
            market = grouped.setdefault(
                race_id,
                {
                    "source_key": str(row["source_key"]),
                    "fetched_at": str(row["fetched_at"]),
                    "source_url": str(row["source_url"]),
                    "payload_sha256": str(row["payload_sha256"]),
                    "parser_version": str(row["parser_version"]),
                    "odds_count": int(row["odds_count"]),
                    "verification_status": str(row["verification_status"]),
                    "raw_json": row["raw_json"],
                    "odds": {},
                },
            )
            if market["source_key"] != str(row["source_key"]):
                continue
            market["odds"][str(row["combination"])] = float(row["odds"])

    result: dict[str, dict[str, Any]] = {}
    for race_id, market in grouped.items():
        odds = dict(market["odds"])
        if len(odds) != 120 or not plausible_trifecta_odds(odds):
            continue
        raw = _json_object(market.get("raw_json"))
        result[race_id] = {
            "official_closing_odds": odds,
            "official_closing_market_probabilities": (
                normalized_market_probabilities(odds)
            ),
            "official_closing_source": market["source_key"],
            "official_closing_source_key": market["source_key"],
            "official_closing_provenance": {
                "mode": (
                    "primary_official_historical_closing"
                    if market["source_key"] == OFFICIAL_SOURCE_KEY
                    else "secondary_archive_of_official_closing_display"
                ),
                "source_key": market["source_key"],
                "fetched_at": market["fetched_at"],
                "source_url": market["source_url"],
                "payload_sha256": market["payload_sha256"],
                "parser_version": market["parser_version"],
                "verification_status": market["verification_status"],
                "source_kind": raw.get("source_kind"),
            },
        }
    return result


def snapshot_age_seconds(snapshot: dict[str, Any]) -> float | None:
    try:
        captured = datetime.fromisoformat(str(snapshot["captured_at"]))
        deadline = datetime.fromisoformat(str(snapshot["odds_deadline_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=deadline.tzinfo or timezone.utc)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=captured.tzinfo or timezone.utc)
    return (deadline - captured.astimezone(deadline.tzinfo)).total_seconds()


def snapshot_captured_after(
    later: dict[str, Any], earlier: dict[str, Any]
) -> bool:
    try:
        later_at = datetime.fromisoformat(str(later["captured_at"]))
        earlier_at = datetime.fromisoformat(str(earlier["captured_at"]))
    except (KeyError, TypeError, ValueError):
        return False
    if later_at.tzinfo is None:
        later_at = later_at.replace(tzinfo=timezone.utc)
    if earlier_at.tzinfo is None:
        earlier_at = earlier_at.replace(tzinfo=timezone.utc)
    return later_at > earlier_at.astimezone(later_at.tzinfo)


def _validate_artifact_before_period(artifact: dict[str, Any], *, from_date: str) -> None:
    trained_through = artifact.get("trained_through")
    if not isinstance(trained_through, (list, tuple)) or len(trained_through) < 2:
        raise ValueError("model artifact lacks trained_through leakage metadata")
    trained_date = str(trained_through[1])
    if trained_date >= from_date:
        raise ValueError(
            f"model training overlaps evaluation period: trained_through={trained_date} "
            f"from_date={from_date}"
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_signature_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _archive_closing_data_signature(
    conn,
    *,
    from_date: str,
    through_date: str | None,
) -> dict[str, Any]:
    empty_fingerprint = _stable_signature_fingerprint([])
    empty = {
        "archive_closing_count": 0,
        "archive_closing_odds_count": 0,
        "archive_closing_update_fingerprint": empty_fingerprint,
    }
    if not _relation_exists(
        conn, "archive_closing_odds_snapshots"
    ) or not _relation_exists(conn, "archive_closing_odds"):
        return empty

    filters = ["r.race_date >= ?"]
    params: list[Any] = [*CLOSING_ODDS_SOURCE_PRIORITY, from_date]
    if through_date is not None:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    rows = conn.execute(
        f"""
        SELECT
          s.race_id,
          s.source_key,
          s.fetched_at,
          s.payload_sha256,
          s.parser_version,
          s.odds_count,
          s.verification_status,
          COUNT(o.combination) AS stored_odds_count
        FROM archive_closing_odds_snapshots s
        JOIN races r ON r.race_id = s.race_id
        JOIN archive_closing_odds o
          ON o.race_id = s.race_id
         AND o.source_key = s.source_key
         AND o.odds IS NOT NULL
         AND o.odds > 0
        WHERE s.source_key IN (?, ?)
          AND {" AND ".join(filters)}
          AND r.deadline_at IS NOT NULL
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = r.race_id
              AND rr.rank IS NOT NULL
          ) = 6
        GROUP BY
          s.race_id, s.source_key, s.fetched_at, s.payload_sha256,
          s.parser_version, s.odds_count, s.verification_status
        HAVING s.odds_count = 120 AND COUNT(o.combination) = 120
        ORDER BY s.race_id
        """,
        params,
    ).fetchall()
    fingerprint_rows = [
        {
            "race_id": str(row["race_id"]),
            "source_key": str(row["source_key"]),
            "fetched_at": str(row["fetched_at"]),
            "payload_sha256": str(row["payload_sha256"]),
            "parser_version": str(row["parser_version"]),
            "odds_count": int(row["odds_count"]),
            "verification_status": str(row["verification_status"]),
            "stored_odds_count": int(row["stored_odds_count"]),
        }
        for row in rows
    ]
    return {
        "archive_closing_count": len(fingerprint_rows),
        "archive_closing_odds_count": sum(
            int(row["stored_odds_count"]) for row in fingerprint_rows
        ),
        "archive_closing_update_fingerprint": (
            _stable_signature_fingerprint(fingerprint_rows)
        ),
    }


def odds_data_signature(
    conn,
    *,
    from_date: str,
    through_date: str | None,
    max_snapshot_age_seconds: float = MARKET_MAX_SNAPSHOT_AGE_SECONDS,
) -> dict[str, Any]:
    filters = ["r.race_date >= ?"]
    params: list[Any] = [TRIFECTA_PARSER_VERSION, from_date]
    if through_date is not None:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    rows = conn.execute(
        f"""
        SELECT
          r.race_id,
          r.deadline_at,
          CASE WHEN EXISTS (
            SELECT 1
            FROM payouts p
            WHERE p.race_id = r.race_id
              AND p.bet_type = '3連単'
              AND p.payout_yen IS NOT NULL
          ) THEN 1 ELSE 0 END AS has_payout,
          os.snapshot_id,
          os.captured_at,
          os.raw_json
        FROM races r
        LEFT JOIN odds_snapshots os
          ON os.race_id = r.race_id
         AND os.bet_type = 'trifecta'
         AND os.parser_version = ?
         AND (
           SELECT COUNT(*)
           FROM odds_trifecta ot
           WHERE ot.snapshot_id = os.snapshot_id
             AND ot.odds IS NOT NULL
             AND ot.odds > 0
         ) = 120
        WHERE {" AND ".join(filters)}
          AND r.deadline_at IS NOT NULL
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = r.race_id
              AND rr.rank IS NOT NULL
          ) = 6
        ORDER BY r.race_id, os.captured_at, os.snapshot_id
        """,
        params,
    ).fetchall()

    race_rows: dict[str, dict[str, Any]] = {}
    jst = timezone(timedelta(hours=9))
    for row in rows:
        race_id = str(row["race_id"])
        race = race_rows.setdefault(
            race_id,
            {
                "deadline_at": row["deadline_at"],
                "has_payout": int(row["has_payout"] or 0),
                "snapshots": [],
            },
        )
        if row["snapshot_id"] is None:
            continue
        captured = _timestamp(row["captured_at"], default_tz=jst)
        if captured is None:
            continue
        race["snapshots"].append(
            {
                "snapshot_id": int(row["snapshot_id"]),
                "captured_at": captured,
                "raw_json": row["raw_json"],
            }
        )

    selected_by_offset: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in (0, *ODDS_CHECKPOINT_OFFSETS_SECONDS)
    }
    collection_fingerprint_rows: list[dict[str, Any]] = []
    for race_id, race in sorted(race_rows.items()):
        start_at = _timestamp(race["deadline_at"], default_tz=jst)
        if start_at is None:
            continue
        betting_deadline = start_at - timedelta(minutes=5)
        for offset in selected_by_offset:
            target_at = betting_deadline - timedelta(seconds=offset)
            eligible = []
            for snapshot in race["snapshots"]:
                captured = snapshot["captured_at"].astimezone(
                    target_at.tzinfo or jst
                )
                age = (target_at - captured).total_seconds()
                if 0.0 <= age <= float(max_snapshot_age_seconds):
                    eligible.append(
                        (captured, int(snapshot["snapshot_id"]), snapshot)
                    )
            if not eligible:
                continue
            _captured, _snapshot_id, selected = max(
                eligible, key=lambda item: (item[0], item[1])
            )
            selected_by_offset[offset].append(
                {
                    "race_id": race_id,
                    "snapshot_id": int(selected["snapshot_id"]),
                }
            )
            if offset in ODDS_CHECKPOINT_OFFSETS_SECONDS:
                collection = _json_object(
                    _json_object(selected.get("raw_json")).get("_collection")
                )
                collection_fingerprint_rows.append(
                    {
                        "race_id": race_id,
                        "target_offset_seconds": offset,
                        "snapshot_id": int(selected["snapshot_id"]),
                        "collection": collection,
                    }
                )

    decision_rows = selected_by_offset[300]
    closing_rows = selected_by_offset[0]
    checkpoint_rows = [
        row
        for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS
        for row in selected_by_offset[offset]
    ]
    checkpoint_ids = [int(row["snapshot_id"]) for row in checkpoint_rows]
    signature: dict[str, Any] = {
        "complete_race_count": len(race_rows),
        "payout_race_count": sum(
            int(race["has_payout"]) for race in race_rows.values()
        ),
        "snapshot_count": len(decision_rows),
        "snapshot_id_sum": sum(
            int(row["snapshot_id"]) for row in decision_rows
        ),
        "max_snapshot_id": max(
            (int(row["snapshot_id"]) for row in decision_rows), default=0
        ),
        "closing_snapshot_count": len(closing_rows),
        "closing_snapshot_id_sum": sum(
            int(row["snapshot_id"]) for row in closing_rows
        ),
        "closing_max_snapshot_id": max(
            (int(row["snapshot_id"]) for row in closing_rows), default=0
        ),
        "checkpoint_snapshot_count": len(checkpoint_rows),
        "checkpoint_snapshot_id_sum": sum(checkpoint_ids),
        "checkpoint_max_snapshot_id": max(checkpoint_ids, default=0),
        "checkpoint_collection_count": sum(
            int(bool(row["collection"])) for row in collection_fingerprint_rows
        ),
        "checkpoint_collection_fingerprint": (
            _stable_signature_fingerprint(collection_fingerprint_rows)
        ),
    }
    for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS:
        selected = selected_by_offset[offset]
        ids = [int(row["snapshot_id"]) for row in selected]
        signature[f"checkpoint_{offset}_snapshot_count"] = len(selected)
        signature[f"checkpoint_{offset}_snapshot_id_sum"] = sum(ids)
        signature[f"checkpoint_{offset}_max_snapshot_id"] = max(ids, default=0)
    signature.update(
        _archive_closing_data_signature(
            conn,
            from_date=from_date,
            through_date=through_date,
        )
    )
    return signature


def complete_race_counts_by_date(
    conn,
    *,
    from_date: str,
    through_date: str | None,
) -> dict[str, dict[str, int]]:
    filters = ["r.race_date >= ?"]
    params: list[Any] = [from_date]
    if through_date is not None:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    rows = conn.execute(
        f"""
        SELECT
          r.race_date,
          COUNT(*) AS complete_race_count,
          COALESCE(SUM(
            CASE WHEN EXISTS (
              SELECT 1
              FROM payouts p
              WHERE p.race_id = r.race_id
                AND p.bet_type = '3連単'
                AND p.payout_yen IS NOT NULL
            ) THEN 1 ELSE 0 END
          ), 0) AS payout_race_count
        FROM races r
        WHERE {" AND ".join(filters)}
          AND r.deadline_at IS NOT NULL
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = r.race_id
              AND rr.rank IS NOT NULL
          ) = 6
        GROUP BY r.race_date
        ORDER BY r.race_date
        """,
        params,
    ).fetchall()
    return {
        str(row["race_date"]): {
            "complete_race_count": int(row["complete_race_count"] or 0),
            "payout_race_count": int(row["payout_race_count"] or 0),
        }
        for row in rows
    }


def registered_evaluation_dates(
    clean_dates: Iterable[str],
    *,
    valid_from: str = MARKET_FORMAL_EVALUATION_FROM,
) -> list[str]:
    if len(valid_from) != 10:
        raise ValueError("formal evaluation start must use YYYY-MM-DD")
    return sorted({str(date) for date in clean_dates if str(date) >= valid_from})


def filter_clean_market_days(
    races: list[dict[str, Any]],
    *,
    day_targets: dict[str, dict[str, int]],
    minimum_day_coverage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold = float(minimum_day_coverage)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_day_coverage must be in (0, 1]")
    eligible_by_day: dict[str, int] = defaultdict(int)
    for race in races:
        eligible_by_day[str(race["race_date"])] += 1

    days = []
    clean_dates: set[str] = set()
    for race_date in sorted(day_targets):
        target = int(day_targets[race_date].get("complete_race_count") or 0)
        payouts = int(day_targets[race_date].get("payout_race_count") or 0)
        eligible = int(eligible_by_day.get(race_date) or 0)
        coverage = eligible / target if target else 0.0
        payout_complete = target > 0 and payouts == target
        clean = payout_complete and coverage >= threshold
        if clean:
            clean_dates.add(race_date)
        days.append(
            {
                "race_date": race_date,
                "complete_races": target,
                "payout_races": payouts,
                "eligible_t5_races": eligible,
                "coverage": coverage,
                "payout_complete": payout_complete,
                "clean": clean,
            }
        )
    filtered = [race for race in races if str(race["race_date"]) in clean_dates]
    return filtered, {
        "minimum_day_coverage": threshold,
        "requires_complete_payouts": True,
        "clean_days": len(clean_dates),
        "excluded_days": len(days) - len(clean_dates),
        "clean_dates": sorted(clean_dates),
        "days": days,
    }


def fixed_benchmark_population(
    races: list[dict[str, Any]],
    *,
    day_targets: dict[str, dict[str, int]],
    evaluation_dates: Iterable[str],
    target_days: int = 7,
) -> dict[str, Any]:
    """Describe the provisional/final holdout denominator independently of odds."""
    if target_days < 1:
        raise ValueError("benchmark target_days must be positive")
    dates = sorted({str(value) for value in evaluation_dates})[-int(target_days):]
    date_set = set(dates)
    eligible_by_day: dict[str, int] = defaultdict(int)
    for race in races:
        race_date = str(race.get("race_date") or "")
        if race_date in date_set:
            eligible_by_day[race_date] += 1
    population = sum(
        int(day_targets[date].get("complete_race_count") or 0) for date in dates
    )
    payouts = sum(
        int(day_targets[date].get("payout_race_count") or 0) for date in dates
    )
    odds_eligible = sum(eligible_by_day.values())
    return {
        "benchmark_target_days": int(target_days),
        "benchmark_days": len(dates),
        "benchmark_status": "final" if len(dates) >= target_days else "provisional",
        "benchmark_dates": dates,
        "benchmark_from": dates[0] if dates else None,
        "benchmark_through": dates[-1] if dates else None,
        "benchmark_population_races": population,
        "benchmark_payout_races": payouts,
        "benchmark_odds_eligible_races": odds_eligible,
        "benchmark_missing_odds_races": max(0, population - odds_eligible),
        "benchmark_odds_coverage": odds_eligible / population if population else 0.0,
    }


def scored_cache_contract(
    *,
    model_path: Path,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str | None,
    max_snapshot_age_seconds: float,
    odds_signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": SCORED_CACHE_VERSION,
        "model_sha256": file_sha256(model_path),
        "trained_through": tuple(artifact.get("trained_through") or ()),
        "feature_variant": artifact.get("feature_variant"),
        "drop_feature_groups": artifact_drop_feature_groups(artifact),
        "from_date": from_date,
        "through_date": through_date,
        "max_snapshot_age_seconds": max_snapshot_age_seconds,
        "local_closing_offset_seconds": 0,
        "checkpoint_schema_version": ODDS_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_offsets_seconds": list(
            ODDS_CHECKPOINT_OFFSETS_SECONDS
        ),
        "official_closing_source_priority": list(CLOSING_ODDS_SOURCE_PRIORITY),
        "official_closing_contract_version": 2,
        "odds_data_signature": dict(odds_signature),
    }


def load_scored_cache(
    path: Path,
    *,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(payload, dict) or payload.get("contract") != contract:
        return None
    races = payload.get("races")
    dataset = payload.get("dataset")
    if not isinstance(races, list) or not isinstance(dataset, dict):
        return None
    return races, {str(key): int(value) for key, value in dataset.items()}


def write_scored_cache(
    path: Path,
    *,
    contract: dict[str, Any],
    races: list[dict[str, Any]],
    dataset: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(
            {"contract": contract, "races": races, "dataset": dataset},
            temporary,
            compress=3,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_build_scored_cache(
    path: Path,
    *,
    contract: dict[str, Any],
    builder: Callable[[], tuple[list[dict[str, Any]], dict[str, int]]],
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    cached = load_scored_cache(path, contract=contract)
    if cached is not None:
        races, dataset = cached
        return races, dataset, "disk"
    with scored_cache_build_lock(path):
        cached = load_scored_cache(path, contract=contract)
        if cached is not None:
            races, dataset = cached
            return races, dataset, "disk_after_wait"
        races, dataset = builder()
        write_scored_cache(
            path,
            contract=contract,
            races=races,
            dataset=dataset,
        )
        return races, dataset, "built"


def baseline_scored_cache_path(
    candidate_cache_path: Path,
    *,
    baseline_model_path: Path,
    baseline_model_sha256: str,
) -> Path:
    suffix = ".races.joblib"
    name = candidate_cache_path.name
    prefix = name[:-len(suffix)] if name.endswith(suffix) else candidate_cache_path.stem
    return candidate_cache_path.with_name(
        f"{prefix}.baseline-{baseline_model_path.stem}-"
        f"{baseline_model_sha256[:16]}{suffix}"
    )


def geometric_blend_model_probabilities(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    candidate_weight: float,
) -> dict[str, float]:
    weight = float(candidate_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("candidate_weight must be finite and in [0, 1]")
    if not candidate or set(candidate) != set(baseline):
        raise ValueError("candidate and baseline probability combinations must match")
    ordered_keys = list(candidate)
    candidate_values = {key: float(candidate[key]) for key in ordered_keys}
    baseline_values = {key: float(baseline[key]) for key in ordered_keys}
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (*candidate_values.values(), *baseline_values.values())
    ):
        raise ValueError("model probabilities must be finite and non-negative")
    candidate_total = sum(candidate_values.values())
    baseline_total = sum(baseline_values.values())
    if not math.isclose(candidate_total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("candidate model probabilities must sum to one")
    if not math.isclose(baseline_total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("baseline model probabilities must sum to one")
    if weight == 0.0:
        return dict(baseline)
    if weight == 1.0:
        return dict(candidate)
    blended = {
        key: math.pow(candidate_values[key], weight)
        * math.pow(baseline_values[key], 1.0 - weight)
        for key in ordered_keys
    }
    total = sum(blended.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("geometric probability blend has no finite positive mass")
    return {key: value / total for key, value in blended.items()}


def validate_fixed_model_blend(
    baseline_model: str | None,
    candidate_weight: float | None,
) -> float | None:
    if (baseline_model is None) != (candidate_weight is None):
        raise ValueError(
            "--baseline-model and --candidate-weight must be provided together"
        )
    if baseline_model is None:
        return None
    if not str(baseline_model).strip():
        raise ValueError("--baseline-model must not be empty")
    weight = float(candidate_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("--candidate-weight must be finite and in [0, 1]")
    return weight


def blend_scored_model_probabilities(
    candidate_races: list[dict[str, Any]],
    baseline_races: list[dict[str, Any]],
    *,
    candidate_weight: float,
    candidate_dataset: dict[str, int] | None = None,
    baseline_dataset: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if (
        candidate_dataset is not None
        and baseline_dataset is not None
        and candidate_dataset != baseline_dataset
    ):
        raise ValueError("candidate and baseline scored datasets differ")

    def index_races(
        races: list[dict[str, Any]], *, label: str
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for race in races:
            race_id = str(race.get("race_id") or "")
            if not race_id:
                raise ValueError(f"{label} scored race is missing race_id")
            if race_id in indexed:
                raise ValueError(f"duplicate {label} scored race_id: {race_id}")
            indexed[race_id] = race
        return indexed

    candidates = index_races(candidate_races, label="candidate")
    baselines = index_races(baseline_races, label="baseline")
    if set(candidates) != set(baselines):
        candidate_only = sorted(set(candidates) - set(baselines))
        baseline_only = sorted(set(baselines) - set(candidates))
        raise ValueError(
            "candidate and baseline scored race_id sets differ: "
            f"candidate_only={candidate_only[:5]}, baseline_only={baseline_only[:5]}"
        )

    blended_races = []
    for candidate_race in candidate_races:
        race_id = str(candidate_race["race_id"])
        baseline_race = baselines[race_id]
        candidate_context = {
            key: value
            for key, value in candidate_race.items()
            if key != "model_probabilities"
        }
        baseline_context = {
            key: value
            for key, value in baseline_race.items()
            if key != "model_probabilities"
        }
        if _stable_signature_fingerprint(candidate_context) != (
            _stable_signature_fingerprint(baseline_context)
        ):
            differing_keys = sorted(
                key
                for key in set(candidate_context) | set(baseline_context)
                if _stable_signature_fingerprint(candidate_context.get(key))
                != _stable_signature_fingerprint(baseline_context.get(key))
            )
            raise ValueError(
                f"candidate and baseline scored race data differ for {race_id}: "
                + ", ".join(differing_keys)
            )
        blended_race = dict(candidate_race)
        blended_race["model_probabilities"] = (
            geometric_blend_model_probabilities(
                candidate_race.get("model_probabilities") or {},
                baseline_race.get("model_probabilities") or {},
                candidate_weight=candidate_weight,
            )
        )
        blended_races.append(blended_race)
    return blended_races


@contextmanager
def scored_cache_build_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_v25_probability_artifact(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    temporal = payload.get("temporal_residual_diagnostic")
    if not isinstance(temporal, dict):
        raise ValueError("V25 source is missing temporal_residual_diagnostic")
    candidate = temporal.get("direct_context_market_residual_v25")
    if not isinstance(candidate, dict):
        raise ValueError("V25 source is missing direct_context_market_residual_v25")
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("V25 source is missing its probability artifact")
    coefficients = artifact.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != FEATURE_DIMENSION:
        raise ValueError(
            f"V25 coefficients must contain exactly {FEATURE_DIMENSION} values"
        )
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("V25 coefficients must all be finite")
    audit = {
        "source": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_through": temporal.get("calibration_through"),
        "evaluation_from": temporal.get("evaluation_from"),
        "evaluation_through": temporal.get("evaluation_through"),
        "inner_fit_through": candidate.get("inner_fit_through"),
        "training_races": artifact.get("training_races"),
        "feature_dimension": len(coefficients),
    }
    return artifact, audit

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe market calibration and bankroll shadow evaluation."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/listwise_newton_cg_v1.joblib")
    parser.add_argument("--baseline-model")
    parser.add_argument("--candidate-weight", type=float)
    parser.add_argument(
        "--output",
        default="data/models/listwise_market_calibrated_shadow.json",
    )
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--min-calibration-days", type=int, default=2)
    parser.add_argument(
        "--calibrator-strategy",
        choices=(
            "grid",
            "newton_residual",
            "orthogonal_residual",
            "odds_path_return",
            "odds_path_probability",
            "odds_path_closing_return",
            "odds_path_observed_closing_return",
            V17_STRATEGY_NAME,
            V18_STRATEGY_NAME,
            V19_STRATEGY_NAME,
            V20_STRATEGY_NAME,
            V21_STRATEGY_NAME,
            V35_STRATEGY_NAME,
            "odds_path_hit_shrunk_return",
            "odds_path_prequential_shrinkage_return",
            "odds_path_crossfit_conservative_ev",
            "odds_path_market_offset_crossfit_conservative_ev",
            "odds_path_market_offset_discrete_log_ev_v9",
            "odds_path_market_offset_selection_conformal_discrete_ev_v10",
            "odds_path_role_integrated_multihorizon_v11",
            "odds_path_role_integrated_t300_nonlinear_v12",
            "odds_path_role_integrated_edge_conditional_lcb_v13",
            "odds_path_role_integrated_registered_band_lcb_v14",
            "odds_path_role_integrated_selection_free_envelope_v15",
            "odds_path_role_integrated_fixed_band_passthrough_v16",
        ),
        default="grid",
    )
    parser.add_argument(
        "--v12-closing-fallback-policy",
        choices=("v11", "no_bet"),
        default="v11",
    )
    parser.add_argument("--v25-probability-artifact")
    parser.add_argument("--scored-cache")
    parser.add_argument(
        "--max-snapshot-age-seconds",
        type=float,
        default=MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    )
    parser.add_argument(
        "--closing-odds-min-training-days",
        type=int,
        default=MIN_CLOSING_ODDS_TRAINING_DAYS,
    )
    parser.add_argument(
        "--closing-odds-min-training-races",
        type=int,
        default=MIN_CLOSING_ODDS_TRAINING_RACES,
    )
    parser.add_argument("--minimum-day-coverage", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_weight = validate_fixed_model_blend(
        args.baseline_model,
        args.candidate_weight,
    )
    init_db(args.db)
    model_path = Path(args.model)
    output_path = Path(args.output)
    cache_path = (
        Path(args.scored_cache)
        if args.scored_cache
        else output_path.with_suffix(".races.joblib")
    )
    artifact = joblib.load(model_path)
    baseline_model_path = Path(args.baseline_model) if args.baseline_model else None
    baseline_artifact = (
        joblib.load(baseline_model_path) if baseline_model_path is not None else None
    )
    v25_probability_artifact = None
    v25_artifact_audit = None
    if args.v25_probability_artifact:
        v25_probability_artifact, v25_artifact_audit = (
            load_v25_probability_artifact(Path(args.v25_probability_artifact))
        )
    with connection(args.db) as conn:
        odds_signature = odds_data_signature(
            conn,
            from_date=args.from_date,
            through_date=args.through_date,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
        )
        day_targets = complete_race_counts_by_date(
            conn,
            from_date=args.from_date,
            through_date=args.through_date,
        )
    contract = scored_cache_contract(
        model_path=model_path,
        artifact=artifact,
        from_date=args.from_date,
        through_date=args.through_date,
        max_snapshot_age_seconds=args.max_snapshot_age_seconds,
        odds_signature=odds_signature,
    )
    def score_artifact(
        source_artifact: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        with connection(args.db) as conn:
            return score_real_odds_races(
                conn,
                artifact=source_artifact,
                from_date=args.from_date,
                through_date=args.through_date,
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            )

    races, dataset, cache_source = load_or_build_scored_cache(
        cache_path,
        contract=contract,
        builder=lambda: score_artifact(artifact),
    )
    baseline_contract = None
    baseline_cache_path = None
    baseline_cache_source = None
    if baseline_model_path is not None and baseline_artifact is not None:
        baseline_contract = scored_cache_contract(
            model_path=baseline_model_path,
            artifact=baseline_artifact,
            from_date=args.from_date,
            through_date=args.through_date,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            odds_signature=odds_signature,
        )
        baseline_cache_path = baseline_scored_cache_path(
            cache_path,
            baseline_model_path=baseline_model_path,
            baseline_model_sha256=baseline_contract["model_sha256"],
        )
        baseline_races, baseline_dataset, baseline_cache_source = (
            load_or_build_scored_cache(
                baseline_cache_path,
                contract=baseline_contract,
                builder=lambda: score_artifact(baseline_artifact),
            )
        )
        races = blend_scored_model_probabilities(
            races,
            baseline_races,
            candidate_weight=float(candidate_weight),
            candidate_dataset=dataset,
            baseline_dataset=baseline_dataset,
        )
    clean_races, coverage_gate = filter_clean_market_days(
        races,
        day_targets=day_targets,
        minimum_day_coverage=args.minimum_day_coverage,
    )
    formal_dates = registered_evaluation_dates(coverage_gate["clean_dates"])
    formal_races = [
        race for race in clean_races if str(race["race_date"]) in formal_dates
    ]
    benchmark = fixed_benchmark_population(
        races,
        day_targets=day_targets,
        evaluation_dates=formal_dates,
        target_days=7,
    )
    coverage_gate.update(
        {
            "calibration_eligible_races": len(races),
            "calibration_eligible_days": len(
                {str(race["race_date"]) for race in races}
            ),
            "formal_evaluation_from": MARKET_FORMAL_EVALUATION_FROM,
            "formal_evaluation_dates": formal_dates,
            "pre_registration_clean_dates": sorted(
                set(coverage_gate["clean_dates"]) - set(formal_dates)
            ),
            "formal_evaluation_eligible_races": len(formal_races),
            **benchmark,
        }
    )
    evaluation_input_races = select_calibrator_evaluation_races(
        args.calibrator_strategy,
        races=races,
        clean_races=clean_races,
    )
    result = walk_forward_evaluate(
        evaluation_input_races,
        daily_budget_yen=args.daily_budget_yen,
        min_calibration_days=args.min_calibration_days,
        calibrator_strategy=args.calibrator_strategy,
        evaluation_dates=formal_dates,
        v12_closing_fallback_policy=args.v12_closing_fallback_policy,
        v25_probability_artifact=v25_probability_artifact,
        closing_odds_min_training_days=args.closing_odds_min_training_days,
        closing_odds_min_training_races=args.closing_odds_min_training_races,
    )
    benchmark_evaluated = sum(
        str(race["race_date"]) in set(benchmark["benchmark_dates"])
        for race in formal_races
    )
    benchmark["benchmark_evaluated_races"] = benchmark_evaluated
    benchmark_race_ids = sorted({
        str(race["race_id"])
        for race in formal_races
        if str(race["race_date"]) in set(benchmark["benchmark_dates"])
    })
    if len(benchmark_race_ids) != benchmark_evaluated:
        raise ValueError("benchmark evaluation race IDs are not unique")
    benchmark["benchmark_evaluation_races_sha256"] = (
        _stable_signature_fingerprint(benchmark_race_ids)
    )
    benchmark["benchmark_evaluation_coverage"] = (
        benchmark_evaluated / benchmark["benchmark_population_races"]
        if benchmark["benchmark_population_races"] else 0.0
    )
    benchmark["population_race_selection_rate"] = (
        int(result.get("selected_races") or 0) / benchmark["benchmark_population_races"]
        if benchmark["benchmark_population_races"] else 0.0
    )
    result.update(
        {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_model": str(args.model),
            "source_model_sha256": contract["model_sha256"],
            "candidate_model_sha256": contract["model_sha256"],
            "baseline_model": (
                str(baseline_model_path) if baseline_model_path is not None else None
            ),
            "baseline_model_sha256": (
                baseline_contract["model_sha256"] if baseline_contract else None
            ),
            "candidate_weight": candidate_weight,
            "source_model_trained_through": artifact.get("trained_through"),
            "v25_probability_artifact": v25_artifact_audit,
            "from_date": args.from_date,
            "through_date": args.through_date,
            "dataset": dataset,
            "evaluation_version": MARKET_EVALUATION_VERSION,
            "closing_odds_training_gate": {
                "minimum_days": args.closing_odds_min_training_days,
                "minimum_races": args.closing_odds_min_training_races,
            },
            "odds_data_signature": odds_signature,
            "coverage_gate": coverage_gate,
            "scored_cache": str(cache_path),
            "scored_cache_source": cache_source,
            "baseline_scored_cache": (
                str(baseline_cache_path) if baseline_cache_path is not None else None
            ),
            "baseline_scored_cache_source": baseline_cache_source,
            "calibration_input_scope": (
                "all_eligible_races_including_partial_market_days"
                if args.calibrator_strategy
                == "odds_path_role_integrated_fixed_band_passthrough_v16"
                else (
                    "complete_market_days_only"
                    if evaluation_input_races is clean_races
                    else "all_eligible_races"
                )
            ),
            **benchmark,
        }
    )
    write_json_atomic(output_path, result)
    compact = {key: value for key, value in result.items() if key not in {"folds", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
