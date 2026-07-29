from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from boatrace_ai.adaptive_allocation import allocate_adaptive_day

from .closing_odds import MAX_ODDS, MIN_ODDS
from .closing_odds_quantile import _paired_race, _trend_design_matrix
from .tail_portfolio_diagnostics import diagnose_tail_portfolio


EPSILON = 1e-12
MODEL_NAME = "odds_path_crossfit_conservative_ev_v7"
STRATEGY_NAME = "odds_path_crossfit_conservative_ev"
REGISTERED_AFTER = "2026-07-29"
MIN_CLOSING_TRAINING_DAYS = 7
MIN_CLOSING_TRAINING_RACES = 500
MIN_PROMOTION_DAYS = 30
MIN_PROMOTION_RACES = 1_000
MIN_PROMOTION_TICKETS = 300
MIN_EFFECTIVE_HITS = 20.0
MAX_LARGEST_HIT_RETURN_SHARE = 0.15
SAFE_EV_THRESHOLD = 1.05
MAX_TICKETS_PER_RACE = 2
FRACTIONAL_KELLY = 0.25
MAX_DAILY_EXPOSURE_FRACTION = 0.20
RACE_CAP_FRACTION = 0.03
TICKET_CAP_FRACTION = 0.01
STAKE_GRANULARITY_YEN = 100
PROBABILITY_REGULARIZATION = 1.0
CLOSING_REGULARIZATION = 0.001
CLOSING_QUANTILE = 0.20
LCB_BOOTSTRAP_SAMPLES = 2_000
LCB_SEED = 20260729

PROBABILITY_FEATURE_NAMES = (
    "base_log_probability_offset",
    "market_to_base_log_probability_offset",
    "base_rank",
    "market_rank",
    "recent_log_probability_slope",
    "long_log_probability_slope",
    "slope_acceleration",
    "path_volatility",
)

FIXED_POLICY: dict[str, Any] = {
    "name": "v7_safe_ev105_q20_lcb_kelly025",
    "safe_ev_threshold": SAFE_EV_THRESHOLD,
    "closing_quantile": CLOSING_QUANTILE,
    "max_tickets_per_race": MAX_TICKETS_PER_RACE,
    "fractional_kelly": FRACTIONAL_KELLY,
    "max_daily_exposure_fraction": MAX_DAILY_EXPOSURE_FRACTION,
    "race_cap_fraction": RACE_CAP_FRACTION,
    "ticket_cap_fraction": TICKET_CAP_FRACTION,
    "stake_granularity_yen": STAKE_GRANULARITY_YEN,
    "zero_bet_allowed": True,
}


def _normalized_descending_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get, reverse=True)
    denominator = max(1, len(ordered) - 1)
    return {key: index / denominator for index, key in enumerate(ordered)}


def _trend_features(
    race: dict[str, Any], combination: str
) -> tuple[float, float, float, float]:
    values: list[tuple[float, float]] = []
    for point in race.get("odds_path") or []:
        probability = (point.get("market_probabilities") or {}).get(combination)
        if probability is None or float(probability) <= 0.0:
            continue
        values.append((
            float(point.get("minutes_before_decision") or 0.0),
            math.log(float(probability)),
        ))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    slopes = [
        (values[index][1] - values[index - 1][1])
        / max(EPSILON, values[index - 1][0] - values[index][0])
        for index in range(1, len(values))
    ]
    recent = slopes[-1]
    long = (values[-1][1] - values[0][1]) / max(
        EPSILON, values[0][0] - values[-1][0]
    )
    previous = (
        sum(slopes[:-1]) / len(slopes[:-1]) if len(slopes) > 1 else long
    )
    volatility = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return recent, long, recent - previous, volatility


def _probability_design(
    race: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    base = race.get("base_model_probabilities") or race["model_probabilities"]
    market = race["market_probabilities"]
    combinations = sorted(set(base) & set(market) & set(race.get("odds") or {}))
    if len(combinations) != 120:
        raise ValueError("v7 probability model requires 120 complete combinations")
    base_ranks = _normalized_descending_ranks(base)
    market_ranks = _normalized_descending_ranks(market)
    features = []
    base_logits = []
    for combination in combinations:
        base_probability = max(EPSILON, float(base[combination]))
        market_probability = max(EPSILON, float(market[combination]))
        recent, long, acceleration, volatility = _trend_features(
            race, combination
        )
        log_base = math.log(base_probability)
        features.append((
            float(np.clip(log_base, -20.0, 0.0)),
            float(np.clip(
                math.log(market_probability / base_probability), -6.0, 6.0
            )),
            base_ranks[combination],
            market_ranks[combination],
            float(np.clip(recent * 10.0, -3.0, 3.0)),
            float(np.clip(long * 20.0, -3.0, 3.0)),
            float(np.clip(acceleration * 10.0, -3.0, 3.0)),
            float(np.clip(volatility * 10.0, 0.0, 3.0)),
        ))
        base_logits.append(log_base)
    return (
        combinations,
        np.asarray(base_logits, dtype=np.float64),
        np.asarray(features, dtype=np.float64),
    )


def _probability_objective(
    base_logits: np.ndarray,
    features: np.ndarray,
    actual_indices: np.ndarray,
    coefficients: np.ndarray,
    regularization: float,
) -> float:
    logits = base_logits + features @ coefficients
    maximum = np.max(logits, axis=1, keepdims=True)
    log_partitions = maximum[:, 0] + np.log(
        np.exp(logits - maximum).sum(axis=1)
    )
    actual_logits = logits[np.arange(len(logits)), actual_indices]
    return float(np.mean(log_partitions - actual_logits)) + (
        0.5 * regularization * float(coefficients @ coefficients)
    )


def fit_t5_residual_probability_model(
    races: list[dict[str, Any]],
    *,
    regularization: float = PROBABILITY_REGULARIZATION,
    max_iterations: int = 40,
) -> dict[str, Any]:
    """Fit a pure outcome model with the ten-year base distribution as an offset."""
    if not races:
        raise ValueError("v7 probability model requires races")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("v7 probability regularization must be positive")
    base_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    actual_indices: list[int] = []
    training_dates: set[str] = set()
    for race in races:
        combinations, base_logits, features = _probability_design(race)
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError("v7 actual combination is missing")
        base_rows.append(base_logits)
        feature_rows.append(features)
        actual_indices.append(combinations.index(actual))
        training_dates.add(str(race["race_date"]))
    base_tensor = np.stack(base_rows)
    feature_tensor = np.stack(feature_rows)
    actual_array = np.asarray(actual_indices, dtype=np.int64)
    coefficients = np.zeros(len(PROBABILITY_FEATURE_NAMES), dtype=np.float64)
    converged = False
    objective = math.inf
    for iteration in range(1, max_iterations + 1):
        logits = base_tensor + feature_tensor @ coefficients
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        means = np.einsum(
            "rc,rcf->rf", probabilities, feature_tensor, optimize=True
        )
        actual_features = feature_tensor[
            np.arange(len(feature_tensor)), actual_array
        ]
        gradient = np.mean(means - actual_features, axis=0)
        gradient += regularization * coefficients
        hessian = np.einsum(
            "rc,rcf,rcg->fg",
            probabilities,
            feature_tensor,
            feature_tensor,
            optimize=True,
        ) / len(feature_tensor)
        hessian -= np.einsum(
            "rf,rg->fg", means, means, optimize=True
        ) / len(feature_tensor)
        hessian += regularization * np.eye(len(coefficients))
        objective = _probability_objective(
            base_tensor,
            feature_tensor,
            actual_array,
            coefficients,
            regularization,
        )
        step = np.linalg.solve(
            hessian + 1e-9 * np.eye(len(coefficients)), gradient
        )
        scale = 1.0
        accepted = False
        while scale >= 1e-7:
            candidate = coefficients - scale * step
            candidate_objective = _probability_objective(
                base_tensor,
                feature_tensor,
                actual_array,
                candidate,
                regularization,
            )
            if candidate_objective <= objective + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            converged = True
            break
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        objective = candidate_objective
        if change <= 1e-7:
            converged = True
            break
    return {
        "model_type": MODEL_NAME,
        "architecture": "ten_year_base_offset_plus_pure_t5_residual",
        "teacher": "actual_120_class_trifecta_only",
        "loss": "multinomial_cross_entropy_plus_zero_centered_l2",
        "feature_names": list(PROBABILITY_FEATURE_NAMES),
        "coefficients": coefficients.tolist(),
        "base_offset_prior": 0.0,
        "market_offset_prior": 0.0,
        "regularization": float(regularization),
        "training_races": len(races),
        "training_days": len(training_dates),
        "trained_through_date": max(training_dates),
        "iterations": iteration,
        "converged": converged,
        "objective": float(objective),
        "uses_return_multiplier": False,
        "uses_historical_hit_lift": False,
    }


def attach_t5_residual_probabilities(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if tuple(model.get("feature_names") or ()) != PROBABILITY_FEATURE_NAMES:
        raise ValueError("v7 probability feature contract mismatch")
    result = []
    for race in races:
        combinations, base_logits, features = _probability_design(race)
        logits = base_logits + features @ coefficients
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(probabilities.sum())
        item = dict(race)
        item["base_model_probabilities"] = dict(
            race.get("base_model_probabilities") or race["model_probabilities"]
        )
        item["model_probabilities"] = dict(
            zip(combinations, probabilities.tolist())
        )
        item.pop("historical_return_multipliers", None)
        item["operational_probability_source"] = MODEL_NAME
        result.append(item)
    return result


def _fit_log_ratio_location(
    races: list[dict[str, Any]], *, regularization: float
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    race_count = 0
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        combinations, current, closing = paired
        matrices.append(_trend_design_matrix(race, combinations, current))
        targets.append(np.log(closing / current))
        race_count += 1
    if not targets:
        raise ValueError("v7 closing model requires complete paired snapshots")
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    scale = np.ones(matrix.shape[1], dtype=np.float64)
    mean[1:] = matrix[:, 1:].mean(axis=0)
    scale[1:] = matrix[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    penalty = regularization * np.eye(matrix.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        standardized.T @ standardized / len(target)
        + penalty
        + 1e-10 * np.eye(matrix.shape[1]),
        standardized.T @ target / len(target),
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": coefficients,
        "training_races": race_count,
        "training_tickets": len(target),
    }


def _log_ratio_residuals(
    races: Iterable[dict[str, Any]], model: dict[str, Any]
) -> list[float]:
    residuals: list[float] = []
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        combinations, current, closing = paired
        matrix = _trend_design_matrix(race, combinations, current)
        location = ((matrix - mean) / scale) @ coefficients
        residuals.extend(np.log(closing / current) - location)
    return residuals


def fit_closing_log_ratio_q20_model(
    races: list[dict[str, Any]],
    *,
    regularization: float = CLOSING_REGULARIZATION,
) -> dict[str, Any]:
    """Fit q20 from leave-one-day-out residuals using quantile trend features."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        if _paired_race(race) is not None:
            by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("v7 q20 closing model requires at least two days")
    final = _fit_log_ratio_location(races, regularization=regularization)
    residuals: list[float] = []
    for held_out_date in dates:
        training = [
            race
            for date in dates
            if date != held_out_date
            for race in by_day[date]
        ]
        fold_model = _fit_log_ratio_location(
            training, regularization=regularization
        )
        residuals.extend(
            _log_ratio_residuals(by_day[held_out_date], fold_model)
        )
    q20 = float(np.quantile(
        np.asarray(residuals, dtype=np.float64), CLOSING_QUANTILE
    ))
    return {
        "model_type": "crossfit_log_closing_to_t5_ratio_q20_v1",
        "teacher": "log(closing_odds / t5_odds)",
        "loss": "crossfit_residual_pinball_q20",
        "quantile": CLOSING_QUANTILE,
        "feature_mean": final["feature_mean"].tolist(),
        "feature_scale": final["feature_scale"].tolist(),
        "coefficients": final["coefficients"].tolist(),
        "residual_q20": q20,
        "regularization": float(regularization),
        "training_races": int(final["training_races"]),
        "training_tickets": int(final["training_tickets"]),
        "training_days": len(dates),
        "trained_through_date": dates[-1],
        "crossfit_days": len(dates),
        "crossfit_tickets": len(residuals),
    }


def forecast_closing_q20(
    race: dict[str, Any], model: dict[str, Any]
) -> dict[str, float]:
    current = race.get("odds") or {}
    combinations = sorted(current)
    if len(combinations) != 120:
        return {}
    values = np.asarray(
        [float(current[key]) for key in combinations], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        return {}
    matrix = _trend_design_matrix(race, combinations, values)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    log_ratio = ((matrix - mean) / scale) @ coefficients
    log_ratio += float(model["residual_q20"])
    forecast = np.clip(values * np.exp(log_ratio), MIN_ODDS, MAX_ODDS)
    return dict(zip(combinations, forecast.tolist()))


def _rank_groups(probabilities: dict[str, float]) -> dict[str, str]:
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    result = {}
    for index, combination in enumerate(ordered, start=1):
        result[combination] = (
            "top2"
            if index <= 2
            else "top5"
            if index <= 5
            else "top20"
            if index <= 20
            else "rest"
        )
    return result


def _crossfit_probability_rows(
    races: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 3:
        return []
    result: list[dict[str, Any]] = []
    for index in range(2, len(dates)):
        held_out_date = dates[index]
        training = [
            race
            for date in dates[:index]
            for race in by_day[date]
        ]
        model = fit_t5_residual_probability_model(training)
        result.extend(
            attach_t5_residual_probabilities(by_day[held_out_date], model)
        )
    return result


def fit_probability_lcb(
    races: list[dict[str, Any]],
    *,
    bootstrap_samples: int = LCB_BOOTSTRAP_SAMPLES,
    seed: int = LCB_SEED,
) -> dict[str, Any]:
    """Estimate one-sided probability haircuts by resampling whole race days."""
    groups = ("top2", "top5", "top20", "rest")
    by_day: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {group: [0.0, 0.0] for group in groups}
    )
    for race in races:
        probabilities = race["model_probabilities"]
        rank_groups = _rank_groups(probabilities)
        actual = str(race["actual_combination"])
        date = str(race["race_date"])
        for combination, probability in probabilities.items():
            group = rank_groups[combination]
            by_day[date][group][0] += float(probability)
            by_day[date][group][1] += float(combination == actual)
    dates = sorted(by_day)
    if not dates:
        return {
            "ready": False,
            "factors": {group: 0.0 for group in groups},
            "training_days": 0,
            "training_races": 0,
            "trained_through_date": None,
        }
    rng = np.random.default_rng(seed)
    factors: dict[str, float] = {}
    prior_expected_hits = 20.0
    for group in groups:
        expected = np.asarray([by_day[date][group][0] for date in dates])
        hits = np.asarray([by_day[date][group][1] for date in dates])
        indices = rng.integers(
            0, len(dates), size=(bootstrap_samples, len(dates))
        )
        sampled_expected = expected[indices].sum(axis=1)
        sampled_hits = hits[indices].sum(axis=1)
        ratios = (sampled_hits + prior_expected_hits) / (
            sampled_expected + prior_expected_hits
        )
        factors[group] = float(np.clip(
            np.quantile(ratios, 0.05), 0.0, 1.0
        ))
    return {
        "ready": True,
        "method": (
            "expanding_prequential_probability_ratio_cluster_bootstrap_lcb95"
        ),
        "factors": factors,
        "training_days": len(dates),
        "training_races": len(races),
        "trained_through_date": dates[-1],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def _probability_lcb(
    probabilities: dict[str, float],
    combination: str,
    artifact: dict[str, Any],
) -> float:
    group = _rank_groups(probabilities)[combination]
    factor = float((artifact.get("factors") or {}).get(group, 0.0))
    return float(probabilities[combination]) * factor


def _policy_candidate(
    race: dict[str, Any],
    *,
    combination: str,
    probability: float,
    estimated_odds: float,
    safe_ev: float,
) -> dict[str, Any]:
    return {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "jcd": race["jcd"],
        "rno": int(race["rno"]),
        "combination": combination,
        "probability": probability,
        "estimated_odds": estimated_odds,
        "estimated_ev": safe_ev,
        "safe_ev": safe_ev,
        "actual_combination": str(race["actual_combination"]),
        "actual_payout_yen": int(race["actual_payout_yen"]),
        "hit": combination == str(race["actual_combination"]),
        "odds_source": "strictly_prior_crossfit_closing_q20",
        "real_odds_snapshot_id": race.get("snapshot_id"),
        "real_odds_captured_at": race.get("captured_at"),
        "real_odds_deadline_at": race.get("odds_deadline_at"),
        "real_odds_combinations": len(race.get("odds") or {}),
    }


def simulate_fixed_safe_ev_policy(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
    daily_budget_yen: int,
) -> dict[str, Any]:
    by_day_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_day_races: dict[str, set[str]] = defaultdict(set)
    for race in races:
        date = str(race["race_date"])
        race_id = str(race["race_id"])
        by_day_races[date].add(race_id)
        closing = closing_forecasts.get(race_id) or {}
        if len(closing) != 120 or not probability_lcb.get("ready"):
            continue
        candidates = []
        probabilities = race["model_probabilities"]
        rank_groups = _rank_groups(probabilities)
        factors = probability_lcb.get("factors") or {}
        for combination, odds in closing.items():
            safe_probability = float(probabilities[combination]) * float(
                factors.get(rank_groups[combination], 0.0)
            )
            safe_ev = safe_probability * float(odds)
            if safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidates.append(
                _policy_candidate(
                    race,
                    combination=combination,
                    probability=safe_probability,
                    estimated_odds=float(odds),
                    safe_ev=safe_ev,
                )
            )
        candidates.sort(
            key=lambda row: (row["safe_ev"], row["probability"]), reverse=True
        )
        by_day_candidates[date].extend(
            candidates[:MAX_TICKETS_PER_RACE]
        )

    daily = []
    cumulative_profit = peak_profit = max_drawdown = 0
    for date in sorted(by_day_races):
        row = allocate_adaptive_day(
            date,
            by_day_candidates.get(date, []),
            by_day_races[date],
            daily_budget_yen=daily_budget_yen,
            fractional_kelly=FRACTIONAL_KELLY,
            max_daily_exposure_fraction=MAX_DAILY_EXPOSURE_FRACTION,
            min_daily_exposure_fraction=0.0,
            race_cap_fraction=RACE_CAP_FRACTION,
            ticket_cap_fraction=TICKET_CAP_FRACTION,
            max_daily_tickets=None,
            allocation_mode="kelly",
            stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN,
        )
        cumulative_profit += int(row["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        row["cumulative_profit_yen"] = cumulative_profit
        daily.append(row)
    return _summarize_bankroll(
        daily, evaluated_races=len(races), max_drawdown_yen=max_drawdown
    )


def _summarize_bankroll(
    daily: list[dict[str, Any]],
    *,
    evaluated_races: int,
    max_drawdown_yen: int | None = None,
) -> dict[str, Any]:
    tickets = sum(int(row.get("tickets") or 0) for row in daily)
    hits = sum(int(row.get("hit_tickets") or 0) for row in daily)
    stake = sum(int(row.get("stake_yen") or 0) for row in daily)
    returned = sum(int(row.get("return_yen") or 0) for row in daily)
    selected_races = sum(int(row.get("races_bet") or 0) for row in daily)
    hit_races = sum(int(row.get("hit_races") or 0) for row in daily)
    profitable_days = sum(
        int(int(row.get("profit_yen") or 0) > 0) for row in daily
    )
    largest_hit = max(
        (int(row.get("largest_hit_return_yen") or 0) for row in daily),
        default=0,
    )
    square_sum = sum(
        int(row.get("hit_return_square_sum_yen2") or 0) for row in daily
    )
    tail_rows = [
        ticket
        for row in daily
        for ticket in (row.get("_tail_portfolio_rows") or [])
    ]
    diagnostics = diagnose_tail_portfolio(
        tail_rows, tail_odds=1_000_000_000.0
    )
    ordinary = diagnostics["normal"]
    if max_drawdown_yen is None:
        cumulative = peak = drawdown = 0
        for row in daily:
            cumulative += int(row.get("profit_yen") or 0)
            peak = max(peak, cumulative)
            drawdown = max(drawdown, peak - cumulative)
        max_drawdown_yen = drawdown
    return_without_largest = max(0, returned - largest_hit)
    return {
        "evaluated_races": evaluated_races,
        "evaluation_days": len(daily),
        "tickets": tickets,
        "hit_tickets": hits,
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else 0.0,
        "max_drawdown_yen": int(max_drawdown_yen),
        "selected_races": selected_races,
        "hit_races": hit_races,
        "profitable_days": profitable_days,
        "profitable_day_fraction": (
            profitable_days / len(daily) if daily else 0.0
        ),
        "race_selection_rate": (
            selected_races / evaluated_races if evaluated_races else 0.0
        ),
        "largest_hit_return_share": largest_hit / returned if returned else None,
        "effective_hit_count": (
            returned * returned / square_sum if square_sum else 0.0
        ),
        "profit_without_largest_hit_yen": return_without_largest - stake,
        "roi_without_largest_hit": (
            return_without_largest / stake if stake else 0.0
        ),
        "daily_cluster_bootstrap_roi_lower_95": ordinary.get(
            "daily_cluster_bootstrap_roi_lower_95"
        ),
        "tail_portfolio_diagnostics": diagnostics,
        "daily": daily,
    }


def _cumulative_daily(
    daily: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    cumulative = 0
    for source in sorted(daily, key=lambda row: str(row["race_date"])):
        row = dict(source)
        cumulative += int(row.get("profit_yen") or 0)
        row["cumulative_profit_yen"] = cumulative
        result.append(row)
    return result


def probability_metrics(races: list[dict[str, Any]]) -> dict[str, Any]:
    losses = {"model": 0.0, "base": 0.0, "market": 0.0}
    top5 = {key: 0 for key in losses}
    winner_losses = {key: 0.0 for key in losses}
    winner_hits = {key: 0 for key in losses}
    for race in races:
        actual = str(race["actual_combination"])
        actual_winner = actual.split("-", 1)[0]
        sources = {
            "model": race["model_probabilities"],
            "base": race.get("base_model_probabilities")
            or race["model_probabilities"],
            "market": race["market_probabilities"],
        }
        for key, probabilities in sources.items():
            losses[key] -= math.log(
                max(EPSILON, float(probabilities[actual]))
            )
            top5[key] += int(
                actual
                in sorted(
                    probabilities, key=probabilities.get, reverse=True
                )[:5]
            )
            winner: dict[str, float] = defaultdict(float)
            for combination, probability in probabilities.items():
                winner[combination.split("-", 1)[0]] += float(probability)
            winner_losses[key] -= math.log(
                max(EPSILON, winner[actual_winner])
            )
            winner_hits[key] += int(
                max(winner, key=winner.get) == actual_winner
            )
    count = len(races)
    return {
        "evaluated_races": count,
        "calibrated_trifecta_log_loss": (
            losses["model"] / count if count else None
        ),
        "model_trifecta_log_loss": losses["base"] / count if count else None,
        "market_trifecta_log_loss": (
            losses["market"] / count if count else None
        ),
        "calibrated_trifecta_top5_hit_rate": (
            top5["model"] / count if count else None
        ),
        "model_trifecta_top5_hit_rate": (
            top5["base"] / count if count else None
        ),
        "market_trifecta_top5_hit_rate": (
            top5["market"] / count if count else None
        ),
        "winner_log_loss": winner_losses["model"] / count if count else None,
        "winner_top1_accuracy": winner_hits["model"] / count if count else None,
        "model_winner_log_loss": (
            winner_losses["base"] / count if count else None
        ),
        "model_winner_top1_accuracy": (
            winner_hits["base"] / count if count else None
        ),
        "market_winner_log_loss": (
            winner_losses["market"] / count if count else None
        ),
        "market_winner_top1_accuracy": (
            winner_hits["market"] / count if count else None
        ),
    }


def closing_q20_metrics(
    races: list[dict[str, Any]],
    forecasts: dict[str, dict[str, float]],
) -> dict[str, Any]:
    losses: list[float] = []
    covered: list[bool] = []
    evaluated_races = 0
    for race in races:
        paired = _paired_race(race)
        forecast = forecasts.get(str(race["race_id"])) or {}
        if paired is None or len(forecast) != 120:
            continue
        combinations, current, closing = paired
        predicted = np.asarray([forecast[key] for key in combinations])
        target = np.log(closing / current)
        estimate = np.log(predicted / current)
        residual = target - estimate
        losses.extend(
            np.maximum(
                CLOSING_QUANTILE * residual,
                (CLOSING_QUANTILE - 1.0) * residual,
            )
        )
        covered.extend(closing >= predicted)
        evaluated_races += 1
    return {
        "closing_q20_evaluation_races": evaluated_races,
        "closing_q20_evaluation_tickets": len(losses),
        "closing_q20_pinball_loss": (
            float(np.mean(losses)) if losses else None
        ),
        "closing_q20_lower_coverage": (
            float(np.mean(covered)) if covered else None
        ),
        "closing_q20_target_coverage": 1.0 - CLOSING_QUANTILE,
    }


def _weighted_probability_metrics(
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = (
        "calibrated_trifecta_log_loss",
        "model_trifecta_log_loss",
        "market_trifecta_log_loss",
        "calibrated_trifecta_top5_hit_rate",
        "model_trifecta_top5_hit_rate",
        "market_trifecta_top5_hit_rate",
        "winner_log_loss",
        "winner_top1_accuracy",
        "model_winner_log_loss",
        "model_winner_top1_accuracy",
        "market_winner_log_loss",
        "market_winner_top1_accuracy",
    )
    count = sum(
        int(fold["probability_metrics"]["evaluated_races"])
        for fold in folds
    )
    result: dict[str, Any] = {"evaluated_races": count}
    for key in keys:
        result[key] = (
            sum(
                float(fold["probability_metrics"][key])
                * int(fold["probability_metrics"]["evaluated_races"])
                for fold in folds
            )
            / count
            if count
            else None
        )
    return result


def _aggregate_closing_metrics(
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    tickets = sum(
        int(
            fold["closing_q20_metrics"][
                "closing_q20_evaluation_tickets"
            ]
        )
        for fold in folds
    )
    races = sum(
        int(
            fold["closing_q20_metrics"]["closing_q20_evaluation_races"]
        )
        for fold in folds
    )
    result = {
        "closing_q20_evaluation_races": races,
        "closing_q20_evaluation_tickets": tickets,
        "closing_q20_target_coverage": 1.0 - CLOSING_QUANTILE,
    }
    for key in (
        "closing_q20_pinball_loss",
        "closing_q20_lower_coverage",
    ):
        weighted = [
            (
                float(fold["closing_q20_metrics"][key]),
                int(
                    fold["closing_q20_metrics"][
                        "closing_q20_evaluation_tickets"
                    ]
                ),
            )
            for fold in folds
            if fold["closing_q20_metrics"][key] is not None
        ]
        denominator = sum(weight for _value, weight in weighted)
        result[key] = (
            sum(value * weight for value, weight in weighted) / denominator
            if denominator
            else None
        )
    return result


def _prospective_summary(
    folds: list[dict[str, Any]],
    daily: list[dict[str, Any]],
) -> dict[str, Any]:
    daily = _cumulative_daily(daily)
    probability = _weighted_probability_metrics(folds)
    closing = _aggregate_closing_metrics(folds)
    bankroll = _summarize_bankroll(
        daily, evaluated_races=int(probability["evaluated_races"])
    )
    coverage = closing.get("closing_q20_lower_coverage")
    cluster_lower = bankroll.get(
        "daily_cluster_bootstrap_roi_lower_95"
    )
    calibrated_loss = probability["calibrated_trifecta_log_loss"]
    base_loss = probability["model_trifecta_log_loss"]
    market_loss = probability["market_trifecta_log_loss"]
    gate = {
        "minimum_evaluation_days": MIN_PROMOTION_DAYS,
        "minimum_evaluation_races": MIN_PROMOTION_RACES,
        "minimum_tickets": MIN_PROMOTION_TICKETS,
        "minimum_effective_hit_count": MIN_EFFECTIVE_HITS,
        "minimum_profitable_day_fraction": 0.60,
        "maximum_largest_hit_return_share": (
            MAX_LARGEST_HIT_RETURN_SHARE
        ),
        "sample_days_pass": len(daily) >= MIN_PROMOTION_DAYS,
        "sample_races_pass": (
            int(probability["evaluated_races"]) >= MIN_PROMOTION_RACES
        ),
        "sample_tickets_pass": (
            int(bankroll["tickets"]) >= MIN_PROMOTION_TICKETS
        ),
        "positive_profit_pass": int(bankroll["profit_yen"]) > 0,
        "roi_pass": float(bankroll["roi"]) > 1.0,
        "largest_hit_excluded_roi_pass": (
            float(bankroll["roi_without_largest_hit"]) > 1.0
        ),
        "cluster_bootstrap_roi_pass": (
            cluster_lower is not None and float(cluster_lower) > 1.0
        ),
        "effective_hits_pass": (
            float(bankroll["effective_hit_count"]) >= MIN_EFFECTIVE_HITS
        ),
        "profitable_day_fraction_pass": (
            float(bankroll["profitable_day_fraction"]) >= 0.60
        ),
        "largest_hit_share_pass": (
            bankroll["largest_hit_return_share"] is not None
            and float(bankroll["largest_hit_return_share"])
            <= MAX_LARGEST_HIT_RETURN_SHARE
        ),
        "probability_log_loss_pass": (
            calibrated_loss is not None
            and base_loss is not None
            and market_loss is not None
            and float(calibrated_loss) < float(base_loss)
            and float(calibrated_loss) < float(market_loss)
        ),
        "quantile_coverage_pass": (
            coverage is not None and 0.75 <= float(coverage) <= 0.90
        ),
        "no_lookahead_pass": all(
            bool((fold.get("leakage_guard") or {}).get("pass"))
            for fold in folds
        ),
    }
    checks = [
        value for key, value in gate.items() if key.endswith("_pass")
    ]
    return {
        "status": (
            "evaluating" if folds else "waiting_for_first_unseen_day"
        ),
        "registered_after": REGISTERED_AFTER,
        "comparison_role": (
            "pre_registered_strict_outer_day_v7_shadow"
        ),
        **probability,
        "trifecta_log_loss": probability[
            "calibrated_trifecta_log_loss"
        ],
        "trifecta_top5_hit_rate": probability[
            "calibrated_trifecta_top5_hit_rate"
        ],
        **closing,
        **{
            key: value
            for key, value in bankroll.items()
            if key != "daily"
        },
        "daily": daily,
        "promotion_gate": gate,
        "promotion_eligible": bool(checks) and all(checks),
    }


def _closing_teachers(
    races: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        race
        for race in races
        if _paired_race(race) is not None
        and (
            race.get("closing_source_changed") is True
            or race.get("closing_odds_changed") is True
        )
    ]


def _deployment_configuration(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    probability_model = fit_t5_residual_probability_model(races)
    teachers = _closing_teachers(races)
    teacher_days = sorted(
        {str(race["race_date"]) for race in teachers}
    )
    closing_ready = (
        len(teacher_days) >= MIN_CLOSING_TRAINING_DAYS
        and len(teachers) >= MIN_CLOSING_TRAINING_RACES
    )
    closing_model = (
        fit_closing_log_ratio_q20_model(teachers)
        if closing_ready
        else None
    )
    lcb_rows = _crossfit_probability_rows(races)
    probability_lcb = fit_probability_lcb(lcb_rows)
    return {
        "role": "next_day_refit_not_evaluation",
        "calibrator_strategy": STRATEGY_NAME,
        "trained_dates": dates,
        "trained_through_date": dates[-1],
        "training_races": len(races),
        "operational_model": probability_model,
        "closing_q20_model": closing_model,
        "closing_training_days": len(teacher_days),
        "closing_training_races": len(teachers),
        "closing_ready": closing_ready,
        "probability_lcb": probability_lcb,
        "daily_budget_yen": daily_budget_yen,
        "candidate_policy": dict(FIXED_POLICY),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": (
            "shadow_only_until_v7_promotion_gate"
        ),
    }


def walk_forward_evaluate_v7(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    candidates = (
        dates
        if evaluation_dates is None
        else sorted(
            {
                str(date)
                for date in evaluation_dates
                if str(date) in by_day
            }
        )
    )
    fold_dates = [
        date
        for date in candidates
        if len([prior for prior in dates if prior < date])
        >= min_calibration_days
    ]
    if not fold_dates:
        empty = _prospective_summary([], [])
        return {
            "model": MODEL_NAME,
            "calibrator_strategy": STRATEGY_NAME,
            "status": "waiting_for_clean_evaluation_day",
            "comparison_role": (
                "real_t5_crossfit_q20_fixed_safe_ev_shadow"
            ),
            "registered_after": REGISTERED_AFTER,
            "available_races": len(races),
            "available_days": len(dates),
            "evaluation_days": 0,
            "evaluation_races": 0,
            "evaluated_races": 0,
            "folds": [],
            "daily": [],
            "promotion_gate": empty["promotion_gate"],
            "promotion_eligible": False,
            "prospective_crossfit_conservative_ev_v7_walk_forward": (
                empty
            ),
        }

    all_crossfit_rows = _crossfit_probability_rows(races)
    folds = []
    daily = []
    for evaluation_date in fold_dates:
        calibration_dates = [
            date for date in dates if date < evaluation_date
        ]
        training = [
            race
            for date in calibration_dates
            for race in by_day[date]
        ]
        holdout = by_day[evaluation_date]
        probability_model = fit_t5_residual_probability_model(training)
        transformed_holdout = attach_t5_residual_probabilities(
            holdout, probability_model
        )
        crossfit_rows = [
            race
            for race in all_crossfit_rows
            if str(race["race_date"]) < evaluation_date
        ]
        probability_lcb = fit_probability_lcb(crossfit_rows)
        teachers = _closing_teachers(training)
        teacher_dates = sorted(
            {str(race["race_date"]) for race in teachers}
        )
        closing_ready = (
            len(teacher_dates) >= MIN_CLOSING_TRAINING_DAYS
            and len(teachers) >= MIN_CLOSING_TRAINING_RACES
        )
        closing_model = None
        closing_forecasts: dict[str, dict[str, float]] = {}
        if closing_ready:
            closing_model = fit_closing_log_ratio_q20_model(teachers)
            closing_forecasts = {
                str(race["race_id"]): forecast_closing_q20(
                    race, closing_model
                )
                for race in transformed_holdout
            }
        bankroll = simulate_fixed_safe_ev_policy(
            transformed_holdout,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            daily_budget_yen=daily_budget_yen,
        )
        metrics = probability_metrics(transformed_holdout)
        closing_metrics = closing_q20_metrics(
            transformed_holdout, closing_forecasts
        )
        daily.extend(bankroll["daily"])
        trained_through = probability_model["trained_through_date"]
        closing_through = (
            closing_model["trained_through_date"]
            if closing_model
            else None
        )
        lcb_through = probability_lcb.get("trained_through_date")
        leakage_pass = all(
            value is None or str(value) < evaluation_date
            for value in (
                trained_through,
                closing_through,
                lcb_through,
            )
        )
        folds.append({
            "fold": len(folds) + 1,
            "calibration_dates": calibration_dates,
            "evaluation_date": evaluation_date,
            "calibration_races": len(training),
            "evaluation_races": len(holdout),
            "operational_model": probability_model,
            "probability_lcb": probability_lcb,
            "closing_ready": closing_ready,
            "closing_training_days": len(teacher_dates),
            "closing_training_races": len(teachers),
            "closing_model": closing_model,
            "selected_policy": (
                dict(FIXED_POLICY)
                if closing_ready
                else {"name": "no_bet", "no_bet": True}
            ),
            "probability_metrics": metrics,
            "closing_q20_metrics": closing_metrics,
            "bankroll": {
                key: value
                for key, value in bankroll.items()
                if key != "daily"
            },
            "leakage_guard": {
                "outer_date": evaluation_date,
                "probability_trained_through": trained_through,
                "closing_trained_through": closing_through,
                "lcb_trained_through": lcb_through,
                "pass": leakage_pass,
            },
        })

    daily = _cumulative_daily(daily)
    probability = _weighted_probability_metrics(folds)
    closing = _aggregate_closing_metrics(folds)
    bankroll = _summarize_bankroll(
        daily, evaluated_races=int(probability["evaluated_races"])
    )
    prospective_folds = [
        fold
        for fold in folds
        if str(fold["evaluation_date"]) > REGISTERED_AFTER
    ]
    prospective_dates = {
        str(fold["evaluation_date"]) for fold in prospective_folds
    }
    prospective_daily = [
        row
        for row in daily
        if str(row["race_date"]) in prospective_dates
    ]
    prospective = _prospective_summary(
        prospective_folds, prospective_daily
    )
    deployment = _deployment_configuration(
        races, daily_budget_yen=daily_budget_yen
    )
    deployment["walk_forward_gate"] = dict(
        prospective["promotion_gate"]
    )
    deployment["walk_forward_gate"]["pass"] = bool(
        prospective["promotion_eligible"]
    )
    if (
        prospective["promotion_eligible"]
        and deployment["closing_ready"]
    ):
        deployment["selected_policy"] = dict(FIXED_POLICY)
        deployment["operational_status"] = (
            "eligible_for_shadow_promotion"
        )
    return {
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": (
            "real_t5_crossfit_q20_fixed_safe_ev_shadow"
        ),
        "validation_design": (
            "Each outer day fits probability, q20 closing, and probability "
            "LCB strictly before the outer date; the purchase policy is "
            "preregistered and fixed"
        ),
        "registered_after": REGISTERED_AFTER,
        "daily_budget_yen": daily_budget_yen,
        "fixed_policy": dict(FIXED_POLICY),
        "available_races": len(races),
        "available_days": len(dates),
        "evaluation_days": len(folds),
        "evaluation_races": int(probability["evaluated_races"]),
        "evaluated_races": int(probability["evaluated_races"]),
        "probability_metrics": probability,
        **probability,
        "trifecta_log_loss": probability[
            "calibrated_trifecta_log_loss"
        ],
        "trifecta_top5_hit_rate": probability[
            "calibrated_trifecta_top5_hit_rate"
        ],
        **closing,
        **{
            key: value
            for key, value in bankroll.items()
            if key != "daily"
        },
        "folds": folds,
        "daily": daily,
        "prospective_crossfit_conservative_ev_v7_walk_forward": (
            prospective
        ),
        "promotion_gate": prospective["promotion_gate"],
        "promotion_eligible": prospective["promotion_eligible"],
        "deployment_configuration": deployment,
    }
