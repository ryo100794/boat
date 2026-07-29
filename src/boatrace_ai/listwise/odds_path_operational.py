from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


EPSILON = 1e-12
FEATURE_NAMES = (
    "intercept", "log_model_probability", "log_market_probability",
    "log_model_market_edge", "model_rank", "market_rank",
    "recent_log_probability_slope", "long_log_probability_slope",
    "slope_acceleration", "path_volatility", "historical_hit_lift",
)


def _ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get, reverse=True)
    denominator = max(1, len(ordered) - 1)
    return {key: index / denominator for index, key in enumerate(ordered)}


def _trend_features(race: dict[str, Any], combination: str) -> tuple[float, float, float, float]:
    values = []
    for point in race.get("odds_path") or []:
        probability = (point.get("market_probabilities") or {}).get(combination)
        if probability is not None and float(probability) > 0.0:
            values.append((float(point.get("minutes_before_decision") or 0.0), math.log(float(probability))))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    slopes = [
        (values[index][1] - values[index - 1][1])
        / max(EPSILON, values[index - 1][0] - values[index][0])
        for index in range(1, len(values))
    ]
    recent = slopes[-1]
    long = (values[-1][1] - values[0][1]) / max(EPSILON, values[0][0] - values[-1][0])
    acceleration = recent - (sum(slopes[:-1]) / len(slopes[:-1]) if len(slopes) > 1 else long)
    volatility = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return recent, long, acceleration, volatility


def _performance_key(odds: float, model_rank: float, market_rank: float, recent_slope: float) -> str:
    odds_bucket = 0 if odds < 10 else 1 if odds < 30 else 2 if odds < 100 else 3
    model_bucket = 0 if model_rank <= 4 / 119 else 1 if model_rank <= 19 / 119 else 2
    market_bucket = 0 if market_rank <= 4 / 119 else 1 if market_rank <= 19 / 119 else 2
    trend_bucket = -1 if recent_slope < -0.002 else 1 if recent_slope > 0.002 else 0
    return f"{odds_bucket}:{model_bucket}:{market_bucket}:{trend_bucket}"


def fit_performance_priors(
    races: list[dict[str, Any]],
    *,
    strength: float = 20.0,
    return_hit_prior: float = 0.0,
    min_return_multiplier: float = 0.25,
    max_return_multiplier: float = 2.0,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"tickets": 0.0, "hits": 0.0, "observed_return": 0.0, "baseline_ev": 0.0}
    )
    for race in races:
        model, market = race["model_probabilities"], race["market_probabilities"]
        return_odds = race.get("performance_return_odds") or race["odds"]
        model_ranks, market_ranks = _ranks(model), _ranks(market)
        actual = str(race["actual_combination"])
        payout_odds = float(race["actual_payout_yen"]) / 100.0
        for combination in sorted(set(model) & set(market) & set(race["odds"])):
            recent, _long, _acceleration, _volatility = _trend_features(race, combination)
            odds = float(race["odds"][combination])
            key = _performance_key(odds, model_ranks[combination], market_ranks[combination], recent)
            row = buckets[key]
            hit = float(combination == actual)
            row["tickets"] += 1.0
            row["hits"] += hit
            row["observed_return"] += hit * payout_odds
            row["baseline_ev"] += float(model[combination]) * float(
                return_odds[combination]
            )
    result = {}
    for key, row in buckets.items():
        tickets = row["tickets"]
        baseline_mean = row["baseline_ev"] / max(1.0, tickets)
        raw_return_multiplier = (
            (row["observed_return"] + strength * baseline_mean)
            / max(EPSILON, row["baseline_ev"] + strength * baseline_mean)
        )
        if return_hit_prior > 0.0:
            hit_weight = row["hits"] / (row["hits"] + return_hit_prior)
            return_multiplier = 1.0 + hit_weight * (raw_return_multiplier - 1.0)
        else:
            hit_weight = 1.0
            return_multiplier = raw_return_multiplier
        result[key] = {
            **row,
            "hit_rate": (row["hits"] + strength / 120.0) / (tickets + strength),
            "raw_return_multiplier": float(raw_return_multiplier),
            "return_hit_shrinkage_weight": float(hit_weight),
            "return_multiplier": float(np.clip(
                return_multiplier,
                min_return_multiplier,
                max_return_multiplier,
            )),
        }
    return {
        "strength": float(strength),
        "return_hit_prior": float(return_hit_prior),
        "min_return_multiplier": float(min_return_multiplier),
        "max_return_multiplier": float(max_return_multiplier),
        "buckets": result,
        "training_races": len(races),
    }


def _feature_matrix(race: dict[str, Any], priors: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    model, market = race["model_probabilities"], race["market_probabilities"]
    combinations = sorted(set(model) & set(market) & set(race["odds"]))
    model_ranks, market_ranks = _ranks(model), _ranks(market)
    rows, multipliers = [], []
    buckets = priors.get("buckets") or {}
    for combination in combinations:
        recent, long, acceleration, volatility = _trend_features(race, combination)
        odds = float(race["odds"][combination])
        key = _performance_key(odds, model_ranks[combination], market_ranks[combination], recent)
        prior = buckets.get(key) or {}
        hit_rate = float(prior.get("hit_rate") or 1.0 / 120.0)
        rows.append((
            1.0,
            math.log(max(EPSILON, float(model[combination]))),
            math.log(max(EPSILON, float(market[combination]))),
            math.log(max(EPSILON, float(model[combination]) / float(market[combination]))),
            model_ranks[combination], market_ranks[combination],
            float(np.clip(recent * 10.0, -2.0, 2.0)),
            float(np.clip(long * 20.0, -2.0, 2.0)),
            float(np.clip(acceleration * 10.0, -2.0, 2.0)),
            float(np.clip(volatility * 10.0, 0.0, 2.0)),
            float(np.clip(math.log(max(EPSILON, hit_rate / float(market[combination]))), -3.0, 3.0)),
        ))
        multipliers.append(float(prior.get("return_multiplier") or 1.0))
    return combinations, np.asarray(rows, dtype=np.float64), np.asarray(multipliers, dtype=np.float64)


def _objective(matrix, actual_indices, weights, prior_weights, regularization):
    logits = matrix @ weights
    maximum = np.max(logits, axis=1, keepdims=True)
    log_partitions = maximum[:, 0] + np.log(
        np.exp(logits - maximum).sum(axis=1)
    )
    actual_logits = logits[np.arange(len(matrix)), actual_indices]
    loss = float(np.mean(log_partitions - actual_logits))
    delta = weights - prior_weights
    return loss + 0.5 * regularization * float(delta @ delta)


def fit_odds_path_model(
    races: list[dict[str, Any]],
    *,
    regularization: float = 0.1,
    max_iterations: int = 40,
    use_return_multipliers: bool = True,
    return_price_basis: str = "decision_t5",
    return_hit_prior: float = 0.0,
    min_return_multiplier: float = 0.25,
    max_return_multiplier: float = 2.0,
) -> dict[str, Any]:
    if not races:
        raise ValueError("odds-path model requires races")
    if return_price_basis not in {
        "decision_t5",
        "forecast_closing",
        "observed_closing",
    }:
        raise ValueError("unsupported return_price_basis")
    if return_price_basis != "decision_t5" and not use_return_multipliers:
        raise ValueError("closing-price basis requires return multipliers")
    if return_price_basis != "decision_t5" and any(
        len(race.get("performance_return_odds") or {}) != 120 for race in races
    ):
        raise ValueError("closing-price basis requires 120 return prices per race")
    priors = fit_performance_priors(
        races,
        return_hit_prior=return_hit_prior,
        min_return_multiplier=min_return_multiplier,
        max_return_multiplier=max_return_multiplier,
    )
    prepared = []
    actual_indices = []
    for race in races:
        combinations, matrix, _multipliers = _feature_matrix(race, priors)
        actual = str(race["actual_combination"])
        if len(combinations) == 120 and actual in combinations:
            prepared.append(matrix)
            actual_indices.append(combinations.index(actual))
    if not prepared:
        raise ValueError("odds-path model requires complete 120-combination races")
    prior_weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    prior_weights[2] = 1.0
    feature_tensor = np.stack(prepared)
    actual_index_array = np.asarray(actual_indices, dtype=np.int64)
    weights = prior_weights.copy()
    converged, objective = False, math.inf
    for iteration in range(1, max_iterations + 1):
        gradient = np.zeros_like(weights)
        hessian = np.zeros((len(weights), len(weights)), dtype=np.float64)
        logits = feature_tensor @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        means = np.einsum("rc,rcf->rf", probabilities, feature_tensor, optimize=True)
        actual_features = feature_tensor[
            np.arange(len(feature_tensor)), actual_index_array
        ]
        gradient = np.sum(means - actual_features, axis=0)
        hessian = np.einsum(
            "rc,rcf,rcg->fg",
            probabilities,
            feature_tensor,
            feature_tensor,
            optimize=True,
        ) - np.einsum("rf,rg->fg", means, means, optimize=True)
        count = len(prepared)
        delta = weights - prior_weights
        gradient = gradient / count + regularization * delta
        hessian = hessian / count + regularization * np.eye(len(weights))
        objective = _objective(
            feature_tensor,
            actual_index_array,
            weights,
            prior_weights,
            regularization,
        )
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(weights)), gradient)
        scale, accepted = 1.0, False
        while scale >= 1e-7:
            candidate = weights - scale * step
            candidate_objective = _objective(
                feature_tensor,
                actual_index_array,
                candidate,
                prior_weights,
                regularization,
            )
            if candidate_objective <= objective + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            converged = True
            break
        change = float(np.max(np.abs(candidate - weights)))
        weights, objective = candidate, candidate_objective
        if change <= 1e-7:
            converged = True
            break
    return {
        "model_type": (
            "odds_path_hit_shrunk_closing_return_v5"
            if return_price_basis == "observed_closing" and return_hit_prior > 0.0
            else
            "odds_path_observed_closing_return_v4"
            if return_price_basis == "observed_closing"
            else
            "odds_path_closing_return_v3"
            if return_price_basis == "forecast_closing"
            else
            "odds_path_probability_and_return_v1"
            if use_return_multipliers
            else "odds_path_probability_only_v2"
        ),
        "return_multiplier_mode": (
            "historical_observed_closing_to_payout_bucket"
            if return_price_basis == "observed_closing"
            else
            "historical_forecast_closing_to_payout_bucket"
            if return_price_basis == "forecast_closing"
            else
            "historical_t5_to_payout_bucket"
            if use_return_multipliers
            else "disabled_for_forecast_closing_price"
        ),
        "return_price_basis": return_price_basis,
        "return_hit_prior": float(return_hit_prior),
        "return_multiplier_bounds": [
            float(min_return_multiplier),
            float(max_return_multiplier),
        ],
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(), "regularization": float(regularization),
        "iterations": iteration, "converged": converged, "objective": float(objective),
        "training_races": len(prepared), "performance_priors": priors,
    }


def attach_odds_path_model(races: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    weights = np.asarray(model["weights"], dtype=np.float64)
    result = []
    for race in races:
        combinations, matrix, multipliers = _feature_matrix(race, model["performance_priors"])
        logits = matrix @ weights
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(probabilities.sum())
        item = dict(race)
        item["base_model_probabilities"] = race["model_probabilities"]
        item["model_probabilities"] = dict(zip(combinations, probabilities.tolist()))
        if model.get("return_multiplier_mode") == "disabled_for_forecast_closing_price":
            multipliers = np.ones_like(multipliers)
        item["historical_return_multipliers"] = dict(zip(combinations, multipliers.tolist()))
        item["operational_probability_source"] = model["model_type"]
        result.append(item)
    return result
