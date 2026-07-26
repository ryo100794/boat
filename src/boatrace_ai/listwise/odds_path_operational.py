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


def fit_performance_priors(races: list[dict[str, Any]], *, strength: float = 20.0) -> dict[str, Any]:
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"tickets": 0.0, "hits": 0.0, "observed_return": 0.0, "baseline_ev": 0.0}
    )
    for race in races:
        model, market = race["model_probabilities"], race["market_probabilities"]
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
            row["baseline_ev"] += float(model[combination]) * odds
    result = {}
    for key, row in buckets.items():
        tickets = row["tickets"]
        baseline_mean = row["baseline_ev"] / max(1.0, tickets)
        result[key] = {
            **row,
            "hit_rate": (row["hits"] + strength / 120.0) / (tickets + strength),
            "return_multiplier": float(np.clip(
                (row["observed_return"] + strength * baseline_mean)
                / max(EPSILON, row["baseline_ev"] + strength * baseline_mean),
                0.25, 2.0,
            )),
        }
    return {"strength": float(strength), "buckets": result, "training_races": len(races)}


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


def _objective(prepared, weights, prior_weights, regularization):
    loss = 0.0
    for matrix, actual_index in prepared:
        logits = matrix @ weights
        maximum = float(np.max(logits))
        loss += maximum + math.log(float(np.exp(logits - maximum).sum())) - float(logits[actual_index])
    delta = weights - prior_weights
    return loss / len(prepared) + 0.5 * regularization * float(delta @ delta)


def fit_odds_path_model(races: list[dict[str, Any]], *, regularization: float = 0.1, max_iterations: int = 40) -> dict[str, Any]:
    if not races:
        raise ValueError("odds-path model requires races")
    priors = fit_performance_priors(races)
    prepared = []
    for race in races:
        combinations, matrix, _multipliers = _feature_matrix(race, priors)
        actual = str(race["actual_combination"])
        if len(combinations) == 120 and actual in combinations:
            prepared.append((matrix, combinations.index(actual)))
    if not prepared:
        raise ValueError("odds-path model requires complete 120-combination races")
    prior_weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    prior_weights[2] = 1.0
    weights = prior_weights.copy()
    converged, objective = False, math.inf
    for iteration in range(1, max_iterations + 1):
        gradient = np.zeros_like(weights)
        hessian = np.zeros((len(weights), len(weights)), dtype=np.float64)
        for matrix, actual_index in prepared:
            logits = matrix @ weights
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            mean = probabilities @ matrix
            gradient += mean - matrix[actual_index]
            hessian += (matrix.T * probabilities) @ matrix - np.outer(mean, mean)
        count = len(prepared)
        delta = weights - prior_weights
        gradient = gradient / count + regularization * delta
        hessian = hessian / count + regularization * np.eye(len(weights))
        objective = _objective(prepared, weights, prior_weights, regularization)
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(weights)), gradient)
        scale, accepted = 1.0, False
        while scale >= 1e-7:
            candidate = weights - scale * step
            candidate_objective = _objective(prepared, candidate, prior_weights, regularization)
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
        "model_type": "odds_path_probability_and_return_v1",
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
        item["historical_return_multipliers"] = dict(zip(combinations, multipliers.tolist()))
        item["operational_probability_source"] = model["model_type"]
        result.append(item)
    return result
