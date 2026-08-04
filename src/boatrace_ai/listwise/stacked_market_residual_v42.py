from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .direct_context_market_residual_v25 import (
    direct_context_probabilities,
    fit_temporal_direct_context_residual,
)
from .nonlinear_context_search_v41 import (
    fit_temporal_nonlinear_context_search,
)
from .nonlinear_market_residual_v38 import (
    EPSILON,
    nonlinear_residual_probabilities,
)


MODEL_NAME = "stacked_market_residual_v42"
STACK_CANDIDATES = (
    {"name": "market", "market": 1.0, "linear": 0.0, "nonlinear": 0.0},
    {"name": "market75_linear25", "market": 0.75, "linear": 0.25, "nonlinear": 0.0},
    {"name": "market50_linear50", "market": 0.5, "linear": 0.5, "nonlinear": 0.0},
    {"name": "market25_linear75", "market": 0.25, "linear": 0.75, "nonlinear": 0.0},
    {"name": "linear", "market": 0.0, "linear": 1.0, "nonlinear": 0.0},
    {"name": "market75_nonlinear25", "market": 0.75, "linear": 0.0, "nonlinear": 0.25},
    {"name": "market50_nonlinear50", "market": 0.5, "linear": 0.0, "nonlinear": 0.5},
    {"name": "market25_nonlinear75", "market": 0.25, "linear": 0.0, "nonlinear": 0.75},
    {"name": "nonlinear", "market": 0.0, "linear": 0.0, "nonlinear": 1.0},
    {"name": "linear50_nonlinear50", "market": 0.0, "linear": 0.5, "nonlinear": 0.5},
    {"name": "market50_linear25_nonlinear25", "market": 0.5, "linear": 0.25, "nonlinear": 0.25},
)


def _market_probabilities(race: Mapping[str, Any]) -> dict[str, float]:
    source = race.get("market_probabilities")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("V42 race requires market probabilities")
    values = {str(key): max(EPSILON, float(value)) for key, value in source.items()}
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V42 market probabilities have invalid mass")
    return {key: value / total for key, value in values.items()}


def _blend(
    sources: Sequence[Mapping[str, float]], weights: Sequence[float]
) -> dict[str, float]:
    if len(sources) != len(weights) or not sources:
        raise ValueError("V42 source and weight dimensions differ")
    combinations = sorted(str(key) for key in sources[0])
    if any(set(source) != set(combinations) for source in sources):
        raise ValueError("V42 probability sources have different tickets")
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if (
        np.any(numeric_weights < 0.0)
        or not np.all(np.isfinite(numeric_weights))
        or not math.isclose(float(np.sum(numeric_weights)), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("V42 weights must be a probability simplex")
    logits = np.zeros(len(combinations), dtype=np.float64)
    for source, weight in zip(sources, numeric_weights, strict=True):
        if weight:
            logits += weight * np.log(np.asarray([
                max(EPSILON, float(source[key])) for key in combinations
            ]))
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return dict(zip(combinations, (float(value) for value in values)))


def stacked_probabilities(
    race: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, float]:
    weights = artifact.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("V42 artifact weights are missing")
    market = _market_probabilities(race)
    linear = direct_context_probabilities(
        dict(race), artifact["linear_artifact"]
    )
    nonlinear = nonlinear_residual_probabilities(
        race,
        artifact["nonlinear_artifact"],
        shrinkage=float(artifact["nonlinear_shrinkage"]),
    )
    return _blend(
        (market, linear, nonlinear),
        (
            float(weights["market"]),
            float(weights["linear"]),
            float(weights["nonlinear"]),
        ),
    )


def stacked_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    loss = market_loss = 0.0
    top5_hits = market_top5_hits = 0
    daily: dict[str, list[float]] = {}
    for race in races:
        probabilities = stacked_probabilities(race, artifact)
        market = _market_probabilities(race)
        actual = str(race["actual_combination"])
        model_item = -math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_item = -math.log(max(EPSILON, market.get(actual, 0.0)))
        loss += model_item
        market_loss += market_item
        daily.setdefault(str(race["race_date"]), []).append(
            model_item - market_item
        )
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5_hits += int(
            actual in sorted(market, key=market.get, reverse=True)[:5]
        )
    count = len(races)
    daily_delta = {
        day: float(np.mean(values)) for day, values in sorted(daily.items())
    }
    return {
        "evaluated_races": count,
        "evaluated_days": len(daily_delta),
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "log_loss_delta_vs_market": (
            (loss - market_loss) / count if count else None
        ),
        "days_better_than_market": sum(value < 0.0 for value in daily_delta.values()),
        "daily_log_loss_delta_vs_market": daily_delta,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }


def _fit_components(
    races: list[dict[str, Any]], *, num_threads: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    linear = fit_temporal_direct_context_residual(races, [])
    nonlinear = fit_temporal_nonlinear_context_search(
        races, [], num_threads=num_threads
    )
    return linear, nonlinear


def _artifact(
    linear: Mapping[str, Any],
    nonlinear: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = {
        "model": MODEL_NAME,
        "role": "market_linear_nonlinear_log_probability_stack",
        "selected_stack": str(candidate["name"]),
        "weights": {
            key: float(candidate[key]) for key in ("market", "linear", "nonlinear")
        },
        "linear_artifact": linear["artifact"],
        "nonlinear_artifact": nonlinear["artifact"],
        "nonlinear_shrinkage": float(nonlinear["selected_shrinkage"]),
        "linear_regularization": float(linear["artifact"]["regularization"]),
        "nonlinear_context_variant": nonlinear["selected_context_variant"],
        "nonlinear_tree_preset": nonlinear["selected_tree_preset"],
    }
    encoded = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return artifact


def fit_temporal_stacked_market_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    stack_candidates: Sequence[Mapping[str, Any]] = STACK_CANDIDATES,
    num_threads: int = 4,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 10:
        raise ValueError("at least ten V42 calibration days are required")
    split_index = max(5, min(len(dates) - 1, int(len(dates) * 0.8)))
    base_dates = set(dates[:split_index])
    stack_dates = set(dates[split_index:])
    base_training = [
        race for race in calibration if str(race["race_date"]) in base_dates
    ]
    stack_validation = [
        race for race in calibration if str(race["race_date"]) in stack_dates
    ]
    inner_linear, inner_nonlinear = _fit_components(
        base_training, num_threads=num_threads
    )
    candidates = []
    for source in stack_candidates:
        candidate = dict(source)
        artifact = _artifact(inner_linear, inner_nonlinear, candidate)
        candidates.append({
            "name": str(candidate["name"]),
            "weights": dict(artifact["weights"]),
            "metrics": stacked_metrics(stack_validation, artifact),
        })
    if not candidates:
        raise ValueError("V42 requires stack candidates")
    selected = min(
        candidates,
        key=lambda row: (
            float(row["metrics"]["trifecta_log_loss"]),
            float(row["weights"]["linear"])
            + float(row["weights"]["nonlinear"]),
            str(row["name"]),
        ),
    )
    selected_source = next(
        value for value in stack_candidates
        if str(value["name"]) == str(selected["name"])
    )
    linear, nonlinear = _fit_components(calibration, num_threads=num_threads)
    artifact = _artifact(linear, nonlinear, selected_source)
    return {
        "model": MODEL_NAME,
        "validation_design": (
            "Base hyperparameters use earlier nested prior-day validation; stack "
            "weights use a separate latest prior-day block; components are refit "
            "on all calibration days before untouched outer scoring"
        ),
        "outer_period_used_for_selection": False,
        "market_is_exact_nested_null": True,
        "base_training_through": dates[split_index - 1],
        "stack_validation_from": dates[split_index],
        "stack_candidates": candidates,
        "selected_stack": selected["name"],
        "selected_weights": selected["weights"],
        "component_selection": {
            "linear_regularization": linear["artifact"]["regularization"],
            "nonlinear_context_variant": nonlinear["selected_context_variant"],
            "nonlinear_tree_preset": nonlinear["selected_tree_preset"],
            "nonlinear_shrinkage": nonlinear["selected_shrinkage"],
        },
        "artifact": artifact,
        "metrics": stacked_metrics(evaluation, artifact),
    }
