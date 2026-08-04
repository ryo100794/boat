from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ticket_utility_ranking_v31 import _lightgbm, ticket_feature_matrix


MODEL_NAME = "nonlinear_market_offset_residual_v38"
EPSILON = 1e-12
SHRINKAGES = (0.0, 0.25, 0.5, 1.0)
TREE_PRESETS = (
    {
        "name": "compact",
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 120,
    },
    {
        "name": "conservative",
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 240,
    },
)
_BOOSTER_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class _TrainingMatrix:
    features: np.ndarray
    labels: np.ndarray
    market_log: np.ndarray
    group_sizes: tuple[int, ...]


def _normalized_market(
    race: Mapping[str, Any], combinations: Sequence[str]
) -> np.ndarray:
    source = race.get("market_probabilities")
    if not isinstance(source, Mapping):
        raise ValueError("V38 race requires market probabilities")
    values = np.asarray(
        [max(EPSILON, float(source[key])) for key in combinations],
        dtype=np.float64,
    )
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V38 market probabilities have invalid mass")
    return values / total


def _training_matrix(races: list[dict[str, Any]]) -> _TrainingMatrix:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    market_logs: list[np.ndarray] = []
    groups: list[int] = []
    for race in races:
        combinations, matrix = ticket_feature_matrix(race)
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError("V38 actual combination is absent from market tickets")
        target = np.zeros(len(combinations), dtype=np.float64)
        target[combinations.index(actual)] = 1.0
        market = _normalized_market(race, combinations)
        matrices.append(matrix)
        labels.append(target)
        market_logs.append(np.log(market))
        groups.append(len(combinations))
    return _TrainingMatrix(
        features=np.vstack(matrices).astype(np.float32, copy=False),
        labels=np.concatenate(labels),
        market_log=np.concatenate(market_logs),
        group_sizes=tuple(groups),
    )


def _group_slices(group_sizes: Sequence[int]) -> tuple[slice, ...]:
    result: list[slice] = []
    start = 0
    for size in group_sizes:
        stop = start + int(size)
        if stop <= start:
            raise ValueError("V38 group sizes must be positive")
        result.append(slice(start, stop))
        start = stop
    return tuple(result)


def _market_offset_objective(
    market_log: np.ndarray, group_sizes: Sequence[int]
):
    slices = _group_slices(group_sizes)

    def objective(predictions: np.ndarray, dataset: Any):
        labels = np.asarray(dataset.get_label(), dtype=np.float64)
        raw = np.asarray(predictions, dtype=np.float64)
        if raw.shape != market_log.shape or labels.shape != market_log.shape:
            raise ValueError("V38 objective input shape mismatch")
        gradient = np.empty_like(raw)
        hessian = np.empty_like(raw)
        for group in slices:
            logits = market_log[group] + raw[group]
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(np.sum(probabilities))
            gradient[group] = probabilities - labels[group]
            # LightGBM accepts the diagonal of the multinomial Hessian.  The
            # omitted within-race covariance is handled by the grouped softmax
            # gradient and conservative leaf/regularization settings.
            hessian[group] = np.maximum(
                probabilities * (1.0 - probabilities), 1e-6
            )
        return gradient, hessian

    return objective


def fit_nonlinear_market_residual(
    races: list[dict[str, Any]],
    *,
    tree_preset: Mapping[str, Any],
    num_threads: int = 4,
    num_boost_round: int = 100,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one V38 race is required")
    prepared = _training_matrix(races)
    lightgbm = _lightgbm()
    dataset = lightgbm.Dataset(
        prepared.features,
        label=prepared.labels,
        free_raw_data=True,
    )
    parameters = {
        "objective": _market_offset_objective(
            prepared.market_log, prepared.group_sizes
        ),
        "metric": "None",
        "learning_rate": 0.035,
        "num_leaves": int(tree_preset["num_leaves"]),
        "max_depth": int(tree_preset["max_depth"]),
        "min_data_in_leaf": int(tree_preset["min_child_samples"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "seed": 20260804,
        "feature_fraction_seed": 20260804,
        "bagging_seed": 20260804,
        "data_random_seed": 20260804,
        "num_threads": max(1, int(num_threads)),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    booster = lightgbm.train(
        parameters,
        dataset,
        num_boost_round=max(1, int(num_boost_round)),
    )
    model_text = booster.model_to_string()
    return {
        "model": MODEL_NAME,
        "role": "market_probability_log_residual_only",
        "objective": "grouped_multinomial_logloss_with_fixed_market_offset",
        "tree_preset": str(tree_preset["name"]),
        "num_leaves": int(tree_preset["num_leaves"]),
        "max_depth": int(tree_preset["max_depth"]),
        "min_child_samples": int(tree_preset["min_child_samples"]),
        "num_boost_round": max(1, int(num_boost_round)),
        "feature_dimension": int(prepared.features.shape[1]),
        "training_races": len(races),
        "training_tickets": int(prepared.features.shape[0]),
        "booster_model": model_text,
        "booster_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
    }


def _booster(artifact: Mapping[str, Any]) -> Any:
    digest = str(artifact["booster_sha256"])
    cached = _BOOSTER_CACHE.get(digest)
    if cached is None:
        model_text = str(artifact["booster_model"])
        if hashlib.sha256(model_text.encode()).hexdigest() != digest:
            raise ValueError("V38 booster digest mismatch")
        cached = _lightgbm().Booster(model_str=model_text)
        _BOOSTER_CACHE[digest] = cached
    return cached


def _softmax_with_market_offset(
    market: np.ndarray, correction: np.ndarray, shrinkage: float
) -> np.ndarray:
    if market.shape != correction.shape:
        raise ValueError("V38 market and correction shapes differ")
    if not 0.0 <= shrinkage <= 1.0 or not math.isfinite(shrinkage):
        raise ValueError("V38 shrinkage must be between zero and one")
    logits = np.log(np.maximum(EPSILON, market)) + shrinkage * correction
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return values


def nonlinear_residual_probabilities(
    race: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    shrinkage: float,
) -> dict[str, float]:
    combinations, matrix = ticket_feature_matrix(race)
    market = _normalized_market(race, combinations)
    correction = np.asarray(
        _booster(artifact).predict(matrix, raw_score=True), dtype=np.float64
    )
    if correction.shape != market.shape or not np.all(np.isfinite(correction)):
        raise ValueError("V38 booster returned invalid residual scores")
    probabilities = _softmax_with_market_offset(market, correction, shrinkage)
    return dict(zip(combinations, (float(value) for value in probabilities)))


def nonlinear_residual_metrics(
    races: list[dict[str, Any]],
    artifact: Mapping[str, Any],
    *,
    shrinkage: float,
) -> dict[str, Any]:
    loss = market_loss = 0.0
    top5_hits = market_top5_hits = 0
    daily_loss: dict[str, list[float]] = {}
    for race in races:
        probabilities = nonlinear_residual_probabilities(
            race, artifact, shrinkage=shrinkage
        )
        market_source = race["market_probabilities"]
        actual = str(race["actual_combination"])
        market_total = sum(max(EPSILON, float(value)) for value in market_source.values())
        model_item = -math.log(max(EPSILON, float(probabilities.get(actual, 0.0))))
        market_item = -math.log(
            max(EPSILON, float(market_source.get(actual, 0.0)) / market_total)
        )
        loss += model_item
        market_loss += market_item
        day = str(race["race_date"])
        daily_loss.setdefault(day, []).append(model_item - market_item)
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5_hits += int(
            actual
            in sorted(
                market_source, key=market_source.get, reverse=True
            )[:5]
        )
    count = len(races)
    daily_deltas = {
        day: float(np.mean(values)) for day, values in sorted(daily_loss.items())
    }
    return {
        "evaluated_races": count,
        "evaluated_days": len(daily_deltas),
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "log_loss_delta_vs_market": (loss - market_loss) / count if count else None,
        "days_better_than_market": sum(value < 0.0 for value in daily_deltas.values()),
        "daily_log_loss_delta_vs_market": daily_deltas,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }


def fit_temporal_nonlinear_market_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    tree_presets: Iterable[Mapping[str, Any]] = TREE_PRESETS,
    shrinkages: Iterable[float] = SHRINKAGES,
    num_threads: int = 4,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 5:
        raise ValueError("at least five V38 calibration days are required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]
    preset_values = tuple(dict(value) for value in tree_presets)
    shrinkage_values = tuple(float(value) for value in shrinkages)
    if not preset_values or not shrinkage_values:
        raise ValueError("V38 requires tree and shrinkage candidates")
    candidates: list[dict[str, Any]] = []
    for preset in preset_values:
        inner_artifact = fit_nonlinear_market_residual(
            inner_fit,
            tree_preset=preset,
            num_threads=num_threads,
        )
        for shrinkage in shrinkage_values:
            metrics = nonlinear_residual_metrics(
                inner_validation,
                inner_artifact,
                shrinkage=shrinkage,
            )
            candidates.append({
                "tree_preset": str(preset["name"]),
                "shrinkage": shrinkage,
                "metrics": metrics,
            })
    selected = min(
        candidates,
        key=lambda row: (
            float(row["metrics"]["trifecta_log_loss"]),
            float(row["shrinkage"]),
            str(row["tree_preset"]),
        ),
    )
    selected_preset = next(
        value
        for value in preset_values
        if str(value["name"]) == str(selected["tree_preset"])
    )
    artifact = fit_nonlinear_market_residual(
        calibration,
        tree_preset=selected_preset,
        num_threads=num_threads,
    )
    selected_shrinkage = float(selected["shrinkage"])
    return {
        "model": MODEL_NAME,
        "validation_design": (
            "Tree complexity and residual shrinkage, including the exact-market "
            "null at shrinkage zero, are selected on the latest inner prior-day "
            "block; the selected tree is refit on all prior days before outer scoring"
        ),
        "market_is_exact_nested_null": True,
        "inner_fit_through": dates[split_index - 1],
        "inner_validation_from": dates[split_index],
        "candidates": candidates,
        "selected_tree_preset": str(selected["tree_preset"]),
        "selected_shrinkage": selected_shrinkage,
        "artifact": artifact,
        "metrics": nonlinear_residual_metrics(
            evaluation, artifact, shrinkage=selected_shrinkage
        ),
    }
