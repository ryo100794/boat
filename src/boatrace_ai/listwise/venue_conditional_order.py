from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction import FeatureHasher

from ..bankroll_backtest import _load_trifecta_payouts
from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import load_hashed_dataset, promote_legacy_hashed_dataset
from .cluster_bootstrap import paired_cluster_mean_bootstrap
from .conditional_order import (
    ConditionalOrderModel,
    _date_boundary,
    _masked_softmax,
    _pack,
    _public_metrics,
    _score_dataset,
    _unpack,
    _validate_scores_orders,
    bankroll_promotion_gate,
    conditional_probabilities,
    evaluate_probabilities,
    fit_conditional_order,
    identity_model,
)
from .direct_bankroll import (
    bootstrap_daily_bankroll,
    simulate_conditional_payout_walk_forward,
)
from .feature_search import load_variant_dataset_with_cache
from .model import ListwiseLinearModel, stable_softmax
from .newton_refine import dump_joblib_atomic
from .paired_bootstrap import paired_mean_bootstrap
from .stagewise_mlp import COMBINATION_LANES, actual_combination_indices


MODEL_NAME = "pastlog_venue_conditional_order"
EPSILON = 1e-15
VENUE_COUNT = 24
DEFAULT_VENUE_REGULARIZATIONS = (0.0001, 0.001, 0.01, 0.1)
LEGACY_PAYOUT_SCHEMA = "conditional_payout_additive_v1"


@dataclass(frozen=True)
class VenueConditionalOrderModel:
    global_model: ConditionalOrderModel
    venue_second_bias: np.ndarray
    venue_third_first_bias: np.ndarray
    venue_third_second_bias: np.ndarray
    venue_regularization: float


def venue_indices(race_keys: list[tuple[str, str, str, int]]) -> np.ndarray:
    values = np.asarray([int(row[2]) - 1 for row in race_keys], dtype=np.int64)
    if np.any(values < 0) or np.any(values >= VENUE_COUNT):
        raise ValueError("venue codes must be between 01 and 24")
    return values


def _load_legacy_bankroll_reference(
    path: Path,
    *,
    evaluation_from: str,
    evaluation_through: str,
    expected_dates: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(payload.get("evaluation_from")) != evaluation_from
        or str(payload.get("evaluation_through")) != evaluation_through
    ):
        raise ValueError("legacy payout reference evaluation period does not match")
    diagnostic = payload.get("conditional_payout_walk_forward")
    bankroll = diagnostic.get("bankroll") if isinstance(diagnostic, dict) else None
    if not isinstance(bankroll, dict) or not isinstance(bankroll.get("daily"), list):
        raise ValueError("legacy payout reference has no daily bankroll results")
    daily = bankroll["daily"]
    dates = [str(row.get("race_date")) for row in daily if isinstance(row, dict)]
    if dates != expected_dates or len(set(dates)) != len(dates):
        raise ValueError("legacy payout reference daily dates do not match")
    for key in ("roi", "profit_yen", "stake_yen", "return_yen"):
        if not isinstance(bankroll.get(key), (int, float)):
            raise ValueError(f"legacy payout reference is missing {key}")
    return bankroll, daily


def _pack_model(model: VenueConditionalOrderModel) -> np.ndarray:
    return np.concatenate(
        (
            _pack(model.global_model),
            np.asarray(model.venue_second_bias, dtype=np.float64).reshape(-1),
            np.asarray(model.venue_third_first_bias, dtype=np.float64).reshape(-1),
            np.asarray(model.venue_third_second_bias, dtype=np.float64).reshape(-1),
        )
    )


def _unpack_model(
    parameters: np.ndarray,
    *,
    global_regularization: float,
    venue_regularization: float,
) -> VenueConditionalOrderModel:
    values = np.asarray(parameters, dtype=np.float64)
    context_size = VENUE_COUNT * 6 * 6
    if values.shape != (111 + context_size * 3,):
        raise ValueError("venue conditional parameter vector has an invalid length")
    offset = 111
    arrays = []
    for _ in range(3):
        arrays.append(values[offset : offset + context_size].reshape(VENUE_COUNT, 6, 6).copy())
        offset += context_size
    return VenueConditionalOrderModel(
        global_model=_unpack(values[:111], regularization=global_regularization),
        venue_second_bias=arrays[0],
        venue_third_first_bias=arrays[1],
        venue_third_second_bias=arrays[2],
        venue_regularization=float(venue_regularization),
    )


def venue_identity_model(
    *,
    global_regularization: float = 0.0,
    venue_regularization: float = 0.0,
) -> VenueConditionalOrderModel:
    zeros = np.zeros((VENUE_COUNT, 6, 6), dtype=np.float64)
    return VenueConditionalOrderModel(
        global_model=identity_model(regularization=global_regularization),
        venue_second_bias=zeros.copy(),
        venue_third_first_bias=zeros.copy(),
        venue_third_second_bias=zeros.copy(),
        venue_regularization=float(venue_regularization),
    )


def objective_gradient(
    parameters: np.ndarray,
    scores: np.ndarray,
    orders: np.ndarray,
    venues: np.ndarray,
    *,
    global_regularization: float,
    venue_regularization: float,
) -> tuple[float, np.ndarray]:
    score_values, order_values = _validate_scores_orders(scores, orders)
    venue_values = np.asarray(venues, dtype=np.int64)
    if venue_values.shape != (len(score_values),):
        raise ValueError("venue indices must align with scores")
    if np.any(venue_values < 0) or np.any(venue_values >= VENUE_COUNT):
        raise ValueError("venue indices are out of range")
    if global_regularization < 0.0 or venue_regularization < 0.0:
        raise ValueError("regularization values must be non-negative")

    model = _unpack_model(
        parameters,
        global_regularization=global_regularization,
        venue_regularization=venue_regularization,
    )
    count = len(score_values)
    rows = np.arange(count)
    first, second, third = order_values.T
    scales_gradient = np.zeros(3, dtype=np.float64)
    second_gradient = np.zeros((6, 6), dtype=np.float64)
    third_first_gradient = np.zeros((6, 6), dtype=np.float64)
    third_second_gradient = np.zeros((6, 6), dtype=np.float64)
    venue_second_gradient = np.zeros((VENUE_COUNT, 6, 6), dtype=np.float64)
    venue_third_first_gradient = np.zeros((VENUE_COUNT, 6, 6), dtype=np.float64)
    venue_third_second_gradient = np.zeros((VENUE_COUNT, 6, 6), dtype=np.float64)

    global_model = model.global_model
    first_probabilities = stable_softmax(global_model.scales[0] * score_values)
    loss = -np.log(np.maximum(first_probabilities[rows, first], EPSILON)).sum()
    first_delta = first_probabilities
    first_delta[rows, first] -= 1.0
    scales_gradient[0] = float(np.sum(first_delta * score_values))

    second_logits = (
        global_model.scales[1] * score_values
        + global_model.second_bias[first]
        + model.venue_second_bias[venue_values, first]
    )
    second_probabilities = _masked_softmax(second_logits, (first,))
    loss -= np.log(np.maximum(second_probabilities[rows, second], EPSILON)).sum()
    second_delta = second_probabilities
    second_delta[rows, second] -= 1.0
    scales_gradient[1] = float(np.sum(second_delta * score_values))
    np.add.at(second_gradient, first, second_delta)
    np.add.at(venue_second_gradient, (venue_values, first), second_delta)

    third_logits = (
        global_model.scales[2] * score_values
        + global_model.third_first_bias[first]
        + global_model.third_second_bias[second]
        + model.venue_third_first_bias[venue_values, first]
        + model.venue_third_second_bias[venue_values, second]
    )
    third_probabilities = _masked_softmax(third_logits, (first, second))
    loss -= np.log(np.maximum(third_probabilities[rows, third], EPSILON)).sum()
    third_delta = third_probabilities
    third_delta[rows, third] -= 1.0
    scales_gradient[2] = float(np.sum(third_delta * score_values))
    np.add.at(third_first_gradient, first, third_delta)
    np.add.at(third_second_gradient, second, third_delta)
    np.add.at(venue_third_first_gradient, (venue_values, first), third_delta)
    np.add.at(venue_third_second_gradient, (venue_values, second), third_delta)

    global_gradient = np.concatenate(
        (
            scales_gradient,
            second_gradient.reshape(-1),
            third_first_gradient.reshape(-1),
            third_second_gradient.reshape(-1),
        )
    ) / max(1, count)
    context_gradient = np.concatenate(
        (
            venue_second_gradient.reshape(-1),
            venue_third_first_gradient.reshape(-1),
            venue_third_second_gradient.reshape(-1),
        )
    ) / max(1, count)
    global_delta = np.asarray(parameters[:111], dtype=np.float64) - _pack(identity_model())
    context_values = np.asarray(parameters[111:], dtype=np.float64)
    objective = float(loss) / max(1, count)
    objective += 0.5 * float(global_regularization) * float(global_delta @ global_delta)
    objective += 0.5 * float(venue_regularization) * float(context_values @ context_values)
    global_gradient += float(global_regularization) * global_delta
    context_gradient += float(venue_regularization) * context_values
    return objective, np.concatenate((global_gradient, context_gradient))


def fit_venue_conditional_order(
    scores: np.ndarray,
    orders: np.ndarray,
    venues: np.ndarray,
    *,
    global_regularization: float,
    venue_regularization: float,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
) -> tuple[VenueConditionalOrderModel, dict[str, Any]]:
    _validate_scores_orders(scores, orders)
    initial = _pack_model(
        venue_identity_model(
            global_regularization=global_regularization,
            venue_regularization=venue_regularization,
        )
    )
    bounds = (
        [(0.05, 5.0)] * 3
        + [(-5.0, 5.0)] * 108
        + [(-3.0, 3.0)] * (VENUE_COUNT * 6 * 6 * 3)
    )
    started = time.perf_counter()
    objective = functools.partial(
        objective_gradient,
        global_regularization=float(global_regularization),
        venue_regularization=float(venue_regularization),
    )
    result = minimize(
        objective,
        initial,
        args=(scores, orders, venues),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(max_iterations), "ftol": float(tolerance)},
    )
    model = _unpack_model(
        result.x,
        global_regularization=global_regularization,
        venue_regularization=venue_regularization,
    )
    return model, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.jac)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def venue_conditional_probabilities(
    scores: np.ndarray,
    model: VenueConditionalOrderModel,
    venues: np.ndarray,
) -> np.ndarray:
    score_values = np.asarray(scores, dtype=np.float64)
    venue_values = np.asarray(venues, dtype=np.int64)
    if score_values.ndim != 2 or score_values.shape[1] != 6:
        raise ValueError("scores must have shape (races, 6)")
    if venue_values.shape != (len(score_values),):
        raise ValueError("venue indices must align with scores")
    first_lanes = COMBINATION_LANES[:, 0]
    second_lanes = COMBINATION_LANES[:, 1]
    third_lanes = COMBINATION_LANES[:, 2]
    global_model = model.global_model
    first_probabilities = stable_softmax(global_model.scales[0] * score_values)
    output = np.empty((len(score_values), len(COMBINATION_LANES)), dtype=np.float64)
    for first in range(6):
        first_column = np.full(len(score_values), first, dtype=np.int64)
        second_logits = (
            global_model.scales[1] * score_values
            + global_model.second_bias[first]
            + model.venue_second_bias[venue_values, first]
        )
        second_probabilities = _masked_softmax(second_logits, (first_column,))
        for second in range(6):
            if second == first:
                continue
            second_column = np.full(len(score_values), second, dtype=np.int64)
            third_logits = (
                global_model.scales[2] * score_values
                + global_model.third_first_bias[first]
                + global_model.third_second_bias[second]
                + model.venue_third_first_bias[venue_values, first]
                + model.venue_third_second_bias[venue_values, second]
            )
            third_probabilities = _masked_softmax(
                third_logits, (first_column, second_column)
            )
            mask = (first_lanes == first) & (second_lanes == second)
            output[:, mask] = (
                first_probabilities[:, first, None]
                * second_probabilities[:, second, None]
                * third_probabilities[:, third_lanes[mask]]
            )
    output /= np.maximum(output.sum(axis=1, keepdims=True), EPSILON)
    return output


def evaluate_venue_model(
    scores: np.ndarray,
    ranks: np.ndarray,
    venues: np.ndarray,
    model: VenueConditionalOrderModel,
    *,
    batch_races: int,
) -> dict[str, Any]:
    losses = []
    top1_hits = 0
    top5 = []
    for start in range(0, len(scores), max(1, int(batch_races))):
        stop = min(len(scores), start + max(1, int(batch_races)))
        probabilities = venue_conditional_probabilities(
            scores[start:stop], model, venues[start:stop]
        )
        actual = actual_combination_indices(ranks[start:stop])
        rows = np.arange(stop - start)
        losses.append(-np.log(np.maximum(probabilities[rows, actual], EPSILON)))
        order = np.argsort(-probabilities, axis=1)
        top1_hits += int(np.sum(order[:, 0] == actual))
        top5.append(np.any(order[:, :5] == actual[:, None], axis=1).astype(np.float64))
    loss_values = np.concatenate(losses)
    top5_values = np.concatenate(top5)
    return {
        "evaluated_races": len(loss_values),
        "trifecta_log_loss": float(loss_values.mean()),
        "trifecta_top1_hit_rate": top1_hits / max(1, len(loss_values)),
        "trifecta_top5_hit_rate": float(top5_values.mean()),
        "race_losses": loss_values,
        "race_top5_hits": top5_values,
    }


def _evaluate_global(
    scores: np.ndarray,
    ranks: np.ndarray,
    model: ConditionalOrderModel,
    *,
    batch_races: int,
) -> dict[str, Any]:
    losses = []
    top1_hits = 0
    top5 = []
    for start in range(0, len(scores), max(1, int(batch_races))):
        stop = min(len(scores), start + max(1, int(batch_races)))
        probabilities = conditional_probabilities(scores[start:stop], model)
        metrics = evaluate_probabilities(probabilities, ranks[start:stop])
        actual = actual_combination_indices(ranks[start:stop])
        top1_hits += int(np.sum(np.argmax(probabilities, axis=1) == actual))
        losses.append(metrics["race_losses"])
        top5.append(metrics["race_top5_hits"])
    loss_values = np.concatenate(losses)
    top5_values = np.concatenate(top5)
    return {
        "evaluated_races": len(loss_values),
        "trifecta_log_loss": float(loss_values.mean()),
        "trifecta_top1_hit_rate": top1_hits / max(1, len(loss_values)),
        "trifecta_top5_hit_rate": float(top5_values.mean()),
        "race_losses": loss_values,
        "race_top5_hits": top5_values,
    }


def _all_probabilities(
    scores: np.ndarray,
    venues: np.ndarray,
    model: VenueConditionalOrderModel,
    *,
    batch_races: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(scores), max(1, int(batch_races))):
        stop = min(len(scores), start + max(1, int(batch_races)))
        parts.append(venue_conditional_probabilities(scores[start:stop], model, venues[start:stop]))
    return np.concatenate(parts)


def _global_probabilities(
    scores: np.ndarray,
    model: ConditionalOrderModel,
    *,
    batch_races: int,
) -> np.ndarray:
    return np.concatenate(
        [
            conditional_probabilities(scores[start : min(len(scores), start + batch_races)], model)
            for start in range(0, len(scores), max(1, int(batch_races)))
        ]
    )


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_artifact = joblib.load(args.baseline_model)
    baseline_model = baseline_artifact.get("model")
    if not isinstance(baseline_model, ListwiseLinearModel):
        raise ValueError("baseline artifact does not contain a ListwiseLinearModel")
    dropped = tuple(str(value) for value in baseline_artifact.get("drop_feature_groups") or ())
    n_features = len(baseline_model.weights)
    race_keys = [
        row
        for row in load_complete_race_ids(conn)
        if str(row[1]) <= args.evaluation_through
    ]
    trained_through = baseline_artifact.get("trained_through")
    trained_date = str(
        trained_through[1]
        if isinstance(trained_through, (list, tuple))
        else trained_through
    )
    if trained_date >= args.evaluation_from:
        raise ValueError("baseline training overlaps evaluation period")
    dataset = None
    if args.cache_prefix:
        cache_prefix = Path(args.cache_prefix)
        hasher = FeatureHasher(
            n_features=n_features,
            input_type="dict",
            alternate_sign=False,
        )
        if args.promote_legacy_cache:
            promote_legacy_hashed_dataset(
                cache_prefix,
                race_keys=race_keys,
                n_features=n_features,
                drop_feature_groups=dropped,
                hasher=hasher,
            )
        dataset = load_hashed_dataset(
            cache_prefix,
            race_keys=race_keys,
            n_features=n_features,
            drop_feature_groups=dropped,
            hasher=hasher,
        )
    if dataset is None:
        dataset, _source, _prefix = load_variant_dataset_with_cache(
            conn,
            race_keys=race_keys,
            cache_dir=Path(args.cache_dir),
            name="venue_context",
            dropped=dropped,
            n_features=n_features,
            batch_races=args.feature_batch_races,
            write_cache=False,
        )
    train_end = _date_boundary(race_keys, args.training_through, inclusive=True)
    evaluation_start = _date_boundary(race_keys, args.evaluation_from, inclusive=False)
    evaluation_end = _date_boundary(race_keys, args.evaluation_through, inclusive=True)
    if train_end <= 0 or train_end != evaluation_start or evaluation_end <= evaluation_start:
        raise ValueError("training and evaluation must be adjacent non-empty full-day ranges")
    scores = _score_dataset(dataset, baseline_model, race_end=evaluation_end, batch_races=args.batch_races)
    orders = np.argsort(dataset.ranks[:evaluation_end], axis=1)[:, :3]
    venues = venue_indices(race_keys[:evaluation_end])
    validation_start_date = (
        datetime.fromisoformat(args.training_through).date()
        - timedelta(days=max(1, int(args.validation_days)) - 1)
    ).isoformat()
    validation_start = _date_boundary(race_keys, validation_start_date, inclusive=False)
    validation_start = min(train_end - 1, max(1, validation_start))
    candidates = []
    calibration_models: dict[float, VenueConditionalOrderModel] = {}
    for venue_regularization in args.venue_regularizations:
        model, fit = fit_venue_conditional_order(
            scores[:validation_start],
            orders[:validation_start],
            venues[:validation_start],
            global_regularization=args.global_regularization,
            venue_regularization=float(venue_regularization),
            max_iterations=args.max_iterations,
        )
        metrics = evaluate_venue_model(
            scores[validation_start:train_end],
            dataset.ranks[validation_start:train_end],
            venues[validation_start:train_end],
            model,
            batch_races=args.batch_races,
        )
        candidates.append(
            {
                "venue_regularization": float(venue_regularization),
                "fit": fit,
                "validation": _public_metrics(metrics),
            }
        )
        calibration_models[float(venue_regularization)] = model
    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation"]["trifecta_log_loss"]),
            -float(row["venue_regularization"]),
        ),
    )
    selected_regularization = float(selected["venue_regularization"])
    model, fit = fit_venue_conditional_order(
        scores[:train_end],
        orders[:train_end],
        venues[:train_end],
        global_regularization=args.global_regularization,
        venue_regularization=selected_regularization,
        max_iterations=args.max_iterations,
    )
    global_model, global_fit = fit_conditional_order(
        scores[:train_end],
        orders[:train_end],
        regularization=args.global_regularization,
        max_iterations=args.max_iterations,
    )
    evaluation_slice = slice(evaluation_start, evaluation_end)
    candidate_metrics = evaluate_venue_model(
        scores[evaluation_slice],
        dataset.ranks[evaluation_slice],
        venues[evaluation_slice],
        model,
        batch_races=args.batch_races,
    )
    global_metrics = _evaluate_global(
        scores[evaluation_slice],
        dataset.ranks[evaluation_slice],
        global_model,
        batch_races=args.batch_races,
    )
    evaluation_keys = race_keys[evaluation_start:evaluation_end]
    dates = [str(row[1]) for row in evaluation_keys]
    loss_delta = candidate_metrics["race_losses"] - global_metrics["race_losses"]
    top5_delta = candidate_metrics["race_top5_hits"] - global_metrics["race_top5_hits"]
    confidence = {
        "race_log_loss": paired_mean_bootstrap(loss_delta),
        "day_log_loss": paired_cluster_mean_bootstrap(loss_delta, dates),
        "race_top5": paired_mean_bootstrap(top5_delta),
        "day_top5": paired_cluster_mean_bootstrap(top5_delta, dates),
    }
    monthly: dict[str, list[float]] = {}
    for race_date, delta in zip(dates, loss_delta):
        monthly.setdefault(race_date[:7], []).append(float(delta))
    monthly_rows = [
        {"month": month, "races": len(values), "mean_log_loss_delta": float(np.mean(values))}
        for month, values in sorted(monthly.items())
    ]
    structure_gate = {
        "day_log_loss_ci_upper_at_most_zero": confidence["day_log_loss"]["ci95_upper"] <= 0.0,
        "day_top5_ci_lower_at_least_zero": confidence["day_top5"]["ci95_lower"] >= 0.0,
        "minimum_non_regressing_months": 8,
        "non_regressing_months": sum(row["mean_log_loss_delta"] <= 0.0 for row in monthly_rows),
    }
    structure_gate["pass"] = bool(
        structure_gate["day_log_loss_ci_upper_at_most_zero"]
        and structure_gate["day_top5_ci_lower_at_least_zero"]
        and structure_gate["non_regressing_months"] >= structure_gate["minimum_non_regressing_months"]
    )

    calibration_slice = slice(validation_start, train_end)
    candidate_probabilities = _all_probabilities(
        scores[evaluation_slice], venues[evaluation_slice], model, batch_races=args.batch_races
    )
    global_probabilities = _global_probabilities(
        scores[evaluation_slice], global_model, batch_races=args.batch_races
    )
    market_probabilities = _global_probabilities(
        scores[evaluation_slice], identity_model(), batch_races=args.batch_races
    )
    calibration_model = calibration_models[selected_regularization]
    calibration_candidate = _all_probabilities(
        scores[calibration_slice],
        venues[calibration_slice],
        calibration_model,
        batch_races=args.batch_races,
    )
    calibration_global_model, _ = fit_conditional_order(
        scores[:validation_start],
        orders[:validation_start],
        regularization=args.global_regularization,
        max_iterations=args.max_iterations,
    )
    calibration_global = _global_probabilities(
        scores[calibration_slice], calibration_global_model, batch_races=args.batch_races
    )
    calibration_market = _global_probabilities(
        scores[calibration_slice], identity_model(), batch_races=args.batch_races
    )
    payouts = _load_trifecta_payouts(conn)
    bankroll_kwargs = {
        "race_keys": evaluation_keys,
        "payouts": payouts,
        "calibration_race_keys": race_keys[validation_start:train_end],
        "market_reference_probabilities": market_probabilities,
        "calibration_market_reference_probabilities": calibration_market,
        "ridge": args.payout_ridge,
        "ridge_candidates": tuple(args.payout_ridges),
        "mean_correction_candidates": tuple(args.payout_mean_corrections),
        "threshold_candidates": tuple(args.payout_threshold_candidates),
        "policy_selection_days": args.payout_policy_selection_days,
    }
    candidate_bankroll = simulate_conditional_payout_walk_forward(
        candidate_probabilities,
        calibration_probabilities=calibration_candidate,
        **bankroll_kwargs,
    )
    global_bankroll = simulate_conditional_payout_walk_forward(
        global_probabilities,
        calibration_probabilities=calibration_global,
        **bankroll_kwargs,
    )
    bankroll_confidence = bootstrap_daily_bankroll(
        candidate_bankroll["daily"], baseline_daily=global_bankroll["daily"]
    )
    bankroll_gate = bankroll_promotion_gate(
        candidate_bankroll, global_bankroll, bankroll_confidence
    )
    expected_dates = sorted({str(row[1]) for row in evaluation_keys})
    legacy_bankroll, legacy_daily = _load_legacy_bankroll_reference(
        Path(args.legacy_evaluation),
        evaluation_from=args.evaluation_from,
        evaluation_through=args.evaluation_through,
        expected_dates=expected_dates,
    )
    payout_feature_confidence = bootstrap_daily_bankroll(
        global_bankroll["daily"], baseline_daily=legacy_daily
    )
    payout_feature_gate = bankroll_promotion_gate(
        global_bankroll, legacy_bankroll, payout_feature_confidence
    )
    promotion_eligible = bool(structure_gate["pass"] and bankroll_gate["pass"])
    artifact = {
        "model": model,
        "base_model": baseline_model,
        "hasher": baseline_artifact.get("hasher"),
        "model_name": MODEL_NAME,
        "source_model": str(args.baseline_model),
        "drop_feature_groups": dropped,
        "n_features": n_features,
        "training_through": args.training_through,
        "global_regularization": float(args.global_regularization),
        "venue_regularization": selected_regularization,
    }
    dump_joblib_atomic(Path(args.model_output), artifact)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "venue-conditioned order transitions on untouched 365-day holdout",
        "source_model": str(args.baseline_model),
        "model_artifact": str(args.model_output),
        "training_through": args.training_through,
        "evaluation_from": args.evaluation_from,
        "evaluation_through": args.evaluation_through,
        "training_races": train_end,
        "validation_races": train_end - validation_start,
        "evaluation_races": evaluation_end - evaluation_start,
        "global_regularization": float(args.global_regularization),
        "venue_regularization_candidates": candidates,
        "selected_venue_regularization": selected_regularization,
        "fit": fit,
        "global_fit": global_fit,
        "venue_conditional_order": _public_metrics(candidate_metrics),
        "global_conditional_order": _public_metrics(global_metrics),
        "paired_confidence": confidence,
        "monthly": monthly_rows,
        "bankroll": candidate_bankroll,
        "baseline_bankroll": {key: value for key, value in global_bankroll.items() if key != "daily"},
        "bankroll_confidence": bankroll_confidence,
        "structure_gate": structure_gate,
        "bankroll_gate": bankroll_gate,
        "payout_feature_comparison": {
            "candidate_schema": global_bankroll["policy"].get("payout_feature_schema"),
            "legacy_schema": LEGACY_PAYOUT_SCHEMA,
            "candidate_bankroll": {
                key: global_bankroll[key]
                for key in ("roi", "profit_yen", "stake_yen", "return_yen")
            },
            "legacy_bankroll": {
                key: legacy_bankroll[key]
                for key in ("roi", "profit_yen", "stake_yen", "return_yen")
            },
            "confidence": payout_feature_confidence,
            "gate": payout_feature_gate,
        },
        "promotion_eligible": promotion_eligible,
        "roi": candidate_bankroll["roi"],
        "profit_yen": candidate_bankroll["profit_yen"],
        "stake_yen": candidate_bankroll["stake_yen"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit venue-conditioned top-three order transitions.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-prefix")
    parser.add_argument("--cache-dir", default="/tmp/boatrace-venue-context")
    parser.add_argument("--feature-batch-races", type=int, default=1_000)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument(
        "--legacy-evaluation",
        default="data/models/conditional_order_365d.json",
    )
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--evaluation-from", required=True)
    parser.add_argument("--evaluation-through", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--global-regularization", type=float, default=0.0001)
    parser.add_argument(
        "--venue-regularizations",
        type=float,
        nargs="+",
        default=DEFAULT_VENUE_REGULARIZATIONS,
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--batch-races", type=int, default=4_000)
    parser.add_argument("--payout-ridge", type=float, default=10.0)
    parser.add_argument("--payout-ridges", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--payout-mean-corrections", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--payout-threshold-candidates", type=float, nargs="+", default=[1.05, 1.10, 1.20])
    parser.add_argument("--payout-policy-selection-days", type=int, default=30)
    parser.add_argument("--promote-legacy-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.global_regularization < 0.0 or any(value <= 0.0 for value in args.venue_regularizations):
        raise ValueError("regularization values must be positive")
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"bankroll", "venue_regularization_candidates"}
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
