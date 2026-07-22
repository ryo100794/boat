from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction import FeatureHasher

from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import load_hashed_dataset, promote_legacy_hashed_dataset
from .cluster_bootstrap import paired_cluster_mean_bootstrap
from .model import ListwiseLinearModel, stable_softmax
from .newton_refine import dump_joblib_atomic
from .paired_bootstrap import paired_mean_bootstrap
from .stagewise_mlp import COMBINATION_LANES, actual_combination_indices


MODEL_NAME = "pastlog_conditional_order"
EPSILON = 1e-15
DEFAULT_REGULARIZATIONS = (0.0001, 0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class ConditionalOrderModel:
    scales: np.ndarray
    second_bias: np.ndarray
    third_first_bias: np.ndarray
    third_second_bias: np.ndarray
    regularization: float


def _validate_scores_orders(
    scores: np.ndarray, orders: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float64)
    order_values = np.asarray(orders, dtype=np.int64)
    if score_values.ndim != 2 or score_values.shape[1] != 6:
        raise ValueError("scores must have shape (races, 6)")
    if order_values.shape != (score_values.shape[0], 3):
        raise ValueError("orders must have shape (races, 3)")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("scores must be finite")
    if np.any(order_values < 0) or np.any(order_values >= 6):
        raise ValueError("order lanes must be zero-based values from zero to five")
    if np.any(np.sort(order_values, axis=1)[:, 1:] == np.sort(order_values, axis=1)[:, :-1]):
        raise ValueError("each order must contain three distinct lanes")
    return score_values, order_values


def _pack(model: ConditionalOrderModel) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(model.scales, dtype=np.float64).reshape(3),
            np.asarray(model.second_bias, dtype=np.float64).reshape(36),
            np.asarray(model.third_first_bias, dtype=np.float64).reshape(36),
            np.asarray(model.third_second_bias, dtype=np.float64).reshape(36),
        )
    )


def _unpack(parameters: np.ndarray, *, regularization: float) -> ConditionalOrderModel:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (111,):
        raise ValueError("conditional order parameter vector must have length 111")
    return ConditionalOrderModel(
        scales=values[:3].copy(),
        second_bias=values[3:39].reshape(6, 6).copy(),
        third_first_bias=values[39:75].reshape(6, 6).copy(),
        third_second_bias=values[75:111].reshape(6, 6).copy(),
        regularization=float(regularization),
    )


def identity_model(*, regularization: float = 0.0) -> ConditionalOrderModel:
    return ConditionalOrderModel(
        scales=np.ones(3, dtype=np.float64),
        second_bias=np.zeros((6, 6), dtype=np.float64),
        third_first_bias=np.zeros((6, 6), dtype=np.float64),
        third_second_bias=np.zeros((6, 6), dtype=np.float64),
        regularization=float(regularization),
    )


def _masked_softmax(logits: np.ndarray, excluded: Iterable[np.ndarray]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    row_indices = np.arange(values.shape[0])
    for lanes in excluded:
        values[row_indices, np.asarray(lanes, dtype=np.int64)] = -np.inf
    maximum = np.max(values, axis=1, keepdims=True)
    numerator = np.exp(values - maximum)
    numerator[~np.isfinite(values)] = 0.0
    return numerator / np.maximum(numerator.sum(axis=1, keepdims=True), EPSILON)


def objective_gradient(
    parameters: np.ndarray,
    scores: np.ndarray,
    orders: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    score_values, order_values = _validate_scores_orders(scores, orders)
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    model = _unpack(parameters, regularization=regularization)
    count = score_values.shape[0]
    rows = np.arange(count)
    first, second, third = order_values.T
    scales_gradient = np.zeros(3, dtype=np.float64)
    second_gradient = np.zeros((6, 6), dtype=np.float64)
    third_first_gradient = np.zeros((6, 6), dtype=np.float64)
    third_second_gradient = np.zeros((6, 6), dtype=np.float64)

    first_probabilities = stable_softmax(model.scales[0] * score_values)
    loss = -np.log(np.maximum(first_probabilities[rows, first], EPSILON)).sum()
    first_delta = first_probabilities
    first_delta[rows, first] -= 1.0
    scales_gradient[0] = float(np.sum(first_delta * score_values))

    second_logits = (
        model.scales[1] * score_values + model.second_bias[first]
    )
    second_probabilities = _masked_softmax(second_logits, (first,))
    loss -= np.log(np.maximum(second_probabilities[rows, second], EPSILON)).sum()
    second_delta = second_probabilities
    second_delta[rows, second] -= 1.0
    scales_gradient[1] = float(np.sum(second_delta * score_values))
    np.add.at(second_gradient, first, second_delta)

    third_logits = (
        model.scales[2] * score_values
        + model.third_first_bias[first]
        + model.third_second_bias[second]
    )
    third_probabilities = _masked_softmax(third_logits, (first, second))
    loss -= np.log(np.maximum(third_probabilities[rows, third], EPSILON)).sum()
    third_delta = third_probabilities
    third_delta[rows, third] -= 1.0
    scales_gradient[2] = float(np.sum(third_delta * score_values))
    np.add.at(third_first_gradient, first, third_delta)
    np.add.at(third_second_gradient, second, third_delta)

    gradient = np.concatenate(
        (
            scales_gradient,
            second_gradient.reshape(-1),
            third_first_gradient.reshape(-1),
            third_second_gradient.reshape(-1),
        )
    ) / max(1, count)
    prior = _pack(identity_model())
    delta = np.asarray(parameters, dtype=np.float64) - prior
    objective = float(loss) / max(1, count)
    objective += 0.5 * float(regularization) * float(delta @ delta)
    gradient += float(regularization) * delta
    return objective, gradient


def fit_conditional_order(
    scores: np.ndarray,
    orders: np.ndarray,
    *,
    regularization: float,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
) -> tuple[ConditionalOrderModel, dict[str, Any]]:
    score_values, order_values = _validate_scores_orders(scores, orders)
    initial = _pack(identity_model(regularization=regularization))
    bounds = [(0.05, 5.0)] * 3 + [(-5.0, 5.0)] * 108
    started = time.perf_counter()
    objective = functools.partial(
        objective_gradient,
        regularization=float(regularization),
    )
    result = minimize(
        objective,
        initial,
        args=(score_values, order_values),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(max_iterations), "ftol": float(tolerance)},
    )
    model = _unpack(result.x, regularization=regularization)
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


def conditional_probabilities(
    scores: np.ndarray,
    model: ConditionalOrderModel,
) -> np.ndarray:
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.ndim != 2 or score_values.shape[1] != 6:
        raise ValueError("scores must have shape (races, 6)")
    first_lanes = COMBINATION_LANES[:, 0]
    second_lanes = COMBINATION_LANES[:, 1]
    third_lanes = COMBINATION_LANES[:, 2]
    first_probabilities = stable_softmax(model.scales[0] * score_values)
    output = np.empty((len(score_values), len(COMBINATION_LANES)), dtype=np.float64)
    for first in range(6):
        second_logits = model.scales[1] * score_values + model.second_bias[first]
        first_column = np.full(len(score_values), first, dtype=np.int64)
        second_probabilities = _masked_softmax(second_logits, (first_column,))
        for second in range(6):
            if second == first:
                continue
            third_logits = (
                model.scales[2] * score_values
                + model.third_first_bias[first]
                + model.third_second_bias[second]
            )
            second_column = np.full(len(score_values), second, dtype=np.int64)
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


def evaluate_probabilities(
    probabilities: np.ndarray,
    ranks: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    rank_values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[1] != len(COMBINATION_LANES):
        raise ValueError("probabilities must have shape (races, 120)")
    actual = actual_combination_indices(rank_values)
    rows = np.arange(len(values))
    losses = -np.log(np.maximum(values[rows, actual], EPSILON))
    order = np.argsort(-values, axis=1)
    top1 = order[:, 0] == actual
    top5 = np.any(order[:, :5] == actual[:, None], axis=1)
    return {
        "evaluated_races": len(values),
        "trifecta_log_loss": float(losses.mean()),
        "trifecta_top1_hit_rate": float(top1.mean()),
        "trifecta_top5_hit_rate": float(top5.mean()),
        "race_losses": losses,
        "race_top5_hits": top5.astype(np.float64),
    }


def evaluate_model(
    scores: np.ndarray,
    ranks: np.ndarray,
    model: ConditionalOrderModel,
    *,
    batch_races: int = 4_000,
) -> dict[str, Any]:
    loss_parts = []
    top5_parts = []
    for start in range(0, len(scores), max(1, int(batch_races))):
        stop = min(len(scores), start + max(1, int(batch_races)))
        metrics = evaluate_probabilities(
            conditional_probabilities(scores[start:stop], model),
            ranks[start:stop],
        )
        loss_parts.append(metrics["race_losses"])
        top5_parts.append(metrics["race_top5_hits"])
    losses = np.concatenate(loss_parts)
    top5 = np.concatenate(top5_parts)
    probabilities = conditional_probabilities(scores, model) if len(scores) <= batch_races else None
    if probabilities is None:
        top1_hits = 0
        for start in range(0, len(scores), max(1, int(batch_races))):
            stop = min(len(scores), start + max(1, int(batch_races)))
            values = conditional_probabilities(scores[start:stop], model)
            actual = actual_combination_indices(ranks[start:stop])
            top1_hits += int(np.sum(np.argmax(values, axis=1) == actual))
        top1_rate = top1_hits / max(1, len(scores))
    else:
        actual = actual_combination_indices(ranks)
        top1_rate = float(np.mean(np.argmax(probabilities, axis=1) == actual))
    return {
        "evaluated_races": len(losses),
        "trifecta_log_loss": float(losses.mean()),
        "trifecta_top1_hit_rate": float(top1_rate),
        "trifecta_top5_hit_rate": float(top5.mean()),
        "race_losses": losses,
        "race_top5_hits": top5,
    }


def _score_dataset(dataset, model: ListwiseLinearModel, *, race_end: int, batch_races: int) -> np.ndarray:
    output = np.empty((race_end, 6), dtype=np.float64)
    for race_start in range(0, race_end, max(1, int(batch_races))):
        race_stop = min(race_end, race_start + max(1, int(batch_races)))
        matrix = dataset.matrix[dataset.row_slice(race_start, race_stop)]
        transformed = model.scaler.transform(matrix)
        output[race_start:race_stop] = np.asarray(
            transformed.dot(model.weights)
        ).reshape(-1, 6)
    return output


def _date_boundary(race_keys: list[tuple[str, str, str, int]], date_value: str, *, inclusive: bool) -> int:
    return sum(
        str(row[1]) <= date_value if inclusive else str(row[1]) < date_value
        for row in race_keys
    )


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"race_losses", "race_top5_hits"}
    }


def _daily_rows(
    race_keys: list[tuple[str, str, str, int]],
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    dates = np.asarray([str(row[1]) for row in race_keys])
    rows = []
    for race_date in sorted(set(dates)):
        mask = dates == race_date
        rows.append(
            {
                "race_date": race_date,
                "races": int(mask.sum()),
                "log_loss": float(candidate["race_losses"][mask].mean()),
                "baseline_log_loss": float(baseline["race_losses"][mask].mean()),
                "log_loss_delta": float(
                    (candidate["race_losses"][mask] - baseline["race_losses"][mask]).mean()
                ),
                "top5_hit_rate": float(candidate["race_top5_hits"][mask].mean()),
                "baseline_top5_hit_rate": float(baseline["race_top5_hits"][mask].mean()),
                "top5_delta": float(
                    (candidate["race_top5_hits"][mask] - baseline["race_top5_hits"][mask]).mean()
                ),
            }
        )
    return rows


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_artifact = joblib.load(args.baseline_model)
    baseline_model = baseline_artifact.get("model")
    if not isinstance(baseline_model, ListwiseLinearModel):
        raise ValueError("baseline artifact does not contain a ListwiseLinearModel")
    manifest = json.loads(Path(f"{args.cache_prefix}.manifest.json").read_text(encoding="utf-8"))
    last_date = str(manifest["last_race_id"])[:10]
    race_keys = [row for row in load_complete_race_ids(conn) if str(row[1]) <= last_date]
    dropped = tuple(str(value) for value in manifest.get("drop_feature_groups") or ())
    n_features = int(manifest["n_features"])
    artifact_dropped = tuple(str(value) for value in baseline_artifact.get("drop_feature_groups") or ())
    if artifact_dropped != dropped or len(baseline_model.weights) != n_features:
        raise ValueError("baseline artifact and feature cache contracts differ")
    trained_through = baseline_artifact.get("trained_through")
    trained_date = str(trained_through[1] if isinstance(trained_through, (list, tuple)) else trained_through)
    if trained_date >= args.evaluation_from:
        raise ValueError("baseline training overlaps evaluation period")
    cache_prefix = Path(args.cache_prefix)
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    cache_promoted = False
    if args.promote_legacy_cache:
        cache_promoted = promote_legacy_hashed_dataset(
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
        raise ValueError("hashed feature cache failed contract validation")
    train_end = _date_boundary(race_keys, args.training_through, inclusive=True)
    evaluation_start = _date_boundary(race_keys, args.evaluation_from, inclusive=False)
    evaluation_end = _date_boundary(race_keys, args.evaluation_through, inclusive=True)
    if train_end <= 0 or train_end != evaluation_start or evaluation_end <= evaluation_start:
        raise ValueError("training and evaluation must be adjacent non-empty full-day ranges")
    scores = _score_dataset(
        dataset, baseline_model, race_end=evaluation_end, batch_races=args.batch_races
    )
    orders = np.argsort(dataset.ranks[:evaluation_end], axis=1)[:, :3]
    validation_start_date = (
        datetime.fromisoformat(args.training_through).date()
        - timedelta(days=max(1, int(args.validation_days)) - 1)
    ).isoformat()
    validation_start = _date_boundary(race_keys, validation_start_date, inclusive=False)
    validation_start = min(train_end - 1, max(1, validation_start))
    candidates = []
    for regularization in args.regularizations:
        model, fit = fit_conditional_order(
            scores[:validation_start],
            orders[:validation_start],
            regularization=float(regularization),
            max_iterations=args.max_iterations,
        )
        metrics = evaluate_model(
            scores[validation_start:train_end],
            dataset.ranks[validation_start:train_end],
            model,
            batch_races=args.batch_races,
        )
        candidates.append(
            {
                "regularization": float(regularization),
                "fit": fit,
                "validation": _public_metrics(metrics),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation"]["trifecta_log_loss"]),
            -float(row["regularization"]),
        ),
    )
    model, fit = fit_conditional_order(
        scores[:train_end],
        orders[:train_end],
        regularization=float(selected["regularization"]),
        max_iterations=args.max_iterations,
    )
    candidate_metrics = evaluate_model(
        scores[evaluation_start:evaluation_end],
        dataset.ranks[evaluation_start:evaluation_end],
        model,
        batch_races=args.batch_races,
    )
    baseline_metrics = evaluate_model(
        scores[evaluation_start:evaluation_end],
        dataset.ranks[evaluation_start:evaluation_end],
        identity_model(),
        batch_races=args.batch_races,
    )
    evaluation_keys = race_keys[evaluation_start:evaluation_end]
    dates = [str(row[1]) for row in evaluation_keys]
    loss_delta = candidate_metrics["race_losses"] - baseline_metrics["race_losses"]
    top5_delta = candidate_metrics["race_top5_hits"] - baseline_metrics["race_top5_hits"]
    confidence = {
        "race_log_loss": paired_mean_bootstrap(loss_delta),
        "day_log_loss": paired_cluster_mean_bootstrap(loss_delta, dates),
        "race_top5": paired_mean_bootstrap(top5_delta),
        "day_top5": paired_cluster_mean_bootstrap(top5_delta, dates),
    }
    daily = _daily_rows(evaluation_keys, candidate_metrics, baseline_metrics)
    positive_months: dict[str, list[float]] = {}
    for row in daily:
        positive_months.setdefault(str(row["race_date"])[:7], []).append(
            float(row["log_loss_delta"])
        )
    monthly = [
        {
            "month": month,
            "days": len(values),
            "mean_daily_log_loss_delta": sum(values) / len(values),
        }
        for month, values in sorted(positive_months.items())
    ]
    gate = {
        "day_log_loss_ci_upper_at_most_zero": confidence["day_log_loss"]["ci95_upper"] <= 0.0,
        "day_top5_ci_lower_at_least_zero": confidence["day_top5"]["ci95_lower"] >= 0.0,
        "minimum_non_regressing_months": 8,
        "non_regressing_months": sum(row["mean_daily_log_loss_delta"] <= 0.0 for row in monthly),
    }
    gate["pass"] = bool(
        gate["day_log_loss_ci_upper_at_most_zero"]
        and gate["day_top5_ci_lower_at_least_zero"]
        and gate["non_regressing_months"] >= gate["minimum_non_regressing_months"]
    )
    artifact = {
        "model": model,
        "base_model": baseline_model,
        "hasher": baseline_artifact.get("hasher"),
        "model_name": MODEL_NAME,
        "source_model": str(args.baseline_model),
        "drop_feature_groups": dropped,
        "n_features": n_features,
        "training_through": args.training_through,
        "selected_regularization": float(selected["regularization"]),
    }
    dump_joblib_atomic(Path(args.model_output), artifact)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "fixed-cutoff conditional order interaction candidate",
        "source_model": str(args.baseline_model),
        "model_artifact": str(args.model_output),
        "training_through": args.training_through,
        "evaluation_from": args.evaluation_from,
        "evaluation_through": args.evaluation_through,
        "training_races": train_end,
        "validation_races": train_end - validation_start,
        "evaluation_races": evaluation_end - evaluation_start,
        "cache_promoted": cache_promoted,
        "regularization_candidates": candidates,
        "selected_regularization": float(selected["regularization"]),
        "fit": fit,
        "scales": model.scales.tolist(),
        "conditional_order": _public_metrics(candidate_metrics),
        "listwise_baseline": _public_metrics(baseline_metrics),
        "paired_confidence": confidence,
        "monthly": monthly,
        "daily": daily,
        "promotion_gate": gate,
        "promotion_eligible": gate["pass"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a conditional top-three order interaction model."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--evaluation-from", required=True)
    parser.add_argument("--evaluation-through", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--regularizations", type=float, nargs="+", default=DEFAULT_REGULARIZATIONS)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--batch-races", type=int, default=4_000)
    parser.add_argument("--promote-legacy-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    compact = {key: value for key, value in result.items() if key not in {"daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
