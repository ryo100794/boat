from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss

from .calibrated_shadow_model import safe_log_loss
from .modeling import _race_level_metrics


EPSILON = 1e-12
DEFAULT_WEIGHTS = tuple(index / 20.0 for index in range(21))


def blend_predictions(
    baseline: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
    *,
    candidate_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    weight = float(candidate_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("candidate_weight must be finite and between zero and one")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate race sets differ")

    blended: dict[str, list[dict[str, Any]]] = {}
    for race_id in baseline:
        baseline_by_lane = _rows_by_lane(baseline[race_id], context="baseline")
        candidate_by_lane = _rows_by_lane(candidate[race_id], context="candidate")
        if set(baseline_by_lane) != set(candidate_by_lane):
            raise ValueError(f"lane sets differ for race {race_id}")
        lanes = sorted(baseline_by_lane)
        values = np.asarray(
            [
                max(EPSILON, float(baseline_by_lane[lane]["probability"]))
                ** (1.0 - weight)
                * max(EPSILON, float(candidate_by_lane[lane]["probability"]))
                ** weight
                for lane in lanes
            ],
            dtype=np.float64,
        )
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"invalid blended probability total for race {race_id}")
        values /= total
        rows = []
        for lane, probability in zip(lanes, values):
            baseline_row = baseline_by_lane[lane]
            candidate_row = candidate_by_lane[lane]
            if int(baseline_row["rank"]) != int(candidate_row["rank"]):
                raise ValueError(f"rank differs for race {race_id} lane {lane}")
            row = dict(candidate_row)
            row["probability"] = float(probability)
            rows.append(row)
        blended[race_id] = rows
    return blended


def prediction_metrics(
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    labels: list[int] = []
    probabilities: list[float] = []
    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race_id, rows in predictions.items():
        by_lane = _rows_by_lane(rows, context="prediction")
        values = np.asarray(
            [float(by_lane[lane]["probability"]) for lane in sorted(by_lane)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid probabilities for race {race_id}")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError(f"probabilities do not sum to one for race {race_id}")
        for lane in sorted(by_lane):
            row = dict(by_lane[lane])
            label = 1 if int(row["rank"]) == 1 else 0
            row["label"] = label
            labels.append(label)
            probabilities.append(float(row["probability"]))
            normalized[race_id].append(row)
    if not labels:
        raise ValueError("at least one prediction race is required")
    return {
        "evaluated_races": len(normalized),
        "entry_log_loss": safe_log_loss(labels, probabilities),
        "entry_brier": float(brier_score_loss(labels, probabilities)),
        **_race_level_metrics(normalized),
    }


def select_protected_blend(
    baseline: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
    *,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    log_loss_tolerance: float = 0.0,
    accuracy_tolerance: float = 0.0,
) -> dict[str, Any]:
    if log_loss_tolerance < 0.0 or accuracy_tolerance < 0.0:
        raise ValueError("protection tolerances must be non-negative")
    unique_weights = sorted({float(value) for value in weights} | {0.0})
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in unique_weights):
        raise ValueError("blend weights must be finite and between zero and one")

    baseline_metrics = prediction_metrics(baseline)
    candidates = []
    for weight in unique_weights:
        predictions = blend_predictions(
            baseline,
            candidate,
            candidate_weight=weight,
        )
        metrics = prediction_metrics(predictions)
        protected = bool(
            float(metrics["entry_log_loss"])
            <= float(baseline_metrics["entry_log_loss"]) + log_loss_tolerance
            and float(metrics["winner_top1_accuracy"])
            + accuracy_tolerance
            >= float(baseline_metrics["winner_top1_accuracy"])
            and float(metrics["trifecta_top5_hit_rate"])
            + accuracy_tolerance
            >= float(baseline_metrics["trifecta_top5_hit_rate"])
        )
        candidates.append(
            {
                "candidate_weight": weight,
                "protected": protected,
                **metrics,
            }
        )
    eligible = [row for row in candidates if row["protected"]]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["entry_log_loss"]),
            -float(row["trifecta_top5_hit_rate"]),
            -float(row["winner_top1_accuracy"]),
            float(row["candidate_weight"]),
        ),
    )
    return {
        "selection_scope": "training-only calibration; holdout untouched",
        "selection_criterion": (
            "minimum entry log loss among blends that do not degrade baseline "
            "winner top1 or trifecta top5"
        ),
        "candidate_weight": float(selected["candidate_weight"]),
        "baseline_metrics": baseline_metrics,
        "selected_metrics": {
            key: value
            for key, value in selected.items()
            if key not in {"protected"}
        },
        "candidate_count": len(candidates),
        "protected_candidate_count": len(eligible),
        "candidates": candidates,
    }


def _rows_by_lane(
    rows: Iterable[dict[str, Any]],
    *,
    context: str,
) -> dict[int, dict[str, Any]]:
    by_lane: dict[int, dict[str, Any]] = {}
    for row in rows:
        lane = int(row["lane"])
        if lane in by_lane:
            raise ValueError(f"duplicate {context} lane {lane}")
        probability = float(row["probability"])
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError(f"invalid {context} probability for lane {lane}")
        by_lane[lane] = row
    if set(by_lane) != set(range(1, 7)):
        raise ValueError(f"{context} race must contain lanes one through six")
    return by_lane
