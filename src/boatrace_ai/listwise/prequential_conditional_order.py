from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from .conditional_order import conditional_probabilities, fit_conditional_order
from .stagewise_mlp import COMBINATION_LANES


DEFAULT_REGULARIZATIONS = (0.001, 0.01, 0.1, 1.0)
DEFAULT_BLENDS = (0.25, 0.5, 0.75, 1.0)
EPSILON = 1e-15


def apply_prequential_conditional_order(
    races: Iterable[dict[str, Any]],
    *,
    minimum_prior_days: int = 4,
    regularizations: Sequence[float] = DEFAULT_REGULARIZATIONS,
    blends: Sequence[float] = DEFAULT_BLENDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine 120-outcome probabilities using only earlier race dates.

    The final prior date selects regularization and the blend against the
    original probabilities. The selected order model is then refit on every
    prior date before transforming the current date.
    """
    if minimum_prior_days < 2:
        raise ValueError("minimum_prior_days must be at least 2")
    regularization_values = tuple(
        sorted({float(value) for value in regularizations})
    )
    blend_values = tuple(sorted({float(value) for value in blends}))
    if not regularization_values or any(
        not math.isfinite(value) or value < 0.0
        for value in regularization_values
    ):
        raise ValueError("regularizations must be finite and non-negative")
    if not blend_values or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in blend_values
    ):
        raise ValueError("blends must be finite values in (0, 1]")

    output = [dict(race) for race in races]
    output.sort(
        key=lambda race: (
            str(race["race_date"]),
            str(race.get("jcd") or ""),
            int(race.get("rno") or 0),
            str(race.get("race_id") or ""),
        )
    )
    if not output:
        return [], _empty_report(minimum_prior_days)
    race_ids = [str(race.get("race_id") or "") for race in output]
    if any(not race_id for race_id in race_ids) or len(set(race_ids)) != len(
        race_ids
    ):
        raise ValueError("conditional order races require unique race_id values")

    dates = sorted({str(race["race_date"]) for race in output})
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in output:
        by_date[str(race["race_date"])].append(race)

    transformed_by_id: dict[str, dict[str, float]] = {}
    daily: list[dict[str, Any]] = []
    for date_index, evaluation_date in enumerate(dates):
        if date_index < minimum_prior_days:
            daily.append({
                "race_date": evaluation_date,
                "status": "insufficient_prior_days",
                "prior_days": date_index,
                "required_prior_days": minimum_prior_days,
                "evaluated_races": 0,
            })
            continue
        prior_dates = dates[:date_index]
        validation_date = prior_dates[-1]
        fit_races = [
            race
            for prior_date in prior_dates[:-1]
            for race in by_date[prior_date]
        ]
        validation_races = list(by_date[validation_date])
        prior_races = [
            race for prior_date in prior_dates for race in by_date[prior_date]
        ]
        evaluation_races = list(by_date[evaluation_date])
        fit_scores, fit_orders, _fit_base, _fit_actual = _arrays(fit_races)
        validation_scores, _validation_orders, validation_base, validation_actual = (
            _arrays(validation_races)
        )
        prior_scores, prior_orders, _prior_base, _prior_actual = _arrays(
            prior_races
        )
        evaluation_scores, _evaluation_orders, evaluation_base, evaluation_actual = (
            _arrays(evaluation_races)
        )

        selected: tuple[float, float] | None = None
        selected_key: tuple[float, float, float, float] | None = None
        selection_rows: list[dict[str, Any]] = []
        for regularization in regularization_values:
            model, fit = fit_conditional_order(
                fit_scores,
                fit_orders,
                regularization=regularization,
            )
            adjusted = conditional_probabilities(validation_scores, model)
            for blend in blend_values:
                mixed = _blend(validation_base, adjusted, blend)
                log_loss, top5 = _metrics(mixed, validation_actual)
                key = (log_loss, -top5, -regularization, blend)
                selection_rows.append({
                    "regularization": regularization,
                    "blend": blend,
                    "log_loss": log_loss,
                    "top5_hit_rate": top5,
                    "fit_success": bool(fit["success"]),
                })
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected = (regularization, blend)
        if selected is None:
            raise ValueError("conditional order selection produced no candidate")
        regularization, blend = selected
        final_model, final_fit = fit_conditional_order(
            prior_scores,
            prior_orders,
            regularization=regularization,
            max_iterations=150,
        )
        if not bool(final_fit["success"]):
            raise ValueError(
                "prequential conditional order optimization did not converge: "
                + str(final_fit.get("message") or final_fit.get("status"))
            )
        adjusted = conditional_probabilities(evaluation_scores, final_model)
        mixed = _blend(evaluation_base, adjusted, blend)
        baseline_log_loss, baseline_top5 = _metrics(
            evaluation_base, evaluation_actual
        )
        conditional_log_loss, conditional_top5 = _metrics(
            mixed, evaluation_actual
        )
        for race, probabilities in zip(evaluation_races, mixed):
            transformed_by_id[str(race["race_id"])] = _probability_mapping(
                probabilities
            )
        daily.append({
            "race_date": evaluation_date,
            "status": "transformed",
            "prior_days": len(prior_dates),
            "fit_through": prior_dates[-2],
            "validation_date": validation_date,
            "fit_races": len(fit_races),
            "validation_races": len(validation_races),
            "evaluated_races": len(evaluation_races),
            "selected_regularization": regularization,
            "selected_blend": blend,
            "baseline_log_loss": baseline_log_loss,
            "conditional_log_loss": conditional_log_loss,
            "log_loss_difference": conditional_log_loss - baseline_log_loss,
            "baseline_top5_hit_rate": baseline_top5,
            "conditional_top5_hit_rate": conditional_top5,
            "top5_hit_rate_difference": conditional_top5 - baseline_top5,
            "selection_candidates": selection_rows,
            "final_fit": final_fit,
        })

    for race in output:
        transformed = transformed_by_id.get(str(race["race_id"]))
        if transformed is not None:
            race["model_probabilities"] = transformed
            race["model_probability_transform"] = (
                "strict_prior_conditional_order"
            )

    transformed_days = [
        row for row in daily if row["status"] == "transformed"
    ]
    total_races = sum(int(row["evaluated_races"]) for row in transformed_days)
    report = {
        "status": "evaluated" if transformed_days else "waiting",
        "method": (
            "last-prior-date selection and all-prior-date refit before each "
            "evaluation date"
        ),
        "minimum_prior_days": minimum_prior_days,
        "regularizations": list(regularization_values),
        "blends": list(blend_values),
        "available_days": len(dates),
        "transformed_days": len(transformed_days),
        "transformed_races": total_races,
        "baseline_log_loss": _weighted_mean(
            transformed_days, "baseline_log_loss", "evaluated_races"
        ),
        "conditional_log_loss": _weighted_mean(
            transformed_days, "conditional_log_loss", "evaluated_races"
        ),
        "baseline_top5_hit_rate": _weighted_mean(
            transformed_days, "baseline_top5_hit_rate", "evaluated_races"
        ),
        "conditional_top5_hit_rate": _weighted_mean(
            transformed_days, "conditional_top5_hit_rate", "evaluated_races"
        ),
        "improving_days": sum(
            float(row["log_loss_difference"]) < 0.0
            for row in transformed_days
        ),
        "daily": daily,
    }
    if transformed_days:
        report["log_loss_difference"] = (
            float(report["conditional_log_loss"])
            - float(report["baseline_log_loss"])
        )
        report["top5_hit_rate_difference"] = (
            float(report["conditional_top5_hit_rate"])
            - float(report["baseline_top5_hit_rate"])
        )
    return output, report


def _arrays(
    races: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not races:
        raise ValueError("conditional order fold must not be empty")
    base = np.asarray(
        [_probability_row(race.get("model_probabilities")) for race in races]
    )
    winner = np.zeros((len(races), 6), dtype=np.float64)
    for race_index, probabilities in enumerate(base):
        np.add.at(winner[race_index], COMBINATION_LANES[:, 0], probabilities)
    winner /= winner.sum(axis=1, keepdims=True)
    scores = np.log(np.clip(winner, EPSILON, 1.0))
    orders = np.asarray(
        [_actual_order(race.get("actual_combination")) for race in races],
        dtype=np.int64,
    )
    actual = np.asarray(
        [_actual_index(order) for order in orders],
        dtype=np.int64,
    )
    return scores, orders, base, actual


def _probability_row(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or len(value) != len(COMBINATION_LANES):
        raise ValueError(
            "conditional order requires 120 model probabilities per race"
        )
    try:
        row = np.asarray(
            [
                float(
                    value[
                        "-".join(
                            str(int(lane) + 1) for lane in combination
                        )
                    ]
                )
                for combination in COMBINATION_LANES
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("conditional order probabilities are invalid") from exc
    if (
        not np.all(np.isfinite(row))
        or np.any(row <= 0.0)
        or not np.isclose(row.sum(), 1.0, rtol=1e-8, atol=1e-10)
    ):
        raise ValueError(
            "conditional order probabilities must be positive and sum to one"
        )
    return row / row.sum()


def _actual_order(value: Any) -> np.ndarray:
    try:
        order = np.asarray(
            [int(item) - 1 for item in str(value).split("-")],
            dtype=np.int64,
        )
    except ValueError as exc:
        raise ValueError("actual_combination is invalid") from exc
    if (
        order.shape != (3,)
        or np.any(order < 0)
        or np.any(order >= 6)
        or len(set(order.tolist())) != 3
    ):
        raise ValueError("actual_combination must contain three distinct lanes")
    return order


def _actual_index(order: np.ndarray) -> int:
    matches = np.flatnonzero(np.all(COMBINATION_LANES == order, axis=1))
    if len(matches) != 1:
        raise ValueError("actual combination is missing from outcome space")
    return int(matches[0])


def _blend(
    baseline: np.ndarray, adjusted: np.ndarray, blend: float
) -> np.ndarray:
    result = (1.0 - blend) * baseline + blend * adjusted
    result /= result.sum(axis=1, keepdims=True)
    return result


def _metrics(
    probabilities: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    rows = np.arange(len(actual))
    log_loss = float(
        -np.log(
            np.clip(probabilities[rows, actual], EPSILON, 1.0)
        ).mean()
    )
    top5 = np.argpartition(probabilities, -5, axis=1)[:, -5:]
    return log_loss, float(np.mean(np.any(top5 == actual[:, None], axis=1)))


def _weighted_mean(
    rows: Sequence[dict[str, Any]], value_key: str, weight_key: str
) -> float | None:
    total_weight = sum(int(row[weight_key]) for row in rows)
    if total_weight <= 0:
        return None
    return sum(
        float(row[value_key]) * int(row[weight_key]) for row in rows
    ) / total_weight


def _probability_mapping(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "-".join(str(int(lane) + 1) for lane in combination): float(
            probability
        )
        for combination, probability in zip(COMBINATION_LANES, probabilities)
    }


def _empty_report(minimum_prior_days: int) -> dict[str, Any]:
    return {
        "status": "waiting",
        "method": (
            "last-prior-date selection and all-prior-date refit before each "
            "evaluation date"
        ),
        "minimum_prior_days": minimum_prior_days,
        "available_days": 0,
        "transformed_days": 0,
        "transformed_races": 0,
        "daily": [],
    }
