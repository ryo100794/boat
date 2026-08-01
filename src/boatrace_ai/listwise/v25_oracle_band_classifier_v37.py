from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .direct_context_market_residual_v25 import direct_context_probabilities
from .v25_top1_closing_policy_v36 import _top1_example
from .v25_top1_narrow_policy_v33 import simulate_v25_top1_narrow_v33


MODEL_NAME = "v25_top1_oracle_band_classifier_v37"
MIN_EV = 0.95
MAX_EV = 1.00
MAX_CLOSING_ODDS = 80.0
SELECTION_EV = 0.975


def _v37_example(
    race: Mapping[str, Any], probability_artifact: Mapping[str, Any]
) -> dict[str, Any] | None:
    base = _top1_example(race, probability_artifact)
    if base is None:
        return None
    probabilities = direct_context_probabilities(dict(race), probability_artifact)
    combination = str(base["combination"])
    probability = float(probabilities[combination])
    return {
        **base,
        "model_probability": probability,
        "current_ev": probability * float(base["current_odds"]),
        "closing_ev": probability * float(base["closing_odds"]),
    }


def _classification_vector(example: Mapping[str, Any]) -> np.ndarray:
    probability = max(float(example["model_probability"]), np.finfo(float).tiny)
    current_ev = max(float(example["current_ev"]), np.finfo(float).tiny)
    return np.concatenate(
        (
            np.asarray(example["features"], dtype=np.float64),
            np.asarray(
                [math.log(probability), current_ev, math.log(current_ev)],
                dtype=np.float64,
            ),
        )
    )


def _oracle_band_label(example: Mapping[str, Any]) -> int:
    probability = max(float(example["model_probability"]), np.finfo(float).tiny)
    return int(
        MIN_EV <= float(example["closing_ev"]) <= MAX_EV
        and float(example["closing_odds"]) <= MAX_CLOSING_ODDS
        and SELECTION_EV / probability <= MAX_CLOSING_ODDS
    )


def _fit_classifier(examples: list[dict[str, Any]]) -> dict[str, Any] | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not examples:
        return None
    labels = np.asarray([_oracle_band_label(row) for row in examples], dtype=np.int8)
    if len(np.unique(labels)) != 2:
        return None
    matrix = np.stack([_classification_vector(row) for row in examples])
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=0.10,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=20260801,
    )
    classifier.fit(scaler.fit_transform(matrix), labels)
    return {
        "scaler": scaler,
        "classifier": classifier,
        "training_examples": len(examples),
        "positive_examples": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "converged": int(classifier.n_iter_[0]) < int(classifier.max_iter),
    }


def _scores(examples: list[dict[str, Any]], fitted: Mapping[str, Any]) -> np.ndarray:
    if not examples:
        return np.asarray([], dtype=np.float64)
    matrix = np.stack([_classification_vector(row) for row in examples])
    return np.asarray(
        fitted["classifier"].predict_proba(
            fitted["scaler"].transform(matrix)
        )[:, 1],
        dtype=np.float64,
    )


def _select_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any] | None:
    if len(scores) < 50 or int(labels.sum()) < 3:
        return None
    rows: list[dict[str, Any]] = []
    for threshold in np.unique(np.quantile(scores, np.linspace(0.50, 0.98, 49))):
        selected = scores >= threshold
        count = int(selected.sum())
        if count < 5:
            continue
        true_positive = int(labels[selected].sum())
        precision = true_positive / count
        recall = true_positive / int(labels.sum())
        beta2 = 0.25
        f_beta = (
            (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
            if precision > 0.0 and recall > 0.0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "selected": count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f0_5": f_beta,
            }
        )
    return (
        max(
            rows,
            key=lambda row: (
                float(row["f0_5"]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["threshold"]),
            ),
        )
        if rows
        else None
    )


def fit_v25_oracle_band_classifier_v37(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    prediction_date: str,
    minimum_training_days: int = 3,
    minimum_training_races: int = 300,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        day = str(race.get("race_date") or "")
        if day < prediction_date:
            by_day[day].append(race)
    dates = sorted(by_day)
    examples_by_day = {
        day: [
            row
            for race in by_day[day]
            if (row := _v37_example(race, probability_artifact)) is not None
        ]
        for day in dates
    }
    examples = [row for day in dates for row in examples_by_day[day]]
    oof_scores: list[float] = []
    oof_labels: list[int] = []
    for index in range(1, len(dates)):
        training = [row for day in dates[:index] for row in examples_by_day[day]]
        validation = examples_by_day[dates[index]]
        fitted = _fit_classifier(training)
        if fitted is None or not validation:
            continue
        oof_scores.extend(_scores(validation, fitted).tolist())
        oof_labels.extend(_oracle_band_label(row) for row in validation)
    threshold = _select_threshold(
        np.asarray(oof_scores, dtype=np.float64),
        np.asarray(oof_labels, dtype=np.int8),
    )
    fitted = _fit_classifier(examples)
    ready = bool(
        len(dates) >= minimum_training_days
        and len(examples) >= minimum_training_races
        and fitted is not None
        and threshold is not None
    )
    return {
        "model": MODEL_NAME,
        "teacher": "official_closing_EV_in_[0.95,1.00]_and_odds_at_most_80",
        "uses_outcome_teacher": False,
        "uses_payout_teacher": False,
        "prediction_date": prediction_date,
        "trained_through_date": dates[-1] if dates else None,
        "training_days": len(dates),
        "training_races": len(examples),
        "strict_prior_boundary": all(day < prediction_date for day in dates),
        "oof_examples": len(oof_scores),
        "oof_positive_examples": int(sum(oof_labels)),
        "threshold_selection": threshold,
        "fitted": fitted,
        "ready": ready,
    }


def walk_forward_v25_oracle_band_classifier_v37(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    evaluation_dates: Iterable[str],
    initial_bankroll_yen: int = 10_000,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    transformed: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    total_selected = total_true_positive = total_positive = 0
    for evaluation_date in sorted(set(evaluation_dates)):
        holdout = by_day.get(evaluation_date, [])
        if not holdout:
            continue
        artifact = fit_v25_oracle_band_classifier_v37(
            races,
            probability_artifact=probability_artifact,
            prediction_date=evaluation_date,
        )
        examples = [
            row
            for race in holdout
            if (row := _v37_example(race, probability_artifact)) is not None
        ]
        selected_count = true_positive = positive_count = 0
        if artifact["ready"]:
            scores = _scores(examples, artifact["fitted"])
            threshold = float(artifact["threshold_selection"]["threshold"])
            rows = {str(row["race_id"]): row for row in examples}
            score_by_race = {
                str(row["race_id"]): float(score)
                for row, score in zip(examples, scores)
            }
            for race in holdout:
                row = rows.get(str(race["race_id"]))
                if row is None:
                    continue
                probability = float(row["model_probability"])
                synthetic_odds = SELECTION_EV / max(
                    probability, np.finfo(float).tiny
                )
                selected = bool(
                    score_by_race[str(race["race_id"])] >= threshold
                    and synthetic_odds <= MAX_CLOSING_ODDS
                )
                label = _oracle_band_label(row)
                item = dict(race)
                estimated = dict(race["odds"])
                estimated[str(row["combination"])] = (
                    synthetic_odds if selected else MIN_EV / 2.0 / probability
                )
                item["estimated_final_odds"] = estimated
                item["oracle_band_classifier_score"] = score_by_race[
                    str(race["race_id"])
                ]
                transformed.append(item)
                selected_count += int(selected)
                true_positive += int(selected and label)
                positive_count += label
        total_selected += selected_count
        total_true_positive += true_positive
        total_positive += positive_count
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "status": (
                    "strict_prior_oracle_band_classifier"
                    if artifact["ready"]
                    else "no_bet_classifier_not_ready"
                ),
                "trained_through_date": artifact["trained_through_date"],
                "training_days": artifact["training_days"],
                "training_races": artifact["training_races"],
                "oof_examples": artifact["oof_examples"],
                "threshold_selection": artifact["threshold_selection"],
                "predicted_positive": selected_count,
                "true_positive": true_positive,
                "actual_positive": positive_count,
                "precision": true_positive / selected_count if selected_count else None,
                "recall": true_positive / positive_count if positive_count else None,
            }
        )
    bankroll = simulate_v25_top1_narrow_v33(
        transformed,
        probability_artifact=probability_artifact,
        initial_bankroll_yen=initial_bankroll_yen,
    )
    return {
        **bankroll,
        "model": MODEL_NAME,
        "classification_predicted_positive": total_selected,
        "classification_true_positive": total_true_positive,
        "classification_actual_positive": total_positive,
        "classification_precision": (
            total_true_positive / total_selected if total_selected else None
        ),
        "classification_recall": (
            total_true_positive / total_positive if total_positive else None
        ),
        "folds": folds,
        "promotion_evidence": False,
        "status": "retrospective_diagnostic_only",
        "real_betting_enabled": False,
    }
