from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import brier_score_loss

from .adaptive_allocation import zero_totals
from .bankroll_backtest import _load_trifecta_payouts
from .calibrated_shadow_model import (
    FEATURE_SET,
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    safe_log_loss,
    score_dataset_fold,
    to_hashable,
    train_bundle_from_dataset,
)
from .db import connection
from .hashed_feature_dataset import HashedRaceDataset, load_or_build_hashed_dataset
from .fast_math import plackett_luce_probabilities
from .feature_tuning import normalize_drop_feature_groups
from .listwise.conditional_order import (
    DEFAULT_REGULARIZATIONS,
    ConditionalOrderModel,
    bankroll_promotion_gate,
    conditional_probabilities,
    evaluate_probabilities,
    fit_conditional_order,
)
from .listwise.direct_bankroll import (
    bootstrap_daily_bankroll,
    simulate_conditional_payout_walk_forward,
)
from .listwise.newton_refine import dump_joblib_atomic
from .listwise.validation import default_policy, evaluate_bankroll_fold
from .modeling import _race_level_metrics
from .standard_evaluation import (
    POLICY as STANDARD_POLICY,
    build_protocol,
    race_set_sha256,
    verify_protocol_against_database,
)


MODEL_NAME = "calibrated_mlp_recency_selected"
DROP_FEATURE_GROUPS = ("research_correlates",)
DEFAULT_FEATURE_CACHE = Path("data/models/calibrated_shadow_features_16384")
DEFAULT_HALF_LIVES: tuple[float | None, ...] = (None, 180.0, 365.0, 730.0)
DAILY_BUDGET_YEN = 10_000
EV_THRESHOLD = 1.20
N_FEATURES = 1 << 14
BATCH_SIZE = 12_000
EPOCHS = 2
ALPHA = 0.0001
ORDER_VALIDATION_DAYS = 60


def parse_evaluation_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation date must use YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("evaluation date must use YYYY-MM-DD")
    return parsed


def parse_drop_feature_groups(value: str) -> tuple[str, ...]:
    try:
        groups = normalize_drop_feature_groups(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not groups:
        raise argparse.ArgumentTypeError("at least one feature group must be dropped")
    return groups


def parse_half_lives(value: str) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if token == "none":
            half_life = None
        else:
            try:
                half_life = float(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid half-life: {raw}"
                ) from exc
            if not np.isfinite(half_life) or half_life <= 0.0:
                raise argparse.ArgumentTypeError(
                    "half-lives must be positive and finite"
                )
        if half_life not in parsed:
            parsed.append(half_life)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one half-life is required")
    return tuple(parsed)


@contextmanager
def frozen_evaluation_max_date(evaluation_date: date) -> Iterator[None]:
    previous = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = evaluation_date.isoformat()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BOATRACE_EVAL_MAX_RACE_DATE", None)
        else:
            os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = previous


def validated_protocol_race_keys(
    conn: Any,
    protocol: dict[str, Any],
) -> tuple[list[tuple[str, str, str, int]], str]:
    race_keys = load_complete_race_ids(conn)
    training_count = int(protocol["training_races"])
    holdout_count = int(protocol["prediction_races"])
    expected_total = training_count + holdout_count
    if len(race_keys) != expected_total:
        raise ValueError(
            "protocol complete race count mismatch: "
            f"expected {expected_total}, got {len(race_keys)}"
        )
    race_ids = [str(row[0]) for row in race_keys]
    if len(set(race_ids)) != expected_total:
        raise ValueError("protocol race IDs are not unique")

    training_rows = race_keys[:training_count]
    holdout_rows = race_keys[training_count:]
    holdout_start = str(protocol["holdout_start"])
    holdout_end = str(protocol["holdout_end"])
    if not training_rows:
        raise ValueError("standard evaluation protocol has no training races")
    if any(str(row[1]) >= holdout_start for row in training_rows):
        raise ValueError("protocol training races cross the holdout boundary")
    if any(
        not holdout_start <= str(row[1]) <= holdout_end
        for row in holdout_rows
    ):
        raise ValueError("protocol holdout races cross the frozen date boundary")

    holdout_hash = race_set_sha256(row[0] for row in holdout_rows)
    if holdout_hash != str(protocol["race_set_sha256"]):
        raise ValueError("protocol holdout race set hash mismatch")
    training_hash = race_set_sha256(row[0] for row in training_rows)
    return race_keys, training_hash


def validate_dataset_races(
    dataset: HashedRaceDataset,
    *,
    race_keys: Sequence[tuple[str, str, str, int]],
    protocol: dict[str, Any],
    training_hash: str,
) -> None:
    if list(dataset.race_keys) != list(race_keys):
        raise ValueError("cached dataset race sequence does not match the protocol")
    training_count = int(protocol["training_races"])
    if race_set_sha256(row[0] for row in dataset.race_keys[:training_count]) != training_hash:
        raise ValueError("cached dataset training race set hash mismatch")
    holdout_hash = race_set_sha256(row[0] for row in dataset.race_keys[training_count:])
    if holdout_hash != str(protocol["race_set_sha256"]):
        raise ValueError("cached dataset holdout race set hash mismatch")


def inner_calibration_boundary(
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    outer_train_end: int,
    calibration_days: int,
) -> tuple[int, str, str]:
    if calibration_days <= 0:
        raise ValueError("calibration_days must be positive")
    if outer_train_end <= 1 or outer_train_end > len(race_keys):
        raise ValueError("outer training fold is too small for calibration")
    training_rows = race_keys[:outer_train_end]
    try:
        end_date = date.fromisoformat(str(training_rows[-1][1]))
        parsed_dates = [date.fromisoformat(str(row[1])) for row in training_rows]
    except ValueError as exc:
        raise ValueError("race date must use YYYY-MM-DD") from exc
    if parsed_dates != sorted(parsed_dates):
        raise ValueError("race keys are not chronological")
    start_date = end_date - timedelta(days=calibration_days - 1)
    inner_train_end = next(
        (index for index, race_date in enumerate(parsed_dates) if race_date >= start_date),
        outer_train_end,
    )
    if inner_train_end <= 0 or inner_train_end >= outer_train_end:
        raise ValueError("calendar calibration split leaves an empty inner fold")
    return inner_train_end, start_date.isoformat(), end_date.isoformat()


def score_range(
    dataset: HashedRaceDataset,
    *,
    bundle: dict[str, Any],
    race_start: int,
    race_end: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    labels: list[int] = []
    probabilities: list[float] = []
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in score_dataset_fold(
        dataset,
        bundle=bundle,
        race_start=race_start,
        race_end=race_end,
        batch_size=batch_size,
    ):
        for row in rows:
            label = int(row["label"])
            probability = float(row["probability"])
            labels.append(label)
            probabilities.append(probability)
            predictions[str(row["race_id"])].append(dict(row))

    expected_ids = {
        str(row[0]) for row in dataset.race_keys[race_start:race_end]
    }
    if set(predictions) != expected_ids:
        raise ValueError("scored race set does not match the requested range")
    if len(labels) != len(expected_ids) * 6:
        raise ValueError("scored entry count does not match the requested range")
    if any(len(rows) != 6 for rows in predictions.values()):
        raise ValueError("scored range contains incomplete races")
    metrics = {
        "entry_log_loss": safe_log_loss(labels, probabilities),
        "entry_brier": float(brier_score_loss(labels, probabilities)),
        **_race_level_metrics(predictions),
    }
    return metrics, dict(predictions)


def select_recency_half_life(
    dataset: HashedRaceDataset,
    *,
    outer_train_end: int,
    half_lives: Sequence[float | None],
    calibration_days: int,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    alpha: float = ALPHA,
    prediction_output: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[float | None, list[dict[str, Any]], dict[str, Any]]:
    if not half_lives:
        raise ValueError("at least one half-life candidate is required")
    inner_train_end, calibration_start, calibration_end = inner_calibration_boundary(
        dataset.race_keys,
        outer_train_end=outer_train_end,
        calibration_days=calibration_days,
    )
    candidates: list[dict[str, Any]] = []
    selected_key: tuple[float, int, float] | None = None
    selected_predictions: dict[str, list[dict[str, Any]]] = {}
    for half_life in half_lives:
        bundle = train_bundle_from_dataset(
            dataset,
            train_race_count=inner_train_end,
            model_kind="mlp",
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
            recency_half_life_days=half_life,
        )
        metrics, candidate_predictions = score_range(
            dataset,
            bundle=bundle,
            race_start=inner_train_end,
            race_end=outer_train_end,
            batch_size=batch_size,
        )
        if not np.isfinite(float(metrics["entry_log_loss"])):
            raise ValueError("candidate calibration log loss is not finite")
        candidate = {
            "recency_half_life_days": half_life,
            "inner_train_races": inner_train_end,
            "calibration_races": outer_train_end - inner_train_end,
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            **metrics,
        }
        candidates.append(candidate)
        candidate_key = (
            float(candidate["entry_log_loss"]),
            0 if half_life is None else 1,
            -float(half_life or 0.0),
        )
        if selected_key is None or candidate_key < selected_key:
            selected_key = candidate_key
            selected_predictions = candidate_predictions

    selected = min(
        candidates,
        key=lambda row: (
            float(row["entry_log_loss"]),
            0 if row["recency_half_life_days"] is None else 1,
            -float(row["recency_half_life_days"] or 0.0),
        ),
    )
    if prediction_output is not None:
        prediction_output.clear()
        prediction_output.update(selected_predictions)
    split = {
        "inner_train_races": inner_train_end,
        "calibration_races": outer_train_end - inner_train_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
    }
    return selected["recency_half_life_days"], candidates, split


def bankroll_summary(
    conn: Any,
    *,
    predictions: dict[str, list[dict[str, Any]]],
    training_races: set[str],
    test_dates: set[str],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    policy = default_policy(
        daily_budget_yen=DAILY_BUDGET_YEN,
        ev_threshold=EV_THRESHOLD,
    )
    policy["require_real_odds"] = False
    for key, expected in STANDARD_POLICY.items():
        if policy.get(key) != expected:
            raise ValueError(f"fixed policy mismatch: {key}")
    policy.update({"model": MODEL_NAME, "feature_set": FEATURE_SET})
    totals = zero_totals()
    daily: list[dict[str, Any]] = []
    bankroll, profit_state = evaluate_bankroll_fold(
        rows_by_race=predictions,
        train_races=training_races,
        test_dates=test_dates,
        payouts=_load_trifecta_payouts(conn),
        policy=policy,
        totals=totals,
        daily_rows=daily,
        profit_state=(0, 0, 0),
    )
    evaluated_races = int(totals["evaluated_races"])
    if evaluated_races != int(protocol["bankroll_evaluable_races"]):
        raise ValueError("bankroll evaluable race count does not match the protocol")
    daily_dates = [str(row["race_date"]) for row in daily]
    if len(daily_dates) != int(protocol["holdout_date_count"]):
        raise ValueError("bankroll daily row count does not match the protocol")
    if not daily_dates or daily_dates[0] != str(protocol["holdout_start"]):
        raise ValueError("bankroll daily start does not match the protocol")
    if daily_dates[-1] != str(protocol["holdout_end"]):
        raise ValueError("bankroll daily end does not match the protocol")

    tickets = int(totals["tickets"])
    selected_races = int(totals["races_bet"])
    stake_yen = int(bankroll["stake_yen"])
    summary = {
        **bankroll,
        "evaluated_races": evaluated_races,
        "race_days": len(daily),
        "selected_races": selected_races,
        "tickets": tickets,
        "hit_tickets": int(totals["hit_tickets"]),
        "ticket_hit_rate": (
            float(totals["hit_tickets"]) / tickets if tickets else 0.0
        ),
        "race_hit_rate": (
            float(totals["hit_races"]) / selected_races
            if selected_races
            else 0.0
        ),
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "budget_utilization": (
            stake_yen / (DAILY_BUDGET_YEN * len(daily)) if daily else 0.0
        ),
        "max_drawdown_yen": int(profit_state[2]),
        "evaluation_race_set_sha256": str(protocol["race_set_sha256"]),
    }
    return policy, summary, daily


def lane_probability_matrix(
    predictions: dict[str, list[dict[str, Any]]],
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    context: str = "trifecta",
) -> np.ndarray:
    matrix = np.empty((len(race_keys), 6), dtype=np.float64)
    for race_index, race_key in enumerate(race_keys):
        rows = predictions.get(str(race_key[0]))
        if rows is None or len(rows) != 6:
            raise ValueError(f"{context} probability input contains an incomplete race")
        lane_probabilities = {
            int(row["lane"]): float(row["probability"])
            for row in rows
        }
        if set(lane_probabilities) != set(range(1, 7)):
            raise ValueError(f"{context} probability input has invalid lanes")
        values = np.asarray(
            [lane_probabilities[lane] for lane in range(1, 7)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"{context} lane probabilities must be finite and non-negative"
            )
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError(f"{context} lane probabilities must have a positive sum")
        matrix[race_index] = values / total
    return matrix


def trifecta_probability_matrix(
    predictions: dict[str, list[dict[str, Any]]],
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    conditional_order_model: ConditionalOrderModel | None = None,
) -> np.ndarray:
    lane_matrix = lane_probability_matrix(predictions, race_keys)
    if conditional_order_model is None:
        matrix = np.asarray(
            [plackett_luce_probabilities(row) for row in lane_matrix],
            dtype=np.float64,
        )
    else:
        matrix = conditional_probabilities(
            np.log(np.clip(lane_matrix, 1e-15, 1.0)),
            conditional_order_model,
        )
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("trifecta probabilities do not sum to one")
    return matrix


def fit_conditional_order_layer(
    predictions: dict[str, list[dict[str, Any]]],
    race_keys: Sequence[tuple[str, str, str, int]],
    ranks: np.ndarray,
    *,
    validation_days: int = ORDER_VALIDATION_DAYS,
    regularizations: Sequence[float] = DEFAULT_REGULARIZATIONS,
) -> tuple[ConditionalOrderModel, dict[str, Any]]:
    if validation_days <= 0:
        raise ValueError("conditional order validation days must be positive")
    if len(race_keys) != len(ranks):
        raise ValueError("conditional order race keys and ranks must align")
    if not race_keys:
        raise ValueError("conditional order calibration rows must not be empty")
    candidates = tuple(sorted({float(value) for value in regularizations}))
    if not candidates or any(
        not np.isfinite(value) or value < 0.0 for value in candidates
    ):
        raise ValueError(
            "conditional order regularizations must be finite and non-negative"
        )
    dates = [date.fromisoformat(str(row[1])) for row in race_keys]
    if dates != sorted(dates):
        raise ValueError("conditional order calibration races must be chronological")
    validation_start_date = dates[-1] - timedelta(days=validation_days - 1)
    validation_start = next(
        (
            index
            for index, race_date in enumerate(dates)
            if race_date >= validation_start_date
        ),
        len(dates),
    )
    if validation_start <= 0 or validation_start >= len(race_keys):
        raise ValueError("conditional order validation split leaves an empty fold")

    lane_probabilities = lane_probability_matrix(
        predictions,
        race_keys,
        context="conditional order calibration",
    )
    scores = np.log(np.clip(lane_probabilities, 1e-15, 1.0))
    rank_values = np.asarray(ranks, dtype=np.int8)
    orders = np.argsort(rank_values, axis=1)[:, :3]

    diagnostics: list[dict[str, Any]] = []
    selected_model: ConditionalOrderModel | None = None
    selected_key: tuple[float, float, float] | None = None
    for regularization in candidates:
        model, fit = fit_conditional_order(
            scores[:validation_start],
            orders[:validation_start],
            regularization=regularization,
        )
        metrics = evaluate_probabilities(
            conditional_probabilities(scores[validation_start:], model),
            rank_values[validation_start:],
        )
        public_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"race_losses", "race_top5_hits"}
        }
        diagnostics.append(
            {
                "regularization": regularization,
                "fit": fit,
                "validation": public_metrics,
            }
        )
        candidate_key = (
            float(metrics["trifecta_log_loss"]),
            -float(metrics["trifecta_top5_hit_rate"]),
            -regularization,
        )
        if selected_key is None or candidate_key < selected_key:
            selected_key = candidate_key
            selected_model = model
    if selected_model is None:
        raise ValueError("conditional order selection produced no model")
    selected_regularization = float(selected_model.regularization)
    final_model, final_fit = fit_conditional_order(
        scores,
        orders,
        regularization=selected_regularization,
    )
    return final_model, {
        "scope": "outer training calibration only; final 365-day holdout untouched",
        "fit_from": str(race_keys[0][1]),
        "fit_through": str(race_keys[validation_start - 1][1]),
        "validation_from": str(race_keys[validation_start][1]),
        "validation_through": str(race_keys[-1][1]),
        "fit_races": validation_start,
        "validation_races": len(race_keys) - validation_start,
        "selected_regularization": selected_regularization,
        "selection_criterion": (
            "minimum trifecta log loss; top5 and stronger regularization tie-breaks"
        ),
        "candidates": diagnostics,
        "final_fit": final_fit,
    }


def fit_deployment_conditional_order_layer(
    predictions: dict[str, list[dict[str, Any]]],
    race_keys: Sequence[tuple[str, str, str, int]],
    ranks: np.ndarray,
    *,
    regularization: float,
) -> tuple[ConditionalOrderModel, dict[str, Any]]:
    if len(race_keys) != len(ranks) or not race_keys:
        raise ValueError("deployment order rows must be non-empty and aligned")
    lane_probabilities = lane_probability_matrix(
        predictions,
        race_keys,
        context="deployment conditional order",
    )
    scores = np.log(np.clip(lane_probabilities, 1e-15, 1.0))
    rank_values = np.asarray(ranks, dtype=np.int8)
    orders = np.argsort(rank_values, axis=1)[:, :3]
    model, fit = fit_conditional_order(
        scores,
        orders,
        regularization=float(regularization),
        max_iterations=200,
    )
    if not bool(fit.get("success")):
        raise ValueError(
            "deployment conditional order optimization did not converge: "
            + str(fit.get("message") or fit.get("status") or "unknown")
        )
    return model, {
        "scope": (
            "post-evaluation deployment refit on untouched holdout predictions; "
            "evaluation metrics not recomputed"
        ),
        "fit_from": str(race_keys[0][1]),
        "fit_through": str(race_keys[-1][1]),
        "fit_races": len(race_keys),
        "regularization": float(regularization),
        "fit": fit,
    }


def load_incumbent_evaluation(
    prediction_path: Path,
    bankroll_path: Path,
    *,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        bankroll = json.loads(bankroll_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("incumbent evaluation artifacts are unreadable") from exc
    if not isinstance(prediction, dict) or not isinstance(bankroll, dict):
        raise ValueError("incumbent evaluation artifacts must be objects")
    expected_hash = str(protocol["race_set_sha256"])
    expected_races = int(protocol["prediction_races"])
    prediction_hash = str(prediction.get("evaluation_race_set_sha256") or "")
    bankroll_hash = str(bankroll.get("evaluation_race_set_sha256") or "")
    if prediction_hash != expected_hash or bankroll_hash != expected_hash:
        raise ValueError("incumbent evaluation race set hash mismatch")
    prediction_races = int(prediction.get("evaluated_races") or 0)
    bankroll_races = int(
        bankroll.get("evaluated_races")
        or bankroll.get("bankroll_evaluated_races")
        or 0
    )
    if prediction_races != expected_races or bankroll_races != int(
        protocol["bankroll_evaluable_races"]
    ):
        raise ValueError("incumbent evaluation race count mismatch")
    if not isinstance(bankroll.get("daily"), list) or not bankroll["daily"]:
        raise ValueError("incumbent bankroll lacks daily rows")
    return prediction, bankroll


def prediction_promotion_gate(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    evaluation_hash: str,
    evaluated_races: int,
) -> dict[str, Any]:
    if incumbent is None:
        return {
            "comparison_ready": False,
            "entry_log_loss_not_worse": False,
            "winner_top1_not_worse": False,
            "trifecta_top5_not_worse": False,
            "pass": False,
        }
    same_hash = str(incumbent.get("evaluation_race_set_sha256") or "") == str(
        evaluation_hash
    )
    same_races = int(incumbent.get("evaluated_races") or 0) == int(
        evaluated_races
    )
    checks = {
        "comparison_ready": bool(same_hash and same_races),
        "entry_log_loss_not_worse": float(candidate["entry_log_loss"])
        <= float(incumbent["entry_log_loss"]),
        "winner_top1_not_worse": float(candidate["winner_top1_accuracy"])
        >= float(incumbent["winner_top1_accuracy"]),
        "trifecta_top5_not_worse": float(candidate["trifecta_top5_hit_rate"])
        >= float(incumbent["trifecta_top5_hit_rate"]),
    }
    checks["pass"] = bool(all(checks.values()))
    return {
        **checks,
        "candidate": {
            key: candidate[key]
            for key in (
                "entry_log_loss",
                "winner_top1_accuracy",
                "trifecta_top5_hit_rate",
            )
        },
        "incumbent": {
            key: incumbent[key]
            for key in (
                "entry_log_loss",
                "winner_top1_accuracy",
                "trifecta_top5_hit_rate",
            )
        },
    }


def conditional_payout_summary(
    conn: Any,
    *,
    race_keys: Sequence[tuple[str, str, str, int]],
    training_count: int,
    inner_train_count: int,
    calibration_predictions: dict[str, list[dict[str, Any]]],
    holdout_predictions: dict[str, list[dict[str, Any]]],
    baseline_bankroll: dict[str, Any],
    baseline_daily: list[dict[str, Any]],
    protocol: dict[str, Any],
    conditional_order_model: ConditionalOrderModel | None = None,
) -> dict[str, Any]:
    calibration_keys = list(race_keys[inner_train_count:training_count])
    holdout_keys = list(race_keys[training_count:])
    calibration_probabilities = trifecta_probability_matrix(
        calibration_predictions,
        calibration_keys,
        conditional_order_model=conditional_order_model,
    )
    holdout_probabilities = trifecta_probability_matrix(
        holdout_predictions,
        holdout_keys,
        conditional_order_model=conditional_order_model,
    )
    candidate = simulate_conditional_payout_walk_forward(
        holdout_probabilities,
        race_keys=holdout_keys,
        payouts=_load_trifecta_payouts(conn),
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
    )
    if int(candidate["evaluated_races"]) != int(
        protocol["bankroll_evaluable_races"]
    ):
        raise ValueError("conditional payout race count does not match the protocol")
    confidence = bootstrap_daily_bankroll(
        candidate["daily"],
        baseline_daily=baseline_daily,
    )
    gate = bankroll_promotion_gate(candidate, baseline_bankroll, confidence)
    return {
        "role": "untouched 365-day payout-policy candidate",
        "promotion_eligible": bool(gate["pass"]),
        "bankroll": candidate,
        "bankroll_confidence": confidence,
        "diagnostic_gate": gate,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_recency_mlp(
    conn: Any,
    *,
    output_path: Path,
    evaluation_date: date,
    feature_cache: Path | None = DEFAULT_FEATURE_CACHE,
    half_lives: Sequence[float | None] = DEFAULT_HALF_LIVES,
    calibration_days: int = 180,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    alpha: float = ALPHA,
    drop_feature_groups: Sequence[str] = DROP_FEATURE_GROUPS,
    model_output_path: Path | None = None,
    deployment_model_output_path: Path | None = None,
    incumbent_prediction_path: Path | None = None,
    incumbent_bankroll_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved_drop_feature_groups = normalize_drop_feature_groups(
        drop_feature_groups
    )
    if not resolved_drop_feature_groups:
        raise ValueError("at least one feature group must be dropped")
    with frozen_evaluation_max_date(evaluation_date):
        protocol = build_protocol(
            conn,
            as_of_date=evaluation_date + timedelta(days=1),
        )
        if int(protocol["calendar_days"]) != 365:
            raise ValueError("recency evaluation requires the standard 365-day protocol")
        verify_protocol_against_database(conn, protocol)
        race_keys, training_hash = validated_protocol_race_keys(conn, protocol)
        training_count = int(protocol["training_races"])
        if (incumbent_prediction_path is None) != (incumbent_bankroll_path is None):
            raise ValueError("both incumbent evaluation paths are required")
        incumbent_prediction: dict[str, Any] | None = None
        incumbent_bankroll: dict[str, Any] | None = None
        if incumbent_prediction_path is not None and incumbent_bankroll_path is not None:
            incumbent_prediction, incumbent_bankroll = load_incumbent_evaluation(
                incumbent_prediction_path,
                incumbent_bankroll_path,
                protocol=protocol,
            )

        hasher = FeatureHasher(
            n_features=N_FEATURES,
            input_type="dict",
            alternate_sign=False,
        )
        dataset, cache_source = load_or_build_hashed_dataset(
            cache_prefix=feature_cache,
            race_keys=race_keys,
            race_rows=lambda: iter_race_feature_rows(
                conn,
                include_races={str(row[0]) for row in race_keys},
                drop_feature_groups=resolved_drop_feature_groups,
            ),
            hasher=hasher,
            to_hashable=to_hashable,
            ensure_sparse_index32=_ensure_sparse_index32,
            drop_feature_groups=resolved_drop_feature_groups,
            batch_size=batch_size,
        )
        validate_dataset_races(
            dataset,
            race_keys=race_keys,
            protocol=protocol,
            training_hash=training_hash,
        )

        calibration_predictions: dict[str, list[dict[str, Any]]] = {}
        selected_half_life, candidates, split = select_recency_half_life(
            dataset,
            outer_train_end=training_count,
            half_lives=half_lives,
            calibration_days=calibration_days,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
            prediction_output=calibration_predictions,
        )
        final_bundle = train_bundle_from_dataset(
            dataset,
            train_race_count=training_count,
            model_kind="mlp",
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
            recency_half_life_days=selected_half_life,
        )
        prediction_metrics, predictions = score_range(
            dataset,
            bundle=final_bundle,
            race_start=training_count,
            race_end=dataset.race_count,
            batch_size=batch_size,
        )
        calibration_start = int(split["inner_train_races"])
        conditional_order_model, conditional_order_selection = (
            fit_conditional_order_layer(
                calibration_predictions,
                race_keys[calibration_start:training_count],
                dataset.ranks[calibration_start:training_count],
            )
        )
        holdout_trifecta_probabilities = trifecta_probability_matrix(
            predictions,
            race_keys[training_count:],
            conditional_order_model=conditional_order_model,
        )
        holdout_trifecta_metrics = evaluate_probabilities(
            holdout_trifecta_probabilities,
            dataset.ranks[training_count:],
        )
        prediction_metrics.update(
            {
                key: value
                for key, value in holdout_trifecta_metrics.items()
                if key not in {"race_losses", "race_top5_hits"}
            }
        )
        evaluation_hash = race_set_sha256(predictions)
        if evaluation_hash != str(protocol["race_set_sha256"]):
            raise ValueError("final holdout race set hash does not match the protocol")
        if int(prediction_metrics["evaluated_races"]) != int(protocol["prediction_races"]):
            raise ValueError("final holdout race count does not match the protocol")

        training_races = {str(row[0]) for row in race_keys[:training_count]}
        test_dates = {str(row[1]) for row in race_keys[training_count:]}
        policy, bankroll, daily = bankroll_summary(
            conn,
            predictions=predictions,
            training_races=training_races,
            test_dates=test_dates,
            protocol=protocol,
        )
        conditional_payout = conditional_payout_summary(
            conn,
            race_keys=race_keys,
            training_count=training_count,
            inner_train_count=int(split["inner_train_races"]),
            calibration_predictions=calibration_predictions,
            holdout_predictions=predictions,
            baseline_bankroll=incumbent_bankroll or bankroll,
            baseline_daily=(incumbent_bankroll or {}).get("daily") or daily,
            protocol=protocol,
            conditional_order_model=conditional_order_model,
        )
        prediction_gate = prediction_promotion_gate(
            prediction_metrics,
            incumbent_prediction,
            evaluation_hash=evaluation_hash,
            evaluated_races=int(protocol["prediction_races"]),
        )
        performance_gate = {
            "prediction_pass": bool(prediction_gate["pass"]),
            "payout_policy_pass": bool(conditional_payout["promotion_eligible"]),
            "conditional_order_converged": bool(
                (conditional_order_selection.get("final_fit") or {}).get("success")
            ),
        }
        performance_gate["pass"] = bool(all(performance_gate.values()))
        promotion_gate = {
            **performance_gate,
            "performance_pass": bool(performance_gate["pass"]),
            "deployable_artifact_pass": False,
            "pass": False,
        }

    bankroll_flat = {
        key: value for key, value in bankroll.items() if key != "evaluated_races"
    }
    result = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_date": evaluation_date.isoformat(),
        "model": MODEL_NAME,
        "model_kind": "mlp",
        "role": "shadow",
        "feature_set": FEATURE_SET,
        "feature_schema_version": dataset.feature_schema_version,
        "drop_feature_groups": list(resolved_drop_feature_groups),
        "include_odds": False,
        "performance_eligible": bool(performance_gate["pass"]),
        "promotion_eligible": False,
        "promotion_note": (
            "performance gates passed; deployable full-data artifact is pending"
            if performance_gate["pass"]
            else "promotion requires prediction, payout, and order-convergence gates"
        ),
        "promotion_gate": promotion_gate,
        "prediction_promotion_gate": prediction_gate,
        "protocol": protocol,
        "training_races": training_count,
        "training_race_set_sha256": training_hash,
        "holdout_races": int(protocol["prediction_races"]),
        "evaluation_race_set_sha256": evaluation_hash,
        "feature_cache_source": cache_source,
        "feature_cache_prefix": str(feature_cache) if feature_cache else None,
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_nnz": int(dataset.matrix.nnz),
        "n_features": N_FEATURES,
        "epochs": max(1, int(epochs)),
        "alpha": float(alpha),
        "selection": {
            "criterion": "inner calibration entry_log_loss",
            "tie_break": "None first, otherwise longer half-life",
            "scope": "outer training only; final 365-day holdout untouched",
            "calibration_days": int(calibration_days),
            **split,
            "candidates": candidates,
            "selected_recency_half_life_days": selected_half_life,
        },
        "selection_candidates": candidates,
        "selected_recency_half_life_days": selected_half_life,
        "conditional_order": conditional_order_selection,
        "evaluated_races": int(prediction_metrics["evaluated_races"]),
        **prediction_metrics,
        "policy": policy,
        "daily_budget_yen": DAILY_BUDGET_YEN,
        "bankroll": bankroll,
        "bankroll_evaluated_races": int(bankroll["evaluated_races"]),
        "daily": daily,
        "conditional_payout_walk_forward": conditional_payout,
        **bankroll_flat,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if model_output_path is not None:
        trained_through = race_keys[training_count - 1]
        artifact = {
            **final_bundle,
            "hasher": hasher,
            "conditional_order_model": conditional_order_model,
            "feature_schema_version": dataset.feature_schema_version,
            "drop_feature_groups": list(resolved_drop_feature_groups),
            "trained_through": trained_through,
            "training_races": training_count,
            "training_race_set_sha256": training_hash,
            "metadata": {
                "trained_at": result["generated_at"],
                "model": MODEL_NAME,
                "role": "shadow",
                "feature_set": FEATURE_SET,
                "feature_schema_version": dataset.feature_schema_version,
                "drop_feature_groups": list(resolved_drop_feature_groups),
                "trained_through": list(trained_through),
                "training_races": training_count,
                "training_race_set_sha256": training_hash,
                "evaluation_races": int(protocol["prediction_races"]),
                "evaluation_race_set_sha256": evaluation_hash,
                "recency_half_life_days": selected_half_life,
                "conditional_order_regularization": (
                    conditional_order_selection["selected_regularization"]
                ),
                "include_odds": False,
            },
        }
        dump_joblib_atomic(model_output_path, artifact)
        result["model_artifact"] = str(model_output_path)
        result["model_artifact_saved"] = bool(
            model_output_path.is_file() and model_output_path.stat().st_size > 0
        )

    deployment_saved = False
    if performance_gate["pass"] and deployment_model_output_path is not None:
        deployment_order_model, deployment_order_fit = (
            fit_deployment_conditional_order_layer(
                predictions,
                race_keys[training_count:],
                dataset.ranks[training_count:],
                regularization=float(
                    conditional_order_selection["selected_regularization"]
                ),
            )
        )
        deployment_bundle = train_bundle_from_dataset(
            dataset,
            train_race_count=dataset.race_count,
            model_kind="mlp",
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
            recency_half_life_days=selected_half_life,
        )
        deployment_training_hash = race_set_sha256(
            row[0] for row in race_keys
        )
        deployment_trained_through = race_keys[-1]
        deployment_artifact = {
            **deployment_bundle,
            "hasher": hasher,
            "conditional_order_model": deployment_order_model,
            "feature_schema_version": dataset.feature_schema_version,
            "drop_feature_groups": list(resolved_drop_feature_groups),
            "trained_through": deployment_trained_through,
            "training_races": dataset.race_count,
            "training_race_set_sha256": deployment_training_hash,
            "metadata": {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_NAME,
                "role": "production_candidate",
                "feature_set": FEATURE_SET,
                "feature_schema_version": dataset.feature_schema_version,
                "drop_feature_groups": list(resolved_drop_feature_groups),
                "trained_through": list(deployment_trained_through),
                "training_races": dataset.race_count,
                "training_race_set_sha256": deployment_training_hash,
                "evaluation_date": evaluation_date.isoformat(),
                "evaluation_races": int(protocol["prediction_races"]),
                "evaluation_race_set_sha256": evaluation_hash,
                "recency_half_life_days": selected_half_life,
                "conditional_order_regularization": (
                    conditional_order_selection["selected_regularization"]
                ),
                "include_odds": False,
            },
        }
        dump_joblib_atomic(deployment_model_output_path, deployment_artifact)
        deployment_saved = bool(
            deployment_model_output_path.is_file()
            and deployment_model_output_path.stat().st_size > 0
        )
        result["deployment_model_artifact"] = str(deployment_model_output_path)
        result["deployment_model_artifact_saved"] = deployment_saved
        result["deployment_refit"] = {
            "trained_through": list(deployment_trained_through),
            "training_races": dataset.race_count,
            "training_race_set_sha256": deployment_training_hash,
            "conditional_order": deployment_order_fit,
        }
    elif deployment_model_output_path is not None:
        result["deployment_model_artifact"] = str(deployment_model_output_path)
        result["deployment_model_artifact_saved"] = False

    promotion_gate["deployable_artifact_pass"] = deployment_saved
    promotion_gate["pass"] = bool(performance_gate["pass"] and deployment_saved)
    result["promotion_eligible"] = bool(promotion_gate["pass"])
    result["promotion_note"] = (
        "all performance gates passed and full-data deployment artifact persisted"
        if promotion_gate["pass"]
        else result["promotion_note"]
    )
    write_json_atomic(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select MLP recency decay on training-only calibration data"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--deployment-model-output", type=Path)
    parser.add_argument("--incumbent-prediction", type=Path)
    parser.add_argument("--incumbent-bankroll", type=Path)
    parser.add_argument(
        "--evaluation-date",
        type=parse_evaluation_date,
        required=True,
    )
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--drop-feature-groups",
        type=parse_drop_feature_groups,
        default=DROP_FEATURE_GROUPS,
    )
    parser.add_argument(
        "--half-lives",
        type=parse_half_lives,
        default=DEFAULT_HALF_LIVES,
        help="comma-separated values such as none,180,365,730",
    )
    parser.add_argument("--calibration-days", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with connection(args.db) as conn:
        result = evaluate_recency_mlp(
            conn,
            output_path=args.output,
            evaluation_date=args.evaluation_date,
            feature_cache=args.feature_cache,
            half_lives=args.half_lives,
            calibration_days=args.calibration_days,
            drop_feature_groups=args.drop_feature_groups,
            model_output_path=args.model_output,
            deployment_model_output_path=args.deployment_model_output,
            incumbent_prediction_path=args.incumbent_prediction,
            incumbent_bankroll_path=args.incumbent_bankroll,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
