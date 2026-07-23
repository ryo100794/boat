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
) -> tuple[float | None, list[dict[str, Any]], dict[str, Any]]:
    if not half_lives:
        raise ValueError("at least one half-life candidate is required")
    inner_train_end, calibration_start, calibration_end = inner_calibration_boundary(
        dataset.race_keys,
        outer_train_end=outer_train_end,
        calibration_days=calibration_days,
    )
    candidates: list[dict[str, Any]] = []
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
        metrics, _predictions = score_range(
            dataset,
            bundle=bundle,
            race_start=inner_train_end,
            race_end=outer_train_end,
            batch_size=batch_size,
        )
        if not np.isfinite(float(metrics["entry_log_loss"])):
            raise ValueError("candidate calibration log loss is not finite")
        candidates.append(
            {
                "recency_half_life_days": half_life,
                "inner_train_races": inner_train_end,
                "calibration_races": outer_train_end - inner_train_end,
                "calibration_start": calibration_start,
                "calibration_end": calibration_end,
                **metrics,
            }
        )

    selected = min(
        candidates,
        key=lambda row: (
            float(row["entry_log_loss"]),
            0 if row["recency_half_life_days"] is None else 1,
            -float(row["recency_half_life_days"] or 0.0),
        ),
    )
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
) -> dict[str, Any]:
    started = time.perf_counter()
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
                drop_feature_groups=DROP_FEATURE_GROUPS,
            ),
            hasher=hasher,
            to_hashable=to_hashable,
            ensure_sparse_index32=_ensure_sparse_index32,
            drop_feature_groups=DROP_FEATURE_GROUPS,
            batch_size=batch_size,
        )
        validate_dataset_races(
            dataset,
            race_keys=race_keys,
            protocol=protocol,
            training_hash=training_hash,
        )

        selected_half_life, candidates, split = select_recency_half_life(
            dataset,
            outer_train_end=training_count,
            half_lives=half_lives,
            calibration_days=calibration_days,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
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
        "drop_feature_groups": list(DROP_FEATURE_GROUPS),
        "include_odds": False,
        "promotion_eligible": False,
        "promotion_note": "training-only half-life selection; no promotion decision",
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
        "evaluated_races": int(prediction_metrics["evaluated_races"]),
        **prediction_metrics,
        "policy": policy,
        "daily_budget_yen": DAILY_BUDGET_YEN,
        "bankroll": bankroll,
        "bankroll_evaluated_races": int(bankroll["evaluated_races"]),
        "daily": daily,
        **bankroll_flat,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json_atomic(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select MLP recency decay on training-only calibration data"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-date",
        type=parse_evaluation_date,
        required=True,
    )
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
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
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
