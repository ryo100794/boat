from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np

from ..db import connection
from ..fast_math import TRIFECTA_COMBINATIONS
from ..feature_tuning import to_hashable
from ..features import MODEL_DECISION_LEAD_MINUTES, latest_trifecta_odds_before_deadline
from ..odds_quality import plausible_trifecta_odds
from .four_head_nested_v22 import (
    DecisionRace,
    LabeledRace,
    RaceOutcome,
    artifact_fingerprint,
    evaluate_outer_outcomes,
    fit_four_head_nested_v22,
)
from .four_head_v22_bankroll import (
    V22BankrollSettlement,
    evaluate_four_head_v22_bankroll,
)
from .market_calibration import (
    MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    iter_scored_artifact_feature_rows,
    prefetch_official_closing_odds,
    prefetch_trifecta_snapshots,
    snapshot_age_seconds,
)


COMBINATIONS = tuple("-".join(map(str, value)) for value in TRIFECTA_COMBINATIONS)
COMBINATION_INDEX = {value: index for index, value in enumerate(COMBINATIONS)}
DEFAULT_PROJECTION_DIMENSIONS = 8
PAYOUT_BET_TYPE = "3連単"


@dataclass(frozen=True)
class DecisionAudit:
    race_id: str
    snapshot_id: int | None
    captured_at: str
    target_at: str
    age_seconds: float
    choices: int = 120


@dataclass(frozen=True)
class V22EvaluationData:
    training_races: tuple[LabeledRace, ...]
    outer_races: tuple[LabeledRace, ...]
    decision_audit: tuple[DecisionAudit, ...]
    diagnostics: Mapping[str, Any]


def load_or_build_v22_evaluation_data(
    cache_path: Path,
    *,
    signature: Mapping[str, Any],
    builder: Callable[[], V22EvaluationData],
) -> V22EvaluationData:
    """Reuse immutable period inputs while serializing concurrent builders."""

    path = cache_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            try:
                envelope = joblib.load(path)
            except Exception:
                envelope = None
            if (
                isinstance(envelope, dict)
                and envelope.get("schema_version") == 1
                and envelope.get("signature") == dict(signature)
                and isinstance(envelope.get("data"), V22EvaluationData)
            ):
                return envelope["data"]
        data = builder()
        if not isinstance(data, V22EvaluationData):
            raise TypeError("V22 evaluation data builder returned an invalid value")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            joblib.dump(
                {
                    "schema_version": 1,
                    "signature": dict(signature),
                    "data": data,
                },
                temporary,
                compress=3,
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return data


def _as_aware_datetime(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _validate_periods(
    *,
    training_from_date: str,
    training_through_date: str,
    outer_from_date: str,
    outer_through_date: str,
) -> None:
    if training_from_date > training_through_date:
        raise ValueError("training period is reversed")
    if outer_from_date > outer_through_date:
        raise ValueError("outer period is reversed")
    if training_through_date >= outer_from_date:
        raise ValueError("outer period must be strictly after the training period")


def _validate_source_artifact(artifact: Mapping[str, Any], earliest_date: str) -> None:
    trained_through = artifact.get("trained_through")
    if not isinstance(trained_through, (tuple, list)) or len(trained_through) < 2:
        raise ValueError("source artifact lacks trained_through leakage metadata")
    if str(trained_through[1]) >= earliest_date:
        raise ValueError("source artifact training overlaps V22 input period")
    if artifact.get("hasher") is None:
        raise ValueError("source artifact lacks the job-2707-compatible feature hasher")


def _project_lane_features(
    feature_rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    dimensions: int,
) -> np.ndarray:
    if dimensions < 1:
        raise ValueError("projection dimensions must be positive")
    lanes = sorted(feature_rows, key=lambda row: int(row["meta"]["lane"]))
    if len(lanes) != 6 or [int(row["meta"]["lane"]) for row in lanes] != list(
        range(1, 7)
    ):
        raise ValueError("exactly six uniquely ordered lane feature rows are required")
    matrix = artifact["hasher"].transform(
        [to_hashable(dict(row["features"])) for row in lanes]
    ).tocsr()
    projected = np.zeros((6, dimensions), dtype=np.float64)
    for lane_index in range(6):
        row = matrix.getrow(lane_index)
        norm = max(float(np.linalg.norm(row.data)), 1.0)
        for source_index, value in zip(row.indices, row.data, strict=True):
            bucket = int(source_index) % dimensions
            sign = 1.0 if (int(source_index) // dimensions) % 2 == 0 else -1.0
            projected[lane_index, bucket] += sign * float(value) / norm
    if not np.isfinite(projected).all():
        raise ValueError("historical feature projection is not finite")
    return projected


def _choice_features(
    feature_rows: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, float],
    artifact: Mapping[str, Any],
    *,
    projection_dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    if set(probabilities) != set(COMBINATIONS):
        raise ValueError("source prediction must contain exactly 120 combinations")
    probability = np.asarray(
        [float(probabilities[key]) for key in COMBINATIONS], dtype=np.float64
    )
    if not np.isfinite(probability).all() or np.any(probability <= 0.0):
        raise ValueError("source prediction probabilities must be finite and positive")
    probability /= float(probability.sum())
    lane_features = _project_lane_features(
        feature_rows, artifact, dimensions=projection_dimensions
    )
    position_marginals = np.zeros((3, 6), dtype=np.float64)
    for index, combination in enumerate(TRIFECTA_COMBINATIONS):
        for position, lane in enumerate(combination):
            position_marginals[position, lane - 1] += probability[index]
    rows: list[tuple[float, ...]] = []
    for index, combination in enumerate(TRIFECTA_COMBINATIONS):
        first, second, third = (lane - 1 for lane in combination)
        structural = (
            float(probability[index]),
            float(math.log(max(probability[index], 1e-15))),
            float(position_marginals[0, first]),
            float(position_marginals[1, second]),
            float(position_marginals[2, third]),
            float((first + 1) / 6.0),
            float((second + 1) / 6.0),
            float((third + 1) / 6.0),
        )
        rows.append(
            tuple(lane_features[first])
            + tuple(lane_features[second])
            + tuple(lane_features[third])
            + structural
        )
    return tuple(rows)


def _complete_odds(values: Any, *, greater_than_one: bool) -> dict[str, float] | None:
    if not isinstance(values, Mapping) or set(values) != set(COMBINATIONS):
        return None
    try:
        odds = {key: float(values[key]) for key in COMBINATIONS}
    except (TypeError, ValueError):
        return None
    minimum = 1.0 if greater_than_one else 0.0
    if any(not math.isfinite(value) or value <= minimum for value in odds.values()):
        return None
    return odds if plausible_trifecta_odds(odds) else None


def _official_winner_from_feature_rows(
    feature_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    finish: dict[int, int] = {}
    try:
        for row in feature_rows:
            rank = int(row["meta"]["rank"])
            lane = int(row["meta"]["lane"])
            if rank in {1, 2, 3}:
                if rank in finish or not 1 <= lane <= 6:
                    return None
                finish[rank] = lane
    except (KeyError, TypeError, ValueError):
        return None
    if set(finish) != {1, 2, 3} or len(set(finish.values())) != 3:
        return None
    combination = "-".join(str(finish[rank]) for rank in (1, 2, 3))
    return combination if combination in COMBINATION_INDEX else None


def _ranking_order(winner_index: int, closing_odds: Sequence[float]) -> tuple[int, ...]:
    remaining = sorted(
        (index for index in range(120) if index != winner_index),
        key=lambda index: (float(closing_odds[index]), index),
    )
    return (winner_index, *remaining)


def _load_target_complete_race_ids(
    conn: Any,
    *,
    training_from_date: str,
    training_through_date: str,
    outer_from_date: str,
    outer_through_date: str,
    max_races_per_day: int | None,
) -> list[tuple[str, str, str, int]]:
    """Select only bounded V22 races before running completeness checks."""

    day_limit = "WHERE day_sequence <= ?" if max_races_per_day is not None else ""
    params: list[Any] = [
        training_from_date,
        training_through_date,
        outer_from_date,
        outer_through_date,
    ]
    if max_races_per_day is not None:
        params.append(max_races_per_day)
    rows = conn.execute(
        f"""
        WITH bounded_complete AS (
          SELECT
            r.race_id,
            r.race_date,
            r.jcd,
            r.rno,
            ROW_NUMBER() OVER (
              PARTITION BY r.race_date
              ORDER BY r.jcd, r.rno, r.race_id
            ) AS day_sequence
          FROM races r
          WHERE (
              r.race_date BETWEEN ? AND ?
              OR r.race_date BETWEEN ? AND ?
            )
            AND (
              SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id
            ) = 6
            AND (
              SELECT COUNT(*)
              FROM race_results rr
              WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL
            ) = 6
        )
        SELECT race_id, race_date, jcd, rno
        FROM bounded_complete
        {day_limit}
        ORDER BY race_date, jcd, rno, race_id
        """,
        params,
    ).fetchall()
    return [
        (str(row["race_id"]), str(row["race_date"]), str(row["jcd"]), int(row["rno"]))
        for row in rows
    ]


def _load_result_available_at(
    conn: Any, target_ids: set[str]
) -> dict[str, str]:
    """Return when each targeted six-lane result became complete."""

    result: dict[str, str] = {}
    race_ids = sorted(target_ids)
    for chunk_start in range(0, len(race_ids), 500):
        chunk = race_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _race_id in chunk)
        rows = conn.execute(
            f"""
            SELECT race_id, COUNT(*) AS result_rows,
                   COUNT(updated_at) AS timestamp_rows,
                   MAX(updated_at) AS result_available_at
            FROM race_results
            WHERE race_id IN ({placeholders})
              AND rank IS NOT NULL
            GROUP BY race_id
            """,
            chunk,
        ).fetchall()
        for row in rows:
            if (
                int(row["result_rows"]) == 6
                and int(row["timestamp_rows"]) == 6
                and row["result_available_at"] is not None
            ):
                result[str(row["race_id"])] = str(row["result_available_at"])
    return result


def _load_target_trifecta_payouts(
    conn: Any, target_ids: set[str]
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Load only targeted official trifecta payouts in bounded chunks."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    race_ids = sorted(target_ids)
    for chunk_start in range(0, len(race_ids), 500):
        chunk = race_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _race_id in chunk)
        rows = conn.execute(
            f"""
            SELECT race_id, combination, payout_yen, popularity
            FROM payouts
            WHERE bet_type = ?
              AND payout_yen IS NOT NULL
              AND race_id IN ({placeholders})
            ORDER BY race_id, combination, payout_yen, popularity
            """,
            [PAYOUT_BET_TYPE, *chunk],
        ).fetchall()
        for row in rows:
            grouped[str(row["race_id"])].append(
                {
                    "combination": str(row["combination"]),
                    "payout_yen": int(row["payout_yen"]),
                    "popularity": row["popularity"],
                }
            )
    return {
        race_id: tuple(rows) for race_id, rows in sorted(grouped.items())
    }


def _build_outer_settlements(
    conn: Any,
    outer_races: Sequence[LabeledRace],
    decision_audit: Sequence[DecisionAudit],
) -> tuple[V22BankrollSettlement, ...]:
    """Load strict post-decision settlement data for the frozen outer universe."""

    race_ids = {race.decision.race_id for race in outer_races}
    audits = {
        audit.race_id: audit
        for audit in decision_audit
        if audit.race_id in race_ids
    }
    if set(audits) != race_ids:
        raise ValueError("outer decision audit universe is incomplete")
    payouts = _load_target_trifecta_payouts(conn, race_ids)
    available_at = _load_result_available_at(conn, race_ids)
    settlements: list[V22BankrollSettlement] = []
    for race in outer_races:
        race_id = race.decision.race_id
        rows = payouts.get(race_id)
        if not isinstance(rows, (tuple, list)) or len(rows) != 1:
            raise ValueError(f"ambiguous or missing official payout: {race_id}")
        winner_index = int(race.outcome.winner_index)
        winner = COMBINATIONS[winner_index]
        row = rows[0]
        try:
            payout_winner = str(row["combination"])
            payout_yen = int(row["payout_yen"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid official payout: {race_id}") from exc
        if payout_winner != winner or payout_yen < 100:
            raise ValueError(f"official payout winner mismatch: {race_id}")
        result_at = available_at.get(race_id)
        if result_at is None:
            raise ValueError(f"missing result_available_at: {race_id}")
        audit = audits[race_id]
        settlements.append(
            V22BankrollSettlement(
                race_id=race_id,
                decision_target_at=audit.target_at,
                odds_captured_at=audit.captured_at,
                result_available_at=result_at,
                official_winner_index=winner_index,
                official_closing_odds=race.outcome.closing_odds,
                official_payout_yen=payout_yen,
                snapshot_id=audit.snapshot_id,
            )
        )
    return tuple(settlements)


def load_v22_evaluation_data(
    conn: Any,
    *,
    source_artifact: Mapping[str, Any],
    training_from_date: str,
    training_through_date: str,
    outer_from_date: str,
    outer_through_date: str,
    max_snapshot_age_seconds: float = MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    projection_dimensions: int = DEFAULT_PROJECTION_DIMENSIONS,
    max_races_per_day: int | None = None,
) -> V22EvaluationData:
    """Load strict decision-time inputs and settlement-only labels for V22."""

    _validate_periods(
        training_from_date=training_from_date,
        training_through_date=training_through_date,
        outer_from_date=outer_from_date,
        outer_through_date=outer_through_date,
    )
    _validate_source_artifact(source_artifact, training_from_date)
    if max_snapshot_age_seconds < 0.0:
        raise ValueError("max_snapshot_age_seconds must be non-negative")
    if max_races_per_day is not None and max_races_per_day < 1:
        raise ValueError("max_races_per_day must be positive")

    race_meta: dict[str, tuple[str, str, int]] = {}
    for race_id, race_date, jcd, rno in _load_target_complete_race_ids(
        conn,
        training_from_date=training_from_date,
        training_through_date=training_through_date,
        outer_from_date=outer_from_date,
        outer_through_date=outer_through_date,
        max_races_per_day=max_races_per_day,
    ):
        date_value = str(race_date)
        race_meta[str(race_id)] = (date_value, str(jcd), int(rno))
    target_ids = set(race_meta)
    counters: Counter[str] = Counter(
        {
            "target_complete_races": len(target_ids),
            "feature_races_seen": 0,
            "eligible_races": 0,
            "excluded_missing_t300": 0,
            "excluded_incomplete_t300": 0,
            "excluded_unsafe_t300": 0,
            "excluded_missing_official_closing_odds": 0,
            "excluded_missing_outcome": 0,
            "excluded_ambiguous_outcome": 0,
            "excluded_invalid_features": 0,
        }
    )
    snapshots = prefetch_trifecta_snapshots(
        conn,
        target_ids=target_ids,
        max_snapshot_age_seconds=max_snapshot_age_seconds,
    )
    official_closing = prefetch_official_closing_odds(conn, target_ids=target_ids)
    training: list[LabeledRace] = []
    outer: list[LabeledRace] = []
    audits: list[DecisionAudit] = []

    for feature_rows, probabilities in iter_scored_artifact_feature_rows(
        conn, target_ids=target_ids, artifact=dict(source_artifact)
    ):
        counters["feature_races_seen"] += 1
        if not feature_rows:
            counters["excluded_invalid_features"] += 1
            continue
        race_id = str(feature_rows[0]["meta"]["race_id"])
        meta = race_meta.get(race_id)
        if meta is None:
            continue
        snapshot = (
            (snapshots.get(race_id) or {}).get(MODEL_DECISION_LEAD_MINUTES)
            if snapshots is not None
            else latest_trifecta_odds_before_deadline(
                conn,
                race_id,
                min_combinations=120,
                decision_lead_minutes=MODEL_DECISION_LEAD_MINUTES,
                max_snapshot_age_seconds=max_snapshot_age_seconds,
            )
        )
        if snapshot is None:
            counters["excluded_missing_t300"] += 1
            continue
        current_odds = _complete_odds(snapshot.get("odds"), greater_than_one=True)
        if current_odds is None:
            counters["excluded_incomplete_t300"] += 1
            continue
        age = snapshot_age_seconds(snapshot)
        try:
            captured = _as_aware_datetime(snapshot["captured_at"])
            target = _as_aware_datetime(snapshot["odds_deadline_at"])
        except (KeyError, TypeError, ValueError):
            counters["excluded_unsafe_t300"] += 1
            continue
        if (
            age is None
            or age < 0.0
            or age > max_snapshot_age_seconds
            or captured > target
        ):
            counters["excluded_unsafe_t300"] += 1
            continue
        official = official_closing.get(race_id) or {}
        closing = _complete_odds(
            official.get("official_closing_odds"), greater_than_one=True
        )
        if closing is None:
            counters["excluded_missing_official_closing_odds"] += 1
            continue
        winner = _official_winner_from_feature_rows(feature_rows)
        if winner is None:
            counters["excluded_missing_outcome"] += 1
            continue
        try:
            features = _choice_features(
                feature_rows,
                probabilities,
                source_artifact,
                projection_dimensions=projection_dimensions,
            )
        except (KeyError, TypeError, ValueError):
            counters["excluded_invalid_features"] += 1
            continue
        winner_index = COMBINATION_INDEX[winner]
        closing_tuple = tuple(closing[key] for key in COMBINATIONS)
        labeled = LabeledRace(
            decision=DecisionRace(
                race_id=race_id,
                race_date=meta[0],
                features=features,
                current_odds=tuple(current_odds[key] for key in COMBINATIONS),
            ),
            outcome=RaceOutcome(
                winner_index=winner_index,
                closing_odds=closing_tuple,
                ranking_order=_ranking_order(winner_index, closing_tuple),
            ),
        )
        (training if meta[0] <= training_through_date else outer).append(labeled)
        audits.append(
            DecisionAudit(
                race_id=race_id,
                snapshot_id=(
                    int(snapshot["snapshot_id"])
                    if snapshot.get("snapshot_id") is not None
                    else None
                ),
                captured_at=captured.isoformat(),
                target_at=target.isoformat(),
                age_seconds=float(age),
            )
        )
        counters["eligible_races"] += 1

    training.sort(key=lambda race: (race.decision.race_date, race.decision.race_id))
    outer.sort(key=lambda race: (race.decision.race_date, race.decision.race_id))
    audits.sort(key=lambda audit: audit.race_id)
    if {race.decision.race_id for race in training} & {
        race.decision.race_id for race in outer
    }:
        raise AssertionError("training and outer race identities overlap")
    if training and outer and max(
        race.decision.race_date for race in training
    ) >= min(race.decision.race_date for race in outer):
        raise AssertionError("outer date boundary was not preserved")
    diagnostics: dict[str, Any] = dict(sorted(counters.items()))
    diagnostics.update(
        {
            "training_races": len(training),
            "outer_races": len(outer),
            "choice_count": 120,
            "projection_dimensions": projection_dimensions,
            "decision_feature_count": (
                len(training[0].decision.features[0])
                if training
                else len(outer[0].decision.features[0]) if outer else 0
            ),
            "eligible_coverage": (
                counters["eligible_races"] / len(target_ids) if target_ids else 0.0
            ),
            "information_boundary": (
                "historical_features_and_complete_t300_at_or_before_target_only;"
                "official_closing_odds_and_outcome_settlement_only"
            ),
            "missing_t300_policy": "exclude_race_fail_closed",
            "ranking_teacher": "winner_first_then_official_closing_odds_order",
            "source_model_trained_through": str(source_artifact["trained_through"][1]),
        }
    )
    return V22EvaluationData(
        training_races=tuple(training),
        outer_races=tuple(outer),
        decision_audit=tuple(audits),
        diagnostics=diagnostics,
    )


def run_v22_smoke_evaluation(
    conn: Any,
    *,
    source_artifact: Mapping[str, Any],
    training_from_date: str,
    training_through_date: str,
    outer_from_date: str,
    outer_through_date: str,
    max_snapshot_age_seconds: float = MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    projection_dimensions: int = DEFAULT_PROJECTION_DIMENSIONS,
    max_races_per_day: int | None = None,
    minimum_inner_training_dates: int = 2,
    minimum_purchase_training_dates: int = 2,
    alpha: float = 1e-3,
    purchase_loss: str = "ridge_capped_net",
    loaded_data: V22EvaluationData | None = None,
) -> dict[str, Any]:
    loaded = loaded_data or load_v22_evaluation_data(
        conn,
        source_artifact=source_artifact,
        training_from_date=training_from_date,
        training_through_date=training_through_date,
        outer_from_date=outer_from_date,
        outer_through_date=outer_through_date,
        max_snapshot_age_seconds=max_snapshot_age_seconds,
        projection_dimensions=projection_dimensions,
        max_races_per_day=max_races_per_day,
    )
    if not loaded.training_races:
        raise ValueError("no leakage-safe V22 training races are available")
    if not loaded.outer_races:
        raise ValueError("no leakage-safe V22 outer races are available")
    artifact = fit_four_head_nested_v22(
        loaded.training_races,
        minimum_inner_training_dates=minimum_inner_training_dates,
        minimum_purchase_training_dates=minimum_purchase_training_dates,
        alpha=alpha,
        purchase_loss=purchase_loss,
    )
    settlements = _build_outer_settlements(
        conn, loaded.outer_races, loaded.decision_audit
    )
    formal_bankroll = evaluate_four_head_v22_bankroll(
        artifact, loaded.outer_races, settlements,
        max_t5_snapshot_age_seconds=max_snapshot_age_seconds,
    )
    outer_evaluation = evaluate_outer_outcomes(artifact, loaded.outer_races)
    return {
        "model_key": artifact.model_key,
        "artifact_sha256": artifact_fingerprint(artifact),
        "purchase_loss": purchase_loss,
        "periods": {
            "training_from": training_from_date,
            "training_through": training_through_date,
            "outer_from": outer_from_date,
            "outer_through": outer_through_date,
        },
        "coverage": dict(loaded.diagnostics),
        "decision_audit": [audit.__dict__ for audit in loaded.decision_audit],
        "evaluation": outer_evaluation,
        "purchase_value_diagnostics": outer_evaluation[
            "purchase_value_diagnostics"
        ],
        "formal_bankroll": formal_bankroll,
        "roi": formal_bankroll["roi"],
        "stake_yen": formal_bankroll["stake_yen"],
        "return_yen": formal_bankroll["return_yen"],
        "profit_yen": formal_bankroll["profit_yen"],
        "max_drawdown_yen": formal_bankroll["max_drawdown_yen"],
        "daily": formal_bankroll["daily"],
        "winner_log_loss": formal_bankroll["winner_log_loss"],
        "winner_top1_accuracy": formal_bankroll["winner_top1_accuracy"],
        "trifecta_log_loss": formal_bankroll["trifecta_log_loss"],
        "trifecta_top5_hit_rate": formal_bankroll["trifecta_top5_hit_rate"],
        "outer_outcomes_used_for_fit_selection_or_threshold": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe actual-data smoke evaluation for V22 four-head"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--training-from", required=True)
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--outer-from", required=True)
    parser.add_argument("--outer-through", required=True)
    parser.add_argument("--output")
    parser.add_argument("--data-cache")
    parser.add_argument("--max-races-per-day", type=int)
    parser.add_argument(
        "--max-snapshot-age-seconds",
        type=float,
        default=MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    )
    parser.add_argument(
        "--projection-dimensions", type=int, default=DEFAULT_PROJECTION_DIMENSIONS
    )
    parser.add_argument("--minimum-inner-training-dates", type=int, default=2)
    parser.add_argument("--minimum-purchase-training-dates", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument(
        "--purchase-loss",
        choices=(
            "ridge_capped_net",
            "poisson_capped_gross",
            "tweedie_capped_gross",
            "hurdle_logistic_lognormal",
            "hurdle_logistic_lognormal_calibrated",
            "hurdle_contextual_lognormal",
            "hurdle_contextual_interactions_lognormal",
            "pairwise_contextual_rank_calibrated",
        ),
        default="ridge_capped_net",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = Path(args.source_model)
    source_artifact = joblib.load(source_path)
    with source_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    cache_signature = {
        "source_sha256": source_sha256,
        "training_from": args.training_from,
        "training_through": args.training_through,
        "outer_from": args.outer_from,
        "outer_through": args.outer_through,
        "max_snapshot_age_seconds": args.max_snapshot_age_seconds,
        "projection_dimensions": args.projection_dimensions,
        "max_races_per_day": args.max_races_per_day,
    }
    with connection(args.db) as conn:
        loaded_data = None
        if args.data_cache:
            loaded_data = load_or_build_v22_evaluation_data(
                Path(args.data_cache),
                signature=cache_signature,
                builder=lambda: load_v22_evaluation_data(
                    conn,
                    source_artifact=source_artifact,
                    training_from_date=args.training_from,
                    training_through_date=args.training_through,
                    outer_from_date=args.outer_from,
                    outer_through_date=args.outer_through,
                    max_snapshot_age_seconds=args.max_snapshot_age_seconds,
                    projection_dimensions=args.projection_dimensions,
                    max_races_per_day=args.max_races_per_day,
                ),
            )
        result = run_v22_smoke_evaluation(
            conn,
            source_artifact=source_artifact,
            training_from_date=args.training_from,
            training_through_date=args.training_through,
            outer_from_date=args.outer_from,
            outer_through_date=args.outer_through,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            projection_dimensions=args.projection_dimensions,
            max_races_per_day=args.max_races_per_day,
            minimum_inner_training_dates=args.minimum_inner_training_dates,
            minimum_purchase_training_dates=args.minimum_purchase_training_dates,
            alpha=args.alpha,
            purchase_loss=args.purchase_loss,
            loaded_data=loaded_data,
        )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

