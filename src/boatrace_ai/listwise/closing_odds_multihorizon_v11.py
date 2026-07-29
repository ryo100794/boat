from __future__ import annotations
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS


MODEL_NAME = "closing_odds_multihorizon_v11"
MODEL_VERSION = 11
EXPECTED_COMBINATIONS = 120
CHECKPOINT_OFFSETS_SECONDS = (300, 120, 60, 30, 10)
CHECKPOINT_LABELS = tuple(f"t{offset}" for offset in CHECKPOINT_OFFSETS_SECONDS)
DEFAULT_LOWER_QUANTILE = 0.20
EPSILON = 1e-12
JST = timezone(timedelta(hours=9))

_BASE_FEATURE_NAMES = (
    "intercept",
    "log_horizon_seconds",
    *(f"horizon_t{offset}" for offset in CHECKPOINT_OFFSETS_SECONDS),
    "log_current_odds",
    "market_rank",
    "log_odds_slope_per_minute",
    "log_odds_curvature_per_minute2",
    "trend_point_count",
    "slope_available",
    "curvature_available",
    "log1p_source_update_staleness_minutes",
    "source_update_staleness_missing",
    "log1p_checkpoint_age_before_target_seconds",
    "checkpoint_age_before_target_missing",
    "hour_sin",
    "hour_cos",
)
_VENUE_FEATURE_NAMES = tuple(f"venue_{value:02d}" for value in range(1, 25))
_RNO_FEATURE_NAMES = tuple(f"rno_{value:02d}" for value in range(1, 13))
_VENUE_INTERACTION_FEATURE_NAMES = (
    *(f"venue_log_odds_{value:02d}" for value in range(1, 25)),
    "venue_log_odds_unknown",
    *(f"venue_rank_{value:02d}" for value in range(1, 25)),
    "venue_rank_unknown",
)
FEATURE_NAMES = (
    *_BASE_FEATURE_NAMES,
    *_VENUE_FEATURE_NAMES,
    "venue_unknown",
    *_RNO_FEATURE_NAMES,
    "rno_unknown",
    *_VENUE_INTERACTION_FEATURE_NAMES,
)
POINT_MODEL_ARCHITECTURES = ("base", "venue_interactions")
VENUE_INTERACTION_PENALTY_MULTIPLIER = 4.0


def checkpoint_label(offset_seconds: int) -> str:
    """Return the canonical label for one supported pre-close checkpoint."""
    offset = int(offset_seconds)
    if offset not in CHECKPOINT_OFFSETS_SECONDS:
        raise ValueError(f"unsupported closing checkpoint: {offset}")
    return f"t{offset}"


def _required_checkpoint_offset(value: object, name: str) -> int:
    offset = _offset_from_label(value)
    if offset is None:
        raise ValueError(
            f"{name} must be one of 300/120/60/30/10 seconds"
        )
    return offset


def _iso_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _finite_positive_odds(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _offset_from_label(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            offset = int(value)
        else:
            text = str(value).strip().lower().replace("seconds", "")
            text = text.replace("second", "").replace("sec", "").replace("s", "")
            text = text.replace("t-", "").replace("t", "").replace("-", "")
            offset = int(text)
    except (TypeError, ValueError, OverflowError):
        return None
    return offset if offset in CHECKPOINT_OFFSETS_SECONDS else None


def _checkpoint_offset(key: object, snapshot: Mapping[str, object]) -> int | None:
    for candidate in (
        snapshot.get("target_offset_seconds"),
        snapshot.get("checkpoint_seconds"),
        snapshot.get("offset_seconds"),
        snapshot.get("label"),
        key,
    ):
        offset = _offset_from_label(candidate)
        if offset is not None:
            return offset
    return None


def _snapshot_odds(snapshot: Mapping[str, object]) -> dict[str, float]:
    parsed = snapshot.get("parsed")
    values = snapshot.get("odds")
    if not isinstance(values, Mapping) and isinstance(parsed, Mapping):
        values = parsed.get("odds")
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in values.items():
        value = _finite_positive_odds(raw)
        if value is not None:
            result[str(key)] = value
    return result


def _snapshot_staleness(snapshot: Mapping[str, object]) -> float | None:
    candidates: list[object] = [snapshot.get("source_update_staleness_seconds")]
    parsed = snapshot.get("parsed")
    if isinstance(parsed, Mapping):
        collection = parsed.get("_collection")
        if isinstance(collection, Mapping):
            candidates.append(collection.get("source_update_staleness_seconds"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value >= 0.0:
            return value
    return None


def _snapshot_captured_age_seconds(
    snapshot: Mapping[str, object],
) -> float | None:
    candidates: list[object] = [snapshot.get("captured_age_seconds")]
    parsed = snapshot.get("parsed")
    if isinstance(parsed, Mapping):
        collection = parsed.get("_collection")
        if isinstance(collection, Mapping):
            candidates.append(collection.get("captured_age_seconds"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            return value
    deadline = snapshot.get("deadline_at")
    captured = snapshot.get("captured_at", snapshot.get("observed_at"))
    if not deadline or not captured:
        return None
    try:
        deadline_at = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        captured_at = datetime.fromisoformat(str(captured).replace("Z", "+00:00"))
    except ValueError:
        return None
    if deadline_at.tzinfo is None:
        deadline_at = deadline_at.replace(tzinfo=timezone.utc)
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    return (deadline_at - captured_at).total_seconds()


def _checkpoint_age_before_target_seconds(
    snapshot: Mapping[str, object], offset: int
) -> float | None:
    captured_age = _snapshot_captured_age_seconds(snapshot)
    return captured_age - offset if captured_age is not None else None


def _snapshot_order(
    snapshot: Mapping[str, object], offset: int
) -> tuple[int, float, int, str]:
    checkpoint_age = _checkpoint_age_before_target_seconds(snapshot, offset)
    try:
        attempt = int(snapshot.get("checkpoint_attempt", snapshot.get("attempt", 1)))
    except (TypeError, ValueError, OverflowError):
        attempt = 1
    captured = str(snapshot.get("captured_at") or snapshot.get("observed_at") or "")
    return (
        0 if checkpoint_age is not None else 1,
        checkpoint_age if checkpoint_age is not None else math.inf,
        max(1, attempt),
        captured,
    )


def normalize_labeled_checkpoints(
    race: Mapping[str, object],
    *,
    as_of_offset_seconds: object | None = None,
) -> dict[str, dict[str, object]]:
    """Normalize checkpoint containers without filling a missing horizon.

    The latest observation at or before each target wins. A negative checkpoint
    age is a future observation and is rejected. Labels outside the contract are
    ignored and missing ages remain explicit.
    """
    as_of = (
        None
        if as_of_offset_seconds is None
        else _required_checkpoint_offset(as_of_offset_seconds, "as_of_offset_seconds")
    )
    container: object = None
    for name in ("closing_odds_checkpoints", "odds_checkpoints", "checkpoints"):
        if race.get(name) is not None:
            container = race.get(name)
            break
    entries: list[tuple[object, Mapping[str, object]]] = []
    if isinstance(container, Mapping):
        entries = [
            (key, value)
            for key, value in container.items()
            if isinstance(value, Mapping)
        ]
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        entries = [
            (index, value)
            for index, value in enumerate(container)
            if isinstance(value, Mapping)
        ]
    elif any(
        race.get(name) is not None
        for name in ("target_offset_seconds", "checkpoint_seconds", "offset_seconds")
    ):
        entries = [(race.get("target_offset_seconds"), race)]

    candidates: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for key, snapshot in entries:
        offset = _checkpoint_offset(key, snapshot)
        if offset is not None and as_of is not None and offset < as_of:
            continue
        checkpoint_age = (
            _checkpoint_age_before_target_seconds(snapshot, offset)
            if offset is not None
            else None
        )
        if (
            offset is not None
            and _snapshot_odds(snapshot)
            and (checkpoint_age is None or checkpoint_age >= 0.0)
        ):
            candidates[offset].append(snapshot)

    result: dict[str, dict[str, object]] = {}
    for offset in CHECKPOINT_OFFSETS_SECONDS:
        snapshots = candidates.get(offset) or []
        if not snapshots:
            continue
        selected = min(snapshots, key=lambda item: _snapshot_order(item, offset))
        item = dict(selected)
        item["label"] = checkpoint_label(offset)
        item["target_offset_seconds"] = offset
        item["odds"] = _snapshot_odds(selected)
        item["source_update_staleness_seconds"] = _snapshot_staleness(selected)
        item["checkpoint_age_before_target_seconds"] = (
            _checkpoint_age_before_target_seconds(selected, offset)
        )
        result[checkpoint_label(offset)] = item
    return result


def missing_checkpoint_labels(race: Mapping[str, object]) -> list[str]:
    checkpoints = normalize_labeled_checkpoints(race)
    return [label for label in CHECKPOINT_LABELS if label not in checkpoints]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _race_hour(race: Mapping[str, object], snapshot: Mapping[str, object]) -> int:
    for value in (snapshot.get("captured_at"), snapshot.get("observed_at")):
        if value:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(JST)
            return parsed.hour
    for name in ("race_hour", "hour"):
        try:
            value = int(race.get(name))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= value <= 23:
            return value
    for name in ("deadline_at", "start_at", "race_time"):
        value = race.get(name)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(JST)
        return parsed.hour
    return 0


def _bounded_integer(value: object, lower: int, upper: int) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if lower <= result <= upper else None


def _context_features(
    race: Mapping[str, object], snapshot: Mapping[str, object]
) -> list[float]:
    hour = _race_hour(race, snapshot)
    angle = 2.0 * math.pi * hour / 24.0
    result = [math.sin(angle), math.cos(angle)]
    venue = _bounded_integer(
        race.get("jcd", race.get("venue_code", race.get("venue"))), 1, 24
    )
    result.extend(1.0 if venue == value else 0.0 for value in range(1, 25))
    result.append(1.0 if venue is None else 0.0)
    rno = _bounded_integer(race.get("rno", race.get("race_no")), 1, 12)
    result.extend(1.0 if rno == value else 0.0 for value in range(1, 13))
    result.append(1.0 if rno is None else 0.0)
    return result


def _trend_features(
    checkpoints: Mapping[str, Mapping[str, object]],
    combination: str,
    horizon: int,
) -> tuple[float, float, float, float, float, tuple[int, ...]]:
    # Larger offsets happened earlier. Smaller offsets are future data and are
    # deliberately unreachable from this feature builder.
    points: list[tuple[int, float]] = []
    for offset in CHECKPOINT_OFFSETS_SECONDS:
        if offset < horizon:
            continue
        snapshot = checkpoints.get(checkpoint_label(offset))
        if not snapshot:
            continue
        value = _snapshot_odds(snapshot).get(combination)
        if value is not None:
            points.append((offset, math.log(value)))
    points.sort(reverse=True)
    slopes: list[tuple[float, float]] = []
    for (older_offset, older), (newer_offset, newer) in zip(points, points[1:]):
        elapsed_minutes = (older_offset - newer_offset) / 60.0
        if elapsed_minutes > 0.0:
            slopes.append(((newer - older) / elapsed_minutes, elapsed_minutes))
    slope = slopes[-1][0] if slopes else 0.0
    curvature = 0.0
    if len(slopes) >= 2:
        distance = max(EPSILON, (slopes[-2][1] + slopes[-1][1]) / 2.0)
        curvature = (slopes[-1][0] - slopes[-2][0]) / distance
    return (
        float(np.clip(slope, -5.0, 5.0)),
        float(np.clip(curvature, -5.0, 5.0)),
        math.log1p(len(points)),
        1.0 if slopes else 0.0,
        1.0 if len(slopes) >= 2 else 0.0,
        tuple(offset for offset, _value in points),
    )


def build_checkpoint_feature_vector(
    race: Mapping[str, object],
    *,
    checkpoint: object,
    combination: str,
    as_of_offset_seconds: object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build one feature vector using only the named checkpoint and its past."""
    horizon = _offset_from_label(checkpoint)
    if horizon is None:
        raise ValueError("checkpoint is not one of t300/t120/t60/t30/t10")
    as_of = (
        horizon
        if as_of_offset_seconds is None
        else _required_checkpoint_offset(as_of_offset_seconds, "as_of_offset_seconds")
    )
    if horizon < as_of:
        raise ValueError("checkpoint is after as_of_offset_seconds")
    checkpoints = normalize_labeled_checkpoints(
        race, as_of_offset_seconds=as_of
    )
    label = checkpoint_label(horizon)
    snapshot = checkpoints.get(label)
    if snapshot is None:
        raise ValueError(f"missing checkpoint: {label}")
    odds = _snapshot_odds(snapshot)
    if combination not in odds:
        raise ValueError(f"checkpoint {label} is missing odds for {combination}")
    keys = sorted(odds)
    values = np.asarray([odds[key] for key in keys], dtype=np.float64)
    ranks = _average_ranks(values) / max(1, len(values) - 1)
    rank = float(ranks[keys.index(combination)])
    current_odds = odds[combination]
    slope, curvature, point_count, has_slope, has_curvature, used = _trend_features(
        checkpoints, combination, horizon
    )
    stale = _snapshot_staleness(snapshot)
    checkpoint_age = snapshot.get("checkpoint_age_before_target_seconds")
    checkpoint_age_value = (
        float(checkpoint_age) if checkpoint_age is not None else None
    )
    if checkpoint_age_value is not None and checkpoint_age_value < 0.0:
        raise ValueError("future checkpoint observation is not allowed")
    current_log = math.log(current_odds)
    context = _context_features(race, snapshot)
    venue_indicators = context[2:27]
    vector = [
        1.0,
        math.log(float(horizon)),
        *(1.0 if horizon == value else 0.0 for value in CHECKPOINT_OFFSETS_SECONDS),
        current_log,
        rank,
        slope,
        curvature,
        point_count,
        has_slope,
        has_curvature,
        math.log1p(stale / 60.0) if stale is not None else 0.0,
        1.0 if stale is None else 0.0,
        math.log1p(checkpoint_age_value) if checkpoint_age_value is not None else 0.0,
        1.0 if checkpoint_age_value is None else 0.0,
        *context,
        *(current_log * value for value in venue_indicators),
        *(rank * value for value in venue_indicators),
    ]
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("v11 checkpoint feature contract mismatch")
    return result, {
        "checkpoint_label": label,
        "target_offset_seconds": horizon,
        "as_of_offset_seconds": as_of,
        "used_checkpoint_offsets": list(used),
        "future_checkpoint_offsets_used": [value for value in used if value < horizon],
        "source_update_staleness_missing": stale is None,
        "checkpoint_age_before_target_seconds": checkpoint_age_value,
        "checkpoint_age_before_target_missing": checkpoint_age_value is None,
    }


def _race_identity(race: Mapping[str, object], race_date: str) -> str:
    return str(
        race.get("race_id")
        or "|".join(
            (
                race_date,
                str(race.get("jcd") or race.get("venue_code") or ""),
                str(race.get("rno") or race.get("race_no") or ""),
            )
        )
    )


def _odds_mapping(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in values.items():
        value = _finite_positive_odds(raw)
        if value is not None:
            result[str(key)] = value
    return result


def select_teacher_final_odds(
    race: Mapping[str, object],
) -> tuple[dict[str, float], str | None]:
    """Select the auditable final-odds teacher for one race.

    Official closing odds win only when all 120 valid trifecta prices are present.
    The legacy final_odds alias is deliberately not a teacher source.
    """
    official = _odds_mapping(race.get("official_closing_odds"))
    if len(official) == EXPECTED_COMBINATIONS:
        return official, "official_closing_odds"
    closing = _odds_mapping(race.get("closing_odds"))
    if closing:
        return closing, "closing_odds_fallback"
    return {}, None


def _teacher_selection(
    race: Mapping[str, object],
) -> tuple[dict[str, float], str | None, bool]:
    official_present = isinstance(race.get("official_closing_odds"), Mapping)
    official = _odds_mapping(race.get("official_closing_odds"))
    final, source = select_teacher_final_odds(race)
    return (
        final,
        source,
        bool(official_present and len(official) != EXPECTED_COMBINATIONS),
    )


def _venue_group(race: Mapping[str, object]) -> str:
    venue = _bounded_integer(
        race.get("jcd", race.get("venue_code", race.get("venue"))), 1, 24
    )
    return f"{venue:02d}" if venue is not None else "unknown"


def _examples_from_race(
    race: Mapping[str, object], race_date: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoints = normalize_labeled_checkpoints(race)
    final, teacher_source, official_incomplete = _teacher_selection(race)
    missing = {label: int(label not in checkpoints) for label in CHECKPOINT_LABELS}
    incomplete = {label: 0 for label in CHECKPOINT_LABELS}
    examples: list[dict[str, object]] = []
    if not final:
        return examples, {
            **{f"missing_{label}": value for label, value in missing.items()},
            **{f"incomplete_{label}": 0 for label in CHECKPOINT_LABELS},
            "teacher_source": teacher_source,
            "teacher_tickets": 0,
            "official_closing_odds_incomplete": official_incomplete,
        }
    identity = _race_identity(race, race_date)
    for horizon in CHECKPOINT_OFFSETS_SECONDS:
        label = checkpoint_label(horizon)
        snapshot = checkpoints.get(label)
        if snapshot is None:
            continue
        current = _snapshot_odds(snapshot)
        if set(current) != set(final):
            incomplete[label] = 1
            continue
        for combination in sorted(final):
            vector, trace = build_checkpoint_feature_vector(
                race, checkpoint=label, combination=combination
            )
            examples.append(
                {
                    "race_date": race_date,
                    "race_id": identity,
                    "horizon": horizon,
                    "label": label,
                    "combination": combination,
                    "features": vector,
                    "target_log_ratio": math.log(final[combination] / current[combination]),
                    "raw_target_log_ratio": math.log(
                        final[combination] / current[combination]
                    ),
                    "trace": trace,
                    "teacher_source": teacher_source,
                    "venue_group": _venue_group(race),
                }
            )
    return examples, {
        **{f"missing_{label}": value for label, value in missing.items()},
        **{f"incomplete_{label}": value for label, value in incomplete.items()},
        "teacher_source": teacher_source,
        "teacher_tickets": len(final),
        "official_closing_odds_incomplete": official_incomplete,
    }


def _winsorize_teachers_by_day_venue(
    examples: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Winsorize only within intact race-day, venue, and horizon clusters."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in examples:
        grouped[
            (
                str(row["race_date"]),
                str(row["venue_group"]),
                str(row["label"]),
            )
        ].append(float(row["raw_target_log_ratio"]))
    bounds: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        bounds[key] = (median - 3.0 * mad, median + 3.0 * mad, mad)
    result: list[dict[str, object]] = []
    clipped = 0
    zero_mad_clusters = 0
    for row in examples:
        key = (
            str(row["race_date"]),
            str(row["venue_group"]),
            str(row["label"]),
        )
        lower, upper, mad = bounds[key]
        raw = float(row["raw_target_log_ratio"])
        robust = min(upper, max(lower, raw)) if mad > 0.0 else raw
        clipped += int(robust != raw)
        item = dict(row)
        item["target_log_ratio"] = robust
        result.append(item)
    zero_mad_clusters = sum(mad <= 0.0 for _lower, _upper, mad in bounds.values())
    return result, {
        "method": "winsorize_within_race_date_venue_horizon_median_plus_minus_3mad",
        "cluster_unit": "race_date_x_venue_x_horizon",
        "clusters": len(bounds),
        "zero_mad_clusters": zero_mad_clusters,
        "training_examples": len(result),
        "clipped_examples": clipped,
        "clipped_fraction": clipped / len(result) if result else None,
    }


def _model_feature_indices(architecture: str) -> np.ndarray:
    if architecture not in POINT_MODEL_ARCHITECTURES:
        raise ValueError(f"unknown point model architecture: {architecture}")
    return np.asarray(
        [
            index
            for index, name in enumerate(FEATURE_NAMES)
            if architecture == "venue_interactions"
            or name not in _VENUE_INTERACTION_FEATURE_NAMES
        ],
        dtype=np.int64,
    )


def _fit_point_model(
    examples: Sequence[Mapping[str, object]],
    regularization: float,
    *,
    architecture: str,
) -> dict[str, object] | None:
    if not examples:
        return None
    feature_indices = _model_feature_indices(architecture)
    matrix = np.stack(
        [np.asarray(row["features"], dtype=np.float64)[feature_indices] for row in examples]
    )
    target = np.asarray([float(row["target_log_ratio"]) for row in examples])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    selected_names = [FEATURE_NAMES[index] for index in feature_indices]
    penalty_weights = np.ones(matrix.shape[1], dtype=np.float64)
    for index, name in enumerate(selected_names):
        if name in _VENUE_INTERACTION_FEATURE_NAMES:
            penalty_weights[index] = VENUE_INTERACTION_PENALTY_MULTIPLIER
    penalty = np.diag(penalty_weights) * regularization
    penalty[0, 0] = 0.0
    gram = standardized.T @ standardized / len(target)
    rhs = standardized.T @ target / len(target)
    system = gram + penalty + 1e-10 * np.eye(matrix.shape[1], dtype=np.float64)
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]
    predicted = standardized @ coefficients
    return {
        "model_type": "robust_ridge_log_selected_closing_to_current_ratio",
        "architecture": architecture,
        "teacher": "winsorized_log(selected_closing_odds/current_odds)",
        "feature_names": selected_names,
        "feature_indices": feature_indices.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "regularization": float(regularization),
        "venue_interaction_penalty_multiplier": (
            VENUE_INTERACTION_PENALTY_MULTIPLIER
            if architecture == "venue_interactions"
            else None
        ),
        "training_examples": len(examples),
        "training_log_ratio_mae": float(np.mean(np.abs(target - predicted))),
    }


def _point_log_ratio(vector: np.ndarray, point_model: Mapping[str, object]) -> float:
    indices = np.asarray(point_model["feature_indices"], dtype=np.int64)
    selected = vector[indices]
    mean = np.asarray(point_model["feature_mean"], dtype=np.float64)
    scale = np.asarray(point_model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(point_model["coefficients"], dtype=np.float64)
    if not (selected.shape == mean.shape == scale.shape == coefficients.shape):
        raise ValueError("v11 point model feature contract mismatch")
    return float(((selected - mean) / scale) @ coefficients)


def _finite_sample_lower_rank(
    values: Sequence[float], *, target_coverage: float
) -> tuple[float, int, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) == 0:
        raise ValueError("finite-sample lower rank requires observations")
    alpha = 1.0 - target_coverage
    rank = max(
        1,
        int(math.floor((len(ordered) + 1) * alpha + 1e-12)),
    )
    rank = min(rank, len(ordered))
    finite_sample_coverage = (
        len(ordered) + 1 - rank
    ) / (len(ordered) + 1)
    return float(ordered[rank - 1]), rank, finite_sample_coverage


def fit_closing_odds_multihorizon_v11(
    races: Iterable[Mapping[str, object]],
    *,
    prediction_date: object,
    regularization: float = 0.01,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    minimum_training_days: int = 5,
    minimum_training_races: int = 100,
    minimum_examples_per_horizon: int = 500,
    calibration_warmup_days: int = 2,
    minimum_calibration_days: int = 2,
    minimum_relative_mae_improvement: float = 0.01,
) -> dict[str, object]:
    """Fit point and conservative odds models using whole days before prediction."""
    target_date = _iso_date(prediction_date, "prediction_date")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be between 0 and 0.5")
    if not 0.0 <= minimum_relative_mae_improvement < 1.0:
        raise ValueError(
            "minimum_relative_mae_improvement must be between zero and one"
        )
    for name, value in (
        ("minimum_training_days", minimum_training_days),
        ("minimum_training_races", minimum_training_races),
        ("minimum_examples_per_horizon", minimum_examples_per_horizon),
        ("calibration_warmup_days", calibration_warmup_days),
        ("minimum_calibration_days", minimum_calibration_days),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")

    source = list(races)
    prior: list[tuple[str, Mapping[str, object]]] = []
    excluded_non_past = excluded_invalid_date = 0
    for race in source:
        try:
            race_date = _iso_date(race.get("race_date"), "race_date")
        except ValueError:
            excluded_invalid_date += 1
            continue
        if race_date >= target_date:
            excluded_non_past += 1
            continue
        prior.append((race_date, race))
    prior.sort(key=lambda item: (item[0], _race_identity(item[1], item[0])))

    examples: list[dict[str, object]] = []
    missing_counts = {label: 0 for label in CHECKPOINT_LABELS}
    incomplete_counts = {label: 0 for label in CHECKPOINT_LABELS}
    eligible_races: set[str] = set()
    selected_teacher_races: dict[str, int] = defaultdict(int)
    selected_teacher_tickets: dict[str, int] = defaultdict(int)
    training_teacher_races: dict[str, set[str]] = defaultdict(set)
    official_incomplete_races = 0
    missing_teacher_races = 0
    for race_date, race in prior:
        rows, audit = _examples_from_race(race, race_date)
        examples.extend(rows)
        teacher_source = audit["teacher_source"]
        official_incomplete_races += int(
            bool(audit["official_closing_odds_incomplete"])
        )
        if teacher_source is None:
            missing_teacher_races += 1
        else:
            selected_teacher_races[str(teacher_source)] += 1
            selected_teacher_tickets[str(teacher_source)] += int(
                audit["teacher_tickets"]
            )
        if rows:
            identity = f"{race_date}|{_race_identity(race, race_date)}"
            eligible_races.add(identity)
            training_teacher_races[str(teacher_source)].add(identity)
        for label in CHECKPOINT_LABELS:
            missing_counts[label] += audit[f"missing_{label}"]
            incomplete_counts[label] += audit[f"incomplete_{label}"]
    examples.sort(
        key=lambda row: (
            str(row["race_date"]),
            str(row["race_id"]),
            -int(row["horizon"]),
            str(row["combination"]),
        )
    )
    examples, robust_teacher = _winsorize_teachers_by_day_venue(examples)
    training_dates = sorted({str(row["race_date"]) for row in examples})
    examples_by_horizon = {
        label: sum(row["label"] == label for row in examples)
        for label in CHECKPOINT_LABELS
    }
    training_dates_by_horizon = {
        label: sorted(
            {
                str(row["race_date"])
                for row in examples
                if row["label"] == label
            }
        )
        for label in CHECKPOINT_LABELS
    }
    training_races_by_horizon = {
        label: len(
            {
                (str(row["race_date"]), str(row["race_id"]))
                for row in examples
                if row["label"] == label
            }
        )
        for label in CHECKPOINT_LABELS
    }

    by_day: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in examples:
        by_day[str(row["race_date"])].append(row)
    baseline_errors: dict[str, list[float]] = {
        label: [] for label in CHECKPOINT_LABELS
    }
    baseline_residuals: dict[str, dict[str, list[float]]] = {
        label: defaultdict(list) for label in CHECKPOINT_LABELS
    }
    candidate_errors: dict[str, dict[str, list[float]]] = {
        label: {architecture: [] for architecture in POINT_MODEL_ARCHITECTURES}
        for label in CHECKPOINT_LABELS
    }
    candidate_residuals: dict[str, dict[str, dict[str, list[float]]]] = {
        label: {
            architecture: defaultdict(list)
            for architecture in POINT_MODEL_ARCHITECTURES
        }
        for label in CHECKPOINT_LABELS
    }
    top5_overprediction: dict[str, dict[str, list[float]]] = {
        label: {architecture: [] for architecture in POINT_MODEL_ARCHITECTURES}
        for label in CHECKPOINT_LABELS
    }
    cv_days: dict[str, set[str]] = {
        label: set() for label in CHECKPOINT_LABELS
    }
    folds: list[dict[str, object]] = []
    for index in range(calibration_warmup_days, len(training_dates)):
        evaluation_date = training_dates[index]
        fold_training_dates = training_dates[:index]
        fold_counts: dict[str, dict[str, int]] = {}
        for label in CHECKPOINT_LABELS:
            fold_training = [
                row
                for day in fold_training_dates
                for row in by_day[day]
                if row["label"] == label
            ]
            holdout = [
                row for row in by_day[evaluation_date] if row["label"] == label
            ]
            fold_counts[label] = {
                "training_examples": len(fold_training),
                "evaluation_examples": len(holdout),
            }
            if not fold_training or not holdout:
                continue
            raw_targets = np.asarray(
                [float(row["raw_target_log_ratio"]) for row in holdout],
                dtype=np.float64,
            )
            baseline_errors[label].extend(np.abs(raw_targets).tolist())
            baseline_residuals[label][evaluation_date].extend(raw_targets.tolist())
            cv_days[label].add(evaluation_date)
            for architecture in POINT_MODEL_ARCHITECTURES:
                fold_model = _fit_point_model(
                    fold_training,
                    regularization,
                    architecture=architecture,
                )
                if fold_model is None:
                    continue
                predictions = np.asarray(
                    [
                        _point_log_ratio(
                            np.asarray(row["features"], dtype=np.float64),
                            fold_model,
                        )
                        for row in holdout
                    ],
                    dtype=np.float64,
                )
                residual_values = raw_targets - predictions
                candidate_errors[label][architecture].extend(
                    np.abs(residual_values).tolist()
                )
                candidate_residuals[label][architecture][evaluation_date].extend(
                    residual_values.tolist()
                )
                by_race: dict[str, list[int]] = defaultdict(list)
                for row_index, row in enumerate(holdout):
                    by_race[str(row["race_id"])].append(row_index)
                for indices in by_race.values():
                    selected = sorted(
                        indices,
                        key=lambda row_index: (
                            -float(predictions[row_index]),
                            str(holdout[row_index]["combination"]),
                        ),
                    )[:5]
                    top5_overprediction[label][architecture].extend(
                        (
                            predictions[selected] - raw_targets[selected]
                        ).tolist()
                    )
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "trained_through_date": fold_training_dates[-1],
                "training_dates": list(fold_training_dates),
                "by_horizon": fold_counts,
                "strict_prior_day": fold_training_dates[-1] < evaluation_date,
            }
        )

    horizon_selection: dict[str, dict[str, object]] = {}
    point_models: dict[str, dict[str, object]] = {}
    selected_residuals: dict[str, dict[str, list[float]]] = {}
    for label in CHECKPOINT_LABELS:
        baseline_mae = (
            float(np.mean(baseline_errors[label]))
            if baseline_errors[label]
            else None
        )
        candidate_metrics: dict[str, dict[str, object]] = {}
        for architecture in POINT_MODEL_ARCHITECTURES:
            errors = candidate_errors[label][architecture]
            model_mae = float(np.mean(errors)) if errors else None
            improvement = (
                1.0 - model_mae / baseline_mae
                if model_mae is not None
                and baseline_mae is not None
                and baseline_mae > 0.0
                else None
            )
            overprediction = top5_overprediction[label][architecture]
            candidate_metrics[architecture] = {
                "strict_prior_model_mae": model_mae,
                "strict_prior_baseline_current_mae": baseline_mae,
                "relative_mae_improvement": improvement,
                "evaluation_examples": len(errors),
                "evaluation_days": len(cv_days[label]),
                "top5_predicted_log_ratio_mean_overestimation": (
                    float(np.mean(overprediction)) if overprediction else None
                ),
                "top5_selection_proxy": "five_largest_predicted_log_ratios_per_race",
            }
        eligible_candidates = [
            architecture
            for architecture in POINT_MODEL_ARCHITECTURES
            if candidate_metrics[architecture]["strict_prior_model_mae"] is not None
        ]
        best_architecture = (
            min(
                eligible_candidates,
                key=lambda architecture: (
                    float(
                        candidate_metrics[architecture][
                            "strict_prior_model_mae"
                        ]
                    ),
                    architecture,
                ),
            )
            if eligible_candidates
            else None
        )
        best_mae = (
            float(
                candidate_metrics[best_architecture]["strict_prior_model_mae"]
            )
            if best_architecture is not None
            else None
        )
        best_improvement = (
            float(candidate_metrics[best_architecture]["relative_mae_improvement"])
            if best_architecture is not None
            and candidate_metrics[best_architecture]["relative_mae_improvement"]
            is not None
            else None
        )
        data_ready = bool(
            len(training_dates_by_horizon[label]) >= minimum_training_days
            and training_races_by_horizon[label] >= minimum_training_races
            and examples_by_horizon[label] >= minimum_examples_per_horizon
            and len(cv_days[label]) >= minimum_calibration_days
        )
        adopt_model = bool(
            data_ready
            and best_architecture is not None
            and best_improvement is not None
            and best_improvement >= minimum_relative_mae_improvement
        )
        if not data_ready:
            selection_reason = "insufficient_strict_prior_data"
        elif adopt_model:
            selection_reason = "strict_prior_mae_improves_baseline"
        else:
            selection_reason = "strict_prior_mae_not_better_than_current_baseline"
        selected_mode = "model" if adopt_model else "current_odds_baseline"
        selected_architecture = best_architecture if adopt_model else None
        final_model = (
            _fit_point_model(
                [row for row in examples if row["label"] == label],
                regularization,
                architecture=str(selected_architecture),
            )
            if selected_architecture is not None
            else None
        )
        selected_mae = best_mae if adopt_model else baseline_mae
        horizon_selection[label] = {
            "ready": data_ready,
            "selected_mode": selected_mode,
            "selected_architecture": selected_architecture,
            "selection_reason": selection_reason,
            "strict_prior_baseline_current_mae": baseline_mae,
            "strict_prior_model_mae": best_mae,
            "strict_prior_selected_mae": selected_mae,
            "strict_prior_selected_relative_improvement": (
                best_improvement if adopt_model else 0.0
            ),
            "minimum_relative_mae_improvement": (
                minimum_relative_mae_improvement
            ),
            "candidate_metrics": candidate_metrics,
        }
        point_models[label] = {
            **horizon_selection[label],
            "model": final_model,
        }
        selected_residuals[label] = (
            candidate_residuals[label][str(selected_architecture)]
            if selected_architecture is not None
            else baseline_residuals[label]
        )

    lower_by_horizon: dict[str, dict[str, object]] = {}
    for label in CHECKPOINT_LABELS:
        residual_by_day = selected_residuals[label]
        days = sorted(residual_by_day)
        daily_lower: dict[str, float] = {}
        daily_counts: dict[str, int] = {}
        for day in days:
            values = residual_by_day[day]
            daily_counts[day] = len(values)
            daily_lower[day] = _finite_sample_lower_rank(
                values, target_coverage=1.0 - lower_quantile
            )[0]
        ready = len(days) >= minimum_calibration_days
        outer = (
            _finite_sample_lower_rank(
                list(daily_lower.values()),
                target_coverage=1.0 - lower_quantile,
            )
            if daily_lower
            else None
        )
        lower_by_horizon[label] = {
            "ready": ready,
            "residual_log_ratio_adjustment": (
                min(0.0, float(outer[0])) if outer is not None else None
            ),
            "finite_sample_unit": "prior_day_cluster",
            "effective_sample_days": len(days),
            "calibration_dates": days,
            "daily_lower_residuals": daily_lower,
            "daily_ticket_counts": daily_counts,
            "finite_sample_rank": outer[1] if outer is not None else None,
            "finite_sample_coverage": outer[2] if outer is not None else None,
        }

    point_ready = all(
        bool(point_models[label]["ready"]) for label in CHECKPOINT_LABELS
    )
    quantile_ready = all(
        bool(lower_by_horizon[label]["ready"]) for label in CHECKPOINT_LABELS
    )
    strict_folds = all(bool(fold["strict_prior_day"]) for fold in folds)
    trained_through = training_dates[-1] if training_dates else None
    strict_training_boundary = trained_through is None or trained_through < target_date
    reasons: list[str] = []
    if len(training_dates) < minimum_training_days:
        reasons.append("insufficient_training_days")
    if len(eligible_races) < minimum_training_races:
        reasons.append("insufficient_training_races")
    if any(
        examples_by_horizon[label] < minimum_examples_per_horizon
        for label in CHECKPOINT_LABELS
    ):
        reasons.append("insufficient_training_examples_by_horizon")
    if not point_ready:
        reasons.append("insufficient_horizon_point_training_or_strict_prior_cv")
    if not quantile_ready:
        reasons.append("insufficient_strict_prior_calibration")
    if not strict_training_boundary or not strict_folds:
        reasons.append("date_boundary_audit_failed")

    training_examples_by_source = {
        source_name: sum(
            row["teacher_source"] == source_name for row in examples
        )
        for source_name in ("official_closing_odds", "closing_odds_fallback")
    }
    teacher_provenance = {
        "selection_policy": (
            "official_closing_odds_when_120_valid_else_closing_odds_fallback"
        ),
        "official_required_points": EXPECTED_COMBINATIONS,
        "selected_races_by_source": dict(sorted(selected_teacher_races.items())),
        "selected_tickets_by_source": dict(
            sorted(selected_teacher_tickets.items())
        ),
        "training_races_by_source": {
            source_name: len(training_teacher_races.get(source_name, set()))
            for source_name in (
                "official_closing_odds",
                "closing_odds_fallback",
            )
        },
        "training_examples_by_source": training_examples_by_source,
        "official_closing_odds_incomplete_races": official_incomplete_races,
        "missing_teacher_races": missing_teacher_races,
        "robustization": robust_teacher,
    }
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "ready": bool(
            point_ready and quantile_ready and strict_training_boundary and strict_folds
        ),
        "not_ready_reasons": reasons,
        "prediction_date": target_date,
        "trained_through_date": trained_through,
        "checkpoint_offsets_seconds": list(CHECKPOINT_OFFSETS_SECONDS),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "feature_names": list(FEATURE_NAMES),
        "teacher": "winsorized_log(selected_closing_odds/current_odds)",
        "teacher_provenance": teacher_provenance,
        "point_models": point_models,
        "selection_conformal_required": True,
        "selection_conformal_requirement_reason": (
            "top5_selection_proxy_is_diagnostic_only_and_may_be_log_overestimated"
        ),
        "lower_quantile_model": {
            "ready": quantile_ready,
            "model_type": (
                "strict_prior_day_cluster_finite_sample_residual_lower_quantile"
            ),
            "quantile": float(lower_quantile),
            "monotone_safe_side": True,
            "finite_sample_unit": "prior_day_cluster",
            "by_horizon": lower_by_horizon,
        },
        "training_summary": {
            "training_dates": training_dates,
            "training_days": len(training_dates),
            "training_races": len(eligible_races),
            "training_examples": len(examples),
            "training_examples_by_horizon": examples_by_horizon,
            "training_dates_by_horizon": training_dates_by_horizon,
            "training_races_by_horizon": training_races_by_horizon,
            "missing_checkpoint_races_by_horizon": missing_counts,
            "incomplete_checkpoint_races_by_horizon": incomplete_counts,
        },
        "minimum_data": {
            "training_days": int(minimum_training_days),
            "training_races": int(minimum_training_races),
            "examples_per_horizon": int(minimum_examples_per_horizon),
            "calibration_warmup_days": int(calibration_warmup_days),
            "calibration_days": int(minimum_calibration_days),
            "relative_mae_improvement_for_model_adoption": float(
                minimum_relative_mae_improvement
            ),
        },
        "boundary_audit": {
            "input_races": len(source),
            "eligible_prior_races": len(prior),
            "excluded_non_past_races": excluded_non_past,
            "excluded_invalid_date_races": excluded_invalid_date,
            "prediction_date": target_date,
            "trained_through_date": trained_through,
            "strict_training_boundary": strict_training_boundary,
            "calibration_folds": folds,
            "strict_calibration_boundaries": strict_folds,
            "future_checkpoint_imputation": False,
        },
    }


def forecast_closing_odds_multihorizon_v11(
    race: Mapping[str, object],
    model: Mapping[str, object],
    *,
    as_of_offset_seconds: object,
    prediction_date: object | None = None,
) -> dict[str, object]:
    """Forecast only checkpoints observable at the explicit as-of boundary."""
    if str(model.get("model_name")) != MODEL_NAME:
        raise ValueError("not a closing odds multihorizon v11 artifact")
    race_date = _iso_date(
        prediction_date if prediction_date is not None else race.get("race_date"),
        "prediction_date",
    )
    artifact_date = _iso_date(model.get("prediction_date"), "artifact prediction_date")
    trained_through = model.get("trained_through_date")
    if race_date < artifact_date:
        raise ValueError("prediction_date precedes artifact boundary")
    if (
        trained_through is not None
        and _iso_date(trained_through, "trained_through_date") >= race_date
    ):
        raise ValueError("artifact is not strictly prior to prediction_date")

    as_of = _required_checkpoint_offset(
        as_of_offset_seconds, "as_of_offset_seconds"
    )
    checkpoints = normalize_labeled_checkpoints(
        race, as_of_offset_seconds=as_of
    )
    point_models = model.get("point_models")
    quantile_model = model.get("lower_quantile_model")
    if not isinstance(point_models, Mapping) or not isinstance(
        quantile_model, Mapping
    ):
        raise ValueError("v11 artifact is missing model components")
    by_horizon = quantile_model.get("by_horizon")
    if not isinstance(by_horizon, Mapping):
        raise ValueError("v11 artifact is missing horizon calibration")
    artifact_ready = bool(model.get("ready"))
    predictions: dict[str, dict[str, object]] = {}
    for horizon in CHECKPOINT_OFFSETS_SECONDS:
        label = checkpoint_label(horizon)
        if horizon < as_of:
            predictions[label] = {
                "ready": False,
                "reason": "after_as_of_checkpoint",
                "target_offset_seconds": horizon,
                "point_final_odds": {},
                "lower_final_odds": {},
                "used_checkpoint_offsets": [],
                "future_checkpoint_offsets_used": [],
            }
            continue
        snapshot = checkpoints.get(label)
        calibration = by_horizon.get(label)
        horizon_model = point_models.get(label)
        local_ready = bool(
            artifact_ready
            and snapshot is not None
            and isinstance(horizon_model, Mapping)
            and horizon_model.get("ready")
            and isinstance(calibration, Mapping)
            and calibration.get("ready")
        )
        if snapshot is None:
            predictions[label] = {
                "ready": False,
                "reason": "missing_checkpoint",
                "target_offset_seconds": horizon,
                "point_final_odds": {},
                "lower_final_odds": {},
                "used_checkpoint_offsets": [],
                "future_checkpoint_offsets_used": [],
            }
            continue
        if not local_ready:
            predictions[label] = {
                "ready": False,
                "reason": "model_not_ready",
                "target_offset_seconds": horizon,
                "point_final_odds": {},
                "lower_final_odds": {},
                "used_checkpoint_offsets": [],
                "future_checkpoint_offsets_used": [],
            }
            continue
        adjustment = min(
            0.0, float(calibration["residual_log_ratio_adjustment"])
        )
        selected_mode = str(horizon_model["selected_mode"])
        fitted_model = horizon_model.get("model")
        if selected_mode == "model" and not isinstance(fitted_model, Mapping):
            raise ValueError(f"v11 {label} selected model is missing")
        if selected_mode not in ("model", "current_odds_baseline"):
            raise ValueError(f"v11 {label} selected mode is invalid")
        current = _snapshot_odds(snapshot)
        point_values: dict[str, float] = {}
        lower_values: dict[str, float] = {}
        traces: list[dict[str, object]] = []
        for combination in sorted(current):
            vector, trace = build_checkpoint_feature_vector(
                race,
                checkpoint=label,
                combination=combination,
                as_of_offset_seconds=as_of,
            )
            log_ratio = (
                float(np.clip(_point_log_ratio(vector, fitted_model), -8.0, 8.0))
                if selected_mode == "model"
                else 0.0
            )
            point = (
                current[combination]
                if selected_mode == "current_odds_baseline"
                else min(
                    MAX_ODDS,
                    max(MIN_ODDS, current[combination] * math.exp(log_ratio)),
                )
            )
            lower = min(
                point,
                min(MAX_ODDS, max(MIN_ODDS, point * math.exp(adjustment))),
            )
            point_values[combination] = point
            lower_values[combination] = lower
            traces.append(trace)
        used = sorted(
            {
                value
                for trace in traces
                for value in trace["used_checkpoint_offsets"]
            },
            reverse=True,
        )
        future = sorted(
            {
                value
                for trace in traces
                for value in trace["future_checkpoint_offsets_used"]
            },
            reverse=True,
        )
        predictions[label] = {
            "ready": True,
            "reason": None,
            "target_offset_seconds": horizon,
            "point_source": selected_mode,
            "point_selection_reason": horizon_model["selection_reason"],
            "point_final_odds": point_values,
            "lower_final_odds": lower_values,
            "lower_residual_log_ratio_adjustment": adjustment,
            "used_checkpoint_offsets": used,
            "future_checkpoint_offsets_used": future,
        }
    allowed_labels = [
        checkpoint_label(horizon)
        for horizon in CHECKPOINT_OFFSETS_SECONDS
        if horizon >= as_of
    ]
    blocked_labels = [
        checkpoint_label(horizon)
        for horizon in CHECKPOINT_OFFSETS_SECONDS
        if horizon < as_of
    ]
    missing = [label for label in allowed_labels if label not in checkpoints]
    boundary_passed = bool(
        trained_through is None
        or _iso_date(trained_through, "trained_through_date") < race_date
    )
    return {
        "model_name": MODEL_NAME,
        "ready": artifact_ready,
        "prediction_date": race_date,
        "artifact_prediction_date": artifact_date,
        "trained_through_date": trained_through,
        "as_of_offset_seconds": as_of,
        "predictions": predictions,
        "missing_checkpoints": missing,
        "after_as_of_checkpoints": blocked_labels,
        "future_checkpoint_imputation": False,
        "checkpoint_access_audit": {
            "allowed_checkpoint_labels": allowed_labels,
            "blocked_future_checkpoint_labels": blocked_labels,
            "future_checkpoint_offsets_used": sorted(
                {
                    offset
                    for row in predictions.values()
                    for offset in row["future_checkpoint_offsets_used"]
                },
                reverse=True,
            ),
        },
        "boundary_audit_passed": boundary_passed,
    }


def closing_odds_multihorizon_v11_metrics(
    races: Iterable[Mapping[str, object]],
    model: Mapping[str, object],
    *,
    as_of_offset_seconds: object,
) -> dict[str, object]:
    """Evaluate the selected point policy at one exact as-of horizon."""
    as_of = _required_checkpoint_offset(
        as_of_offset_seconds, "as_of_offset_seconds"
    )
    label = checkpoint_label(as_of)
    baseline_errors: list[float] = []
    selected_errors: list[float] = []
    lower_covered: list[bool] = []
    top5_overprediction: list[float] = []
    source_races: dict[str, int] = defaultdict(int)
    source_tickets: dict[str, int] = defaultdict(int)
    point_sources: dict[str, int] = defaultdict(int)
    official_incomplete = missing_teacher = missing_prediction = 0
    evaluated_races = 0
    for race in races:
        teacher, teacher_source, incomplete_official = _teacher_selection(race)
        official_incomplete += int(incomplete_official)
        if teacher_source is None:
            missing_teacher += 1
            continue
        source_races[str(teacher_source)] += 1
        source_tickets[str(teacher_source)] += len(teacher)
        forecast = forecast_closing_odds_multihorizon_v11(
            race,
            model,
            as_of_offset_seconds=as_of,
        )
        row = forecast["predictions"][label]
        if not row["ready"]:
            missing_prediction += 1
            continue
        current_snapshot = normalize_labeled_checkpoints(
            race, as_of_offset_seconds=as_of
        ).get(label)
        current = (
            _snapshot_odds(current_snapshot)
            if isinstance(current_snapshot, Mapping)
            else {}
        )
        point = row["point_final_odds"]
        lower = row["lower_final_odds"]
        if (
            set(current) != set(teacher)
            or set(point) != set(teacher)
            or set(lower) != set(teacher)
        ):
            missing_prediction += 1
            continue
        combinations = sorted(teacher)
        target_log = np.asarray(
            [math.log(teacher[key]) for key in combinations],
            dtype=np.float64,
        )
        current_log = np.asarray(
            [math.log(current[key]) for key in combinations],
            dtype=np.float64,
        )
        point_log = np.asarray(
            [math.log(float(point[key])) for key in combinations],
            dtype=np.float64,
        )
        lower_values = np.asarray(
            [float(lower[key]) for key in combinations],
            dtype=np.float64,
        )
        baseline_errors.extend(np.abs(target_log - current_log).tolist())
        selected_errors.extend(np.abs(target_log - point_log).tolist())
        lower_covered.extend(
            (
                np.asarray([teacher[key] for key in combinations])
                >= lower_values
            ).tolist()
        )
        predicted_log_ratios = point_log - current_log
        selected = sorted(
            range(len(combinations)),
            key=lambda index: (
                -float(predicted_log_ratios[index]),
                combinations[index],
            ),
        )[:5]
        top5_overprediction.extend(
            (point_log[selected] - target_log[selected]).tolist()
        )
        point_sources[str(row["point_source"])] += 1
        evaluated_races += 1
    baseline_mae = (
        float(np.mean(baseline_errors)) if baseline_errors else None
    )
    selected_mae = (
        float(np.mean(selected_errors)) if selected_errors else None
    )
    return {
        "model_name": MODEL_NAME,
        "as_of_offset_seconds": as_of,
        "checkpoint_label": label,
        "evaluation_races": evaluated_races,
        "evaluation_tickets": len(selected_errors),
        "baseline_current_log_mae": baseline_mae,
        "selected_point_log_mae": selected_mae,
        "selected_relative_mae_improvement": (
            1.0 - selected_mae / baseline_mae
            if selected_mae is not None
            and baseline_mae is not None
            and baseline_mae > 0.0
            else None
        ),
        "lower_bound_coverage": (
            float(np.mean(lower_covered)) if lower_covered else None
        ),
        "top5_predicted_log_ratio_mean_overestimation": (
            float(np.mean(top5_overprediction))
            if top5_overprediction
            else None
        ),
        "top5_selection_proxy": "five_largest_predicted_log_ratios_per_race",
        "selection_conformal_required": True,
        "point_source_races": dict(sorted(point_sources.items())),
        "missing_prediction_races": missing_prediction,
        "teacher_provenance": {
            "selection_policy": (
                "official_closing_odds_when_120_valid_else_closing_odds_fallback"
            ),
            "races_by_source": dict(sorted(source_races.items())),
            "tickets_by_source": dict(sorted(source_tickets.items())),
            "official_closing_odds_incomplete_races": official_incomplete,
            "missing_teacher_races": missing_teacher,
        },
    }


def attach_closing_odds_multihorizon_v11(
    races: Iterable[Mapping[str, object]],
    model: Mapping[str, object],
    *,
    as_of_offset_seconds: object,
) -> list[dict[str, object]]:
    """Pure convenience adapter for later v10 integration."""
    result: list[dict[str, object]] = []
    for race in races:
        item = dict(race)
        item["closing_odds_multihorizon_v11"] = (
            forecast_closing_odds_multihorizon_v11(
                race,
                model,
                as_of_offset_seconds=as_of_offset_seconds,
            )
        )
        result.append(item)
    return result
