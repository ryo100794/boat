from __future__ import annotations

import math
from typing import Any


Accumulator = dict[str, dict[str, Any]]


def new_roi_attribution() -> Accumulator:
    return {}


def merge_roi_attribution(target: Accumulator, source: Accumulator) -> None:
    for dimension, source_dimension in source.items():
        target_dimension = target.setdefault(
            dimension,
            {"family": source_dimension.get("family") or "other", "buckets": {}},
        )
        for bucket, source_bucket in source_dimension.get("buckets", {}).items():
            target_bucket = target_dimension["buckets"].setdefault(
                bucket,
                {
                    "tickets": 0.0,
                    "hits": 0.0,
                    "stake_yen": 0.0,
                    "return_yen": 0.0,
                    "probability_sum": 0.0,
                    "estimated_ev_sum": 0.0,
                    "estimated_odds_sum": 0.0,
                },
            )
            for key in target_bucket:
                target_bucket[key] += float(source_bucket.get(key) or 0.0)


def update_roi_attribution(accumulator: Accumulator, ticket: dict[str, Any]) -> None:
    stake = _finite_float(ticket.get("stake_yen"))
    if stake is None or stake <= 0:
        return
    returned = _finite_float(ticket.get("return_yen")) or 0.0
    hit = 1.0 if ticket.get("hit") else 0.0
    probability = _finite_float(ticket.get("probability")) or 0.0
    estimated_ev = _finite_float(ticket.get("estimated_ev")) or 0.0
    estimated_odds = _finite_float(ticket.get("estimated_odds")) or 0.0

    for dimension, family, bucket in _ticket_dimensions(ticket):
        dimension_row = accumulator.setdefault(
            dimension,
            {"family": family, "buckets": {}},
        )
        bucket_row = dimension_row["buckets"].setdefault(
            bucket,
            {
                "tickets": 0.0,
                "hits": 0.0,
                "stake_yen": 0.0,
                "return_yen": 0.0,
                "probability_sum": 0.0,
                "estimated_ev_sum": 0.0,
                "estimated_odds_sum": 0.0,
            },
        )
        bucket_row["tickets"] += 1.0
        bucket_row["hits"] += hit
        bucket_row["stake_yen"] += stake
        bucket_row["return_yen"] += returned
        bucket_row["probability_sum"] += probability
        bucket_row["estimated_ev_sum"] += estimated_ev
        bucket_row["estimated_odds_sum"] += estimated_odds


def summarize_roi_attribution(
    accumulator: Accumulator,
    *,
    min_tickets: int = 100,
    min_stake_yen: int = 10_000,
) -> dict[str, Any]:
    dimensions = []
    for dimension, dimension_row in accumulator.items():
        buckets = []
        for bucket, raw in dimension_row.get("buckets", {}).items():
            tickets = int(raw.get("tickets") or 0)
            hits = int(raw.get("hits") or 0)
            stake = int(round(raw.get("stake_yen") or 0.0))
            returned = int(round(raw.get("return_yen") or 0.0))
            buckets.append(
                {
                    "bucket": bucket,
                    "tickets": tickets,
                    "hits": hits,
                    "hit_rate": hits / tickets if tickets else 0.0,
                    "stake_yen": stake,
                    "return_yen": returned,
                    "profit_yen": returned - stake,
                    "roi": returned / stake if stake else None,
                    "avg_probability": (raw.get("probability_sum") or 0.0) / tickets if tickets else None,
                    "avg_estimated_ev": (raw.get("estimated_ev_sum") or 0.0) / tickets if tickets else None,
                    "avg_estimated_odds": (raw.get("estimated_odds_sum") or 0.0) / tickets if tickets else None,
                    "eligible": tickets >= min_tickets and stake >= min_stake_yen,
                }
            )
        total_stake = sum(row["stake_yen"] for row in buckets)
        total_return = sum(row["return_yen"] for row in buckets)
        for row in buckets:
            rest_stake = total_stake - row["stake_yen"]
            rest_return = total_return - row["return_yen"]
            rest_roi = rest_return / rest_stake if rest_stake else None
            row["rest_stake_yen"] = rest_stake
            row["rest_roi"] = rest_roi
            row["roi_vs_rest"] = row["roi"] - rest_roi if row["roi"] is not None and rest_roi is not None else None
        buckets.sort(key=lambda row: (-row["stake_yen"], str(row["bucket"])))
        eligible = [row for row in buckets if row["eligible"] and row["roi"] is not None]
        best = max(eligible, key=lambda row: row["roi"], default=None)
        worst = min(eligible, key=lambda row: row["roi"], default=None)
        roi_spread = (best["roi"] - worst["roi"]) if best and worst else None
        dimensions.append(
            {
                "dimension": dimension,
                "family": dimension_row.get("family") or "other",
                "tickets": sum(row["tickets"] for row in buckets),
                "stake_yen": sum(row["stake_yen"] for row in buckets),
                "eligible_buckets": len(eligible),
                "roi_spread": roi_spread,
                "best_bucket": _compact_bucket(best),
                "worst_bucket": _compact_bucket(worst),
                "status": "signal" if roi_spread is not None and roi_spread >= 0.15 else ("weak" if roi_spread is not None else "insufficient"),
                "buckets": buckets,
            }
        )
    dimensions.sort(
        key=lambda row: (
            row["roi_spread"] is not None,
            row["roi_spread"] or -1.0,
            row["stake_yen"],
        ),
        reverse=True,
    )
    signals = [row for row in dimensions if row["status"] == "signal"]
    return {
        "method": "selected-ticket stake-weighted ROI buckets",
        "minimum_evidence": {"tickets": min_tickets, "stake_yen": min_stake_yen},
        "dimensions": dimensions,
        "top_signals": [
            {
                "dimension": row["dimension"],
                "family": row["family"],
                "roi_spread": row["roi_spread"],
                "best_bucket": row["best_bucket"],
                "worst_bucket": row["worst_bucket"],
            }
            for row in signals[:12]
        ],
        "diagnosis": (
            "ROI differs materially across selected-ticket feature buckets. Validate the strongest signals on later time folds before changing the primary model."
            if signals
            else "No stable ROI separation is proven yet; collect more selected tickets or revise the purchase policy and features."
        ),
    }


def summarize_fold_signal_stability(fold_attributions: list[dict[str, Any]]) -> dict[str, Any]:
    fold_count = len(fold_attributions)
    required_folds = min(3, max(2, fold_count)) if fold_count else 2
    by_bucket: dict[tuple[str, str], dict[str, Any]] = {}

    for fold_index, attribution in enumerate(fold_attributions, start=1):
        for dimension_row in (attribution or {}).get("dimensions") or []:
            dimension = str(dimension_row.get("dimension") or "")
            if not dimension:
                continue
            for bucket in dimension_row.get("buckets") or []:
                lift = _finite_float(bucket.get("roi_vs_rest"))
                if not bucket.get("eligible") or lift is None:
                    continue
                bucket_name = str(bucket.get("bucket") or "missing")
                row = by_bucket.setdefault(
                    (dimension, bucket_name),
                    {
                        "dimension": dimension,
                        "bucket": bucket_name,
                        "family": dimension_row.get("family") or "other",
                        "folds": [],
                        "lifts": [],
                        "stakes": [],
                        "returns": [],
                    },
                )
                row["folds"].append(fold_index)
                row["lifts"].append(lift)
                row["stakes"].append(int(bucket.get("stake_yen") or 0))
                row["returns"].append(int(bucket.get("return_yen") or 0))

    rows = []
    for row in by_bucket.values():
        observations = len(row["folds"])
        positive = sum(1 for value in row["lifts"] if value > 0)
        negative = sum(1 for value in row["lifts"] if value < 0)
        direction_count = max(positive, negative)
        direction = "beneficial" if positive >= negative else "harmful"
        consistency = direction_count / observations if observations else 0.0
        total_stake = sum(row["stakes"])
        weighted_lift = (
            sum(lift * stake for lift, stake in zip(row["lifts"], row["stakes"])) / total_stake
            if total_stake
            else 0.0
        )
        stable = observations >= required_folds and consistency >= 0.75 and abs(weighted_lift) >= 0.05
        rows.append(
            {
                "dimension": row["dimension"],
                "bucket": row["bucket"],
                "family": row["family"],
                "signal_folds": row["folds"],
                "signal_fold_rate": observations / fold_count if fold_count else 0.0,
                "direction": direction,
                "positive_folds": positive,
                "negative_folds": negative,
                "direction_consistency": consistency,
                "weighted_roi_lift_vs_rest": weighted_lift,
                "stake_yen": total_stake,
                "return_yen": sum(row["returns"]),
                "status": "stable" if stable else "unstable",
            }
        )

    if not rows:
        rows = _legacy_fold_signal_stability(fold_attributions, required_folds)
    rows.sort(
        key=lambda row: (
            row["status"] == "stable",
            row["signal_fold_rate"],
            row["direction_consistency"],
            abs(row.get("weighted_roi_lift_vs_rest") or row.get("mean_roi_spread") or 0.0),
        ),
        reverse=True,
    )
    stable_count = sum(1 for row in rows if row["status"] == "stable")
    return {
        "method": "bucket ROI lift versus same-dimension remainder across forward time folds",
        "folds": fold_count,
        "required_folds": required_folds,
        "minimum_direction_consistency": 0.75,
        "minimum_abs_roi_lift": 0.05,
        "stable_signals": stable_count,
        "signals": rows[:48],
        "gate": "candidate" if stable_count else "insufficient",
    }


def _legacy_fold_signal_stability(
    fold_attributions: list[dict[str, Any]], required_folds: int
) -> list[dict[str, Any]]:
    by_dimension: dict[str, dict[str, Any]] = {}
    for fold_index, attribution in enumerate(fold_attributions, start=1):
        for signal in (attribution or {}).get("top_signals") or []:
            dimension = str(signal.get("dimension") or "")
            if not dimension:
                continue
            row = by_dimension.setdefault(
                dimension,
                {"family": signal.get("family") or "other", "folds": [], "spreads": [], "directions": []},
            )
            best = str((signal.get("best_bucket") or {}).get("bucket") or "")
            worst = str((signal.get("worst_bucket") or {}).get("bucket") or "")
            row["folds"].append(fold_index)
            row["spreads"].append(float(signal.get("roi_spread") or 0.0))
            row["directions"].append((best, worst))
    rows = []
    fold_count = len(fold_attributions)
    for dimension, raw in by_dimension.items():
        observations = len(raw["folds"])
        counts: dict[tuple[str, str], int] = {}
        for direction in raw["directions"]:
            counts[direction] = counts.get(direction, 0) + 1
        consistency = max(counts.values(), default=0) / observations if observations else 0.0
        mean_spread = sum(raw["spreads"]) / observations if observations else 0.0
        stable = observations >= required_folds and consistency >= 0.75 and mean_spread >= 0.05
        rows.append({
            "dimension": dimension,
            "family": raw["family"],
            "signal_folds": raw["folds"],
            "signal_fold_rate": observations / fold_count if fold_count else 0.0,
            "direction_consistency": consistency,
            "mean_roi_spread": mean_spread,
            "status": "stable" if stable else "unstable",
        })
    return rows

def _ticket_dimensions(ticket: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = [
        ("venue", "race", _text_bucket(ticket.get("jcd"))),
        ("race_no", "race", _number_bucket(ticket.get("rno"), ((4, "1-4"), (8, "5-8"), (12, "9-12")))),
        ("first_lane", "lane", _combination_lane(ticket.get("combination"), 0)),
        ("second_lane", "lane", _combination_lane(ticket.get("combination"), 1)),
        ("third_lane", "lane", _combination_lane(ticket.get("combination"), 2)),
        ("odds_source", "market", _text_bucket(ticket.get("odds_source"))),
        ("probability", "purchase", _number_bucket(ticket.get("probability"), ((0.005, "<0.005"), (0.01, "0.005-0.010"), (0.02, "0.010-0.020"), (0.04, "0.020-0.040"), (0.08, "0.040-0.080"), (math.inf, ">=0.080")))),
        ("estimated_odds", "market", _number_bucket(ticket.get("estimated_odds"), ((10, "<10"), (20, "10-20"), (40, "20-40"), (80, "40-80"), (150, "80-150"), (math.inf, ">=150")))),
        ("estimated_ev", "purchase", _number_bucket(ticket.get("estimated_ev"), ((1.10, "1.00-1.10"), (1.20, "1.10-1.20"), (1.50, "1.20-1.50"), (2.00, "1.50-2.00"), (3.00, "2.00-3.00"), (math.inf, ">=3.00")))),
        ("kelly_fraction", "purchase", _number_bucket(ticket.get("kelly_fraction"), ((0.001, "<0.001"), (0.0025, "0.001-0.0025"), (0.005, "0.0025-0.005"), (0.01, "0.005-0.010"), (0.02, "0.010-0.020"), (math.inf, ">=0.020")))),
        ("stake_yen", "purchase", _number_bucket(ticket.get("stake_yen"), ((100, "100"), (200, "200"), (300, "300"), (500, "400-500"), (1000, "600-1000"), (math.inf, ">1000")), inclusive=True)),
        ("payout_history_count", "market", _number_bucket(ticket.get("payout_history_count"), ((0, "0"), (25, "1-25"), (100, "26-100"), (500, "101-500"), (math.inf, ">500")), inclusive=True)),
    ]
    context = ticket.get("feature_context") or {}
    if isinstance(context, dict):
        for key, value in context.items():
            rows.append((f"feature:{key}", _feature_family(str(key)), _feature_bucket(str(key), value)))
    return rows


def _feature_bucket(key: str, value: Any) -> str:
    if isinstance(value, str):
        return value or "missing"
    number = _finite_float(value)
    if number is None or number < 0:
        return "missing"
    if key.endswith("_rank"):
        return str(int(round(number)))
    if "win_rate_s" in key or "top2_rate_s" in key or "top3_rate_s" in key or "series_win_rate" in key:
        return _number_bucket(number, ((0.1, "0.0-0.1"), (0.2, "0.1-0.2"), (0.4, "0.2-0.4"), (0.6, "0.4-0.6"), (0.8, "0.6-0.8"), (math.inf, ">=0.8")))
    if "avg_finish" in key or "avg_rank" in key:
        return _number_bucket(number, ((2, "<2"), (3, "2-3"), (4, "3-4"), (5, "4-5"), (math.inf, ">=5")))
    return _number_bucket(number, ((0, "0"), (1, "0-1"), (2, "1-2"), (4, "2-4"), (8, "4-8"), (math.inf, ">=8")), inclusive=True)


def _feature_family(key: str) -> str:
    if "racer" in key or "class" in key or "origin" in key or "national" in key or "local" in key:
        return "racer"
    if "motor" in key:
        return "motor"
    if "boat" in key:
        return "boat"
    if "series" in key:
        return "series"
    if "venue" in key or "race_" in key or "month" in key or "weekday" in key:
        return "race"
    return "feature"


def _number_bucket(value: Any, bounds: tuple[tuple[float, str], ...], *, inclusive: bool = False) -> str:
    number = _finite_float(value)
    if number is None:
        return "missing"
    for upper, label in bounds:
        if number <= upper if inclusive else number < upper:
            return label
    return bounds[-1][1]


def _combination_lane(combination: Any, index: int) -> str:
    parts = str(combination or "").split("-")
    return parts[index] if len(parts) > index and parts[index] else "missing"


def _text_bucket(value: Any) -> str:
    text = str(value or "").strip()
    return text or "missing"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _compact_bucket(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {key: row.get(key) for key in ("bucket", "tickets", "stake_yen", "profit_yen", "roi", "hit_rate")}
