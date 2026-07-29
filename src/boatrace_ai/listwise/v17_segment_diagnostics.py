from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_NAME = "v17_segment_diagnostics"
RESULT_FIELDS = frozenset({
    "hit", "is_hit", "return_yen", "payout_yen", "actual_payout_yen",
    "result", "actual_combination",
})
SELECTED_KEYS = frozenset({"selected_candidates", "selected_sample"})
STAKE_UNIT_YEN = 100


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _first(row: Mapping[str, Any], names: Sequence[str]) -> object | None:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _date(row: Mapping[str, Any], inherited: Mapping[str, Any]) -> str:
    value = _first(row, ("race_date", "date", "evaluation_date"))
    if value is None:
        value = _first(inherited, ("race_date", "date", "evaluation_date"))
    return str(value or "")[:10]


def _rno(row: Mapping[str, Any], race_id: str) -> int | None:
    value = _first(row, ("rno", "race_no", "race_number"))
    try:
        result = int(value) if value is not None else int(race_id[-2:])
    except (TypeError, ValueError):
        return None
    return result if 1 <= result <= 12 else None


def _trend_value(row: Mapping[str, Any]) -> tuple[str, float | str] | None:
    raw = _first(row, (
        "t5_trend", "t5_odds_trend", "odds_t5_trend", "odds_trend",
        "t5_slope", "t5_change_ratio",
    ))
    if isinstance(raw, Mapping):
        raw = _first(raw, ("slope", "change_ratio", "value", "direction"))
    number = _finite(raw)
    if number is not None:
        return "numeric", number
    if raw is None:
        return None
    label = str(raw).strip().lower()
    return ("label", label) if label else None


def _settlement(row: Mapping[str, Any], stake_yen: int) -> dict[str, Any]:
    returned = _finite(row.get("return_yen"))
    if returned is None:
        payout = _finite(_first(row, ("payout_yen", "actual_payout_yen")))
        hit = bool(_first(row, ("hit", "is_hit")))
        returned = payout * (stake_yen / STAKE_UNIT_YEN) if hit and payout else 0.0
    returned_yen = max(0, int(round(returned)))
    return {
        "hit": bool(_first(row, ("hit", "is_hit"))) or returned_yen > 0,
        "return_yen": returned_yen,
    }


def _normalise_candidate(
    row: Mapping[str, Any], inherited: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    race_id = str(_first(row, ("race_id", "race")) or "")
    combination = str(_first(row, ("combination", "bet", "trifecta")) or "")
    race_date = _date(row, inherited)
    if not race_date or not race_id or len(combination) != 3 or not combination.isdigit():
        return None
    probability = _finite(_first(row, (
        "probability", "raw_probability", "model_probability",
    )))
    odds = _finite(_first(row, (
        "estimated_odds", "odds", "estimated_closing_odds",
        "point_final_odds", "t300_odds",
    )))
    estimated_ev = _finite(row.get("estimated_ev"))
    if estimated_ev is None and probability is not None and odds is not None:
        estimated_ev = probability * odds
    ratio = _finite(_first(row, (
        "model_market_ratio", "probability_market_ratio", "model_to_market_ratio",
    )))
    market_probability = _finite(_first(row, (
        "market_probability", "t300_market_probability",
    )))
    if ratio is None and probability is not None and market_probability:
        ratio = probability / market_probability
    if ratio is None and probability is not None and odds:
        ratio = probability * odds
    stake = _finite(_first(row, ("stake_yen", "stake")))
    stake_yen = max(STAKE_UNIT_YEN, int(round(stake or STAKE_UNIT_YEN)))
    venue = str(_first(row, ("jcd", "venue", "venue_code")) or "").strip()
    decision = {
        "race_date": race_date,
        "race_id": race_id,
        "combination": combination,
        "stake_yen": stake_yen,
        "estimated_ev": estimated_ev,
        "odds": odds,
        "probability": probability,
        "model_market_ratio": ratio,
        "venue": venue or None,
        "rno": _rno(row, race_id),
        "t5_trend": _trend_value(row),
    }
    return decision, _settlement(row, stake_yen)


def _selected_lists(
    value: object,
    inherited: Mapping[str, Any] | None = None,
) -> Iterable[tuple[list[Mapping[str, Any]], Mapping[str, Any], str]]:
    if not isinstance(value, Mapping):
        return
    context = dict(inherited or {})
    for key in ("race_date", "date", "evaluation_date", "jcd", "venue"):
        if value.get(key) is not None:
            context[key] = value[key]
    found = False
    for key in SELECTED_KEYS:
        rows = value.get(key)
        if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
            found = True
            yield rows, context, key
    for key, child in value.items():
        if key in SELECTED_KEYS:
            continue
        if isinstance(child, Mapping):
            yield from _selected_lists(child, context)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    yield from _selected_lists(item, context)


def extract_selected_candidates(
    job: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    """Extract selected candidates while keeping outcomes in a separate map."""
    decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    settlements: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources: dict[str, int] = defaultdict(int)
    invalid = duplicates = 0
    for rows, inherited, source in _selected_lists(job):
        sources[source] += 1
        for raw in rows:
            normalised = _normalise_candidate(raw, inherited)
            if normalised is None:
                invalid += 1
                continue
            decision, settlement = normalised
            key = decision["race_date"], decision["race_id"], decision["combination"]
            if key in decisions:
                duplicates += 1
                if decisions[key] != decision or settlements[key] != settlement:
                    raise ValueError(f"conflicting duplicate selected candidate: {key}")
                continue
            decisions[key] = decision
            settlements[key] = settlement
    ordered = [decisions[key] for key in sorted(decisions)]
    return ordered, settlements, {
        "selected_list_sources": dict(sorted(sources.items())),
        "invalid_rows": invalid,
        "duplicate_rows_ignored": duplicates,
    }


def _bin(value: float | None, cuts: Sequence[float], labels: Sequence[str]) -> str | None:
    if value is None:
        return None
    for upper, label in zip(cuts, labels):
        if value < upper:
            return label
    return labels[-1]


def _lane_pattern(combination: str) -> str:
    lanes = [int(value) for value in combination]
    def direction(left: int, right: int) -> str:
        return "out" if right > left else "in"
    return f"first={lanes[0]}|{direction(lanes[0], lanes[1])}-{direction(lanes[1], lanes[2])}"


def segment_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    values = (
        ("estimated_ev", _bin(row.get("estimated_ev"), (0.9, 1.0, 1.1, 1.25),
                              ("<0.90", "0.90-0.99", "1.00-1.09", "1.10-1.24", ">=1.25"))),
        ("odds", _bin(row.get("odds"), (10.0, 30.0, 100.0),
                      ("<10", "10-29.9", "30-99.9", ">=100"))),
        ("model_market_ratio", _bin(row.get("model_market_ratio"), (1.0, 1.5, 2.0),
                                    ("<1.0", "1.0-1.49", "1.5-1.99", ">=2.0"))),
    )
    for family, value in values:
        if value is not None:
            labels.append(f"{family}:{value}")
    if row.get("venue") is not None:
        labels.append(f"venue:{row['venue']}")
    if row.get("rno") is not None:
        labels.append(f"rno:{row['rno']}")
    labels.append(f"lane_pattern:{_lane_pattern(str(row['combination']))}")
    trend = row.get("t5_trend")
    if trend:
        kind, value = trend
        if kind == "numeric":
            trend_label = _bin(float(value), (-0.05, 0.05), ("falling", "stable", "rising"))
        else:
            trend_label = str(value)
        labels.append(f"t5_trend:{trend_label}")
    return tuple(labels)


def _metrics(
    rows: Iterable[Mapping[str, Any]],
    settlements: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    selected = list(rows)
    returns = [int(settlements[(row["race_date"], row["race_id"], row["combination"])]["return_yen"])
               for row in selected]
    stake = sum(int(row["stake_yen"]) for row in selected)
    returned = sum(returns)
    largest = max(returns, default=0)
    by_day: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, value in zip(selected, returns):
        day = by_day[str(row["race_date"])]
        day[0] += int(row["stake_yen"])
        day[1] += value
    active_days = len(by_day)
    profitable_days = sum(returned_day > staked for staked, returned_day in by_day.values())
    return {
        "days": active_days,
        "tickets": len(selected),
        "hits": sum(value > 0 for value in returns),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "largest_hit_return_yen": largest,
        "roi_excluding_largest_hit": (returned - largest) / stake if stake else None,
        "profitable_days": profitable_days,
        "profitable_day_rate": profitable_days / active_days if active_days else None,
    }


def _sufficiency(metrics: Mapping[str, Any], *, min_days: int, min_tickets: int,
                 min_hits: int) -> dict[str, Any]:
    missing = {
        "days": max(0, min_days - int(metrics["days"])),
        "tickets": max(0, min_tickets - int(metrics["tickets"])),
        "hits": max(0, min_hits - int(metrics["hits"])),
    }
    return {"sufficient": not any(missing.values()), "missing": missing}


def _fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = [{key: row.get(key) for key in (
        "race_date", "race_id", "combination", "stake_yen", "estimated_ev",
        "odds", "probability", "model_market_ratio", "venue", "rno", "t5_trend",
    )} for row in sorted(rows, key=lambda item: (
        item["race_date"], item["race_id"], item["combination"]
    ))]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def diagnose_segments(
    job: Mapping[str, Any], *, min_days: int = 3, min_tickets: int = 20,
    min_hits: int = 2, max_prequential_segments: int = 5,
) -> dict[str, Any]:
    decisions, settlements, extraction = extract_selected_candidates(job)
    dates = sorted({str(row["race_date"]) for row in decisions})
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_key: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for row in decisions:
        key = row["race_date"], row["race_id"], row["combination"]
        labels = segment_labels(row)
        labels_by_key[key] = labels
        for label in labels:
            by_segment[label].append(row)

    segments = []
    for label in sorted(by_segment):
        rows = by_segment[label]
        aggregate = _metrics(rows, settlements)
        segments.append({
            "segment": label,
            "family": label.split(":", 1)[0],
            "aggregate": aggregate,
            "sample_sufficiency": _sufficiency(
                aggregate, min_days=min_days, min_tickets=min_tickets, min_hits=min_hits
            ),
            "leave_one_day_out": [{
                "holdout_date": date,
                "training_excluding_holdout": _metrics(
                    (row for row in rows if row["race_date"] != date), settlements
                ),
                "holdout": _metrics(
                    (row for row in rows if row["race_date"] == date), settlements
                ),
            } for date in dates],
        })

    prequential_days = []
    all_prequential_rows: list[dict[str, Any]] = []
    for date in dates:
        prior_dates = [value for value in dates if value < date]
        eligible = []
        rejected = defaultdict(int)
        for label, rows in by_segment.items():
            prior_rows = [row for row in rows if row["race_date"] < date]
            prior = _metrics(prior_rows, settlements)
            sufficient = _sufficiency(
                prior, min_days=min_days, min_tickets=min_tickets, min_hits=min_hits
            )
            if not sufficient["sufficient"]:
                rejected["insufficient_prior_sample"] += 1
                continue
            if (prior["roi_excluding_largest_hit"] or 0.0) <= 1.0:
                rejected["prior_roi_excluding_largest_not_above_one"] += 1
                continue
            if (prior["profitable_day_rate"] or 0.0) < 0.5:
                rejected["prior_profitable_day_rate_below_half"] += 1
                continue
            eligible.append((label, prior))
        eligible.sort(key=lambda item: (
            -float(item[1]["roi_excluding_largest_hit"]),
            -float(item[1]["profitable_day_rate"]), -int(item[1]["hits"]), item[0],
        ))
        chosen = eligible[:max_prequential_segments]
        chosen_labels = {label for label, _ in chosen}
        holdout_rows = [row for row in decisions if row["race_date"] == date]
        selected = [row for row in holdout_rows if chosen_labels.intersection(
            labels_by_key[(row["race_date"], row["race_id"], row["combination"])]
        )]
        all_prequential_rows.extend(selected)
        prequential_days.append({
            "holdout_date": date,
            "prior_dates": prior_dates,
            "selected_segments": [{"segment": label, "prior_metrics": metrics}
                                  for label, metrics in chosen],
            "rejected_segment_counts": dict(sorted(rejected.items())),
            "sample_insufficient": not chosen,
            "sample_insufficient_reason": (
                "no segment passed prior-only sample and robustness gates" if not chosen else None
            ),
            "selected_ticket_fingerprint": _fingerprint(selected),
            "holdout": _metrics(selected, settlements),
        })

    return {
        "model_name": MODEL_NAME,
        "status": "research_diagnostic_real_betting_disabled",
        "real_betting_enabled": False,
        "input": {**extraction, "dates": dates, "tickets": len(decisions)},
        "decision_information_fingerprint": _fingerprint(decisions),
        "settlement_separation": {
            "result_or_payout_used_for_segment_membership": False,
            "result_or_payout_used_for_evaluation_only": True,
        },
        "thresholds": {
            "min_prior_days": min_days, "min_prior_tickets": min_tickets,
            "min_prior_hits": min_hits,
            "required_prior_roi_excluding_largest_hit_above": 1.0,
            "required_prior_profitable_day_rate_at_least": 0.5,
            "max_prequential_segments": max_prequential_segments,
        },
        "segments": segments,
        "prequential": {
            "selection_uses_strictly_prior_dates_only": True,
            "aggregate": _metrics(all_prequential_rows, settlements),
            "daily": prequential_days,
        },
        "post_hoc_best_is_promotion_evidence": False,
        "promotion_warning": (
            "LODO and post-hoc best segments are exploratory only. Promotion requires a "
            "pre-registered rule evaluated on untouched prospective dates."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V17 selected-ticket segment diagnostics")
    parser.add_argument("input", type=Path, help="market evaluation job JSON")
    parser.add_argument("output", type=Path, help="diagnostic JSON output")
    parser.add_argument("--min-days", type=int, default=3)
    parser.add_argument("--min-tickets", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-prequential-segments", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.min_days, args.min_tickets, args.min_hits) < 0:
        parser.error("sample thresholds must be non-negative")
    if args.max_prequential_segments <= 0:
        parser.error("--max-prequential-segments must be positive")
    job = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(job, Mapping):
        parser.error("input JSON must be an object")
    result = diagnose_segments(
        job, min_days=args.min_days, min_tickets=args.min_tickets,
        min_hits=args.min_hits,
        max_prequential_segments=args.max_prequential_segments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
