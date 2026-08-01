from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260801
DEFAULT_SAMPLES = 20_000
DEFAULT_MIN_TICKET_JACCARD = 0.80
DEFAULT_MAX_STAKE_TURNOVER = 0.10

TicketKey = tuple[str, str, str]


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _assert_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{field} is inconsistent: {actual} != {expected}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    return dict(_require_mapping(value, str(path)))


def _contract(result: Mapping[str, Any], label: str) -> dict[str, Any]:
    version = result.get("evaluation_version")
    if isinstance(version, bool) or not isinstance(version, (int, str)):
        raise ValueError(f"{label}.evaluation_version must be an integer or string")
    strategy = _require_string(
        result.get("calibrator_strategy"), f"{label}.calibrator_strategy"
    )
    raw_dates = _require_list(
        result.get("benchmark_dates"), f"{label}.benchmark_dates"
    )
    dates = [
        _require_string(value, f"{label}.benchmark_dates[{index}]")
        for index, value in enumerate(raw_dates)
    ]
    if not dates or dates != sorted(set(dates)):
        raise ValueError(f"{label}.benchmark_dates must be non-empty, unique, sorted")
    odds_signature = dict(
        _require_mapping(
            result.get("odds_data_signature"),
            f"{label}.odds_data_signature",
        )
    )
    if not odds_signature:
        raise ValueError(f"{label}.odds_data_signature must not be empty")
    evaluation_races = _require_int(
        result.get("evaluation_races"), f"{label}.evaluation_races"
    )
    return {
        "evaluation_version": version,
        "calibrator_strategy": strategy,
        "benchmark_dates": dates,
        "odds_data_signature": odds_signature,
        "evaluation_races": evaluation_races,
    }


def _extract_bankroll(result: Mapping[str, Any], label: str) -> dict[str, Any]:
    bankroll = _require_mapping(
        result.get("chronological_bankroll"),
        f"{label}.chronological_bankroll",
    )
    daily = _require_list(bankroll.get("daily"), f"{label}.chronological_bankroll.daily")
    if not daily:
        raise ValueError(f"{label}.chronological_bankroll.daily must not be empty")

    tickets: dict[TicketKey, int] = {}
    race_keys: set[tuple[str, str]] = set()
    daily_profits: dict[str, int] = {}
    total_stake = 0
    total_return = 0
    total_ticket_count = 0
    largest_hit_return = 0

    for day_index, raw_day in enumerate(daily):
        day_field = f"{label}.chronological_bankroll.daily[{day_index}]"
        day = _require_mapping(raw_day, day_field)
        race_date = _require_string(day.get("race_date"), f"{day_field}.race_date")
        if race_date in daily_profits:
            raise ValueError(f"{label} has duplicate daily date: {race_date}")
        day_stake = _require_int(day.get("stake_yen"), f"{day_field}.stake_yen")
        day_return = _require_int(day.get("return_yen"), f"{day_field}.return_yen")
        day_profit = _require_int(
            day.get("profit_yen"), f"{day_field}.profit_yen", minimum=-10**18
        )
        if day_profit != day_return - day_stake:
            raise ValueError(f"{day_field}.profit_yen is inconsistent")
        day_tickets = _require_int(day.get("tickets"), f"{day_field}.tickets")
        day_largest_hit = _require_int(
            day.get("largest_hit_return_yen"),
            f"{day_field}.largest_hit_return_yen",
        )
        if day_largest_hit > day_return:
            raise ValueError(f"{day_field}.largest_hit_return_yen exceeds return")

        ledger = _require_list(day.get("ledger"), f"{day_field}.ledger")
        decision_count = 0
        selected_count = 0
        selected_stake = 0
        for ledger_index, raw_entry in enumerate(ledger):
            entry_field = f"{day_field}.ledger[{ledger_index}]"
            entry = _require_mapping(raw_entry, entry_field)
            event = _require_string(entry.get("event"), f"{entry_field}.event")
            if event != "decision":
                continue
            decision_count += 1
            race_id = _require_string(entry.get("race_id"), f"{entry_field}.race_id")
            race_key = (race_date, race_id)
            if race_key in race_keys:
                raise ValueError(f"{label} has duplicate decision race: {race_key}")
            race_keys.add(race_key)
            selections = _require_list(
                entry.get("selections"), f"{entry_field}.selections"
            )
            entry_tickets = _require_int(
                entry.get("tickets"), f"{entry_field}.tickets"
            )
            entry_stake = _require_int(
                entry.get("stake_yen"), f"{entry_field}.stake_yen"
            )
            for selection_index, raw_selection in enumerate(selections):
                selection_field = f"{entry_field}.selections[{selection_index}]"
                selection = _require_mapping(raw_selection, selection_field)
                combination = _require_string(
                    selection.get("combination"),
                    f"{selection_field}.combination",
                )
                stake = _require_int(
                    selection.get("stake_yen"),
                    f"{selection_field}.stake_yen",
                    minimum=1,
                )
                key = (race_date, race_id, combination)
                if key in tickets:
                    raise ValueError(f"{label} has duplicate ticket: {key}")
                tickets[key] = stake
                selected_count += 1
                selected_stake += stake
            if entry_tickets != len(selections):
                raise ValueError(f"{entry_field}.tickets is inconsistent")
            if entry_stake != sum(
                int(selection["stake_yen"])
                for selection in selections
                if isinstance(selection, Mapping)
            ):
                raise ValueError(f"{entry_field}.stake_yen is inconsistent")
        evaluated_races = _require_int(
            day.get("evaluated_races"), f"{day_field}.evaluated_races"
        )
        if evaluated_races != decision_count:
            raise ValueError(f"{day_field}.evaluated_races does not match decisions")
        if selected_count != day_tickets or selected_stake != day_stake:
            raise ValueError(f"{day_field} ticket totals do not match ledger")
        daily_profits[race_date] = day_profit
        total_stake += day_stake
        total_return += day_return
        total_ticket_count += day_tickets
        largest_hit_return = max(largest_hit_return, day_largest_hit)

    dates = sorted(daily_profits)
    if dates != list(daily_profits):
        raise ValueError(f"{label}.chronological_bankroll.daily must be date-sorted")
    aggregate_stake = _require_int(
        bankroll.get("stake_yen"), f"{label}.chronological_bankroll.stake_yen"
    )
    aggregate_return = _require_int(
        bankroll.get("return_yen"), f"{label}.chronological_bankroll.return_yen"
    )
    aggregate_profit = _require_int(
        bankroll.get("profit_yen"),
        f"{label}.chronological_bankroll.profit_yen",
        minimum=-10**18,
    )
    aggregate_tickets = _require_int(
        bankroll.get("tickets"), f"{label}.chronological_bankroll.tickets"
    )
    aggregate_roi = _require_number(
        bankroll.get("roi"), f"{label}.chronological_bankroll.roi"
    )
    expected_roi = total_return / total_stake if total_stake else 0.0
    if (
        aggregate_stake != total_stake
        or aggregate_return != total_return
        or aggregate_profit != total_return - total_stake
        or aggregate_tickets != total_ticket_count
    ):
        raise ValueError(f"{label}.chronological_bankroll totals are inconsistent")
    _assert_close(aggregate_roi, expected_roi, f"{label}.chronological_bankroll.roi")
    largest_excluded_roi = (
        (total_return - largest_hit_return) / total_stake if total_stake else 0.0
    )
    return {
        "dates": dates,
        "race_keys": race_keys,
        "tickets": tickets,
        "daily_profits": daily_profits,
        "ticket_count": total_ticket_count,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": expected_roi,
        "largest_hit_return_yen": largest_hit_return,
        "largest_hit_excluded_roi": largest_excluded_roi,
    }


def _validate_pair(
    anchor_contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    anchor_bankroll: Mapping[str, Any],
    candidate_bankroll: Mapping[str, Any],
) -> None:
    for field in (
        "evaluation_version",
        "calibrator_strategy",
        "benchmark_dates",
        "odds_data_signature",
        "evaluation_races",
    ):
        if anchor_contract[field] != candidate_contract[field]:
            raise ValueError(f"anchor/candidate {field} mismatch")
    if anchor_bankroll["dates"] != candidate_bankroll["dates"]:
        raise ValueError("anchor/candidate evaluation dates mismatch")
    if anchor_bankroll["race_keys"] != candidate_bankroll["race_keys"]:
        raise ValueError("anchor/candidate evaluation races mismatch")
    if len(anchor_bankroll["race_keys"]) != anchor_contract["evaluation_races"]:
        raise ValueError("evaluation_races does not match chronological decisions")


def _bootstrap_daily_profit_difference(
    differences: Sequence[float], *, samples: int, seed: int
) -> dict[str, Any]:
    if not differences:
        raise ValueError("daily profit differences must not be empty")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    values = np.asarray(differences, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("daily profit differences must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sampled].mean(axis=1)
    return {
        "method": "daily_cluster_nonparametric_bootstrap",
        "unit": "evaluation_day",
        "alternative": "candidate_profit_difference_greater_than_or_equal_to_zero",
        "seed": seed,
        "samples": samples,
        "days": len(values),
        "mean_profit_difference_yen": float(values.mean()),
        "one_sided_5pct_lower_yen": float(np.quantile(means, 0.05)),
        "probability_difference_greater_than_or_equal_to_zero": float(
            np.mean(means >= 0.0)
        ),
    }


def compare_market_results(
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
    min_ticket_jaccard: float = DEFAULT_MIN_TICKET_JACCARD,
    max_stake_turnover: float = DEFAULT_MAX_STAKE_TURNOVER,
) -> dict[str, Any]:
    min_jaccard = _require_number(min_ticket_jaccard, "min_ticket_jaccard")
    max_turnover = _require_number(max_stake_turnover, "max_stake_turnover")
    if not 0.0 <= min_jaccard <= 1.0:
        raise ValueError("min_ticket_jaccard must be between zero and one")
    if not 0.0 <= max_turnover <= 1.0:
        raise ValueError("max_stake_turnover must be between zero and one")

    anchor_contract = _contract(anchor, "anchor")
    candidate_contract = _contract(candidate, "candidate")
    anchor_bankroll = _extract_bankroll(anchor, "anchor")
    candidate_bankroll = _extract_bankroll(candidate, "candidate")
    _validate_pair(
        anchor_contract, candidate_contract, anchor_bankroll, candidate_bankroll
    )

    anchor_tickets = anchor_bankroll["tickets"]
    candidate_tickets = candidate_bankroll["tickets"]
    anchor_keys = set(anchor_tickets)
    candidate_keys = set(candidate_tickets)
    common = anchor_keys & candidate_keys
    union = anchor_keys | candidate_keys
    jaccard = len(common) / len(union) if union else 1.0
    ticket_turnover = len(anchor_keys ^ candidate_keys) / len(union) if union else 0.0
    absolute_stake_difference = sum(
        abs(anchor_tickets.get(key, 0) - candidate_tickets.get(key, 0))
        for key in union
    )
    maximum_stake = sum(
        max(anchor_tickets.get(key, 0), candidate_tickets.get(key, 0))
        for key in union
    )
    stake_turnover = (
        absolute_stake_difference / maximum_stake if maximum_stake else 0.0
    )

    daily = []
    differences = []
    for race_date in anchor_bankroll["dates"]:
        anchor_profit = anchor_bankroll["daily_profits"][race_date]
        candidate_profit = candidate_bankroll["daily_profits"][race_date]
        difference = candidate_profit - anchor_profit
        differences.append(float(difference))
        daily.append({
            "race_date": race_date,
            "anchor_profit_yen": anchor_profit,
            "candidate_profit_yen": candidate_profit,
            "profit_difference_yen": difference,
        })
    bootstrap = _bootstrap_daily_profit_difference(
        differences, samples=samples, seed=seed
    )

    checks = {
        "ticket_jaccard": jaccard >= min_jaccard,
        "stake_turnover": stake_turnover <= max_turnover,
        "candidate_roi_non_degradation": (
            candidate_bankroll["roi"] >= anchor_bankroll["roi"]
        ),
        "candidate_largest_hit_excluded_roi_non_degradation": (
            candidate_bankroll["largest_hit_excluded_roi"]
            >= anchor_bankroll["largest_hit_excluded_roi"]
        ),
        "bootstrap_profit_difference_lower": (
            bootstrap["one_sided_5pct_lower_yen"] >= 0.0
        ),
    }
    race_keys = [list(key) for key in sorted(anchor_bankroll["race_keys"])]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            **anchor_contract,
            "odds_data_signature_sha256": _canonical_sha256(
                anchor_contract["odds_data_signature"]
            ),
            "evaluation_dates": anchor_bankroll["dates"],
            "evaluation_race_count": len(race_keys),
            "evaluation_races_sha256": _canonical_sha256(race_keys),
        },
        "tickets": {
            "anchor_count": len(anchor_keys),
            "candidate_count": len(candidate_keys),
            "common_count": len(common),
            "union_count": len(union),
            "jaccard": jaccard,
            "turnover": ticket_turnover,
        },
        "stakes": {
            "anchor_yen": sum(anchor_tickets.values()),
            "candidate_yen": sum(candidate_tickets.values()),
            "absolute_difference_yen": absolute_stake_difference,
            "union_maximum_yen": maximum_stake,
            "turnover": stake_turnover,
        },
        "anchor": {
            key: anchor_bankroll[key]
            for key in (
                "ticket_count",
                "stake_yen",
                "return_yen",
                "profit_yen",
                "roi",
                "largest_hit_return_yen",
                "largest_hit_excluded_roi",
            )
        },
        "candidate": {
            key: candidate_bankroll[key]
            for key in (
                "ticket_count",
                "stake_yen",
                "return_yen",
                "profit_yen",
                "roi",
                "largest_hit_return_yen",
                "largest_hit_excluded_roi",
            )
        },
        "daily_profit_differences": daily,
        "daily_cluster_bootstrap": bootstrap,
        "gate": {
            "thresholds": {
                "minimum_ticket_jaccard": min_jaccard,
                "maximum_stake_turnover": max_turnover,
                "minimum_bootstrap_profit_difference_lower_yen": 0.0,
                "roi_non_degradation": True,
                "largest_hit_excluded_roi_non_degradation": True,
            },
            "checks": checks,
            "pass": all(checks.values()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired market-calibration bankroll results."
    )
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--min-ticket-jaccard", type=float, default=DEFAULT_MIN_TICKET_JACCARD
    )
    parser.add_argument(
        "--max-stake-turnover", type=float, default=DEFAULT_MAX_STAKE_TURNOVER
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_market_results(
            _load_json(args.anchor),
            _load_json(args.candidate),
            seed=args.seed,
            samples=args.samples,
            min_ticket_jaccard=args.min_ticket_jaccard,
            max_stake_turnover=args.max_stake_turnover,
        )
    except ValueError as exc:
        raise SystemExit(f"paired market comparison failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
