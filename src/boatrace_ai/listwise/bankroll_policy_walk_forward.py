from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from ..bankroll_bootstrap import bootstrap_daily_roi
from ..bankroll_policy_search import (
    canonicalize_policy_candidates,
    compact_metrics,
    policy_candidates,
    successive_halving_search,
)
from ..packed_bankroll import PackedCandidates, _evaluate_day, evaluate_packed_policy


PROTOCOL_VERSION = "nested-annual-v1"
FORBIDDEN_LEGACY_JOB_IDS = frozenset({3995})

NESTED_FOLD_MIN_TICKETS = 60
NESTED_FOLD_MIN_HITS = 4
NESTED_FOLD_MIN_SELECTED_RACES = 20
NESTED_FOLD_MIN_PURCHASE_DAYS = 12

NESTED_MIN_TICKETS = 300
NESTED_MIN_HITS = 20
NESTED_MIN_SELECTED_RACES = 100
NESTED_MIN_PURCHASE_DAYS = 60


def build_annual_walk_forward_folds(
    dates: Sequence[str | date],
    *,
    outer_days: int = 365,
    selection_days: int = 365,
    folds: int = 5,
    embargo_days: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Build oldest-to-newest annual policy selection/holdout boundaries.

    Boundaries are whole calendar days. Later selection windows may contain an
    earlier outer holdout because it is then legitimate historical data, while
    outer holdouts themselves are adjacent, non-overlapping, and never reused.
    """
    for value, name, minimum in (
        (outer_days, "outer_days", 1),
        (selection_days, "selection_days", 4),
        (folds, "folds", 1),
        (embargo_days, "embargo_days", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")

    normalized = sorted({_as_date(value) for value in dates})
    required = selection_days + embargo_days + folds * outer_days
    if len(normalized) < required:
        raise ValueError(
            f"at least {required} unique dates are required for {folds} folds"
        )
    window = normalized[-required:]
    source_is_contiguous = _is_calendar_contiguous(window)
    rows: list[dict[str, Any]] = []
    previous_holdout: tuple[date, ...] | None = None
    for index in range(folds):
        holdout_start = selection_days + embargo_days + index * outer_days
        holdout = tuple(window[holdout_start : holdout_start + outer_days])
        embargo = tuple(window[holdout_start - embargo_days : holdout_start])
        selection_stop = holdout_start - embargo_days
        selection = tuple(
            window[selection_stop - selection_days : selection_stop]
        )
        audit = {
            "whole_day_boundaries": True,
            "selection_day_count": len(selection) == selection_days,
            "holdout_day_count": len(holdout) == outer_days,
            "embargo_day_count": len(embargo) == embargo_days,
            "selection_contiguous": _is_calendar_contiguous(selection),
            "holdout_contiguous": _is_calendar_contiguous(holdout),
            "selection_before_holdout": selection[-1] < holdout[0],
            "selection_holdout_disjoint": not set(selection) & set(holdout),
            "holdout_non_overlapping": (
                previous_holdout is None
                or not set(previous_holdout) & set(holdout)
            ),
            "holdout_contiguous_with_previous": (
                previous_holdout is None
                or previous_holdout[-1] + timedelta(days=1) == holdout[0]
            ),
            "source_window_contiguous": source_is_contiguous,
        }
        audit["passed"] = all(audit.values())
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "fold": index + 1,
                "selection_dates": tuple(value.isoformat() for value in selection),
                "embargo_dates": tuple(value.isoformat() for value in embargo),
                "holdout_dates": tuple(value.isoformat() for value in holdout),
                "selection_date_from": selection[0].isoformat(),
                "selection_date_through": selection[-1].isoformat(),
                "holdout_date_from": holdout[0].isoformat(),
                "holdout_date_through": holdout[-1].isoformat(),
                "boundary_audit": audit,
            }
        )
        previous_holdout = holdout
    return tuple(rows)


def evaluate_annual_walk_forward(
    fold_inputs: Sequence[Mapping[str, Any]],
    base_policy: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    candidate_count: int = 64,
    finalists: int = 8,
    selection_bootstrap_samples: int = 20_000,
    aggregate_bootstrap_samples: int = 20_000,
    outer_days: int = 365,
    selection_days: int = 365,
    embargo_days: int = 0,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Select on each fold's selection data and evaluate its holdout once.

    This is deliberately a pure policy layer. Building each fold's
    ``PackedCandidates``, including prediction-model fitting/refitting and all
    feature construction, belongs to the external fold input builder.
    """
    if not fold_inputs:
        raise ValueError("fold_inputs must not be empty")
    if outer_days < 1 or selection_days < 4 or embargo_days < 0:
        raise ValueError("invalid annual fold day counts")
    _reject_forbidden_provenance(fold_inputs)

    generated = (
        policy_candidates(base_policy, count=candidate_count, seed=seed)
        if candidates is None
        else candidates
    )
    registry, registry_sha256 = canonicalize_policy_candidates(generated)
    fold_results: list[dict[str, Any]] = []
    seen_holdout_dates: set[str] = set()
    previous_holdout_through: date | None = None
    for expected_fold, fold_input in enumerate(fold_inputs, start=1):
        selection = _packed_value(fold_input, "selection")
        holdout = _packed_value(fold_input, "holdout")
        fold_number = int(fold_input.get("fold", expected_fold))
        boundary_audit = _audit_fold_input(
            selection,
            holdout,
            fold_input.get("boundary_audit"),
            expected_selection_days=selection_days,
            expected_holdout_days=outer_days,
            expected_embargo_days=embargo_days,
        )
        overlap = seen_holdout_dates.intersection(holdout.dates)
        if overlap:
            raise ValueError(
                f"outer holdout dates are reused: {', '.join(sorted(overlap))}"
            )
        seen_holdout_dates.update(holdout.dates)
        current_holdout_from = _as_date(holdout.dates[0])
        boundary_audit["fold_number_in_sequence"] = fold_number == expected_fold
        boundary_audit["holdout_non_overlapping"] = not overlap
        boundary_audit["holdout_contiguous_with_previous"] = (
            previous_holdout_through is None
            or previous_holdout_through + timedelta(days=1) == current_holdout_from
        )
        boundary_audit["passed"] = all(
            value for key, value in boundary_audit.items() if key != "passed"
        )
        previous_holdout_through = _as_date(holdout.dates[-1])

        search = successive_halving_search(
            selection,
            base_policy,
            candidate_count=len(registry),
            finalists=finalists,
            bootstrap_samples=selection_bootstrap_samples,
            seed=seed,
            candidates=registry,
        )
        if search["policy_candidates_sha256"] != registry_sha256:
            raise RuntimeError("policy candidate registry changed within evaluation")
        selected_policy = dict(search["selected"]["policy"])
        holdout_result = evaluate_packed_policy(holdout, selected_policy)
        metrics = compact_metrics(holdout_result)
        minimum_evidence = _fold_minimum_evidence(metrics)
        largest_hit_return = _largest_selected_hit_return(
            holdout, selected_policy
        )
        stake = int(metrics["stake_yen"])
        return_without_largest = int(metrics["return_yen"]) - largest_hit_return
        selected_sha256 = canonicalize_policy_candidates(
            (selected_policy,)
        )[1]
        fold_results.append(
            {
                "fold": fold_number,
                "boundary_audit": boundary_audit,
                "policy_candidates_sha256": registry_sha256,
                "selected_policy_sha256": selected_sha256,
                "selected_policy": selected_policy,
                "selection_search": search,
                "holdout_metrics": metrics,
                "holdout_daily": list(holdout_result["daily"]),
                "largest_hit_return_yen": largest_hit_return,
                "largest_hit_excluded_roi": (
                    return_without_largest / stake if stake else None
                ),
                "minimum_evidence": minimum_evidence,
            }
        )

    aggregate = aggregate_outer_folds(
        fold_results,
        bootstrap_samples=aggregate_bootstrap_samples,
        seed=seed,
    )
    gate = nested_promotion_gate(aggregate)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_candidates_sha256": registry_sha256,
        "candidate_count": len(registry),
        "folds": fold_results,
        "aggregate": aggregate,
        "promotion_gate": gate,
        "promotion_eligible": all(gate.values()),
    }


def aggregate_outer_folds(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260728,
) -> dict[str, Any]:
    if not fold_results:
        raise ValueError("fold_results must not be empty")
    ordered = sorted(fold_results, key=lambda row: int(row["fold"]))
    daily: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    policy_counts: Counter[str] = Counter()
    policy_rows: dict[str, Mapping[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    largest_hit_return = 0
    for row in ordered:
        metrics = row["holdout_metrics"]
        fold_daily = [dict(value) for value in row["holdout_daily"]]
        daily_stake = sum(int(value["stake_yen"]) for value in fold_daily)
        daily_return = sum(int(value["return_yen"]) for value in fold_daily)
        daily_purchase_days = sum(int(value["stake_yen"]) > 0 for value in fold_daily)

        if daily_stake != int(metrics["stake_yen"]) or daily_return != int(
            metrics["return_yen"]
        ):
            raise ValueError("holdout daily totals do not match fold metrics")
        if daily_purchase_days != int(metrics["days_with_bets"]):
            raise ValueError("holdout purchase days do not match fold metrics")

        for value in fold_daily:
            race_date = str(value["race_date"])
            if race_date in seen_dates:
                raise ValueError(f"outer holdout date is duplicated: {race_date}")
            seen_dates.add(race_date)
            daily.append(value)
        policy_hash = str(row["selected_policy_sha256"])
        policy_counts[policy_hash] += 1
        policy_rows[policy_hash] = row["selected_policy"]
        evidence = dict(row.get("minimum_evidence") or _fold_minimum_evidence(metrics))
        fold_rows.append(
            {
                "fold": int(row["fold"]),
                "roi": daily_return / daily_stake if daily_stake else 0.0,
                "tickets": int(metrics["tickets"]),
                "hit_tickets": int(metrics["hit_tickets"]),
                "selected_races": int(metrics["selected_races"]),
                "purchase_days": int(metrics["days_with_bets"]),
                "minimum_evidence": evidence,
                "boundary_passed": bool(row["boundary_audit"]["passed"]),
                "selected_policy_sha256": policy_hash,
            }
        )
        largest_hit_return = max(
            largest_hit_return, int(row.get("largest_hit_return_yen", 0))
        )

    daily.sort(key=lambda row: str(row["race_date"]))
    totals = {
        "evaluated_races": sum(int(row["holdout_metrics"]["evaluated_races"]) for row in ordered),
        "selected_races": sum(int(row["holdout_metrics"]["selected_races"]) for row in ordered),
        "tickets": sum(int(row["holdout_metrics"]["tickets"]) for row in ordered),
        "hit_tickets": sum(int(row["holdout_metrics"]["hit_tickets"]) for row in ordered),
        "hit_races": sum(int(row["holdout_metrics"]["hit_races"]) for row in ordered),
        "stake_yen": sum(int(row["holdout_metrics"]["stake_yen"]) for row in ordered),
        "return_yen": sum(int(row["holdout_metrics"]["return_yen"]) for row in ordered),
        "purchase_days": sum(int(row["holdout_metrics"]["days_with_bets"]) for row in ordered),
    }
    totals["profit_yen"] = totals["return_yen"] - totals["stake_yen"]
    totals["roi"] = (
        totals["return_yen"] / totals["stake_yen"]
        if totals["stake_yen"]
        else None
    )
    totals["largest_hit_return_yen"] = largest_hit_return
    totals["largest_hit_excluded_roi"] = (
        (totals["return_yen"] - largest_hit_return) / totals["stake_yen"]
        if totals["stake_yen"]
        else None
    )
    confidence = bootstrap_daily_roi(
        daily,
        samples=bootstrap_samples,
        seed=seed,
    )
    selection_frequency = [
        {
            "policy_sha256": policy_hash,
            "count": count,
            "fraction": count / len(ordered),
            "policy": dict(policy_rows[policy_hash]),
        }
        for policy_hash, count in sorted(
            policy_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    rois = [row["roi"] for row in fold_rows]
    return {
        **totals,
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "minimum_fold_roi": min(rois),
        "profitable_folds": sum(roi > 1.0 for roi in rois),
        "all_fold_boundaries_passed": all(row["boundary_passed"] for row in fold_rows),
        "all_folds_minimum_evidence": all(
            all(row["minimum_evidence"].values()) for row in fold_rows
        ),
        "bootstrap": confidence,
        "daily": daily,
        "selection_frequency": selection_frequency,
    }


def nested_promotion_gate(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    confidence = aggregate["bootstrap"]
    lower = confidence.get("roi_ci95_lower")
    probability = confidence.get("probability_roi_above_one")
    roi = aggregate.get("roi")
    excluded_roi = aggregate.get("largest_hit_excluded_roi")
    return {
        "five_outer_folds_completed": int(aggregate["fold_count"]) == 5,
        "all_fold_boundaries_passed": bool(aggregate["all_fold_boundaries_passed"]),
        "all_fold_roi_above_one": float(aggregate["minimum_fold_roi"]) > 1.0,
        "all_folds_minimum_evidence": bool(aggregate["all_folds_minimum_evidence"]),
        "minimum_tickets": int(aggregate["tickets"]) >= NESTED_MIN_TICKETS,
        "minimum_hits": int(aggregate["hit_tickets"]) >= NESTED_MIN_HITS,
        "minimum_selected_races": int(aggregate["selected_races"]) >= NESTED_MIN_SELECTED_RACES,
        "minimum_purchase_days": int(aggregate["purchase_days"]) >= NESTED_MIN_PURCHASE_DAYS,
        "roi_above_one": roi is not None and float(roi) > 1.0,
        "bootstrap_lower_above_one": lower is not None and float(lower) > 1.0,
        "bootstrap_probability_at_least_95pct": (
            probability is not None and float(probability) >= 0.95
        ),
        "largest_hit_excluded_roi_above_one": (
            excluded_roi is not None and float(excluded_roi) > 1.0
        ),
    }


def _fold_minimum_evidence(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "minimum_tickets": int(metrics["tickets"]) >= NESTED_FOLD_MIN_TICKETS,
        "minimum_hits": int(metrics["hit_tickets"]) >= NESTED_FOLD_MIN_HITS,
        "minimum_selected_races": int(metrics["selected_races"]) >= NESTED_FOLD_MIN_SELECTED_RACES,
        "minimum_purchase_days": int(metrics["days_with_bets"]) >= NESTED_FOLD_MIN_PURCHASE_DAYS,
    }


def _largest_selected_hit_return(
    packed: PackedCandidates,
    policy: Mapping[str, Any],
) -> int:
    """Return the largest realized contribution of one selected winning ticket."""
    probe_hits = np.zeros_like(packed.hit)
    probe = replace(packed, hit=probe_hits)
    largest = 0
    for day_index in range(len(packed.dates)):
        start, stop = map(int, packed.offsets[day_index : day_index + 2])
        for ticket_index in np.flatnonzero(packed.hit[start:stop]) + start:
            probe_hits[ticket_index] = True
            contribution = int(
                _evaluate_day(probe, start, stop, policy)["return_yen"]
            )
            probe_hits[ticket_index] = False
            largest = max(largest, contribution)
    return largest


def _audit_fold_input(
    selection: PackedCandidates,
    holdout: PackedCandidates,
    supplied: object,
    *,
    expected_selection_days: int,
    expected_holdout_days: int,
    expected_embargo_days: int,
) -> dict[str, bool]:
    selection_dates = tuple(_as_date(value) for value in selection.dates)
    holdout_dates = tuple(_as_date(value) for value in holdout.dates)
    audit = {
        "whole_day_boundaries": len(selection_dates) == len(set(selection_dates)) and len(holdout_dates) == len(set(holdout_dates)),
        "selection_day_count": len(selection_dates) == expected_selection_days,
        "holdout_day_count": len(holdout_dates) == expected_holdout_days,
        "selection_contiguous": _is_calendar_contiguous(selection_dates),
        "holdout_contiguous": _is_calendar_contiguous(holdout_dates),
        "selection_before_holdout": bool(selection_dates and holdout_dates and selection_dates[-1] < holdout_dates[0]),
        "selection_holdout_disjoint": not set(selection_dates) & set(holdout_dates),
        "embargo_day_count": bool(
            selection_dates
            and holdout_dates
            and (holdout_dates[0] - selection_dates[-1]).days - 1
            == expected_embargo_days
        ),
    }
    if isinstance(supplied, Mapping):
        audit["declared_boundary_passed"] = bool(supplied.get("passed", False))
    audit["passed"] = all(audit.values())
    return audit


def _packed_value(row: Mapping[str, Any], key: str) -> PackedCandidates:
    value = row.get(key)
    if not isinstance(value, PackedCandidates):
        raise ValueError(f"fold {key} must be PackedCandidates")
    if not value.dates:
        raise ValueError(f"fold {key} must contain at least one date")
    return value


def _reject_forbidden_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    provenance_keys = ("source_job_id", "policy_source_job_id", "checkpoint_job_id")
    for row in rows:
        sources = [row]
        nested = row.get("provenance")
        if isinstance(nested, Mapping):
            sources.append(nested)
        for source in sources:
            for key in provenance_keys:
                value = source.get(key)
                try:
                    normalized = int(value) if value is not None else None
                except (TypeError, ValueError):
                    continue
                if normalized in FORBIDDEN_LEGACY_JOB_IDS:
                    raise ValueError(
                        f"legacy job {normalized} cannot source nested evaluation"
                    )


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid ISO race date: {value!r}") from exc


def _is_calendar_contiguous(values: Sequence[date]) -> bool:
    return bool(values) and all(
        right - left == timedelta(days=1)
        for left, right in zip(values, values[1:])
    )
