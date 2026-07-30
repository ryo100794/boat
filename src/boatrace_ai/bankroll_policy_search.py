from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_allocation import validate_policy
from .listwise.direct_bankroll import bootstrap_daily_bankroll
from .packed_bankroll import PackedCandidates, evaluate_packed_policy


PROMOTION_MIN_TICKETS = 300
PROMOTION_MIN_HITS = 20
PROMOTION_MIN_SELECTED_RACES = 100
PROMOTION_MIN_BETTING_DAYS = 60
PROMOTION_MAX_DRAWDOWN_STAKE_FRACTION = 0.50


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "ev_threshold": (1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50),
    "min_ticket_probability": (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05),
    "min_estimated_odds": (None, 5.0, 6.0, 30.0, 100.0, 101.0),
    "max_estimated_odds": (None, 30.0, 50.0, 100.0, 200.0),
    "fractional_kelly": (0.10, 0.25, 0.50),
    "max_daily_exposure_fraction": (0.30, 0.60, 0.80),
    "min_daily_exposure_fraction": (0.0, 0.20, 0.40),
    "race_cap_fraction": (0.05, 0.10, 0.15),
    "ticket_cap_fraction": (0.01, 0.02, 0.03),
    "max_daily_tickets": (1, 3, 5, 10, 20, 30, 50),
}


# Preserve the sparse, conservative region when new search dimensions are added.
CONSERVATIVE_POLICY_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "ev_threshold": 1.75,
        "min_ticket_probability": 0.0,
        "max_estimated_odds": None,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
    },
    {
        "ev_threshold": 2.00,
        "min_ticket_probability": 0.0,
        "max_estimated_odds": 100.0,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.02,
        "max_daily_tickets": 30,
    },
    {
        "ev_threshold": 1.75,
        "min_ticket_probability": 0.0,
        "min_estimated_odds": 6.0,
        "max_estimated_odds": 30.0,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.02,
        "max_daily_tickets": 30,
    },
    {
        "ev_threshold": 2.00,
        "min_ticket_probability": 0.0,
        "min_estimated_odds": 101.0,
        "max_estimated_odds": 200.0,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 30,
    },
)


TAIL_POLICY_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "ev_threshold": 1.35,
        "min_ticket_probability": 0.0,
        "min_estimated_odds": 101.0,
        "max_estimated_odds": None,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 20,
    },
    {
        "ev_threshold": 2.00,
        "min_ticket_probability": 0.002,
        "min_estimated_odds": 101.0,
        "max_estimated_odds": None,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 20,
    },
    {
        "ev_threshold": 2.50,
        "min_ticket_probability": 0.0,
        "min_estimated_odds": 200.0,
        "max_estimated_odds": None,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 20,
    },
)


SPARSE_POLICY_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "ev_threshold": 1.50,
        "min_ticket_probability": 0.02,
        "min_estimated_odds": 5.0,
        "max_estimated_odds": 30.0,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 3,
    },
    {
        "ev_threshold": 2.00,
        "min_ticket_probability": 0.002,
        "min_estimated_odds": 101.0,
        "max_estimated_odds": None,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": 3,
    },
)


def _conservative_anchor_policies(
    base_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    anchors = []
    seen = set()
    for overrides in CONSERVATIVE_POLICY_ANCHORS:
        candidate = {**base_policy, **overrides}
        key = _policy_key(candidate)
        if _valid_caps(candidate) and key not in seen:
            seen.add(key)
            anchors.append(candidate)
    return anchors


def _tail_anchor_policies(
    base_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    anchors = []
    seen = set()
    for overrides in TAIL_POLICY_ANCHORS:
        candidate = {**base_policy, **overrides}
        key = _policy_key(candidate)
        if _valid_caps(candidate) and key not in seen:
            seen.add(key)
            anchors.append(candidate)
    return anchors


def _sparse_anchor_policies(
    base_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    anchors = []
    seen = set()
    for overrides in SPARSE_POLICY_ANCHORS:
        candidate = {**base_policy, **overrides}
        key = _policy_key(candidate)
        if _valid_caps(candidate) and key not in seen:
            seen.add(key)
            anchors.append(candidate)
    return anchors


def _registered_anchor_policies(
    base_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return (
        _conservative_anchor_policies(base_policy)
        + _tail_anchor_policies(base_policy)
        + _sparse_anchor_policies(base_policy)
    )


def policy_candidates(
    base_policy: Mapping[str, Any],
    *,
    count: int = 64,
    seed: int = 20260726,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = np.random.default_rng(seed)
    canonical_base = dict(base_policy)
    canonical_base.setdefault("min_ticket_probability", 0.0)
    canonical_base.setdefault("min_estimated_odds", None)
    canonical_base.setdefault("max_estimated_odds", None)
    candidates = [canonical_base]
    seen = {_policy_key(candidates[0])}
    anchors = _registered_anchor_policies(canonical_base)
    for candidate in anchors:
        if len(candidates) >= count:
            break
        key = _policy_key(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    attempts = 0
    while len(candidates) < count and attempts < count * 100:
        attempts += 1
        candidate = dict(canonical_base)
        for name, values in SEARCH_SPACE.items():
            candidate[name] = values[int(rng.integers(0, len(values)))]
        if not _valid_caps(candidate):
            continue
        key = _policy_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    if len(candidates) != count:
        raise RuntimeError("could not generate enough unique policy candidates")
    return candidates


def canonicalize_policy_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Validate and fingerprint one immutable policy candidate registry."""
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("policy candidates must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    canonical_rows: list[str] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"policy candidate {index} must be a mapping")
        policy = dict(candidate)
        policy.setdefault("min_estimated_odds", None)
        try:
            _validate_policy_candidate(policy)
            canonical = json.dumps(
                policy,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid policy candidate at index {index}: {exc}") from exc
        if canonical in seen:
            raise ValueError(f"duplicate policy candidate at index {index}")
        seen.add(canonical)
        normalized.append(policy)
        canonical_rows.append(canonical)
    if not normalized:
        raise ValueError("policy candidates must be a non-empty sequence")
    payload = "[" + ",".join(canonical_rows) + "]"
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return tuple(normalized), digest


def _retain_registered_anchors(
    rows: Sequence[dict[str, Any]],
    *,
    keep: int,
    base_policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    selected = list(rows[:keep])
    # Anchors prevent premature elimination on short early windows, but must
    # not occupy nearly every finalist slot regardless of observed results.
    registered = _registered_anchor_policies(base_policy)
    registered_keys = {_policy_key(policy) for policy in registered}
    protection_limit = min(len(registered), keep // 4)
    protected = [
        row["policy"]
        for row in rows
        if _policy_key(row["policy"]) in registered_keys
    ][:protection_limit]
    anchor_keys = tuple(
        _policy_key(policy)
        for policy in protected
    )
    anchor_key_set = set(anchor_keys)
    selected_keys = {_policy_key(row["policy"]) for row in selected}
    rows_by_key = {_policy_key(row["policy"]): row for row in rows}
    for anchor_key in anchor_keys:
        row = rows_by_key.get(anchor_key)
        if row is None:
            continue
        key = _policy_key(row["policy"])
        if key not in anchor_key_set or key in selected_keys:
            continue
        replace_at = next(
            (
                index for index in range(len(selected) - 1, -1, -1)
                if _policy_key(selected[index]["policy"]) not in anchor_key_set
            ),
            None,
        )
        if replace_at is None:
            break
        selected_keys.discard(_policy_key(selected[replace_at]["policy"]))
        selected[replace_at] = row
        selected_keys.add(key)
    return selected, len(protected)


def slice_days(packed: PackedCandidates, stop_day: int) -> PackedCandidates:
    stop_day = max(0, min(int(stop_day), len(packed.dates)))
    ticket_stop = int(packed.offsets[stop_day])
    return PackedCandidates(
        dates=packed.dates[:stop_day],
        offsets=packed.offsets[: stop_day + 1].copy(),
        evaluated_races=packed.evaluated_races[:stop_day].copy(),
        race_codes=packed.race_codes[:ticket_stop],
        estimated_odds=packed.estimated_odds[:ticket_stop],
        estimated_ev=packed.estimated_ev[:ticket_stop],
        probability=packed.probability[:ticket_stop],
        actual_payout_yen=packed.actual_payout_yen[:ticket_stop],
        hit=packed.hit[:ticket_stop],
    )

def slice_day_range(
    packed: PackedCandidates,
    start_day: int,
    stop_day: int,
) -> PackedCandidates:
    start_day = max(0, min(int(start_day), len(packed.dates)))
    stop_day = max(start_day, min(int(stop_day), len(packed.dates)))
    ticket_start = int(packed.offsets[start_day])
    ticket_stop = int(packed.offsets[stop_day])
    return PackedCandidates(
        dates=packed.dates[start_day:stop_day],
        offsets=packed.offsets[start_day : stop_day + 1].copy() - ticket_start,
        evaluated_races=packed.evaluated_races[start_day:stop_day].copy(),
        race_codes=packed.race_codes[ticket_start:ticket_stop],
        estimated_odds=packed.estimated_odds[ticket_start:ticket_stop],
        estimated_ev=packed.estimated_ev[ticket_start:ticket_stop],
        probability=packed.probability[ticket_start:ticket_stop],
        actual_payout_yen=packed.actual_payout_yen[ticket_start:ticket_stop],
        hit=packed.hit[ticket_start:ticket_stop],
    )



def successive_halving_search(
    packed: PackedCandidates,
    base_policy: Mapping[str, Any],
    *,
    candidate_count: int = 64,
    finalists: int = 8,
    bootstrap_samples: int = 20_000,
    seed: int = 20260726,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(packed.dates) < 4:
        raise ValueError("at least four policy-selection days are required")
    if candidates is None:
        generated = policy_candidates(base_policy, count=candidate_count, seed=seed)
    else:
        generated = candidates
    registered, candidates_sha256 = canonicalize_policy_candidates(generated)
    actual_candidate_count = len(registered)
    if not 1 <= finalists <= actual_candidate_count:
        raise ValueError("finalists must be between one and candidate count")
    active = list(registered)
    stages = []
    for fraction in (0.25, 0.50, 1.0):
        stage_data = slice_days(
            packed, max(1, int(round(len(packed.dates) * fraction)))
        )
        rows = []
        for policy in active:
            result = evaluate_packed_policy(stage_data, policy)
            rows.append({
                "policy": policy,
                "metrics": compact_metrics(result),
                "score": screening_score(result),
            })
        rows.sort(
            key=lambda row: (
                row["score"],
                row["metrics"]["profit_yen"],
                row["metrics"]["hit_tickets"],
            ),
            reverse=True,
        )
        keep = (
            max(finalists, int(np.ceil(len(rows) / 3)))
            if fraction < 1.0
            else finalists
        )
        retained, protected_anchor_count = (
            _retain_registered_anchors(rows, keep=keep, base_policy=base_policy)
            if fraction < 1.0
            else (list(rows[:keep]), 0)
        )
        active = [row["policy"] for row in retained]
        stages.append({
            "fraction": fraction,
            "days": len(stage_data.dates),
            "evaluated_candidates": len(rows),
            "retained_candidates": len(active),
            "protected_anchor_count": protected_anchor_count,
            "leaders": rows[: min(10, len(rows))],
        })

    final_rows = []
    for policy in active:
        result = evaluate_packed_policy(packed, policy)
        confidence = bootstrap_daily_bankroll(
            result["daily"], samples=bootstrap_samples, seed=seed
        )
        stability = temporal_stability(packed, policy)
        final_rows.append({
            "policy": policy,
            "metrics": compact_metrics(result),
            "confidence": confidence,
            "temporal_stability": stability,
        })
    final_rows.sort(
        key=lambda row: (
            row["temporal_stability"]["all_minimum_evidence"],
            row["temporal_stability"]["minimum_roi"],
            row["temporal_stability"]["mean_roi_minus_std"],
            row["confidence"]["roi_ci95_lower"],
            row["confidence"]["probability_roi_above_one"],
            row["metrics"]["profit_yen"],
            row["metrics"]["hit_tickets"],
        ),
        reverse=True,
    )
    selected = final_rows[0]
    selected["promotion_gate"] = promotion_gate(selected)
    return {
        "method": "chronological_successive_halving_then_daily_bootstrap",
        "candidate_count": actual_candidate_count,
        "policy_candidates_sha256": candidates_sha256,
        "bootstrap_samples": bootstrap_samples,
        "stages": stages,
        "finalists": final_rows,
        "selected": selected,
        "promotion_eligible": all(selected["promotion_gate"].values()),
    }


def temporal_stability(
    packed: PackedCandidates,
    policy: Mapping[str, Any],
    *,
    folds: int = 3,
) -> dict[str, Any]:
    if folds < 2 or len(packed.dates) < folds:
        raise ValueError("temporal stability requires at least two folds")
    boundaries = np.linspace(0, len(packed.dates), folds + 1, dtype=int)
    rows = []
    for fold in range(folds):
        fold_data = slice_day_range(
            packed, int(boundaries[fold]), int(boundaries[fold + 1])
        )
        metrics = compact_metrics(evaluate_packed_policy(fold_data, policy))
        metrics["fold"] = fold + 1
        metrics["date_from"] = fold_data.dates[0]
        metrics["date_through"] = fold_data.dates[-1]
        metrics["minimum_evidence"] = (
            int(metrics["tickets"]) >= 100
            and int(metrics["hit_tickets"]) >= 10
        )
        rows.append(metrics)
    rois = np.asarray([row["roi"] for row in rows], dtype=np.float64)
    return {
        "folds": rows,
        "minimum_roi": float(rois.min()),
        "mean_roi": float(rois.mean()),
        "roi_std": float(rois.std()),
        "mean_roi_minus_std": float(rois.mean() - rois.std()),
        "all_minimum_evidence": all(
            bool(row["minimum_evidence"]) for row in rows
        ),
    }


def screening_score(result: Mapping[str, Any]) -> float:
    stake = float(result["stake_yen"])
    returned = float(result["return_yen"])
    prior_stake = 100_000.0
    shrunk_roi = (returned + prior_stake * 0.75) / (stake + prior_stake)
    drawdown = float(result["max_drawdown_yen"]) / max(stake, 1.0)
    evidence = min(1.0, float(result["hit_tickets"]) / 30.0)
    return shrunk_roi - 0.10 * drawdown + 0.02 * evidence


def compact_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "evaluated_races", "selected_races", "tickets", "hit_tickets",
        "hit_races", "stake_yen", "return_yen", "profit_yen", "roi",
        "ticket_hit_rate", "race_hit_rate", "max_drawdown_yen",
        "days_with_bets", "winning_days", "losing_days",
    )
    return {key: result[key] for key in keys}


def recent_allocation_diagnostics(
    daily: Sequence[Mapping[str, Any]],
    *,
    window_days: int = 7,
) -> dict[str, Any]:
    if window_days < 1 or len(daily) <= window_days:
        raise ValueError("recent allocation diagnostics require a baseline window")
    baseline = daily[:-window_days]
    recent = daily[-window_days:]

    def average(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(sum(float(row[key]) for row in rows) / len(rows))

    baseline_stake = average(baseline, "stake_yen")
    recent_stake = average(recent, "stake_yen")
    baseline_tickets = average(baseline, "tickets")
    recent_tickets = average(recent, "tickets")

    def ratio(current: float, previous: float) -> float:
        return current / previous if previous > 0.0 else (float("inf") if current else 1.0)

    stake_multiplier = ratio(recent_stake, baseline_stake)
    ticket_multiplier = ratio(recent_tickets, baseline_tickets)
    return {
        "window_days": window_days,
        "baseline_days": len(baseline),
        "baseline_average_stake_yen": baseline_stake,
        "recent_average_stake_yen": recent_stake,
        "stake_multiplier": stake_multiplier,
        "baseline_average_tickets": baseline_tickets,
        "recent_average_tickets": recent_tickets,
        "ticket_multiplier": ticket_multiplier,
        "stable": stake_multiplier <= 3.0 and ticket_multiplier <= 3.0,
    }


def promotion_gate(row: Mapping[str, Any]) -> dict[str, bool]:
    metrics = row["metrics"]
    confidence = row["confidence"]
    gate = {
        "minimum_tickets": int(metrics["tickets"]) >= PROMOTION_MIN_TICKETS,
        "minimum_hits": int(metrics["hit_tickets"]) >= PROMOTION_MIN_HITS,
        "minimum_selected_races": (
            int(metrics["selected_races"]) >= PROMOTION_MIN_SELECTED_RACES
        ),
        "minimum_betting_days": (
            int(metrics["days_with_bets"]) >= PROMOTION_MIN_BETTING_DAYS
        ),
        "drawdown_within_stake_limit": (
            int(metrics["max_drawdown_yen"])
            <= float(metrics["stake_yen"])
            * PROMOTION_MAX_DRAWDOWN_STAKE_FRACTION
        ),
        "roi_above_one": float(metrics["roi"]) > 1.0,
        "roi_ci95_lower_above_one": float(confidence["roi_ci95_lower"]) > 1.0,
        "probability_roi_above_one": (
            float(confidence["probability_roi_above_one"]) >= 0.95
        ),
    }
    stability = row.get("temporal_stability")
    if isinstance(stability, Mapping):
        gate["temporal_fold_evidence"] = bool(
            stability["all_minimum_evidence"]
        )
        gate["minimum_temporal_roi_above_one"] = (
            float(stability["minimum_roi"]) > 1.0
        )
    recent = row.get("recent_allocation")
    if isinstance(recent, Mapping):
        gate["recent_allocation_stable"] = bool(recent["stable"])
    return gate


def _valid_caps(policy: Mapping[str, Any]) -> bool:
    minimum_odds = policy.get("min_estimated_odds")
    maximum_odds = policy.get("max_estimated_odds")
    return (
        policy["min_daily_exposure_fraction"]
        <= policy["max_daily_exposure_fraction"]
        and policy["ticket_cap_fraction"] <= policy["race_cap_fraction"]
        and policy["race_cap_fraction"]
        <= policy["max_daily_exposure_fraction"]
        and (
            minimum_odds is None
            or maximum_odds is None
            or float(minimum_odds) <= float(maximum_odds)
        )
    )


def _validate_policy_candidate(policy: Mapping[str, Any]) -> None:
    required = {
        "daily_budget_yen",
        "fractional_kelly",
        "max_daily_exposure_fraction",
        "min_daily_exposure_fraction",
        "race_cap_fraction",
        "ticket_cap_fraction",
        "allocation_mode",
        "stake_granularity_yen",
        "min_stake_yen",
        *SEARCH_SPACE,
    }
    missing = sorted(required - policy.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    for name in (
        "daily_budget_yen",
        "fractional_kelly",
        "max_daily_exposure_fraction",
        "min_daily_exposure_fraction",
        "race_cap_fraction",
        "ticket_cap_fraction",
        "stake_granularity_yen",
        "min_stake_yen",
        "ev_threshold",
        "min_ticket_probability",
    ):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be numeric")
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    minimum_odds = policy["min_estimated_odds"]
    if minimum_odds is not None and (
        isinstance(minimum_odds, bool)
        or not isinstance(minimum_odds, (int, float, np.number))
        or not isfinite(float(minimum_odds))
        or float(minimum_odds) <= 1.0
    ):
        raise ValueError("min_estimated_odds must be null or finite and above one")
    maximum_odds = policy["max_estimated_odds"]
    if maximum_odds is not None and (
        isinstance(maximum_odds, bool)
        or not isinstance(maximum_odds, (int, float, np.number))
        or not isfinite(float(maximum_odds))
        or float(maximum_odds) <= 1.0
    ):
        raise ValueError("max_estimated_odds must be null or finite and above one")
    if (
        minimum_odds is not None
        and maximum_odds is not None
        and float(minimum_odds) > float(maximum_odds)
    ):
        raise ValueError("min_estimated_odds must not exceed max_estimated_odds")
    max_tickets = policy["max_daily_tickets"]
    if max_tickets is not None and (
        isinstance(max_tickets, bool)
        or not isinstance(max_tickets, (int, np.integer))
    ):
        raise ValueError("max_daily_tickets must be null or an integer")
    if max_tickets is not None and int(max_tickets) <= 0:
        raise ValueError("max_daily_tickets must be positive when set")
    if not _valid_caps(policy):
        raise ValueError("policy exposure caps are inconsistent")
    if float(policy["ev_threshold"]) < 0.0:
        raise ValueError("ev_threshold must be non-negative")
    if not 0.0 <= float(policy["min_ticket_probability"]) <= 1.0:
        raise ValueError("min_ticket_probability must be between zero and one")
    validate_policy(
        daily_budget_yen=int(policy["daily_budget_yen"]),
        fractional_kelly=float(policy["fractional_kelly"]),
        max_daily_exposure_fraction=float(policy["max_daily_exposure_fraction"]),
        min_daily_exposure_fraction=float(policy["min_daily_exposure_fraction"]),
        race_cap_fraction=float(policy["race_cap_fraction"]),
        ticket_cap_fraction=float(policy["ticket_cap_fraction"]),
        max_daily_tickets=(int(max_tickets) if max_tickets is not None else None),
        allocation_mode=str(policy["allocation_mode"]),
        stake_granularity_yen=int(policy["stake_granularity_yen"]),
        min_stake_yen=int(policy["min_stake_yen"]),
    )


def _policy_key(policy: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(policy.get(name) for name in SEARCH_SPACE)
