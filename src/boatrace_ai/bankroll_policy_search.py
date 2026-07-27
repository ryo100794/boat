from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .listwise.direct_bankroll import bootstrap_daily_bankroll
from .packed_bankroll import PackedCandidates, evaluate_packed_policy


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "ev_threshold": (1.00, 1.10, 1.20, 1.35, 1.50),
    "min_ticket_probability": (0.0, 0.002, 0.005, 0.01),
    "max_estimated_odds": (None, 30.0, 50.0, 100.0, 200.0),
    "fractional_kelly": (0.10, 0.25, 0.50),
    "max_daily_exposure_fraction": (0.30, 0.60, 0.80),
    "min_daily_exposure_fraction": (0.0, 0.20, 0.40),
    "race_cap_fraction": (0.05, 0.10, 0.15),
    "ticket_cap_fraction": (0.01, 0.02, 0.03),
    "max_daily_tickets": (10, 20, 30, 50),
}


# Preserve the sparse, conservative region when new search dimensions are added.
CONSERVATIVE_POLICY_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "ev_threshold": 1.35,
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
        "ev_threshold": 1.35,
        "min_ticket_probability": 0.0,
        "max_estimated_odds": 100.0,
        "fractional_kelly": 0.10,
        "max_daily_exposure_fraction": 0.30,
        "min_daily_exposure_fraction": 0.0,
        "race_cap_fraction": 0.05,
        "ticket_cap_fraction": 0.02,
        "max_daily_tickets": 30,
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


def policy_candidates(
    base_policy: Mapping[str, Any],
    *,
    count: int = 64,
    seed: int = 20260726,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = np.random.default_rng(seed)
    candidates = [dict(base_policy)]
    seen = {_policy_key(candidates[0])}
    for candidate in _conservative_anchor_policies(base_policy):
        if len(candidates) >= count:
            break
        key = _policy_key(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    attempts = 0
    while len(candidates) < count and attempts < count * 100:
        attempts += 1
        candidate = dict(base_policy)
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


def _retain_conservative_anchors(
    rows: Sequence[dict[str, Any]],
    *,
    keep: int,
    base_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = list(rows[:keep])
    anchor_keys = {
        _policy_key(policy)
        for policy in _conservative_anchor_policies(base_policy)
    }
    selected_keys = {_policy_key(row["policy"]) for row in selected}
    for row in rows:
        key = _policy_key(row["policy"])
        if key not in anchor_keys or key in selected_keys:
            continue
        replace_at = next(
            (
                index for index in range(len(selected) - 1, -1, -1)
                if _policy_key(selected[index]["policy"]) not in anchor_keys
            ),
            None,
        )
        if replace_at is None:
            break
        selected_keys.discard(_policy_key(selected[replace_at]["policy"]))
        selected[replace_at] = row
        selected_keys.add(key)
    return selected


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
) -> dict[str, Any]:
    if len(packed.dates) < 4:
        raise ValueError("at least four policy-selection days are required")
    if not 1 <= finalists <= candidate_count:
        raise ValueError("finalists must be between one and candidate count")
    active = policy_candidates(base_policy, count=candidate_count, seed=seed)
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
        retained = _retain_conservative_anchors(
            rows, keep=keep, base_policy=base_policy
        )
        active = [row["policy"] for row in retained]
        anchor_keys = {
            _policy_key(policy)
            for policy in _conservative_anchor_policies(base_policy)
        }
        stages.append({
            "fraction": fraction,
            "days": len(stage_data.dates),
            "evaluated_candidates": len(rows),
            "retained_candidates": len(active),
            "protected_anchor_count": sum(
                _policy_key(policy) in anchor_keys for policy in active
            ),
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
        "candidate_count": candidate_count,
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
        "minimum_tickets": int(metrics["tickets"]) >= 300,
        "minimum_hits": int(metrics["hit_tickets"]) >= 20,
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
    return (
        policy["min_daily_exposure_fraction"]
        <= policy["max_daily_exposure_fraction"]
        and policy["ticket_cap_fraction"] <= policy["race_cap_fraction"]
        and policy["race_cap_fraction"]
        <= policy["max_daily_exposure_fraction"]
    )


def _policy_key(policy: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(policy.get(name) for name in SEARCH_SPACE)
