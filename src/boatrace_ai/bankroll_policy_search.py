from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .listwise.direct_bankroll import bootstrap_daily_bankroll
from .packed_bankroll import PackedCandidates, evaluate_packed_policy


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "ev_threshold": (1.00, 1.10, 1.20, 1.35, 1.50),
    "fractional_kelly": (0.10, 0.25, 0.50),
    "max_daily_exposure_fraction": (0.30, 0.60, 0.80),
    "min_daily_exposure_fraction": (0.0, 0.20, 0.40),
    "race_cap_fraction": (0.05, 0.10, 0.15),
    "ticket_cap_fraction": (0.01, 0.02, 0.03),
    "max_daily_tickets": (10, 20, 30, 50),
}


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
        active = [row["policy"] for row in rows[:keep]]
        stages.append({
            "fraction": fraction,
            "days": len(stage_data.dates),
            "evaluated_candidates": len(rows),
            "retained_candidates": len(active),
            "leaders": rows[: min(10, len(rows))],
        })

    final_rows = []
    for policy in active:
        result = evaluate_packed_policy(packed, policy)
        confidence = bootstrap_daily_bankroll(
            result["daily"], samples=bootstrap_samples, seed=seed
        )
        final_rows.append({
            "policy": policy,
            "metrics": compact_metrics(result),
            "confidence": confidence,
        })
    final_rows.sort(
        key=lambda row: (
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


def promotion_gate(row: Mapping[str, Any]) -> dict[str, bool]:
    metrics = row["metrics"]
    confidence = row["confidence"]
    return {
        "minimum_tickets": int(metrics["tickets"]) >= 300,
        "minimum_hits": int(metrics["hit_tickets"]) >= 20,
        "roi_above_one": float(metrics["roi"]) > 1.0,
        "roi_ci95_lower_above_one": float(confidence["roi_ci95_lower"]) > 1.0,
        "probability_roi_above_one": (
            float(confidence["probability_roi_above_one"]) >= 0.95
        ),
    }


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
