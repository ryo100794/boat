from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..listwise.direct_bankroll import bootstrap_daily_bankroll
from ..listwise.market_calibration import write_json_atomic


MINIMUM_DAYS = 30
MINIMUM_RACES = 1_000
MINIMUM_TICKETS = 200
MINIMUM_EFFECTIVE_HITS = 20.0
MAXIMUM_LARGEST_HIT_RETURN_SHARE = 0.15
MINIMUM_CONFIDENCE = 0.95


def prospective_promotion_payload(
    source: dict[str, Any],
    *,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any] | None:
    track_name = "trend_point_market_offset_kelly_walk_forward"
    track = source.get(track_name)
    if not isinstance(track, dict):
        track_name = "prospective_normalized_ev_walk_forward"
        track = source.get(track_name)
    if not isinstance(track, dict) or not isinstance(track.get("policy"), dict):
        return None
    daily = list(track.get("daily") or [])
    if not daily:
        return None

    confidence = bootstrap_daily_bankroll(
        daily,
        samples=max(100, int(bootstrap_samples)),
    )
    days = int(track.get("evaluation_days") or len(daily))
    races = int(track.get("evaluated_races") or 0)
    tickets = int(track.get("tickets") or 0)
    stake_yen = int(track.get("stake_yen") or 0)
    return_yen = int(track.get("return_yen") or 0)
    profit_yen = return_yen - stake_yen
    roi = return_yen / stake_yen if stake_yen else 0.0
    effective_hits = float(track.get("effective_hit_count") or 0.0)
    roi_without_largest = float(track.get("roi_without_largest_hit") or 0.0)
    raw_largest_hit_return_share = track.get("largest_hit_return_share")
    largest_hit_return_share = (
        float(raw_largest_hit_return_share)
        if raw_largest_hit_return_share is not None
        else None
    )
    track_gate = track.get("promotion_gate")
    track_gate = track_gate if isinstance(track_gate, dict) else {}
    source_gate = source.get("promotion_gate") or {}
    source_formal_gate_pass = (
        track_gate.get("pass") is True
        if track_name == "trend_point_market_offset_kelly_walk_forward"
        else True
    )

    sample_size_pass = bool(
        days >= MINIMUM_DAYS
        and races >= MINIMUM_RACES
        and tickets >= MINIMUM_TICKETS
        and effective_hits >= MINIMUM_EFFECTIVE_HITS
    )
    stability_pass = bool(
        roi_without_largest > 1.0
        and float(confidence["roi_ci95_lower"]) > 1.0
        and float(confidence["probability_roi_above_one"]) >= MINIMUM_CONFIDENCE
    )
    gate = {
        "minimum_evaluation_races": MINIMUM_RACES,
        "minimum_evaluation_days": MINIMUM_DAYS,
        "minimum_tickets": MINIMUM_TICKETS,
        "minimum_effective_hits": MINIMUM_EFFECTIVE_HITS,
        "maximum_largest_hit_return_share": MAXIMUM_LARGEST_HIT_RETURN_SHARE,
        "sample_size_pass": sample_size_pass,
        "positive_profit_pass": profit_yen > 0 and stake_yen > 0,
        "roi_pass": roi > 1.0,
        "fold_stability_pass": stability_pass,
        "largest_hit_return_share_pass": (
            largest_hit_return_share is not None
            and largest_hit_return_share <= MAXIMUM_LARGEST_HIT_RETURN_SHARE
        ),
        "source_formal_gate_pass": source_formal_gate_pass,
        "calibration_pass": (
            source_gate.get("calibration_pass") is True
            if track_name == "prospective_normalized_ev_walk_forward"
            else source_formal_gate_pass
        ),
        "market_confidence_pass": (
            source_gate.get("market_confidence_pass") is True
            if track_name == "prospective_normalized_ev_walk_forward"
            else source_formal_gate_pass
        ),
        "no_lookahead_pass": (
            source_gate.get("no_lookahead_pass") is True
            if track_name == "prospective_normalized_ev_walk_forward"
            else track_gate.get("no_lookahead_pass") is True
        ),
    }
    gate["pass"] = all(
        value for key, value in gate.items() if key.endswith("_pass") and key != "pass"
    )

    deployment = copy.deepcopy(source.get("deployment_configuration") or {})
    policy = copy.deepcopy(track["policy"])
    deployment["candidate_policy"] = policy
    deployment["selected_policy"] = (
        policy if gate["pass"] else {"name": "no_bet", "no_bet": True}
    )
    deployment["operational_status"] = (
        "eligible_for_prospective_promotion"
        if gate["pass"]
        else "shadow_only_insufficient_prospective_evidence"
    )
    deployment["walk_forward_gate"] = {
        **gate,
        "evaluation_days": days,
        "evaluation_races": races,
        "tickets": tickets,
        "effective_hit_count": effective_hits,
        "largest_hit_return_share": largest_hit_return_share,
        "roi": roi,
        "roi_without_largest_hit": roi_without_largest,
        "confidence": confidence,
    }

    cumulative_profit = peak_profit = max_drawdown = 0
    for row in sorted(daily, key=lambda item: str(item.get("race_date") or "")):
        cumulative_profit += int(row.get("profit_yen") or 0)
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)

    result = copy.deepcopy(source)
    result.update(
        {
            "model": f"{source.get('model') or 'market'}_prospective_normalized_ev_v2",
            "comparison_role": "prospective_only_pre_registered_policy_production_candidate",
            "from_date": min(str(row["race_date"]) for row in daily),
            "evaluation_races": races,
            "evaluated_races": races,
            "evaluation_days": days,
            "tickets": tickets,
            "hit_tickets": int(track.get("hit_tickets") or 0),
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": profit_yen,
            "roi": roi,
            "roi_without_largest_hit": roi_without_largest,
            "largest_hit_return_share": largest_hit_return_share,
            "effective_hit_count": effective_hits,
            "max_drawdown_yen": max_drawdown,
            "daily": daily,
            "deployment_configuration": deployment,
            "promotion_gate": gate,
            "promotion_eligible": bool(gate["pass"]),
            "prospective_promotion_confidence": confidence,
            "prospective_source_result": source.get("result_path"),
            "prospective_source_track": track_name,
        }
    )
    return result


def write_prospective_candidate(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    bootstrap_samples: int = 5_000,
) -> str | None:
    source_file = Path(source_path).resolve()
    try:
        source = json.loads(source_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(source, dict):
        return None
    payload = prospective_promotion_payload(
        source,
        bootstrap_samples=bootstrap_samples,
    )
    if payload is None:
        return None
    payload["prospective_source_result"] = str(source_file)
    destination = Path(output_dir).resolve() / (
        source_file.stem + ".prospective-normalized-v2.json"
    )
    write_json_atomic(destination, payload)
    return str(destination)
