from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib

from .empirical_lcb_policy import policy_edge_records, simulate_empirical_lcb_policy
from .market_calibration import (
    BANDWISE_DIAGNOSTIC_MIN_TRAINING_DAYS,
    _attach_oof_closing_odds_forecast,
    _attach_t5_policy_fallback,
    _fit_prior_empirical_ev_artifact,
    _summarize_empirical_lcb_walk_forward,
    apply_prequential_closing_odds_policy_inputs,
    attach_odds_path_model,
    blend_probabilities,
    verifiable_closing_odds_races,
)
from .market_edge_diagnostics import edge_records, summarize_edge_records


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_scored_races(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("races"), list):
        raise ValueError("scored cache must contain a races list")
    races = payload["races"]
    if not all(isinstance(race, dict) for race in races):
        raise ValueError("scored cache races must be objects")
    return payload, races


def _strict_prior_closing_teachers(
    races: list[dict[str, Any]], evaluation_date: str
) -> list[dict[str, Any]]:
    return verifiable_closing_odds_races(
        [race for race in races if str(race["race_date"]) < evaluation_date]
    )


def _reconstruct_policy_races(
    all_races: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    fold: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evaluation_date = str(fold["evaluation_date"])
    operational_model = fold.get("operational_model")
    if not isinstance(operational_model, dict):
        raise ValueError(f"{evaluation_date}: operational_model is missing")
    transformed = attach_odds_path_model(holdout, operational_model)

    selection = fold.get("closing_odds_selection")
    if isinstance(selection, dict):
        teachers = _strict_prior_closing_teachers(all_races, evaluation_date)
        teacher_dates = sorted({str(race["race_date"]) for race in teachers})
        if not teacher_dates:
            raise ValueError(f"{evaluation_date}: closing teachers are missing")
        attached = _attach_oof_closing_odds_forecast(
            holdout,
            selection,
            training_dates=teacher_dates,
            training_races=len(teachers),
        )
    else:
        reason = str(
            fold.get("closing_odds_policy_fallback_reason")
            or "source_result_fallback"
        )
        attached = _attach_t5_policy_fallback(holdout, reason=reason)
    inputs = {
        "races_by_id": {str(race["race_id"]): race for race in attached},
    }
    return apply_prequential_closing_odds_policy_inputs(transformed, inputs)


def replay_bandwise_empirical_policy(
    source_result: Mapping[str, Any],
    scored_cache: Mapping[str, Any],
    *,
    daily_budget_yen: int = 10_000,
) -> dict[str, Any]:
    races = scored_cache.get("races")
    if not isinstance(races, list) or not all(isinstance(row, dict) for row in races):
        raise ValueError("scored cache must contain object races")
    folds = source_result.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("source result must contain evaluation folds")
    expected_edges = source_result.get("edge_diagnostics")
    if not isinstance(expected_edges, dict):
        raise ValueError("source result edge diagnostics are required")

    by_day: dict[str, list[dict[str, Any]]] = {}
    for race in races:
        by_day.setdefault(str(race["race_date"]), []).append(race)

    teacher_history: list[dict[str, Any]] = []
    replay_edges: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    evaluated_races = 0
    for fold_number, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict):
            raise ValueError("source result folds must be objects")
        evaluation_date = str(fold["evaluation_date"])
        holdout = by_day.get(evaluation_date, [])
        if len(holdout) != int(fold.get("evaluation_races") or 0):
            raise ValueError(f"{evaluation_date}: holdout race count mismatch")
        purchase_calibrator = fold.get("purchase_calibrator")
        if not isinstance(purchase_calibrator, dict):
            raise ValueError(f"{evaluation_date}: purchase calibrator is missing")
        policy_races = _reconstruct_policy_races(races, holdout, fold)
        replay_edges.extend(
            edge_records(
                policy_races,
                calibrator=purchase_calibrator,
                probability_blender=blend_probabilities,
            )
        )
        artifact = _fit_prior_empirical_ev_artifact(
            teacher_history,
            evaluation_date,
            shape_constraint="bandwise",
        )
        bankroll = simulate_empirical_lcb_policy(
            policy_races,
            purchase_calibrator,
            blend_probabilities,
            artifact,
            daily_budget_yen,
        )
        daily_rows.extend(bankroll["daily"])
        evaluated_races += len(policy_races)
        fold_rows.append(
            {
                "fold": fold_number,
                "evaluation_date": evaluation_date,
                "shape_constraint": "bandwise",
                "calibration_ready": artifact.ready,
                "trained_through_date": artifact.trained_through_date,
                "training_days": artifact.training_days,
                "training_tickets": artifact.tickets,
                "candidate_days": artifact.candidate_days,
                "ready_reasons": list(artifact.ready_reasons),
            }
        )
        teacher_history.extend(
            policy_edge_records(
                policy_races,
                purchase_calibrator,
                blend_probabilities,
            )
        )

    actual_edges = summarize_edge_records(replay_edges)
    expected_hash = _canonical_sha256(expected_edges)
    actual_hash = _canonical_sha256(actual_edges)
    if actual_hash != expected_hash:
        raise ValueError(
            "policy replay does not match source edge diagnostics: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    result = _summarize_empirical_lcb_walk_forward(
        daily_rows,
        evaluated_races=evaluated_races,
        folds=fold_rows,
    )
    result.update(
        {
            "comparison_role": (
                "source_verified_policy_only_nonmonotonic_bandwise_lcb95"
            ),
            "shape_constraint": "bandwise",
            "evaluation_mode": "provisional_prior_only_diagnostic",
            "minimum_training_days": BANDWISE_DIAGNOSTIC_MIN_TRAINING_DAYS,
            "promotion_eligible": False,
            "source_edge_diagnostics_sha256": expected_hash,
            "replayed_edge_diagnostics_sha256": actual_hash,
            "source_edge_diagnostics_match": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay only the empirical purchase policy from a verified result."
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    args = parser.parse_args()

    source = _load_json(args.source_result)
    cache, _races = _load_scored_races(args.scored_cache)
    result = replay_bandwise_empirical_policy(
        source,
        cache,
        daily_budget_yen=args.daily_budget_yen,
    )
    result["source_result"] = str(args.source_result)
    result["source_result_sha256"] = hashlib.sha256(
        args.source_result.read_bytes()
    ).hexdigest()
    result["scored_cache"] = str(args.scored_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
