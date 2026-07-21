from __future__ import annotations

from datetime import date, timedelta
import json

from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    PROTOCOL_ID,
    evaluate_promotions,
    protocol_sha256,
)
from boatrace_ai.web.dashboard import (
    _load_standardized_v2_bundle,
    _merge_standardized_v2_status,
    model_performance_report,
)


def _write_bundle(model_dir, *, protocol_hash_override: str | None = None):
    root = model_dir / "standardized_365d_v2"
    root.mkdir(parents=True)
    expected_ids = [source.model_id for source in MODEL_SOURCES]
    holdout_start = date(2025, 7, 20)
    holdout_end = date(2026, 7, 19)
    race_hash = "a" * 64
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": "2026-07-20T00:00:00+00:00",
        "as_of_date_jst": "2026-07-20",
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": holdout_end.isoformat(),
        "calendar_days": 365,
        "holdout_date_count": 365,
        "training_races": 1_000,
        "prediction_races": 365,
        "bankroll_evaluable_races": 365,
        "prediction_universe": "exactly 6 entries and 6 non-null finish ranks",
        "bankroll_universe": "prediction universe with a non-null official trifecta payout",
        "race_set_sha256": race_hash,
        "full_day_boundary": True,
        "model_selection": "training period only; final 365-day holdout untouched",
        "policy": dict(POLICY),
    }
    protocol["protocol_sha256"] = protocol_sha256(protocol)
    daily = []
    cumulative = 0
    for offset in range(365):
        cumulative -= 20
        daily.append(
            {
                "race_date": (holdout_start + timedelta(days=offset)).isoformat(),
                "evaluated_races": 1,
                "candidate_tickets": 1,
                "tickets": 1,
                "races_bet": 1,
                "hit_tickets": 0,
                "hit_races": 0,
                "stake_yen": 100,
                "return_yen": 80,
                "profit_yen": -20,
                "cumulative_profit_yen": cumulative,
                "roi": 0.8,
                "budget_used_fraction": 0.01,
            }
        )
    models = []
    for index, source in enumerate(MODEL_SOURCES):
        validation = {
            "prediction_race_count": 365,
            "expected_prediction_races": 365,
            "bankroll_race_count": 365,
            "expected_bankroll_races": 365,
            "prediction_race_set_sha256": race_hash,
            "bankroll_race_set_sha256": race_hash,
            "expected_race_set_sha256": race_hash,
            "daily_date_count": 365,
            "expected_daily_date_count": 365,
            "daily_start": holdout_start.isoformat(),
            "daily_end": holdout_end.isoformat(),
            "policy_mismatches": [],
            "passed": True,
        }
        model = {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluation_scope": PROTOCOL_ID,
            "generated_at": protocol["generated_at"],
            "model_id": source.model_id,
            "model": source.model_id,
            "role": source.role,
            "feature_set": source.model_id,
            "include_odds": False,
            "protocol": dict(protocol),
            "policy": dict(POLICY),
            "validation": validation,
            "evaluated_races": 365,
            "bankroll_evaluated_races": 365,
            "evaluation_race_set_sha256": race_hash,
            "daily": list(daily),
            "entry_log_loss": 0.3 + index / 100,
            "entry_brier": 0.1,
            "winner_top1_accuracy": 0.57 - index / 100,
            "trifecta_top1_hit_rate": 0.08,
            "trifecta_top5_hit_rate": 0.31 - index / 100,
            "ranking_log_loss": 1.2,
            "race_days": 365,
            "candidate_tickets": 365,
            "selected_races": 365,
            "tickets": 365,
            "hit_tickets": 0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "stake_yen": 36_500,
            "return_yen": 29_200,
            "profit_yen": -7_300,
            "roi": 0.8,
            "winning_days": 0,
            "losing_days": 365,
            "budget_utilization": 0.01,
            "max_drawdown_yen": 7_300,
        }
        model["bankroll"] = {
            key: model[key]
            for key in (
                "race_days",
                "candidate_tickets",
                "selected_races",
                "tickets",
                "hit_tickets",
                "ticket_hit_rate",
                "race_hit_rate",
                "stake_yen",
                "return_yen",
                "profit_yen",
                "roi",
                "winning_days",
                "losing_days",
                "budget_utilization",
                "max_drawdown_yen",
            )
        }
        models.append(model)
    decision = evaluate_promotions(models, comparison_ready=True)
    manifest = {
        **protocol,
        "comparison_model_ids": expected_ids,
        "registry_coverage_passed": True,
        "model_count": len(models),
        "valid_model_count": len(models),
        "failed_models": [],
        "comparison_ready": True,
        "promotion_decision": decision,
        "models": [
            {
                "model_id": model["model_id"],
                "role": model["role"],
                "validation": model["validation"],
                "promotion": model["promotion"],
                "entry_log_loss": model["entry_log_loss"],
                "winner_top1_accuracy": model["winner_top1_accuracy"],
                "trifecta_top5_hit_rate": model["trifecta_top5_hit_rate"],
                "roi": model["roi"],
                "profit_yen": model["profit_yen"],
            }
            for model in models
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stored_protocol = dict(protocol)
    if protocol_hash_override:
        stored_protocol["protocol_sha256"] = protocol_hash_override
    (root / "protocol.json").write_text(
        json.dumps(stored_protocol), encoding="utf-8"
    )
    for model in models:
        (root / f"{model['model_id']}.json").write_text(
            json.dumps(model), encoding="utf-8"
        )
    return manifest


def test_report_accepts_only_complete_current_protocol(tmp_path) -> None:
    manifest = _write_bundle(tmp_path)

    bundle = _load_standardized_v2_bundle(tmp_path)
    merged = _merge_standardized_v2_status({"jobs": []}, bundle)

    assert bundle["ready"] is True
    assert len(bundle["models"]) == len(MODEL_SOURCES)
    assert {
        row["name"]
        for row in merged["jobs"]
        if row["kind"] == "standardized_365d_v2_model"
    } == {
        f"standardized_365d_v2_{source.model_id}"
        for source in MODEL_SOURCES
    }
    assert merged["generated_at"] == manifest.get("generated_at")


def test_model_report_contains_exactly_seven_unified_series(tmp_path) -> None:
    model_dir = tmp_path / "models"
    _write_bundle(model_dir)

    report = model_performance_report(
        tmp_path / "boatrace.sqlite",
        {"model_dir": [str(model_dir)]},
    )

    expected = {
        f"standardized_365d_v2_{source.model_id}"
        for source in MODEL_SOURCES
    }
    backtests = [
        row["name"]
        for row in report["backtests"]
        if row.get("evaluation_scope") == PROTOCOL_ID
    ]
    bankroll = [
        row["name"]
        for row in report["bankroll"]
        if row.get("evaluation_scope") == PROTOCOL_ID
    ]
    assert len(backtests) == len(expected) == len(set(backtests))
    assert len(bankroll) == len(expected) == len(set(bankroll))
    assert set(backtests) == expected
    assert set(bankroll) == expected
    assert set(report["bankroll_daily"]) == expected
    assert all(len(rows) == 365 for rows in report["bankroll_daily"].values())


def test_report_rejects_corrupt_daily_race_total(tmp_path) -> None:
    _write_bundle(tmp_path)
    path = tmp_path / "standardized_365d_v2" / "pastlog_v7.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["daily"][0]["evaluated_races"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")

    bundle = _load_standardized_v2_bundle(tmp_path)

    assert bundle["ready"] is False
    assert any("daily evaluated race aggregate mismatch" in error for error in bundle["errors"])


def test_report_rejects_stale_manifest_for_new_protocol(tmp_path) -> None:
    _write_bundle(tmp_path, protocol_hash_override="new-protocol")

    bundle = _load_standardized_v2_bundle(tmp_path)
    merged = _merge_standardized_v2_status(
        {
            "jobs": [
                {
                    "kind": "standardized_365d_v2_model",
                    "name": "standardized_365d_v2_no_odds_v8",
                    "status": "完了",
                }
            ]
        },
        bundle,
    )

    assert bundle["ready"] is False
    assert "current protocol is not consolidated" in bundle["errors"]
    assert not any(
        row["kind"] == "standardized_365d_v2_model"
        for row in merged["jobs"]
    )
    assert any(
        row["kind"] == "standardized_365d_v2_queue"
        and row["status"] == "実行中"
        for row in merged["jobs"]
    )
