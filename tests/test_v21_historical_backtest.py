from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from boatrace_ai.web.v21_historical_backtest import (
    build_projection,
    load_dashboard_payload,
    write_projection,
)


def _result() -> dict:
    return {
        "model": "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
        "generated_at": "2026-07-31T00:00:00+00:00",
        "validation_design": "prior dates only",
        "coverage_gate": {
            "minimum_day_coverage": 0.99,
            "days": [{
                "race_date": "2026-07-30",
                "complete_races": 2,
                "eligible_t5_races": 1,
            }],
        },
        "chronological_bankroll": {"daily": [{
            "race_date": "2026-07-30",
            "initial_bankroll_yen": 10_000,
            "closing_bankroll_yen": 13_800,
            "evaluated_races": 1,
            "tickets": 1,
            "races_bet": 1,
            "hit_tickets": 1,
            "stake_yen": 200,
            "return_yen": 4_000,
            "profit_yen": 3_800,
            "roi": 20.0,
            "max_drawdown_yen": 200,
            "ledger": [
                {"event": "decision", "race_id": "r1", "selections": [{}]},
                {
                    "event": "settlement", "race_id": "r1", "stake_yen": 200,
                    "return_yen": 4_000, "cash_after_yen": 13_800,
                    "outstanding_stake_yen": 0,
                },
            ],
        }]},
    }


def test_builds_and_loads_v21_historical_dashboard_projection(tmp_path: Path) -> None:
    path = tmp_path / "2026-07-30.json"
    projection = build_projection(_result(), race_date="2026-07-30", source_job_id=9271)
    write_projection(path, projection)
    payload = load_dashboard_payload(
        path,
        race_date="2026-07-30",
        models=[{"id": "v21_daily"}],
        selected_model={"label": "V21 主系"},
        schedule=[{
            "race_id": "r1", "venue": "桐生", "jcd": "01", "rno": 1,
            "race_time_at": "2026-07-30T10:00:00+09:00",
        }],
        now_jst=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )

    assert payload is not None
    assert payload["selected_model"] == "v21_daily"
    assert payload["stats"]["profit_yen"] == 3_800
    assert payload["series"][0]["venue"] == "桐生"
    assert payload["backtest_source_job_id"] == 9271


def test_rejects_non_v21_result() -> None:
    result = _result()
    result["model"] = "v20"
    try:
        build_projection(result, race_date="2026-07-30", source_job_id=1)
    except ValueError as exc:
        assert "not V21" in str(exc)
    else:
        raise AssertionError("non-V21 result was accepted")
