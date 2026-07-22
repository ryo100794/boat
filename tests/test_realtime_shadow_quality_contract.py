import json

from boatrace_ai.runtime.model_cycle import evaluation_baseline
from boatrace_ai.web.dashboard import _model_track_summaries


def test_old_quality_contract_resets_incremental_baseline() -> None:
    last_races, last_at, reason = evaluation_baseline(
        {
            "last_evaluated_races": 546,
            "last_evaluated_at": "2026-07-22T03:37:03+00:00",
        }
    )

    assert last_races == 0
    assert last_at is None
    assert reason == "data_quality_contract_changed"


def test_current_quality_contract_preserves_incremental_baseline() -> None:
    last_races, last_at, reason = evaluation_baseline(
        {
            "evaluation_version": 2,
            "last_evaluated_races": 450,
            "last_evaluated_at": "2026-07-23T03:00:00+00:00",
        }
    )

    assert last_races == 450
    assert last_at == "2026-07-23T03:00:00+00:00"
    assert reason is None


def test_model_report_hides_pre_quality_filter_odds_metrics(tmp_path) -> None:
    (tmp_path / "realtime_odds_shadow_t5_safe_candidate_state.json").write_text(
        json.dumps(
            {
                "eligible_races": 232,
                "required_races": 450,
                "last_evaluated_races": 546,
                "status": "waiting_for_data",
            }
        ),
        encoding="utf-8",
    )
    backtests = [
        {
            "file": "realtime_odds_shadow_t5_safe_candidate_backtest.json",
            "races": 547,
            "evaluated_races": 147,
            "entry_log_loss": 0.39755,
            "winner_top1_accuracy": 0.55102,
            "trifecta_top5_hit_rate": 0.31293,
        }
    ]

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, backtests, {"jobs": []})
    }
    shadow = tracks["realtime_odds_shadow"]

    assert shadow["status"] == "品質更新後の再蓄積中"
    assert shadow["eligible_races"] == 232
    assert shadow["backtest_available"] is False
    assert shadow["entry_log_loss"] is None
    assert shadow["winner_top1_accuracy"] is None
    assert shadow["trifecta_top5_hit_rate"] is None
