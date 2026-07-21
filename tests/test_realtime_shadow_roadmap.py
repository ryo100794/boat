from boatrace_ai.web.dashboard import _roadmap_milestones


def test_m5_tracks_shadow_process_and_realtime_readiness() -> None:
    milestones = _roadmap_milestones(
        {
            "realtime": {
                "eligible_races": 214,
                "target_eligible_races": 1000,
                "readiness": 0.214,
            }
        },
        [{"kind": "リアルタイムshadow"}],
        {},
    )
    m5 = next(row for row in milestones if row["id"] == "M5")

    assert m5["status"] == "並走/蓄積中"
    assert m5["progress"] == 24
    assert "214/1,000" in m5["next"]


def test_m6_exposes_provisional_real_odds_evaluation() -> None:
    from boatrace_ai.web.dashboard import _roadmap_improvements

    progress = {
        "realtime": {
            "eligible_races": 629,
            "target_eligible_races": 1000,
            "readiness": 0.629,
        },
        "realtime_shadow_evaluation": {
            "available": True,
            "evaluated_races": 95,
            "entry_log_loss": 0.345016,
            "winner_top1_accuracy": 0.589474,
            "trifecta_top5_hit_rate": 0.326316,
            "bankroll_evaluated_races": 105,
            "roi": 0.84,
            "profit_yen": -1600,
        },
    }
    rows = {
        row["id"]: row
        for row in _roadmap_improvements(
            progress,
            [{"kind": "リアルタイムshadow"}, {"kind": "予測ループ"}],
            {},
        )
    }

    assert rows["M6-1"]["status"] == "暫定評価済み/正式1,000R待ち"
    assert "LogLoss 0.3450" in rows["M6-1"]["next"]
    assert rows["M6-10"]["status"] == "暫定評価済み/要改善"
    assert "ROI 0.8400" in rows["M6-10"]["next"]
