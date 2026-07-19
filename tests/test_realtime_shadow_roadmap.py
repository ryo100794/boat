from boatrace_ai.web_dashboard import _roadmap_milestones


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
    assert m5["progress"] == 35
    assert "214/1,000" in m5["next"]
