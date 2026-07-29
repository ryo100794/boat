from __future__ import annotations

import json

from boatrace_ai.listwise.v17_segment_diagnostics import (
    diagnose_segments,
    extract_selected_candidates,
    main,
    segment_labels,
)


def _ticket(
    race_id: str,
    combination: str,
    *,
    hit: bool,
    returned: int,
    venue: str = "01",
) -> dict:
    return {
        "race_id": race_id,
        "combination": combination,
        "jcd": venue,
        "stake_yen": 100,
        "probability": 0.04,
        "market_probability": 0.02,
        "estimated_odds": 30.0,
        "estimated_ev": 1.2,
        "t5_trend": 0.08,
        "hit": hit,
        "return_yen": returned,
    }


def _job() -> dict:
    return {
        "outer_folds": [
            {
                "evaluation_date": "2026-07-27",
                "bankroll": {
                    "selected_candidates": [
                        _ticket("202607270101", "123", hit=True, returned=300),
                        _ticket("202607270102", "132", hit=True, returned=300),
                    ]
                },
            },
            {
                "evaluation_date": "2026-07-28",
                "bankroll": {
                    "daily": [{
                        "race_date": "2026-07-28",
                        "selected_sample": [
                            _ticket("202607280101", "123", hit=False, returned=0),
                            _ticket("202607280102", "145", hit=True, returned=900),
                        ],
                    }]
                },
            },
        ]
    }


def test_extracts_both_supported_shapes_and_separates_settlement() -> None:
    decisions, settlements, metadata = extract_selected_candidates(_job())

    assert len(decisions) == 4
    assert metadata["selected_list_sources"] == {
        "selected_candidates": 1,
        "selected_sample": 1,
    }
    assert all("hit" not in row and "return_yen" not in row for row in decisions)
    assert settlements[("2026-07-28", "202607280102", "145")] == {
        "hit": True,
        "return_yen": 900,
    }


def test_normalizes_hyphenated_real_world_combination() -> None:
    job = {
        "evaluation_date": "2026-07-29",
        "selected_sample": [
            _ticket("202607290101", "1-2-3", hit=True, returned=500)
        ],
    }

    decisions, settlements, metadata = extract_selected_candidates(job)

    assert metadata["invalid_rows"] == 0
    assert decisions[0]["combination"] == "123"
    assert settlements[("2026-07-29", "202607290101", "123")]["return_yen"] == 500


def test_attributes_are_segmented_with_fixed_decision_time_bins() -> None:
    decisions, _, _ = extract_selected_candidates(_job())
    labels = segment_labels(decisions[0])

    assert "estimated_ev:1.10-1.24" in labels
    assert "odds:30-99.9" in labels
    assert "model_market_ratio:>=2.0" in labels
    assert "venue:01" in labels
    assert "rno:1" in labels
    assert "lane_pattern:first=1|out-out" in labels
    assert "t5_trend:rising" in labels


def test_lodo_reports_holdout_and_largest_hit_excluded_metrics() -> None:
    result = diagnose_segments(_job(), min_days=1, min_tickets=2, min_hits=2)
    segment = next(
        row for row in result["segments"]
        if row["segment"] == "estimated_ev:1.10-1.24"
    )

    assert segment["aggregate"]["tickets"] == 4
    assert segment["aggregate"]["hits"] == 3
    assert segment["aggregate"]["roi"] == 3.75
    assert segment["aggregate"]["roi_excluding_largest_hit"] == 1.5
    folds = {row["holdout_date"]: row for row in segment["leave_one_day_out"]}
    assert folds["2026-07-27"]["holdout"]["profitable_day_rate"] == 1.0
    assert folds["2026-07-28"]["training_excluding_holdout"]["tickets"] == 2


def test_prequential_selection_uses_only_strictly_earlier_days() -> None:
    result = diagnose_segments(
        _job(), min_days=1, min_tickets=2, min_hits=2,
        max_prequential_segments=2,
    )
    first, second = result["prequential"]["daily"]

    assert first["prior_dates"] == []
    assert first["sample_insufficient"] is True
    assert first["holdout"]["tickets"] == 0
    assert second["prior_dates"] == ["2026-07-27"]
    assert second["sample_insufficient"] is False
    assert second["selected_segments"]
    assert second["holdout"]["tickets"] == 2
    assert result["post_hoc_best_is_promotion_evidence"] is False
    assert result["real_betting_enabled"] is False


def test_future_outcomes_cannot_change_earlier_prequential_decisions() -> None:
    original = diagnose_segments(
        _job(), min_days=1, min_tickets=2, min_hits=2,
        max_prequential_segments=2,
    )
    changed = _job()
    future = changed["outer_folds"][1]["bankroll"]["daily"][0]["selected_sample"]
    for row in future:
        row["hit"] = True
        row["return_yen"] = 1_000_000
    modified = diagnose_segments(
        changed, min_days=1, min_tickets=2, min_hits=2,
        max_prequential_segments=2,
    )

    assert original["decision_information_fingerprint"] == (
        modified["decision_information_fingerprint"]
    )
    for index in range(2):
        assert original["prequential"]["daily"][index]["selected_segments"] == (
            modified["prequential"]["daily"][index]["selected_segments"]
        )
        assert original["prequential"]["daily"][index][
            "selected_ticket_fingerprint"
        ] == modified["prequential"]["daily"][index][
            "selected_ticket_fingerprint"
        ]


def test_sample_shortage_is_explicit() -> None:
    result = diagnose_segments(_job(), min_days=10, min_tickets=100, min_hits=20)

    assert all(not row["sample_sufficiency"]["sufficient"] for row in result["segments"])
    assert result["segments"][0]["sample_sufficiency"]["missing"]["days"] > 0
    assert all(row["sample_insufficient"] for row in result["prequential"]["daily"])


def test_cli_reads_job_json_and_writes_diagnostic_json(tmp_path) -> None:
    input_path = tmp_path / "job.json"
    output_path = tmp_path / "v17.json"
    input_path.write_text(json.dumps(_job()), encoding="utf-8")

    assert main([
        str(input_path), str(output_path), "--min-days", "1",
        "--min-tickets", "2", "--min-hits", "2",
    ]) == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output["model_name"] == "v17_segment_diagnostics"
    assert output["input"]["tickets"] == 4
    assert output["prequential"]["selection_uses_strictly_prior_dates_only"] is True
