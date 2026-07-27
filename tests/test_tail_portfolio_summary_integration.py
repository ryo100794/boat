from __future__ import annotations

import json

from boatrace_ai.adaptive_allocation import allocate_adaptive_day
from boatrace_ai.evaluation_queue import _load_result


def test_adaptive_allocation_retains_all_tickets_for_tail_diagnostics() -> None:
    candidates = [
        {
            "race_id": f"race-{index // 7}",
            "combination": f"ticket-{index}",
            "probability": 0.2,
            "estimated_odds": 10.0,
            "estimated_ev": 2.0,
            "actual_payout_yen": 2_000,
            "hit": index == 0,
        }
        for index in range(14)
    ]

    result = allocate_adaptive_day(
        "2026-07-20",
        candidates,
        {"race-0", "race-1"},
        daily_budget_yen=10_000,
        fractional_kelly=0.25,
        max_daily_exposure_fraction=1.0,
        min_daily_exposure_fraction=0.0,
        race_cap_fraction=1.0,
        ticket_cap_fraction=0.1,
        max_daily_tickets=None,
        allocation_mode="kelly",
        stake_granularity_yen=100,
        min_stake_yen=100,
    )

    assert result["tickets"] == 14
    assert len(result["selected_sample"]) == 12
    assert len(result["_tail_portfolio_rows"]) == 14
    assert result["_tail_portfolio_rows"][0] == {
        "date": "2026-07-20",
        "race_id": "race-0",
        "odds": 10.0,
        "stake": 200,
        "return": 4_000,
    }


def test_load_result_persists_tail_diagnostics_in_json_and_db_summary(
    tmp_path,
) -> None:
    result_path = tmp_path / "evaluation.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "candidate",
                "bankroll": {
                    "roi": 1.5,
                    "daily": [
                        {
                            "race_date": "2026-07-20",
                            "_tail_portfolio_rows": [
                                {
                                    "date": "2026-07-20",
                                    "race_id": "normal-hit",
                                    "odds": 20.0,
                                    "stake": 100,
                                    "return": 2_000,
                                },
                                {
                                    "date": "2026-07-20",
                                    "race_id": "tail-loss",
                                    "odds": 101.0,
                                    "stake": 100,
                                    "return": 0,
                                },
                            ],
                        },
                        {
                            "race_date": "2026-07-21",
                            "_tail_portfolio_rows": [
                                {
                                    "date": "2026-07-21",
                                    "race_id": "tail-hit",
                                    "odds": 150.0,
                                    "stake": 100,
                                    "return": 15_000,
                                }
                            ],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    payload, summary = _load_result(result_path)

    diagnostics = payload["bankroll"]["tail_portfolio_diagnostics"]
    assert diagnostics["odds_field"] == "estimated_odds_at_purchase"
    assert diagnostics["normal"]["tickets"] == 1
    assert diagnostics["normal"]["roi"] == 20.0
    assert diagnostics["normal"]["roi_excluding_largest_hit"] == 0.0
    assert diagnostics["tail"]["tickets"] == 2
    assert diagnostics["tail"]["roi"] == 75.0
    assert diagnostics["tail"]["roi_excluding_largest_hit"] == 0.0
    assert diagnostics["tail"]["daily_cluster_bootstrap_roi_lower_95"] == 0.0
    assert summary["tail_portfolio_diagnostics"] == diagnostics

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["bankroll"]["tail_portfolio_diagnostics"] == diagnostics
    assert all(
        "_tail_portfolio_rows" not in day
        for day in persisted["bankroll"]["daily"]
    )
