from __future__ import annotations

import copy

import pytest

from boatrace_ai.listwise import fixed_policy_replay as replay


POLICY = {
    "name": "fixed",
    "ev_threshold": 1.5,
    "max_estimated_ev": None,
    "max_odds": 40.0,
    "max_tickets_per_race": 1,
    "min_model_market_ratio": 1.2,
    "staking_mode": "kelly_100",
    "v18_ticket_control": {
        "learned_daily_ticket_limit": 13,
        "schedule_quota_rounding": "ceil",
        "schedule_quota_opportunity": None,
        "result_or_payout_fields_used": False,
    },
}


def _evaluation() -> dict:
    return {
        "model": "v21",
        "calibrator_strategy": "v21",
        "folds": [
            {
                "evaluation_date": "2026-07-31",
                "evaluation_races": 1,
                "calibration_dates": ["2026-07-30"],
                "purchase_calibrator": {"model_weight": 0.1},
                "operational_model": {"training_races": 10},
            },
            {
                "evaluation_date": "2026-08-01",
                "evaluation_races": 1,
                "calibration_dates": ["2026-07-31"],
                "purchase_calibrator": {"model_weight": 0.2},
                "operational_model": {"training_races": 11},
            },
        ],
    }


def test_replay_uses_each_fold_calibrator_and_fixed_policy(monkeypatch) -> None:
    cache = {
        "races": [
            {"race_id": "a", "race_date": "2026-07-31"},
            {"race_id": "b", "race_date": "2026-08-01"},
        ]
    }
    observed = []

    monkeypatch.setattr(
        replay,
        "_reconstruct_policy_races",
        lambda _all, holdout, _fold: holdout,
    )

    def simulate(races, *, calibrator, policy, **_kwargs):
        observed.append((races[0]["race_date"], calibrator, copy.deepcopy(policy)))
        returned = 200 if races[0]["race_date"] == "2026-07-31" else 0
        daily = {
            "race_date": races[0]["race_date"],
            "evaluated_races": 1,
            "tickets": 1,
            "races_bet": 1,
            "hit_races": int(returned > 0),
            "hit_tickets": int(returned > 0),
            "stake_yen": 100,
            "return_yen": returned,
            "profit_yen": returned - 100,
            "max_drawdown_yen": int(returned == 0) * 100,
            "largest_hit_return_yen": returned,
            "hit_return_square_sum_yen2": returned * returned,
        }
        return {"chronological_bankroll": {"daily": [daily]}}

    monkeypatch.setattr(replay, "simulate_policy", simulate)
    result = replay.replay_fixed_policy(_evaluation(), cache, POLICY)

    assert [row[1]["model_weight"] for row in observed] == [0.1, 0.2]
    assert all(row[2] == POLICY for row in observed)
    bankroll = result["chronological_bankroll"]
    assert bankroll["race_days"] == 2
    assert bankroll["evaluated_races"] == 2
    assert bankroll["stake_yen"] == 200
    assert bankroll["return_yen"] == 200
    assert bankroll["roi"] == 1.0
    assert bankroll["roi_without_largest_hit"] == 0.0
    assert result["information_boundary"]["outer_holdout_used_to_fit_or_select_policy"] is False


def test_replay_rejects_holdout_count_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        replay,
        "_reconstruct_policy_races",
        lambda _all, holdout, _fold: holdout,
    )
    with pytest.raises(ValueError, match="holdout race count mismatch"):
        replay.replay_fixed_policy(_evaluation(), {"races": []}, POLICY)


def test_replay_rejects_policy_using_results() -> None:
    policy = copy.deepcopy(POLICY)
    policy["v18_ticket_control"]["result_or_payout_fields_used"] = True
    with pytest.raises(ValueError, match="exclude result and payout"):
        replay.replay_fixed_policy(_evaluation(), {"races": []}, policy)
