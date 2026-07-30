from __future__ import annotations

import sqlite3

import pytest

import boatrace_ai.discrete_log_allocation as allocation
import boatrace_ai.bankroll_backtest as backtest
import boatrace_ai.listwise.odds_path_conservative_v7 as v7
import boatrace_ai.listwise.odds_path_discrete_v9 as v9
import boatrace_ai.listwise.odds_path_selection_conformal_v10 as v10
from boatrace_ai.bankroll_backtest import (
    _allocate_daily_budget,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from boatrace_ai.discrete_log_allocation import (
    allocate_discrete_log_day,
    candidate_with_settlements,
)


FORBIDDEN = {"actual_combination", "actual_payout_yen", "hit"}


def _decision(race_id: str, combination: str) -> dict:
    return {
        "race_id": race_id,
        "race_date": "2026-07-29",
        "combination": combination,
        "probability": 0.20,
        "estimated_odds": 10.0,
        "estimated_ev": 2.0,
    }


def _allocate(candidates: list[dict]) -> dict:
    return allocate_discrete_log_day(
        "2026-07-29",
        candidates,
        {str(candidate["race_id"]) for candidate in candidates},
        daily_budget_yen=10_000,
        max_daily_exposure_fraction=0.02,
        race_cap_fraction=0.02,
        ticket_cap_fraction=0.01,
        max_daily_tickets=None,
        stake_granularity_yen=100,
        min_stake_yen=100,
        max_tickets_per_race=2,
    )


def _lane_rows(race_id: str = "2026-07-29-01-01") -> list[dict]:
    return [
        {
            "race_id": race_id,
            "race_date": "2026-07-29",
            "jcd": "01",
            "rno": 1,
            "lane": lane,
            "probability": 0.70 if lane == 1 else 0.06,
        }
        for lane in range(1, 7)
    ]


def _payout_model() -> dict[str, dict[str, float]]:
    return {
        f"{first}-{second}-{third}": {
            "estimated_odds": 10.0,
            "estimated_payout_yen": 1_000.0,
            "history_count": 100.0,
        }
        for first in range(1, 7)
        for second in range(1, 7)
        for third in range(1, 7)
        if len({first, second, third}) == 3
    }


def test_loader_retains_every_dead_heat_payout_deterministically() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE races (
            race_id TEXT PRIMARY KEY,
            race_date TEXT NOT NULL,
            jcd TEXT NOT NULL,
            rno INTEGER NOT NULL
        );
        CREATE TABLE payouts (
            race_id TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            combination TEXT NOT NULL,
            payout_yen INTEGER,
            popularity INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO races VALUES (?, ?, ?, ?)",
        ("2026-07-29-01-01", "2026-07-29", "01", 1),
    )
    conn.executemany(
        "INSERT INTO payouts VALUES (?, ?, ?, ?, ?)",
        [
            ("2026-07-29-01-01", "3連単", "2-1-3", 2_500, 4),
            ("2026-07-29-01-01", "3連単", "1-2-3", 1_000, 1),
        ],
    )

    loaded = _load_trifecta_payouts(conn)
    payout = loaded["2026-07-29-01-01"]

    assert payout["combination"] == "1-2-3"
    assert payout["payout_yen"] == 1_000
    assert [
        (row["combination"], row["payout_yen"])
        for row in payout["settlements"]
    ] == [("1-2-3", 1_000), ("2-1-3", 2_500)]


def test_optimizer_receives_no_result_fields(monkeypatch) -> None:
    seen_sources = []
    original = allocation._enumerate_race_options

    def inspect_inputs(race_id, candidates, **kwargs):
        for candidate in candidates:
            seen_sources.append(dict(candidate.source))
            assert FORBIDDEN.isdisjoint(candidate.source)
        return original(race_id, candidates, **kwargs)

    monkeypatch.setattr(allocation, "_enumerate_race_options", inspect_inputs)
    legacy = {
        **_decision("2026-07-29-01-01", "1-2-3"),
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1_000,
        "hit": True,
    }

    result = _allocate([legacy])

    assert seen_sources
    assert result["tickets"] == 1
    assert result["hit_tickets"] == 1
    assert result["return_yen"] == 1_000


def test_both_dead_heat_combinations_settle_at_official_payouts() -> None:
    race_id = "2026-07-29-01-01"
    settlements = (
        {
            "race_id": race_id,
            "combination": "1-2-3",
            "payout_yen": 1_000,
        },
        {
            "race_id": race_id,
            "combination": "2-1-3",
            "payout_yen": 2_500,
        },
    )
    candidates = [
        candidate_with_settlements(
            _decision(race_id, combination),
            settlements,
        )
        for combination in ("1-2-3", "2-1-3")
    ]

    result = _allocate(candidates)
    sample = {
        row["combination"]: row for row in result["selected_sample"]
    }

    assert result["tickets"] == 2
    assert result["hit_tickets"] == 2
    assert result["hit_races"] == 1
    assert result["stake_yen"] == 200
    assert result["return_yen"] == 3_500
    assert sample["1-2-3"]["return_yen"] == 1_000
    assert sample["2-1-3"]["return_yen"] == 2_500


def test_single_payout_legacy_and_additive_roi_are_identical() -> None:
    race_id = "2026-07-29-01-01"
    decision = _decision(race_id, "1-2-3")
    legacy = {
        **decision,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1_200,
        "hit": True,
    }
    additive = candidate_with_settlements(
        decision,
        [{
            "race_id": race_id,
            "combination": "1-2-3",
            "payout_yen": 1_200,
        }],
    )

    legacy_result = _allocate([legacy])
    additive_result = _allocate([additive])

    for key in (
        "tickets",
        "hit_tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "expected_log_growth",
    ):
        assert additive_result[key] == pytest.approx(legacy_result[key])


def test_bankroll_candidates_are_decision_only_before_allocation() -> None:
    candidates = _candidate_tickets(
        _lane_rows(),
        payout_model=_payout_model(),
        ev_threshold=0.0,
        decision_only=True,
    )

    assert candidates
    assert all(type(candidate) is dict for candidate in candidates)
    assert all(FORBIDDEN.isdisjoint(candidate) for candidate in candidates)
    assert all(not hasattr(candidate, "settlement_rows") for candidate in candidates)


def test_separate_settlement_preserves_legacy_result_and_cannot_change_decision() -> None:
    race_id = "2026-07-29-01-01"
    decision_candidates = _candidate_tickets(
        _lane_rows(race_id),
        payout_model=_payout_model(),
        ev_threshold=0.0,
        decision_only=True,
    )[:5]
    legacy_candidates = _candidate_tickets(
        _lane_rows(race_id),
        actual={"combination": "1-2-3", "payout_yen": 1_200},
        payout_model=_payout_model(),
        ev_threshold=0.0,
    )[:5]
    common = {
        "evaluated_races": {race_id},
        "daily_budget_yen": 1_000,
        "unit_yen": 100,
    }

    legacy = _allocate_daily_budget(legacy_candidates, **common)
    separate = _allocate_daily_budget(
        decision_candidates,
        settlements={(race_id, "1-2-3"): 1_200},
        **common,
    )
    changed_result = _allocate_daily_budget(
        decision_candidates,
        settlements={(race_id, "6-5-4"): 99_900},
        **common,
    )

    for key in ("tickets", "stake_yen", "return_yen", "profit_yen", "roi"):
        assert separate[key] == pytest.approx(legacy[key])
    assert changed_result["tickets"] == separate["tickets"]
    assert changed_result["stake_yen"] == separate["stake_yen"]
    assert changed_result["avg_selected_odds"] == separate["avg_selected_odds"]
    assert changed_result["avg_selected_probability"] == separate["avg_selected_probability"]
    assert changed_result["return_yen"] != separate["return_yen"]


def test_backtest_fold_passes_results_only_to_post_decision_settlement(
    tmp_path,
    monkeypatch,
) -> None:
    features = []
    labels = []
    meta = []
    payouts = {}
    for day in range(1, 21):
        race_id = f"2026-07-{day:02d}-01-01"
        payouts[race_id] = {
            "race_id": race_id,
            "race_date": f"2026-07-{day:02d}",
            "jcd": "01",
            "rno": 1,
            "combination": "1-2-3",
            "payout_yen": 1_200,
        }
        for lane in range(1, 7):
            features.append({"lane": lane})
            labels.append(lane == 1)
            meta.append(
                {
                    "race_id": race_id,
                    "race_date": f"2026-07-{day:02d}",
                    "jcd": "01",
                    "rno": 1,
                    "lane": lane,
                    "rank": lane,
                }
            )

    class Pipeline:
        def fit(self, X, y):
            return self

    original_candidate_tickets = backtest._candidate_tickets
    original_allocate = backtest._allocate_daily_budget
    candidate_calls = []
    allocation_calls = []

    def inspect_candidate_tickets(rows, **kwargs):
        assert "actual" not in kwargs
        result = original_candidate_tickets(rows, **kwargs)
        assert all(FORBIDDEN.isdisjoint(candidate) for candidate in result)
        candidate_calls.append(result)
        return result

    def inspect_allocate(candidates, **kwargs):
        assert all(FORBIDDEN.isdisjoint(candidate) for candidate in candidates)
        assert all(not hasattr(candidate, "settlement_rows") for candidate in candidates)
        assert kwargs["settlements"] == backtest._payout_settlement_map(payouts)
        allocation_calls.append(list(candidates))
        return original_allocate(candidates, **kwargs)

    monkeypatch.setattr(
        backtest,
        "load_training_examples",
        lambda *args, **kwargs: (features, labels, meta),
    )
    monkeypatch.setattr(backtest, "_load_trifecta_payouts", lambda conn: payouts)
    monkeypatch.setattr(backtest, "_make_pipeline", Pipeline)
    monkeypatch.setattr(
        backtest,
        "_positive_probs",
        lambda pipeline, rows: [0.70, 0.12, 0.08, 0.05, 0.03, 0.02],
    )
    monkeypatch.setattr(backtest, "_candidate_tickets", inspect_candidate_tickets)
    monkeypatch.setattr(backtest, "_allocate_daily_budget", inspect_allocate)

    result = backtest.bankroll_backtest(
        object(),
        output_path=tmp_path / "bankroll.json",
        folds=1,
        min_train_races=19,
        ev_threshold=0.0,
        max_tickets_per_race=5,
    )

    assert len(candidate_calls) == 1
    assert len(allocation_calls) == 1
    assert result["tickets"] == 5
    assert result["hit_tickets"] == 1
    assert result["return_yen"] > 0


def test_v9_and_v10_use_the_shared_decision_only_candidate_factory() -> None:
    assert v9._policy_candidate is v7._policy_candidate
    assert v10._policy_candidate is v7._policy_candidate


def test_v7_shared_candidate_boundary_supports_multiple_settlements() -> None:
    race_id = "2026-07-29-01-01"
    race = {
        "race_id": race_id,
        "race_date": "2026-07-29",
        "jcd": "01",
        "rno": 1,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1_000,
        "odds": {"1-2-3": 10.0, "2-1-3": 10.0},
        "settlements": [
            {"combination": "1-2-3", "payout_yen": 1_000},
            {"combination": "2-1-3", "payout_yen": 2_500},
        ],
    }
    candidates = [
        v7._policy_candidate(
            race,
            combination=combination,
            probability=0.20,
            estimated_odds=10.0,
            safe_ev=2.0,
        )
        for combination in ("1-2-3", "2-1-3")
    ]

    assert all(FORBIDDEN.isdisjoint(candidate) for candidate in candidates)
    result = v7._allocate_adaptive_day_with_settlement_boundary(
        "2026-07-29",
        candidates,
        {race_id},
        daily_budget_yen=10_000,
    )

    assert result["tickets"] == 2
    assert result["hit_tickets"] == 2
    assert result["return_yen"] == 3_500
