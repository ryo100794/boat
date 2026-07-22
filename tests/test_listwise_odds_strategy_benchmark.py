from __future__ import annotations

from itertools import permutations

import pytest

from boatrace_ai.db import connection, init_db
from boatrace_ai.listwise.odds_strategy_benchmark import evaluate_odds_strategies


TOP5 = ("1-2-3", "1-2-4", "1-3-2", "2-1-3", "2-3-1")
LANE_PROBABILITIES = (0.50, 0.20, 0.10, 0.08, 0.07, 0.05)


def _all_odds(value: float = 10.0) -> dict[str, float]:
    return {"-".join(map(str, combo)): value for combo in permutations(range(1, 7), 3)}


def _prediction(race_id: str, *, actual_order: tuple[int, ...]) -> dict:
    return {
        "race_id": race_id,
        "race_date": race_id[:10],
        "trifecta_top5": list(TOP5),
        "lane_probabilities": [
            {"lane": lane, "probability": probability}
            for lane, probability in enumerate(LANE_PROBABILITIES, start=1)
        ],
        "actual_order": list(actual_order),
    }


def _insert_race(
    conn,
    race_id: str,
    *,
    winning_combination: str,
    payout_yen: int,
    odds: dict[str, float] | None,
) -> None:
    race_date = race_id[:10]
    jcd, rno = race_id.rsplit("-", 2)[-2:]
    conn.execute(
        """
        INSERT INTO races (race_id, race_date, jcd, venue_name, rno, deadline_at, status)
        VALUES (?, ?, ?, '桐生', ?, ?, 'final')
        """,
        (race_id, race_date, jcd, int(rno), f"{race_date}T12:00:00+09:00"),
    )
    conn.execute(
        """
        INSERT INTO payouts (race_id, bet_type, combination, payout_yen, popularity)
        VALUES (?, '3連単', ?, ?, 1)
        """,
        (race_id, winning_combination, payout_yen),
    )
    if odds is None:
        return
    cursor = conn.execute(
        """
        INSERT INTO odds_snapshots (race_id, bet_type, captured_at, source_update_time, parser_version)
        VALUES (?, 'trifecta', ?, '11:54', 'odds3t_dom_v2')
        """,
        (race_id, f"{race_date}T11:54:00+09:00"),
    )
    snapshot_id = cursor.lastrowid
    conn.executemany(
        """
        INSERT INTO odds_trifecta (snapshot_id, race_id, combination, odds)
        VALUES (?, ?, ?, ?)
        """,
        [(snapshot_id, race_id, combination, value) for combination, value in odds.items()],
    )


def _evaluate(tmp_path, race_specs: list[tuple[dict, str, int, dict[str, float] | None]]):
    database = tmp_path / "benchmark.sqlite"
    init_db(database)
    with connection(database) as conn:
        for prediction, winning_combination, payout_yen, odds in race_specs:
            _insert_race(
                conn,
                prediction["race_id"],
                winning_combination=winning_combination,
                payout_yen=payout_yen,
                odds=odds,
            )
        return evaluate_odds_strategies(
            conn,
            prediction={"model": "test", "predictions": [row[0] for row in race_specs]},
        )


def test_flat_top5_profit_and_prediction_metrics(tmp_path) -> None:
    first = _prediction("2026-07-19-01-01", actual_order=(1, 2, 3, 4, 5, 6))
    second = _prediction("2026-07-19-01-02", actual_order=(6, 5, 4, 3, 2, 1))
    result = _evaluate(
        tmp_path,
        [
            (first, "1-2-3", 1_200, _all_odds()),
            (second, "6-5-4", 900, _all_odds()),
        ],
    )

    strategy = result["strategies"]["A_top5_flat"]
    assert result["daily_budget_applied"] is False
    assert strategy["evaluated_races"] == 2
    assert strategy["tickets"] == 10
    assert strategy["stake_yen"] == 1_000
    assert strategy["return_yen"] == 1_200
    assert strategy["profit_yen"] == 200
    assert strategy["roi"] == pytest.approx(1.2)
    assert strategy["hit_races"] == 1
    assert strategy["break_even_average_odds"] == pytest.approx(10.0)
    assert result["prediction_metrics"]["entry_log_loss"] is not None
    assert result["prediction_metrics"]["winner_top1_accuracy"] == pytest.approx(0.5)
    assert result["prediction_metrics"]["trifecta_top5_hit_rate"] == pytest.approx(0.5)


def test_five_times_odds_is_not_break_even_at_thirty_percent_top5_hit_rate(tmp_path) -> None:
    specs = []
    odds = _all_odds(10.0)
    odds.update({combination: 5.0 for combination in TOP5})
    for index in range(10):
        hit = index < 3
        race_id = f"2026-07-19-01-{index + 1:02d}"
        actual = (1, 2, 3, 4, 5, 6) if hit else (6, 5, 4, 3, 2, 1)
        winning = "1-2-3" if hit else "6-5-4"
        specs.append((_prediction(race_id, actual_order=actual), winning, 500, odds))

    result = _evaluate(tmp_path, specs)
    strategy = result["strategies"]["B_top5_odds_gte_5"]

    assert strategy["tickets"] == 50
    assert strategy["stake_yen"] == 5_000
    assert strategy["return_yen"] == 1_500
    assert strategy["roi"] == pytest.approx(0.3)
    assert strategy["race_hit_rate"] == pytest.approx(0.3)
    assert strategy["break_even_average_odds"] == pytest.approx(50 / 3)


def test_race_without_complete_real_odds_is_excluded_from_all_strategies(tmp_path) -> None:
    complete = _prediction("2026-07-19-01-01", actual_order=(1, 2, 3, 4, 5, 6))
    missing = _prediction("2026-07-19-01-02", actual_order=(1, 2, 3, 4, 5, 6))
    result = _evaluate(
        tmp_path,
        [
            (complete, "1-2-3", 1_000, _all_odds()),
            (missing, "1-2-3", 1_000, None),
        ],
    )

    assert result["evaluated_races"] == 1
    assert result["skipped_no_real_odds"] == 1
    for strategy in result["strategies"].values():
        assert strategy["evaluated_races"] == 1
        assert strategy["skipped_no_real_odds"] == 1


def test_ev_strategy_uses_pl_probability_times_pre_deadline_odds(tmp_path) -> None:
    odds = _all_odds(1.1)
    odds["1-2-3"] = 20.0
    prediction = _prediction("2026-07-19-01-01", actual_order=(1, 2, 3, 4, 5, 6))

    result = _evaluate(tmp_path, [(prediction, "1-2-3", 1_500, odds)])
    strategy = result["strategies"]["C_top5_ev_gte_1"]

    assert strategy["tickets"] == 1
    assert strategy["stake_yen"] == 100
    assert strategy["return_yen"] == 1_500
    assert strategy["profit_yen"] == 1_400
    assert strategy["roi"] == pytest.approx(15.0)
