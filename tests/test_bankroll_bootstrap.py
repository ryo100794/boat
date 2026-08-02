from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from boatrace_ai.bankroll_bootstrap import (
    MAX_EXACT_YEN,
    bootstrap_daily_roi,
    leave_one_venue_out_roi,
    moving_block_bootstrap_roi,
)


def _row(day: str, stake: float, returned: float) -> dict[str, object]:
    return {"race_date": day, "stake_yen": stake, "return_yen": returned}


def _venue_row(
    day: str, jcd: str, stake: float, returned: float
) -> dict[str, object]:
    return {
        "race_date": day,
        "jcd": jcd,
        "stake_yen": stake,
        "return_yen": returned,
    }


class _ExhaustiveRng:
    def __init__(self, combinations: list[tuple[int, ...]]) -> None:
        self._combinations = combinations
        self._offset = 0

    def integers(self, low: int, high: int, *, size: tuple[int, int]) -> np.ndarray:
        assert low == 0
        assert high == size[1]
        stop = self._offset + size[0]
        result = np.asarray(self._combinations[self._offset : stop], dtype=np.int64)
        self._offset = stop
        return result


def test_two_day_bootstrap_matches_all_possible_cluster_draws(monkeypatch) -> None:
    combinations = list(product(range(2), repeat=2))
    monkeypatch.setattr(
        "boatrace_ai.bankroll_bootstrap.np.random.default_rng",
        lambda seed: _ExhaustiveRng(combinations),
    )
    result = bootstrap_daily_roi(
        [_row("2026-07-01", 100, 0), _row("2026-07-02", 100, 200)],
        samples=4,
        seed=17,
        chunk_size=2,
    )
    exhaustive_roi = np.asarray([0.0, 1.0, 1.0, 2.0])
    assert result["roi"] == 1.0
    assert result["profit_yen"] == 0.0
    assert result["samples"] == 4
    assert result["valid_samples"] == 4
    assert result["roi_ci95_lower"] == pytest.approx(
        np.quantile(exhaustive_roi, 0.05)
    )
    assert result["probability_roi_above_one"] == 0.25


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    rows = [
        _row("2026-07-01", 100, 0),
        _row("2026-07-02", 200, 500),
        _row("2026-07-03", 0, 0),
    ]
    first = bootstrap_daily_roi(rows, samples=2_003, seed=91, chunk_size=37)
    second = bootstrap_daily_roi(rows, samples=2_003, seed=91, chunk_size=37)
    assert first == second


def test_result_is_invariant_to_input_and_day_order() -> None:
    rows = [
        _row("2026-07-03", 0, 0),
        _row("2026-07-01", 100, 130),
        _row("2026-07-02", 200, 0),
    ]
    forward = bootstrap_daily_roi(rows, samples=1_001, seed=7, chunk_size=23)
    reverse = bootstrap_daily_roi(
        list(reversed(rows)), samples=1_001, seed=7, chunk_size=23
    )
    assert forward == reverse


def test_zero_bet_days_are_sampled_but_undefined_draws_are_excluded(
    monkeypatch,
) -> None:
    combinations = list(product(range(2), repeat=2))
    monkeypatch.setattr(
        "boatrace_ai.bankroll_bootstrap.np.random.default_rng",
        lambda seed: _ExhaustiveRng(combinations),
    )
    result = bootstrap_daily_roi(
        [_row("2026-07-01", 0, 0), _row("2026-07-02", 100, 150)],
        samples=4,
        chunk_size=3,
    )
    assert result["days"] == 2
    assert result["valid_samples"] == 3
    assert result["roi_ci95_lower"] == 1.5
    assert result["probability_roi_above_one"] == 1.0


def test_all_zero_stake_draws_report_undefined_roi_statistics() -> None:
    result = bootstrap_daily_roi(
        [_row("2026-07-01", 0, 0), _row("2026-07-02", 0, 0)],
        samples=101,
    )
    assert result == {
        "days": 2,
        "samples": 101,
        "valid_samples": 0,
        "stake_yen": 0.0,
        "return_yen": 0.0,
        "profit_yen": 0.0,
        "roi": None,
        "roi_ci95_lower": None,
        "probability_roi_above_one": None,
    }


def test_multiple_rows_per_day_are_aggregated_before_resampling() -> None:
    split = [
        _row("2026-07-02", 100, 0),
        _row("2026-07-01", 100, 50),
        _row("2026-07-01", 200, 400),
        _row("2026-07-02", 0, 0),
    ]
    aggregated = [
        _row("2026-07-01", 300, 450),
        _row("2026-07-02", 100, 0),
    ]
    split_result = bootstrap_daily_roi(split, samples=1_007, seed=23)
    reordered_result = bootstrap_daily_roi(
        [split[2], split[0], split[3], split[1]],
        samples=1_007,
        seed=23,
    )
    aggregated_result = bootstrap_daily_roi(aggregated, samples=1_007, seed=23)
    assert split_result == reordered_result
    assert split_result == aggregated_result
    assert split_result["days"] == 2
    assert split_result["stake_yen"] == 400.0
    assert split_result["return_yen"] == 450.0
    assert split_result["profit_yen"] == 50.0
    assert split_result["roi"] == 1.125


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("stake_yen", -1),
        ("return_yen", -1),
        ("stake_yen", float("nan")),
        ("return_yen", float("inf")),
        ("stake_yen", 1e308),
        ("return_yen", MAX_EXACT_YEN + 1),
    ],
)
def test_invalid_or_extreme_money_values_are_rejected(key, value) -> None:
    row = _row("2026-07-01", 100, 100)
    row[key] = value
    with pytest.raises(ValueError):
        bootstrap_daily_roi([row], samples=10)


def test_extreme_daily_and_observed_aggregates_are_rejected() -> None:
    with pytest.raises(ValueError, match="daily aggregate"):
        bootstrap_daily_roi(
            [
                _row("2026-07-01", MAX_EXACT_YEN, 0),
                _row("2026-07-01", 1, 0),
            ],
            samples=10,
        )
    with pytest.raises(ValueError, match="observed aggregate"):
        bootstrap_daily_roi(
            [
                _row("2026-07-01", MAX_EXACT_YEN, 0),
                _row("2026-07-02", 1, 0),
            ],
            samples=10,
        )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"race_date": "2026-07-01", "stake_yen": 100}],
        [{"race_date": "", "stake_yen": 100, "return_yen": 100}],
        [None],
    ],
)
def test_malformed_rows_are_rejected(rows) -> None:
    with pytest.raises(ValueError):
        bootstrap_daily_roi(rows, samples=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"samples": 1.5},
        {"chunk_size": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_bootstrap_controls_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        bootstrap_daily_roi([_row("2026-07-01", 100, 100)], **kwargs)


def test_default_uses_twenty_thousand_memory_bounded_samples(monkeypatch) -> None:
    observed_shapes: list[tuple[int, int]] = []
    original = np.random.default_rng

    class _RecordingRng:
        def __init__(self) -> None:
            self._rng = original(3)

        def integers(self, low, high, *, size):
            observed_shapes.append(size)
            return self._rng.integers(low, high, size=size)

    monkeypatch.setattr(
        "boatrace_ai.bankroll_bootstrap.np.random.default_rng",
        lambda seed: _RecordingRng(),
    )
    result = bootstrap_daily_roi([_row("2026-07-01", 100, 120)])
    assert result["samples"] == 20_000
    assert sum(shape[0] for shape in observed_shapes) == 20_000
    assert max(shape[0] for shape in observed_shapes) == 2_000


def test_moving_block_bootstrap_preserves_complete_consecutive_days(
    monkeypatch,
) -> None:
    class _StartsRng:
        def integers(self, low, high, *, size):
            assert (low, high, size) == (0, 3, (2, 2))
            return np.asarray([[0, 2], [1, 0]], dtype=np.int64)

    monkeypatch.setattr(
        "boatrace_ai.bankroll_bootstrap.np.random.default_rng",
        lambda seed: _StartsRng(),
    )
    rows = [
        _row("2026-07-01", 40, 0),
        _row("2026-07-01", 60, 100),
        _row("2026-07-02", 100, 0),
        _row("2026-07-03", 100, 300),
        _row("2026-07-04", 100, 0),
    ]
    result = moving_block_bootstrap_roi(
        rows, block_days=2, samples=2, seed=19, chunk_size=2
    )
    assert result["days"] == 4
    assert result["blocks_per_sample"] == 2
    assert result["valid_samples"] == 2
    assert result["roi_ci95_lower"] == pytest.approx(1.0)
    assert result["probability_roi_above_one"] == 0.0


def test_moving_block_bootstrap_is_seeded_and_chunk_bounded(monkeypatch) -> None:
    observed_shapes: list[tuple[int, int]] = []
    original = np.random.default_rng

    class _RecordingRng:
        def __init__(self, seed: int) -> None:
            self._rng = original(seed)

        def integers(self, low, high, *, size):
            observed_shapes.append(size)
            return self._rng.integers(low, high, size=size)

    monkeypatch.setattr(
        "boatrace_ai.bankroll_bootstrap.np.random.default_rng",
        lambda seed: _RecordingRng(seed),
    )
    rows = [_row(f"2026-07-{day:02d}", 100, day * 10) for day in range(1, 8)]
    first = moving_block_bootstrap_roi(
        rows, block_days=3, samples=101, seed=31, chunk_size=17
    )
    shapes_after_first = list(observed_shapes)
    observed_shapes.clear()
    second = moving_block_bootstrap_roi(
        rows, block_days=3, samples=101, seed=31, chunk_size=17
    )
    assert first == second
    assert shapes_after_first == observed_shapes
    assert max(shape[0] for shape in observed_shapes) == 17
    assert all(shape[1] == 3 for shape in observed_shapes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_days": 0},
        {"block_days": 3},
        {"block_days": 1, "samples": 0},
        {"block_days": 1, "chunk_size": 0},
        {"block_days": 1, "seed": -1},
    ],
)
def test_invalid_moving_block_controls_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        moving_block_bootstrap_roi(
            [_row("2026-07-01", 100, 100), _row("2026-07-02", 100, 100)],
            **kwargs,
        )


def test_leave_one_venue_out_reports_whole_venue_sensitivity() -> None:
    rows = [
        _venue_row("2026-07-01", "01", 100, 300),
        _venue_row("2026-07-02", "01", 100, 0),
        _venue_row("2026-07-01", "02", 200, 100),
        _venue_row("2026-07-02", "03", 0, 0),
    ]
    result = leave_one_venue_out_roi(rows)
    assert result["venues"] == 3
    assert result["roi"] == 1.0
    assert result["profit_yen"] == 0.0
    by_venue = {row["jcd"]: row for row in result["leave_one_venue_out"]}
    assert by_venue["01"] == {
        "jcd": "01",
        "omitted_stake_yen": 200.0,
        "omitted_return_yen": 300.0,
        "omitted_profit_yen": 100.0,
        "remaining_stake_yen": 200.0,
        "remaining_return_yen": 100.0,
        "remaining_profit_yen": -100.0,
        "remaining_roi": 0.5,
    }
    assert by_venue["02"]["remaining_roi"] == 1.5
    assert by_venue["03"]["remaining_roi"] == 1.0


def test_leave_one_venue_out_is_order_invariant_and_handles_one_venue() -> None:
    rows = [
        _venue_row("2026-07-02", "01", 100, 150),
        _venue_row("2026-07-01", "01", 0, 0),
    ]
    assert leave_one_venue_out_roi(rows) == leave_one_venue_out_roi(
        list(reversed(rows))
    )
    diagnostic = leave_one_venue_out_roi(rows)["leave_one_venue_out"][0]
    assert diagnostic["remaining_stake_yen"] == 0.0
    assert diagnostic["remaining_roi"] is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [_row("2026-07-01", 100, 100)],
        [_venue_row("2026-07-01", "", 100, 100)],
        [
            {
                "race_date": "2026-07-01",
                "jcd": "01",
                "stake_yen": -1,
                "return_yen": 0,
            }
        ],
        [None],
    ],
)
def test_malformed_venue_rows_are_rejected(rows) -> None:
    with pytest.raises(ValueError):
        leave_one_venue_out_roi(rows)
