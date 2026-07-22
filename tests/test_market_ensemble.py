from __future__ import annotations

import pytest

from boatrace_ai.listwise.market_ensemble import (
    align_scored_races,
    probability_metrics,
    select_source_subset_prequential,
)


COMBINATIONS = ("1-2-3", "1-3-2", "2-1-3", "2-3-1")


def _probabilities(actual: str, strength: float) -> dict[str, float]:
    remainder = (1.0 - strength) / (len(COMBINATIONS) - 1)
    return {
        combination: strength if combination == actual else remainder
        for combination in COMBINATIONS
    }


def _scored(date: str, index: int, source: str) -> dict:
    actual = COMBINATIONS[index % len(COMBINATIONS)]
    return {
        "race_id": f"{date}-01-{index:02d}",
        "race_date": date,
        "actual_combination": actual,
        "model_probabilities": _probabilities(
            actual,
            0.70 if source == "signal" else 0.10,
        ),
        "market_probabilities": _probabilities(actual, 0.35),
    }


def test_forward_subset_selection_finds_signal_before_holdout() -> None:
    dates = ("2026-07-20", "2026-07-21", "2026-07-22")
    named = {
        source: [
            _scored(date, index, source)
            for date in dates
            for index in range(1, 13)
        ]
        for source in ("signal", "noise")
    }
    aligned = align_scored_races(named)
    calibration = [row for row in aligned if row["race_date"] < dates[-1]]
    evaluation = [row for row in aligned if row["race_date"] == dates[-1]]
    selection = select_source_subset_prequential(
        calibration,
        available_sources=("signal", "noise"),
    )["selected"]
    metrics = probability_metrics(
        evaluation,
        source_names=selection["source_names"],
        calibrator=selection["final_calibrator"],
    )

    assert selection["source_names"] == ["signal"]
    assert metrics["trifecta_log_loss"] < 1.0
    assert metrics["trifecta_top5_hit_rate"] == 1.0


def test_alignment_uses_common_races_and_rejects_result_mismatch() -> None:
    first = [_scored("2026-07-20", index, "signal") for index in (1, 2)]
    second = [_scored("2026-07-20", index, "noise") for index in (2, 3)]
    aligned = align_scored_races({"first": first, "second": second})
    assert [row["race_id"] for row in aligned] == ["2026-07-20-01-02"]

    second[0] = {**second[0], "actual_combination": "6-5-4"}
    with pytest.raises(ValueError, match="actual result differs"):
        align_scored_races({"first": first, "second": second})
