from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from boatrace_ai.listwise import market_offset_selection as selection


ACTUAL = "1-2-3"
OTHER = "1-3-2"


def _race(day: str, race_no: int = 1) -> dict[str, object]:
    return {
        "race_id": f"{day}-{race_no:02d}",
        "race_date": day,
        "model_probabilities": {ACTUAL: 0.7, OTHER: 0.3},
        "market_probabilities": {ACTUAL: 0.55, OTHER: 0.45},
        "forecast_odds": {ACTUAL: 2.0, OTHER: 3.0},
        "actual_combination": ACTUAL,
        "profit_yen": 10**9,
        "payout_yen": 10**9,
    }


def _records() -> list[dict[str, object]]:
    return [
        _race(day, race_no)
        for day in ("2026-07-20", "2026-07-21", "2026-07-22")
        for race_no in (1, 2)
    ] + [_race("2026-07-23", 1), _race("2026-07-23", 2)]


def _fake_fit_factory(
    probabilities: dict[float, float],
    *,
    nonconverged: set[float] | None = None,
    calls: list[dict[str, object]] | None = None,
):
    nonconverged = nonconverged or set()

    def fake_fit(records, *, prediction_date, regularization, min_training_races):
        rows = list(records)
        if calls is not None:
            calls.append(
                {
                    "dates": tuple(row["race_date"] for row in rows),
                    "prediction_date": prediction_date,
                    "regularization": regularization,
                    "min_training_races": min_training_races,
                }
            )
        dates = tuple(sorted({str(row["race_date"]) for row in rows}))
        converged = regularization not in nonconverged

        class Artifact:
            fitted = True
            fallback_reason = None
            training_dates = dates
            training_races = len(rows)

            def __init__(self):
                self.converged = converged

            def predict(self, model, market, odds, *, prediction_date):
                del model, market, odds, prediction_date
                probability = probabilities[regularization]
                return SimpleNamespace(
                    probabilities={ACTUAL: probability, OTHER: 1.0 - probability}
                )

        return Artifact()

    return fake_fit


def _select(records, **kwargs):
    return selection.select_market_offset_regularization(
        records,
        prediction_date="2026-07-24",
        candidates=kwargs.pop("candidates", (0.1, 1.0, 10.0)),
        min_inner_training_days=kwargs.pop("min_inner_training_days", 3),
        min_inner_training_races=kwargs.pop("min_inner_training_races", 6),
        **kwargs,
    )


def test_selects_lowest_validation_log_loss_and_stronger_tie(monkeypatch) -> None:
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory({0.1: 0.6, 1.0: 0.8, 10.0: 0.8}),
    )

    result = _select(_records())

    assert result["selected_regularization"] == 10.0
    assert result["fallback_reason"] is None
    losses = {
        row["regularization"]: row["validation_log_loss"]
        for row in result["candidates"]
    }
    assert losses[1.0] == pytest.approx(-math.log(0.8))
    assert losses[10.0] == losses[1.0]


def test_inner_split_is_strictly_prior_and_complete_day(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory({0.1: 0.6, 1.0: 0.7, 10.0: 0.65}, calls=calls),
    )
    records = _records() + [
        {"race_date": "2026-07-24"},
        {"race_date": "2099-01-01"},
    ]

    result = _select(records)

    assert result["validation_date"] == "2026-07-23"
    assert result["training_through"] == "2026-07-22"
    assert all(
        set(call["dates"])
        == {"2026-07-20", "2026-07-21", "2026-07-22"}
        and call["prediction_date"] == "2026-07-23"
        for call in calls
    )
    assert result["audit"]["excluded_non_past_records"] == 2
    assert result["audit"]["excluded_non_past_dates"] == [
        "2026-07-24",
        "2099-01-01",
    ]


def test_future_and_same_day_value_changes_cannot_change_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory({0.1: 0.61, 1.0: 0.72, 10.0: 0.69}),
    )
    baseline = _select(
        _records() + [_race("2026-07-24"), _race("2026-07-25")]
    )
    malformed_non_past = [
        {"race_date": "2026-07-24", "actual_combination": object()},
        {"race_date": "2026-07-25", "model_probabilities": object()},
    ]
    mutated = _select(_records() + malformed_non_past)

    assert mutated == baseline


def test_profit_and_payout_columns_are_not_used(monkeypatch) -> None:
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory({0.1: 0.62, 1.0: 0.71, 10.0: 0.68}),
    )
    records = _records()
    mutated = copy.deepcopy(records)
    for index, row in enumerate(mutated):
        row["profit_yen"] = float("nan") if index % 2 else -(10**30)
        row["payout_yen"] = float("inf") if index % 2 else object()
        row["return_multiplier"] = object()

    assert _select(mutated) == _select(records)


def test_input_and_candidate_order_do_not_change_result(monkeypatch) -> None:
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory({0.1: 0.64, 1.0: 0.73, 10.0: 0.68}),
    )
    forward = _select(_records())
    reverse = _select(
        list(reversed(_records())), candidates=(10.0, 1.0, 0.1, 1.0)
    )

    assert reverse == forward


@pytest.mark.parametrize(
    ("records", "min_days", "min_races", "reason"),
    [
        ([], 3, 6, "no_strictly_prior_validation_day"),
        (
            [_race("2026-07-22"), _race("2026-07-23")],
            3,
            1,
            "insufficient_inner_training_days",
        ),
        (_records(), 3, 7, "insufficient_inner_training_races"),
    ],
)
def test_insufficient_data_explicitly_falls_back_to_default(
    monkeypatch, records, min_days, min_races, reason
) -> None:
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("fit must not run when inner data are insufficient")

    monkeypatch.setattr(
        selection, "fit_market_offset_calibration", unexpected_fit
    )

    result = _select(
        records,
        min_inner_training_days=min_days,
        min_inner_training_races=min_races,
    )

    assert result["selected_regularization"] == 1.0
    assert result["fallback_reason"] == reason
    assert all(
        row["validation_log_loss"] is None and not row["eligible"]
        for row in result["candidates"]
    )


def test_all_nonconverged_candidates_fall_back_to_default(monkeypatch) -> None:
    monkeypatch.setattr(
        selection,
        "fit_market_offset_calibration",
        _fake_fit_factory(
            {0.1: 0.6, 1.0: 0.7, 10.0: 0.8},
            nonconverged={0.1, 1.0, 10.0},
        ),
    )

    result = _select(_records())

    assert result["selected_regularization"] == 1.0
    assert result["fallback_reason"] == "no_converged_candidates"
    assert all(
        row["validation_log_loss"] is None
        and row["fitted"]
        and not row["converged"]
        and row["fallback_reason"] == "candidate_did_not_converge"
        for row in result["candidates"]
    )
