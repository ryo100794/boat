from itertools import permutations

from scripts import analyze_market_momentum as probe


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _uniform() -> dict[str, float]:
    return {combination: 1.0 / len(COMBINATIONS) for combination in COMBINATIONS}


def _race(race_date: str, index: int) -> dict:
    actual = COMBINATIONS[index % len(COMBINATIONS)]
    earlier = {combination: 0.8 / 119 for combination in COMBINATIONS}
    earlier[actual] = 0.2
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": _uniform(),
        "market_probabilities": _uniform(),
        "earlier_market_probabilities": earlier,
    }


def test_probe_keeps_evaluation_day_out_of_selection() -> None:
    races = [
        _race(race_date, index)
        for race_date in ("2026-07-20", "2026-07-21", "2026-07-22")
        for index in range(8)
    ]

    result = probe.evaluate_momentum_candidate(
        races,
        evaluation_date="2026-07-22",
    )

    assert result["calibration_dates"] == ["2026-07-20", "2026-07-21"]
    assert result["evaluation_date"] == "2026-07-22"
    assert result["evaluation_races"] == 8
    assert result["momentum_vs_baseline"]["log_loss_difference"][
        "mean_difference"
    ] < 0


def test_earlier_snapshot_must_be_fresh_and_complete(monkeypatch) -> None:
    odds = {combination: 10.0 + index for index, combination in enumerate(COMBINATIONS)}
    monkeypatch.setattr(
        probe,
        "latest_trifecta_odds_before_deadline",
        lambda *_args, **_kwargs: {
            "snapshot_id": 7,
            "captured_at": "2026-07-22T10:00:30+09:00",
            "odds_deadline_at": "2026-07-22T10:01:00+09:00",
            "odds": odds,
        },
    )
    race = {
        "race_id": "race-1",
        "race_date": "2026-07-22",
        "captured_at": "2026-07-22T10:05:30+09:00",
        "market_probabilities": _uniform(),
    }

    augmented, dataset = probe.attach_earlier_market_probabilities(
        object(),
        [race],
        earlier_decision_lead_minutes=10,
        max_snapshot_age_seconds=60.0,
    )

    assert dataset["eligible_momentum_races"] == 1
    assert dataset["eligible_by_day"] == {"2026-07-22": 1}
    assert augmented[0]["earlier_snapshot_age_seconds"] == 30.0
    assert augmented[0]["momentum_interval_seconds"] == 300.0
    assert augmented[0]["momentum_scale"] == 1.0
    assert len(augmented[0]["earlier_market_probabilities"]) == 120
