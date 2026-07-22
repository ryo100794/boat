from boatrace_ai.runtime.collector import prediction_due


def test_history_prediction_does_not_wait_for_valid_odds() -> None:
    assert prediction_due(odds_collected=False, latest_prediction_at=None) is True


def test_failed_odds_does_not_duplicate_existing_history_prediction() -> None:
    assert (
        prediction_due(
            odds_collected=False,
            latest_prediction_at="2026-07-23T00:00:00+00:00",
        )
        is False
    )


def test_valid_odds_refreshes_existing_prediction() -> None:
    assert prediction_due(odds_collected=True, latest_prediction_at="existing") is True
