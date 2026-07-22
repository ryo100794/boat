from boatrace_ai.runtime.collector import beforeinfo_interval, odds_interval


def test_beforeinfo_polling_targets_model_decision_window() -> None:
    assert beforeinfo_interval(31 * 60, has_rows=False) is None
    assert beforeinfo_interval(20 * 60, has_rows=False) == 30.0
    assert beforeinfo_interval(20 * 60, has_rows=True) == 90.0
    assert beforeinfo_interval(10 * 60, has_rows=True) == 30.0
    assert beforeinfo_interval(4 * 60, has_rows=False) is None


def test_odds_polling_does_not_probe_before_the_collection_window() -> None:
    assert odds_interval(61 * 60) is None
    assert odds_interval(60 * 60) == 90.0
    assert odds_interval(15 * 60) == 45.0
    assert odds_interval(5 * 60) == 20.0
    assert odds_interval(90) == 10.0
    assert odds_interval(-1) is None
