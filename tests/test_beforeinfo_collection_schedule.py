from boatrace_ai.runtime.collector import beforeinfo_interval


def test_beforeinfo_polling_targets_model_decision_window() -> None:
    assert beforeinfo_interval(31 * 60, has_rows=False) is None
    assert beforeinfo_interval(20 * 60, has_rows=False) == 30.0
    assert beforeinfo_interval(20 * 60, has_rows=True) == 90.0
    assert beforeinfo_interval(10 * 60, has_rows=True) == 30.0
    assert beforeinfo_interval(4 * 60, has_rows=False) is None
