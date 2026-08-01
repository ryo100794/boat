from pathlib import Path


def test_quota_ceil_runner_is_shadow_only_and_date_registered() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-quota-ceil-shadow.sh"
    ).read_text()

    assert "quota_ceil_daily:quota_ceil_v21_t300" in script
    assert 'prediction_date" < "2026-08-02"' in script
    assert "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0" in script
    assert "boatrace_ai.runtime.quota_ceil_shadow" in script
