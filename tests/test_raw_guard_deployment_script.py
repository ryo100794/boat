from pathlib import Path


def test_raw_guard_runner_is_shadow_only_and_date_registered() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-raw-guard-shadow.sh"
    ).read_text()

    assert "raw_guard_daily:raw_guard_v21_t300" in script
    assert 'prediction_date" < "2026-08-02"' in script
    assert "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0" in script
    assert "boatrace_ai.runtime.raw_guard_shadow" in script
