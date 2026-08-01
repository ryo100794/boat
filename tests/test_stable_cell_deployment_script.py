from pathlib import Path


def test_stable_cell_runner_is_shadow_only_and_date_registered() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-stable-cell-shadow.sh"
    ).read_text()

    assert "stable_cell_top5_lt20_t300" in script
    assert 'prediction_date" < "2026-08-02"' in script
    assert "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0" in script
    assert "run-boatrace-intraday-t300-shadow.sh" in script
