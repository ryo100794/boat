from pathlib import Path


def test_v32_shadow_wrapper_reloads_daily_specs_and_stays_shadow_only() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/deployment/run-boatrace-intraday-v32-shadow.sh"
    ).read_text()

    assert "sha256sum" in script
    assert 'identity="${BOATRACE_T300_SHADOW_DATE:-}:${spec_hash}"' in script
    assert "stop_child" in script
    assert 'kill -TERM "$child_pid"' in script
    assert 'wait "$child_pid"' in script
    assert "v32_daily:v32_uncertainty_adjusted_top5_t300" in script
    assert "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0" in script
    assert "run-boatrace-intraday-v32-cycle.sh" in script
