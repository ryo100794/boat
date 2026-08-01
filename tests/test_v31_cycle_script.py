from pathlib import Path


def test_v31_cycle_passes_required_runtime_arguments() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/deployment/run-boatrace-intraday-v31-cycle.sh"
    ).read_text()

    assert '--db "$DSN"' in script
    assert '--model-spec "$MODEL_SPEC"' in script
    assert '--date "$BOATRACE_T300_SHADOW_DATE"' in script
    assert "boatrace_ai.runtime.v31_uncertainty_adjusted_shadow" in script
    assert "PGPASSFILE" in script
