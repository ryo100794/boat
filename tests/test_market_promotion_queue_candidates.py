from argparse import Namespace
from pathlib import Path

from boatrace_ai.runtime import market_promotion_cycle


def test_cycle_adds_discovered_queue_candidates(monkeypatch, tmp_path: Path) -> None:
    observed = {}
    discovered = tmp_path / "job-00000042.json"

    monkeypatch.setattr(
        market_promotion_cycle,
        "discover_market_evaluation_candidates",
        lambda directory: [str(discovered)] if directory == str(tmp_path) else [],
    )

    def fake_promote(candidates, *, output_path):
        observed["candidates"] = candidates
        observed["output"] = output_path
        return {"status": "no_eligible_candidate"}

    monkeypatch.setattr(
        market_promotion_cycle,
        "promote_best_candidate",
        fake_promote,
    )
    args = Namespace(
        candidate=["fixed.json", str(discovered)],
        evaluation_queue_dir=str(tmp_path),
        output=str(tmp_path / "active.json"),
        state=str(tmp_path / "state.json"),
    )

    event = market_promotion_cycle.run_once(args)

    assert event["status"] == "ok"
    assert observed["candidates"] == ["fixed.json", str(discovered)]
    assert observed["output"] == str(tmp_path / "active.json")


def test_deployment_enables_evaluation_queue_discovery() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "deployment"
        / "run-boatrace-market-promotion.sh"
    ).read_text(encoding="utf-8")

    assert '--evaluation-queue-dir "$APP_ROOT/data/models/evaluation_queue"' in script
