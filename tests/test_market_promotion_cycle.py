from argparse import Namespace
from pathlib import Path

from boatrace_ai.runtime import market_promotion_cycle


def test_cycle_records_rejection_without_creating_active_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    observed = {}

    def fake_promote(candidates, *, output_path):
        observed["candidates"] = candidates
        observed["output"] = output_path
        return {"status": "no_eligible_candidate"}

    monkeypatch.setattr(
        market_promotion_cycle, "promote_best_candidate", fake_promote
    )
    args = Namespace(
        candidate=["a.json", "b.json"],
        output=str(tmp_path / "active.json"),
        state=str(tmp_path / "state.json"),
    )

    event = market_promotion_cycle.run_once(args)

    assert event["status"] == "ok"
    assert event["promotion"]["status"] == "no_eligible_candidate"
    assert observed == {
        "candidates": ["a.json", "b.json"],
        "output": str(tmp_path / "active.json"),
    }
    assert not (tmp_path / "active.json").exists()
    assert (tmp_path / "state.json").is_file()


def test_deployment_cycle_includes_all_production_market_tracks() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "deployment"
        / "run-boatrace-market-promotion.sh"
    ).read_text(encoding="utf-8")

    assert script.count("--candidate") == 4
    assert "active_market_model.json" in script
