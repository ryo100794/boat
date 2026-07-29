from __future__ import annotations

from pathlib import Path

import boatrace_ai.evaluation_queue as evaluation_queue
import boatrace_ai.listwise.market_calibration as market_calibration
import boatrace_ai.listwise.odds_path_role_integrated_v15 as integrated_v15
from boatrace_ai.evaluation_queue import build_command, seed_daily_market_jobs


STRATEGY = "odds_path_role_integrated_selection_free_envelope_v15"


def _job() -> dict[str, object]:
    return {
        "job_id": 15,
        "status": "running",
        "task_type": "market_residual_walk_forward",
        "model_key": "v15-candidate",
        "parameters": {
            "model_input": "data/models/source.joblib",
            "from_date": "2026-07-18",
            "through_date": "2026-08-01",
            "daily_budget_yen": 10_000,
            "min_calibration_days": 2,
            "calibrator_strategy": STRATEGY,
            "v12_closing_fallback_policy": "no_bet",
        },
    }


def test_v15_parser_and_queue_command_are_reproducible(tmp_path: Path) -> None:
    model = tmp_path / "data/models/source.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    python = tmp_path / ".venv/bin/python"

    first, first_output = build_command(
        _job(), app_root=tmp_path, python=python, db="postgresql://test"
    )
    second, second_output = build_command(
        _job(), app_root=tmp_path, python=python, db="postgresql://test"
    )

    assert first == second
    assert first_output == second_output
    assert first[first.index("--calibrator-strategy") + 1] == STRATEGY
    assert first[first.index("--v12-closing-fallback-policy") + 1] == "no_bet"
    parsed = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", STRATEGY,
    ])
    assert parsed.calibrator_strategy == STRATEGY


def test_v15_dispatcher_preserves_the_exact_evaluation_population(
    monkeypatch,
) -> None:
    fallback_policies: list[str] = []

    def fake_v12(_races, **kwargs):
        fallback_policies.append(str(kwargs["closing_fallback_policy"]))
        return {}

    monkeypatch.setattr(integrated_v15, "walk_forward_evaluate_v12", fake_v12)
    races = [
        {"race_date": "2026-07-30", "race_id": "2026-07-30-02-01"},
        {"race_date": "2026-07-30", "race_id": "2026-07-30-01-01"},
    ]
    direct = integrated_v15.walk_forward_evaluate_v15(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=["2026-07-30"],
    )
    dispatched = market_calibration.walk_forward_evaluate(
        list(reversed(races)),
        daily_budget_yen=10_000,
        min_calibration_days=2,
        calibrator_strategy=STRATEGY,
        evaluation_dates=["2026-07-30"],
        v12_closing_fallback_policy="no_bet",
    )

    assert integrated_v15.MODEL_NAME == STRATEGY
    assert integrated_v15.STRATEGY_NAME == STRATEGY
    assert direct["evaluation_population_races"] == 2
    assert dispatched["evaluation_population_races"] == 2
    assert direct["evaluation_population_hash"] == dispatched[
        "evaluation_population_hash"
    ]
    assert direct["model"] == dispatched["model"] == STRATEGY
    assert direct["calibrator_strategy"] == STRATEGY
    assert dispatched["calibrator_strategy"] == STRATEGY
    assert fallback_policies == ["no_bet", "no_bet"]


def test_daily_v15_is_queued_ahead_of_v14(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "data/models/evaluation_queue"
    model_dir.mkdir(parents=True)
    source_path = model_dir / "source.json"
    source_path.with_suffix(".joblib").write_bytes(b"model")

    class FakeResult:
        def fetchone(self):
            return {"result_path": str(source_path)}

    class FakeConnection:
        def execute(self, _sql, _parameters):
            return FakeResult()

    calls: list[dict[str, object]] = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)
    seed_daily_market_jobs(
        FakeConnection(), app_root=tmp_path, evaluation_date="2026-07-30"
    )

    by_strategy = {
        row["parameters"]["calibrator_strategy"]: row for row in calls
    }
    v15 = by_strategy[STRATEGY]
    v14 = by_strategy[
        "odds_path_role_integrated_registered_band_lcb_v14"
    ]
    assert v15["priority"] > v14["priority"]
    assert v15["parameters"]["timeout_seconds"] == 14_400
    assert v15["parameters"]["min_calibration_days"] == 5
    assert v15["parameters"]["v12_closing_fallback_policy"] == "no_bet"
    assert v15["parameters"]["model_input"] == v14["parameters"]["model_input"]
    assert v15["parameters"]["from_date"] == v14["parameters"]["from_date"]
    assert v15["parameters"]["through_date"] == v14["parameters"]["through_date"]
    assert calls.index(v15) < calls.index(v14)
