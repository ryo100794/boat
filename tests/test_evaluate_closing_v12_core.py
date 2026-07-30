from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import joblib
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "modeling"
    / "evaluate_closing_v12_core.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_closing_v12_core", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _races() -> list[dict[str, object]]:
    return [
        {
            "race_date": day,
            "race_id": f"{day.replace('-', '')}0101",
            "jcd": 1,
            "rno": 1,
        }
        for day in ("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04")
    ]


def _install_fake_v12(module: ModuleType, calls: list[dict[str, object]]) -> None:
    def fake_fit(
        training: list[dict[str, object]], *, prediction_date: str
    ) -> dict[str, object]:
        calls.append(
            {
                "prediction_date": prediction_date,
                "training_dates": [race["race_date"] for race in training],
            }
        )
        adopted = prediction_date == "2026-07-04"
        return {
            "ready": True,
            "trained_through_date": (
                max(str(race["race_date"]) for race in training)
                if training
                else None
            ),
            "challenger_adopted": adopted,
            "selected_mode": (
                "nonlinear_model" if adopted else "current_odds_baseline"
            ),
            "actual_engine": "lightgbm" if adopted else "sklearn_hist_gradient_boosting",
            "selection_reason": "test",
            "strict_prior_baseline_current_mae": 0.5,
            "strict_prior_challenger_mae": 0.4,
            "strict_prior_relative_mae_improvement": 0.2,
        }

    def fake_metrics(
        holdout: list[dict[str, object]], model: dict[str, object]
    ) -> dict[str, object]:
        evaluation_date = str(holdout[0]["race_date"])
        if evaluation_date == "2026-07-03":
            return {
                "evaluation_races": 1,
                "evaluation_tickets": 100,
                "baseline_current_log_mae": 0.5,
                "selected_point_log_mae": 0.4,
                "selected_relative_mae_improvement": 0.2,
                "lower_bound_coverage": 0.8,
            }
        return {
            "evaluation_races": 1,
            "evaluation_tickets": 300,
            "baseline_current_log_mae": 0.3,
            "selected_point_log_mae": 0.24,
            "selected_relative_mae_improvement": 0.2,
            "lower_bound_coverage": 0.9,
        }

    module.fit_closing_odds_t300_nonlinear_v12 = fake_fit
    module.closing_odds_t300_nonlinear_v12_metrics = fake_metrics


def test_evaluate_uses_strict_prior_outer_day_and_v12_metrics(tmp_path: Path) -> None:
    module = _load_module()
    cache = tmp_path / "scored.joblib"
    joblib.dump({"contract": {"version": 1}, "races": _races()}, cache)
    calls: list[dict[str, object]] = []
    _install_fake_v12(module, calls)

    result = module.evaluate(cache, first_evaluation_date="2026-07-03")

    assert result["model"] == "closing_odds_t300_nonlinear_v12"
    assert result["validation"] == "strict_prior_outer_day"
    assert [fold["evaluation_date"] for fold in result["daily"]] == [
        "2026-07-03",
        "2026-07-04",
    ]
    assert calls == [
        {
            "prediction_date": "2026-07-03",
            "training_dates": ["2026-07-01", "2026-07-02"],
        },
        {
            "prediction_date": "2026-07-04",
            "training_dates": [
                "2026-07-01",
                "2026-07-02",
                "2026-07-03",
            ],
        },
    ]
    assert all(fold["strict_prior_boundary"] for fold in result["daily"])
    assert result["daily"][0]["metrics"]["selected_point_log_mae"] == 0.4
    assert "v11" not in json.dumps(result).lower()


def test_weighted_aggregate_uses_evaluation_ticket_weights(tmp_path: Path) -> None:
    module = _load_module()
    cache = tmp_path / "scored.joblib"
    joblib.dump({"races": _races()}, cache)
    _install_fake_v12(module, [])

    aggregate = module.evaluate(
        cache, first_evaluation_date="2026-07-03"
    )["aggregate"]

    assert aggregate["weighting"] == "evaluation_tickets"
    assert aggregate["evaluation_tickets"] == 400
    assert aggregate["baseline_current_log_mae"] == pytest.approx(0.35)
    assert aggregate["selected_point_log_mae"] == pytest.approx(0.28)
    assert aggregate["selected_relative_mae_improvement"] == pytest.approx(0.2)
    assert aggregate["lower_bound_coverage"] == pytest.approx(0.875)
    assert aggregate["adopted_folds"] == 1
    assert aggregate["adopted_evaluation_tickets"] == 300
    assert aggregate["adopted_evaluation_ticket_rate"] == pytest.approx(0.75)
    assert aggregate["engines_by_evaluation_tickets"] == {
        "lightgbm": 300,
        "sklearn_hist_gradient_boosting": 100,
    }


def test_cli_writes_stable_json_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    cache = tmp_path / "scored.joblib"
    output = tmp_path / "reports" / "v12.json"
    joblib.dump({"races": _races()}, cache)
    _install_fake_v12(module, [])
    replacements: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", recording_replace)

    assert module.main(
        [
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--first-evaluation-date",
            "2026-07-03",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["first_evaluation_date"] == "2026-07-03"
    assert replacements and replacements[0][1] == output
    assert replacements[0][0].parent == output.parent
    assert not replacements[0][0].exists()
    assert output.read_bytes().endswith(b"\n")


def test_atomic_write_preserves_existing_output_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    output = tmp_path / "result.json"
    output.write_text("old\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        module._atomic_write_json(output, {"new": True})

    assert output.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_invalid_cache_and_date_are_rejected(tmp_path: Path) -> None:
    module = _load_module()
    cache = tmp_path / "bad.joblib"
    joblib.dump({"races": "not-a-list"}, cache)

    with pytest.raises(ValueError, match="race list"):
        module.evaluate(cache, first_evaluation_date=None)

    good = tmp_path / "good.joblib"
    joblib.dump({"races": _races()}, good)
    with pytest.raises(ValueError, match="first_evaluation_date"):
        module.evaluate(good, first_evaluation_date="2026/07/03")


def test_evaluate_rejects_non_prior_fitted_artifact(tmp_path: Path) -> None:
    module = _load_module()
    cache = tmp_path / "scored.joblib"
    joblib.dump({"races": _races()}, cache)

    def leaking_fit(
        training: list[dict[str, object]], *, prediction_date: str
    ) -> dict[str, object]:
        return {
            "ready": True,
            "trained_through_date": prediction_date,
            "challenger_adopted": False,
            "actual_engine": "test",
        }

    module.fit_closing_odds_t300_nonlinear_v12 = leaking_fit
    module.closing_odds_t300_nonlinear_v12_metrics = lambda holdout, model: {}

    with pytest.raises(ValueError, match="leaked non-prior data"):
        module.evaluate(cache, first_evaluation_date="2026-07-03")
