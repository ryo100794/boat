from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import boatrace_ai.listwise.learned_purchase_allocation_v33_evaluation as module
from boatrace_ai.listwise.four_head_nested_v22 import (
    DecisionRace,
    LabeledRace,
    RaceOutcome,
    RacePrediction,
)
from boatrace_ai.listwise.four_head_v22_bankroll import V22BankrollSettlement
from boatrace_ai.listwise.four_head_v22_evaluation import (
    DecisionAudit,
    V22EvaluationData,
)
from boatrace_ai.listwise.learned_purchase_allocation_v33 import AllocationDecision


CHOICES = 120


class _AllocationArtifact:
    teacher = "daily_weighted_realized_log_bankroll_growth"
    trained_through_date = "2026-01-09"

    def summary(self):
        return {"teacher": self.teacher, "trained_through_date": self.trained_through_date}


def _race(date: str, sequence: int, *, outer_winner: int = 0) -> LabeledRace:
    race_id = f"{date}-01-{sequence:02d}"
    odds = tuple(10.0 + index / 10.0 for index in range(CHOICES))
    order = (outer_winner, *(i for i in range(CHOICES) if i != outer_winner))
    return LabeledRace(
        DecisionRace(
            race_id,
            date,
            tuple((float(sequence), float(index)) for index in range(CHOICES)),
            odds,
        ),
        RaceOutcome(outer_winner, odds, order),
    )


def _prediction(decision: DecisionRace) -> RacePrediction:
    probability = np.arange(CHOICES, 0, -1, dtype=np.float64)
    probability /= probability.sum()
    return RacePrediction(
        decision.race_id,
        decision.race_date,
        tuple(probability),
        tuple(probability),
        decision.current_odds,
        tuple(0.0 for _ in range(CHOICES)),
        (),
    )


def _settlement(
    race: LabeledRace,
    *,
    hour: int,
    minute: int = 0,
    result_minute: int | None = None,
    payout: int = 1_000,
    winner: int | None = None,
) -> V22BankrollSettlement:
    result = result_minute if result_minute is not None else minute + 5
    return V22BankrollSettlement(
        race_id=race.decision.race_id,
        decision_target_at=f"{race.decision.race_date}T{hour:02d}:{minute:02d}:00+00:00",
        odds_captured_at=f"{race.decision.race_date}T{hour:02d}:{minute:02d}:00+00:00",
        result_available_at=f"{race.decision.race_date}T{hour:02d}:{result:02d}:00+00:00",
        official_winner_index=(race.outcome.winner_index if winner is None else winner),
        official_closing_odds=race.outcome.closing_odds,
        official_payout_yen=payout,
        snapshot_id=1,
    )


def _data(*, outer_count: int = 3):
    training = tuple(_race(f"2026-01-{day:02d}", 1) for day in range(1, 10))
    outer = tuple(_race("2026-01-10", index + 1, outer_winner=119) for index in range(outer_count))
    settlements = [
        _settlement(race, hour=10, payout=2_000)
        for race in training
    ]
    for index, race in enumerate(outer):
        settlements.append(
            _settlement(
                race,
                hour=11,
                minute=index,
                result_minute=(2 if index == 0 else index + 5),
                payout=1_000,
                winner=0,
            )
        )
    audits = tuple(
        DecisionAudit(
            settlement.race_id,
            settlement.snapshot_id,
            settlement.odds_captured_at,
            settlement.decision_target_at,
            0.0,
        )
        for settlement in settlements
    )
    return V22EvaluationData(training, outer, audits, {}), tuple(settlements)


def _install_models(monkeypatch: pytest.MonkeyPatch, *, allocations=None):
    fit_calls = []
    lpa_call = {}

    def fit_base(races, **kwargs):
        races = tuple(races)
        fit_calls.append(tuple(race.decision.race_id for race in races))
        return SimpleNamespace(
            trained_through_date=races[-1].decision.race_date,
            race_ids=fit_calls[-1],
        )

    def fit_lpa(races, predictions, payout_map, **kwargs):
        lpa_call.update(
            races=tuple(race.decision.race_id for race in races),
            predictions=tuple(row.race_id for row in predictions),
            payout_map=dict(payout_map),
            kwargs=kwargs,
        )
        return _AllocationArtifact()

    seen_available = []

    def allocate(artifact, decision, prediction, *, available_bankroll_yen, stake_unit_yen):
        seen_available.append((decision.race_id, available_bankroll_yen))
        stake = (
            allocations.get(decision.race_id, 0)
            if allocations is not None
            else available_bankroll_yen * 5 // 100 // 100 * 100
        )
        stakes = [0] * CHOICES
        stakes[0] = stake
        return AllocationDecision(
            decision.race_id,
            stake / max(available_bankroll_yen, 1),
            stake,
            tuple(stakes),
            tuple(1.0 if i == 0 else 0.0 for i in range(CHOICES)),
            1.0,
        )

    monkeypatch.setattr(module, "fit_four_head_nested_v22", fit_base)
    monkeypatch.setattr(module, "predict_race", lambda artifact, decision: _prediction(decision))
    monkeypatch.setattr(module, "fit_learned_allocation_head", fit_lpa)
    monkeypatch.setattr(module, "allocation_decision", allocate)
    return fit_calls, lpa_call, seen_available


def test_strict_date_split_official_payout_teacher_and_outer_is_never_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, settlements = _data(outer_count=1)
    fit_calls, lpa_call, _seen = _install_models(monkeypatch)

    result = module.evaluate_learned_purchase_allocation_v33(
        data,
        settlements,
        base_training_fraction=5 / 9,
        bootstrap_samples=100,
    )

    assert fit_calls[0] == tuple(race.decision.race_id for race in data.training_races[:5])
    assert fit_calls[1] == tuple(race.decision.race_id for race in data.training_races)
    assert lpa_call["races"] == tuple(race.decision.race_id for race in data.training_races[5:])
    assert lpa_call["predictions"] == lpa_call["races"]
    assert lpa_call["kwargs"]["base_predictions_trained_through_date"] == "2026-01-05"
    assert lpa_call["payout_map"] == {
        race.decision.race_id: 2_000 for race in data.training_races
    }
    assert not set(fit_calls[1]) & {race.decision.race_id for race in data.outer_races}
    assert result["teacher"]["closing_odds_used_as_payout_teacher"] is False
    assert result["information_boundary"]["outer_outcomes_used_for_fit_or_selection"] is False


def test_pending_stakes_reduce_cash_and_returns_arrive_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, settlements = _data(outer_count=3)
    _fits, _lpa, seen = _install_models(monkeypatch)

    result = module.evaluate_learned_purchase_allocation_v33(
        data,
        settlements,
        base_training_fraction=5 / 9,
        bootstrap_samples=100,
    )

    primary_seen = seen[:3]
    assert [value for _race_id, value in primary_seen] == [10_000, 9_500, 14_100]
    ledger = result["daily"][0]["ledger"]
    assert [row["event"] for row in ledger[:4]] == [
        "decision",
        "decision",
        "settlement",
        "decision",
    ]
    assert ledger[0]["outstanding_stake_yen"] == 500
    assert ledger[1]["outstanding_stake_yen"] == 900
    assert ledger[2]["return_yen"] == 5_000
    assert result["policy"]["pending_stakes_reduce_available_cash"] is True
    assert result["policy"]["official_payout_role"] == "settlement_only"


def test_uses_settlement_result_not_outer_outcome_and_reports_stress_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, settlements = _data(outer_count=1)
    race_id = data.outer_races[0].decision.race_id
    _install_models(monkeypatch, allocations={race_id: 500})

    result = module.evaluate_learned_purchase_allocation_v33(
        data,
        settlements,
        base_training_fraction=5 / 9,
        bootstrap_samples=500,
    )

    assert data.outer_races[0].outcome.winner_index == 119
    assert result["stake_yen"] == 500
    assert result["return_yen"] == 5_000
    assert result["profit_yen"] == 4_500
    assert result["roi"] == 10.0
    assert result["roi_without_largest_hit"] == 0.0
    assert result["effective_hit_count"] == 1.0
    assert result["daily_bootstrap"]["roi_lower_95"] == 10.0
    assert result["odds_stress_5pct"]["return_yen"] == 4_750
    assert result["odds_stress_5pct"]["roi"] == 9.5
    assert result["promotion_eligible"] is False
    assert result["promotion_gate"]["roi_without_largest_hit_above_one"] is False


def test_writes_reusable_deployment_artifact_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import hashlib
    import joblib

    data, settlements = _data(outer_count=1)
    _install_models(monkeypatch)
    destination = tmp_path / "models" / "v33-lpa.joblib"

    result = module.evaluate_learned_purchase_allocation_v33(
        data,
        settlements,
        base_training_fraction=5 / 9,
        bootstrap_samples=10,
        artifact_output=destination,
    )

    assert destination.is_file()
    assert not destination.with_name(f".{destination.name}.tmp").exists()
    with destination.open("rb") as artifact_file:
        digest = hashlib.file_digest(artifact_file, "sha256").hexdigest()
    assert result["deployment_artifact"] == {
        "ready": True,
        "path": str(destination),
        "sha256": digest,
    }
    bundle = joblib.load(destination)
    assert bundle["schema_version"] == 1
    assert bundle["model_key"] == module.MODEL_KEY
    assert bundle["information_boundary"][
        "outer_outcomes_used_for_fit_or_selection"
    ] is False
    assert bundle["allocation_model"].teacher == (
        "daily_weighted_realized_log_bankroll_growth"
    )


def test_missing_training_official_payout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, settlements = _data(outer_count=1)
    _install_models(monkeypatch)
    incomplete = settlements[1:]

    with pytest.raises(ValueError, match="settlement universe mismatch"):
        module.evaluate_learned_purchase_allocation_v33(
            data,
            incomplete,
            base_training_fraction=5 / 9,
            bootstrap_samples=10,
        )


def test_outer_result_change_does_not_change_frozen_prediction_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, settlements = _data(outer_count=1)
    _install_models(monkeypatch)
    first = module.evaluate_learned_purchase_allocation_v33(
        data, settlements, base_training_fraction=5 / 9, bootstrap_samples=10
    )
    changed_outer = replace(
        data.outer_races[0],
        outcome=replace(data.outer_races[0].outcome, winner_index=42),
    )
    changed = replace(data, outer_races=(changed_outer,))
    second = module.evaluate_learned_purchase_allocation_v33(
        changed, settlements, base_training_fraction=5 / 9, bootstrap_samples=10
    )

    assert first["frozen_outer_prediction_sha256"] == second["frozen_outer_prediction_sha256"]
    assert first["stake_yen"] == second["stake_yen"]


def test_rejects_snapshot_after_t5_target(monkeypatch: pytest.MonkeyPatch) -> None:
    data, settlements = _data(outer_count=1)
    _install_models(monkeypatch)
    unsafe = replace(settlements[-1], odds_captured_at="2026-01-10T11:00:01+00:00")
    audit = replace(data.decision_audit[-1], captured_at=unsafe.odds_captured_at)
    changed = replace(data, decision_audit=(*data.decision_audit[:-1], audit))

    with pytest.raises(ValueError, match="unsafe T-5 snapshot"):
        module.evaluate_learned_purchase_allocation_v33(
            changed,
            (*settlements[:-1], unsafe),
            base_training_fraction=5 / 9,
            bootstrap_samples=10,
        )


def test_reuses_legacy_main_pickled_v20_cache_and_restores_aliases(tmp_path) -> None:
    import __main__
    import joblib

    original_data = getattr(__main__, "V22EvaluationData", module._MISSING)
    original_audit = getattr(__main__, "DecisionAudit", module._MISSING)
    try:
        legacy_data_type = type("V22EvaluationData", (), {})
        legacy_data_type.__module__ = "__main__"
        legacy_audit_type = type("DecisionAudit", (), {})
        legacy_audit_type.__module__ = "__main__"
        setattr(__main__, "V22EvaluationData", legacy_data_type)
        setattr(__main__, "DecisionAudit", legacy_audit_type)

        legacy_audit = legacy_audit_type()
        legacy_audit.race_id = "legacy-race"
        legacy_audit.snapshot_id = 7
        legacy_audit.captured_at = "2026-01-01T00:00:00+00:00"
        legacy_audit.target_at = "2026-01-01T00:00:10+00:00"
        legacy_audit.age_seconds = 10.0
        legacy_audit.choices = 120
        legacy_data = legacy_data_type()
        legacy_data.training_races = ()
        legacy_data.outer_races = ()
        legacy_data.decision_audit = (legacy_audit,)
        legacy_data.diagnostics = {"source": "v20-cache"}

        cache = tmp_path / "legacy-v20.joblib"
        joblib.dump(
            {
                "schema_version": 1,
                "signature": {"model": "v20"},
                "data": legacy_data,
            },
            cache,
        )

        data_sentinel = object()
        audit_sentinel = object()
        setattr(__main__, "V22EvaluationData", data_sentinel)
        setattr(__main__, "DecisionAudit", audit_sentinel)
        loaded = module.load_v22_evaluation_cache_compat(
            cache, expected_signature={"model": "v20"}
        )

        assert isinstance(loaded, V22EvaluationData)
        assert isinstance(loaded.decision_audit[0], DecisionAudit)
        assert loaded.diagnostics == {"source": "v20-cache"}
        assert getattr(__main__, "V22EvaluationData") is data_sentinel
        assert getattr(__main__, "DecisionAudit") is audit_sentinel
    finally:
        if original_data is module._MISSING:
            if hasattr(__main__, "V22EvaluationData"):
                delattr(__main__, "V22EvaluationData")
        else:
            setattr(__main__, "V22EvaluationData", original_data)
        if original_audit is module._MISSING:
            if hasattr(__main__, "DecisionAudit"):
                delattr(__main__, "DecisionAudit")
        else:
            setattr(__main__, "DecisionAudit", original_audit)
