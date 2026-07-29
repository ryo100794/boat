from __future__ import annotations

from itertools import permutations

import pytest

from boatrace_ai.listwise import market_calibration
from boatrace_ai.listwise import odds_path_role_integrated_v12 as v12
from boatrace_ai.listwise import odds_path_role_integrated_v15 as v15


COMBINATIONS = [
    "".join(str(lane) for lane in lanes)
    for lanes in permutations(range(1, 7), 3)
]


def _odds(base: float) -> dict[str, float]:
    return {combination: base + index for index, combination in enumerate(COMBINATIONS)}


def _prewarm_race(
    race_date: str, race_no: int, *, checkpoint_odds=None, captured_age=305.0
):
    checkpoint = {}
    if checkpoint_odds is not None:
        checkpoint = {
            "t300": {
                "target_offset_seconds": 300,
                "captured_age_seconds": captured_age,
                "odds": checkpoint_odds,
            }
        }
    return {
        "race_date": race_date,
        "race_id": f"{race_date}-{race_no:02d}",
        "closing_odds_checkpoints": checkpoint,
        "official_closing_odds": _odds(11.0),
    }


def test_prewarm_uses_only_first_calibration_days_and_t300_baseline():
    races = [
        _prewarm_race(f"2026-07-{day:02d}", race_no, checkpoint_odds=_odds(10.0))
        for day in range(1, 7)
        for race_no in range(1, 7)
    ]

    observations = v15.build_strict_prior_prewarm_observations_v15(
        races, min_calibration_days=5
    )

    assert len(observations) == 30
    assert {row["race_date"] for row in observations} == {
        f"2026-07-{day:02d}" for day in range(1, 6)
    }
    assert all(len(row["predicted_closing_odds"]) == 120 for row in observations)
    assert all(row["strict_prior_prewarm"] is True for row in observations)
    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-06"
    )
    assert artifact["ready"] is True
    assert artifact["training_days"] == 5
    assert artifact["training_races"] == 30
    assert artifact["training_observations"] == 3600


def test_prewarm_rejects_future_checkpoint_and_audits_missing_t300():
    future = _prewarm_race(
        "2026-07-01", 1, checkpoint_odds=_odds(10.0), captured_age=299.0
    )
    missing = _prewarm_race("2026-07-01", 2, checkpoint_odds=None)

    observations = v15.build_strict_prior_prewarm_observations_v15(
        [future, missing], min_calibration_days=1
    )
    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-02"
    )

    assert [row["predicted_closing_odds"] for row in observations] == [{}, {}]
    audit = artifact["missing_audit"]
    assert audit["accepted_races"] == 0
    assert audit["rejected_races"] == 2
    assert audit["rejection_reasons"] == {"missing_predicted_closing_odds": 2}


def test_observation_append_uses_all_120_after_decision(monkeypatch):
    races = [{"race_id": "202607300101", "race_date": "2026-07-30"}]
    predicted = _odds(10.0)
    actual = _odds(11.0)
    monkeypatch.setattr(v15, "select_teacher_final_odds", lambda race: (actual, "final"))
    observations = []

    appended = v15.append_closing_envelope_observations_v15(
        observations,
        races,
        closing_forecasts={"202607300101": predicted},
        probability_lcb={"ready": False},
        evaluation_date="2026-07-30",
    )

    assert appended == 1
    assert observations == [{
        "race_date": "2026-07-30",
        "race_id": "202607300101",
        "predicted_closing_odds": predicted,
        "actual_closing_odds": actual,
        "teacher_population": "all_120_complete_combinations",
        "teacher_appended_after_purchase_decision": True,
    }]


def test_observation_append_preserves_incomplete_race_for_missing_audit(monkeypatch):
    actual = _odds(11.0)
    actual.pop(COMBINATIONS[-1])
    monkeypatch.setattr(v15, "select_teacher_final_odds", lambda race: (actual, "final"))
    observations = []

    appended = v15.append_closing_envelope_observations_v15(
        observations,
        [{"race_id": "r1", "race_date": "2026-07-30"}],
        closing_forecasts={"r1": _odds(10.0)},
        probability_lcb={},
        evaluation_date="2026-07-30",
    )

    assert appended == 1
    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-31"
    )
    audit = artifact["missing_audit"]
    assert audit["accepted_races"] == 0
    assert audit["rejected_races"] == 1
    assert audit["rejection_reasons"] == {
        "incomplete_actual_closing_odds": 1
    }


def test_observation_append_preserves_missing_forecast_for_missing_audit(monkeypatch):
    monkeypatch.setattr(
        v15,
        "select_teacher_final_odds",
        lambda race: (_odds(11.0), "final"),
    )
    observations = []

    appended = v15.append_closing_envelope_observations_v15(
        observations,
        [{"race_id": "r1", "race_date": "2026-07-30"}],
        closing_forecasts={},
        probability_lcb={},
        evaluation_date="2026-07-30",
    )
    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-31"
    )

    assert appended == 1
    assert artifact["missing_audit"]["rejection_reasons"] == {
        "missing_predicted_closing_odds": 1
    }


def test_fit_adapter_is_strict_prior_and_selection_free():
    observations = []
    for day in range(1, 6):
        for race_no in range(1, 7):
            observations.append({
                "race_date": f"2026-07-{day:02d}",
                "race_id": f"{day}-{race_no}",
                "predicted_closing_odds": _odds(10.0),
                "actual_closing_odds": _odds(9.0),
            })

    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-06"
    )

    assert artifact["ready"] is True
    assert artifact["selection_free"] is True
    assert artifact["training_races"] == 30
    assert artifact["training_observations"] == 3600
    assert artifact["trained_through_date"] == "2026-07-05"


def test_v15_point_forecast_is_independent_of_v12_lower_forecast(monkeypatch):
    point = _odds(20.0)
    response = {"lower": _odds(10.0)}

    def fake_forecast(race, model, *, prediction_date):
        del race, model, prediction_date
        return {
            "ready": True,
            "point_final_odds": point,
            "lower_final_odds": response["lower"],
            "future_checkpoint_offsets_used": [],
        }

    monkeypatch.setattr(
        v12,
        "forecast_closing_odds_t300_nonlinear_v12",
        fake_forecast,
    )
    model = {"ready": True, "challenger_adopted": True}
    races = [{"race_id": "r1"}]

    first, first_audit = v12._strict_prior_forecasts(
        races,
        model,
        None,
        evaluation_date="2026-07-30",
        fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
        forecast_field="point_final_odds",
    )
    response["lower"] = _odds(1_000.0)
    second, second_audit = v12._strict_prior_forecasts(
        races,
        model,
        None,
        evaluation_date="2026-07-30",
        fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
        forecast_field="point_final_odds",
    )

    assert first == {"r1": point}
    assert second == first
    assert first_audit["forecast_field"] == "point_final_odds"
    assert second_audit["forecast_field"] == "point_final_odds"


def test_walk_forward_delegates_to_v12_and_normalizes_v15_result(monkeypatch):
    captured = {}

    def fake_v12(races, **kwargs):
        captured.update(kwargs)
        return {
            "model": "v12",
            "fixed_policy": {"name": "v12-policy"},
            "selection_conformal": {"ready_folds": 1},
            "selection_conformal_artifacts_by_date": {"2026-07-30": {"ready": True}},
            "folds": [{
                "evaluation_date": "2026-07-30",
                "selected_policy": {"name": "v12-policy"},
                "selection_conformal": {
                    "ready": True,
                    "haircut": 0.9,
                    "training_days": 5,
                    "training_races": 30,
                    "training_observations": 3600,
                    "trained_through_date": "2026-07-29",
                    "missing_audit": {
                        "input_races": 30,
                        "accepted_races": 30,
                        "rejected_races": 0,
                    },
                },
                "selection_observations_appended_after_decision": 12,
                "leakage_guard": {"selection_conformal_trained_through": "2026-07-29"},
                "bankroll": {"selection_conformal": {"ready": True}},
            }],
            v15.V12_PROSPECTIVE_OUTPUT_KEY: {
                "selection_conformal": {"ready_folds": 1},
                "promotion_gate": {"base_roi_pass": True},
                "promotion_eligible": True,
            },
            "deployment_configuration": {
                "selection_conformal": {"ready": True},
                "candidate_policy": {"name": "v12-policy"},
            },
        }

    monkeypatch.setattr(v15, "walk_forward_evaluate_v12", fake_v12)
    result = v15.walk_forward_evaluate_v15(
        [], daily_budget_yen=10_000, min_calibration_days=5
    )

    assert captured["closing_fallback_policy"] == v15.CLOSING_FALLBACK_NO_BET
    assert captured["closing_forecast_field"] == "point_final_odds"
    assert captured["selection_conformal_fit"] is v15._fit_closing_envelope
    assert captured["selection_observation_append"] is v15.append_closing_envelope_observations_v15
    assert captured["initial_selection_observations"] == []
    assert result["model"] == v15.MODEL_NAME
    assert result["selection_free"] is True
    assert result["zero_bet_allowed"] is True
    assert result["real_betting_enabled"] is False
    assert "selection_conformal" not in result
    assert result["closing_envelope_conformal"]["ready_folds"] == 1
    assert result["closing_envelope_conformal"]["training_races_latest"] == 30
    assert result["closing_envelope_conformal"]["training_observations_latest"] == 3600
    assert result["closing_envelope_conformal"]["missing_audit_rejected_races"] == 0
    assert result["folds"][0]["closing_envelope_races_appended_after_decision"] == 12
    assert result["folds"][0]["leakage_guard"]["closing_envelope_trained_through"] == "2026-07-29"
    assert "selection_conformal" not in result["folds"][0]["bankroll"]
    assert "closing_envelope_conformal" in result["folds"][0]["bankroll"]
    assert result["folds"][0]["selected_policy"]["real_betting_enabled"] is False
    deployment = result["deployment_configuration"]
    assert deployment["selected_policy"] == {"name": "no_bet", "no_bet": True}
    assert deployment["real_betting_enabled"] is False
    assert deployment["missing_real_t300_action"] == "no_bet"
    assert result["promotion_gate"]["closing_envelope_ready_pass"] is True
    assert result["promotion_gate"]["closing_envelope_no_missing_races_pass"] is True
    assert result["promotion_eligible"] is True


def test_v15_rejects_non_no_bet_closing_fallback():
    with pytest.raises(ValueError, match="requires closing_fallback_policy='no_bet'"):
        v15.walk_forward_evaluate_v15(
            [],
            daily_budget_yen=10_000,
            min_calibration_days=5,
            closing_fallback_policy=v12.CLOSING_FALLBACK_V11,
        )


def test_market_calibration_dispatcher_calls_v15_without_type_error(monkeypatch):
    captured = {}

    def fake_v12(races, **kwargs):
        del races
        captured.update(kwargs)
        return {
            "fixed_policy": {},
            "selection_conformal": {},
            "selection_conformal_artifacts_by_date": {},
            "folds": [],
            v15.V12_PROSPECTIVE_OUTPUT_KEY: {},
            "deployment_configuration": {},
        }

    monkeypatch.setattr(v15, "walk_forward_evaluate_v12", fake_v12)
    result = market_calibration.walk_forward_evaluate(
        [],
        daily_budget_yen=10_000,
        min_calibration_days=5,
        calibrator_strategy=(
            "odds_path_role_integrated_selection_free_envelope_v15"
        ),
        v12_closing_fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
    )

    assert captured["closing_fallback_policy"] == v12.CLOSING_FALLBACK_NO_BET
    assert captured["closing_forecast_field"] == "point_final_odds"
    assert result["real_betting_enabled"] is False


def test_ready_fit_with_one_missing_race_is_not_promotion_eligible(monkeypatch):
    observations = []
    for day in range(25, 30):
        for race_no in range(1, 7):
            observations.append({
                "race_date": f"2026-07-{day:02d}",
                "race_id": f"{day}-{race_no}",
                "predicted_closing_odds": _odds(10.0),
                "actual_closing_odds": _odds(9.0),
            })
    observations.append({
        "race_date": "2026-07-29",
        "race_id": "29-missing",
        "predicted_closing_odds": {},
        "actual_closing_odds": _odds(9.0),
    })
    artifact = v15._fit_closing_envelope(
        observations, evaluation_date="2026-07-30"
    )

    def fake_v12(races, **kwargs):
        del races, kwargs
        return {
            "fixed_policy": {},
            "selection_conformal": {},
            "selection_conformal_artifacts_by_date": {},
            "folds": [{
                "evaluation_date": "2026-07-30",
                "selection_conformal": artifact,
                "selected_policy": {"name": "policy"},
                "leakage_guard": {},
            }],
            v15.V12_PROSPECTIVE_OUTPUT_KEY: {
                "promotion_gate": {"base_roi_pass": True},
                "promotion_eligible": True,
            },
            "deployment_configuration": {},
        }

    monkeypatch.setattr(v15, "walk_forward_evaluate_v12", fake_v12)
    result = v15.walk_forward_evaluate_v15(
        [], daily_budget_yen=10_000, min_calibration_days=5
    )

    summary = result["closing_envelope_conformal"]
    assert artifact["ready"] is True
    assert artifact["training_races"] == 30
    assert artifact["training_observations"] == 3600
    assert summary["ready_folds"] == 1
    assert summary["missing_audit_input_races"] == 31
    assert summary["missing_audit_accepted_races"] == 30
    assert summary["missing_audit_rejected_races"] == 1
    gate = result["promotion_gate"]
    assert gate["closing_envelope_ready_pass"] is True
    assert gate["closing_envelope_no_missing_races_pass"] is False
    assert result["promotion_eligible"] is False


def test_six_complete_days_prewarm_is_ready_even_when_purchase_count_is_zero(
    monkeypatch,
):
    races = [
        _prewarm_race(
            f"2026-07-{day:02d}", race_no, checkpoint_odds=_odds(10.0)
        )
        for day in range(25, 31)
        for race_no in range(1, 7)
    ]

    def fake_v12(rows, **kwargs):
        initial = [dict(row) for row in kwargs["initial_selection_observations"]]
        assert len(initial) == 30
        artifact = kwargs["selection_conformal_fit"](
            initial, evaluation_date="2026-07-30"
        )
        assert artifact["ready"] is True
        holdout = [row for row in rows if row["race_date"] == "2026-07-30"]
        point_forecasts = {
            row["race_id"]: _odds(10.0) for row in holdout
        }
        appended = kwargs["selection_observation_append"](
            initial,
            holdout,
            closing_forecasts=point_forecasts,
            probability_lcb={"ready": False},
            evaluation_date="2026-07-30",
        )
        assert appended == 6
        assert len(initial) == 36
        return {
            "fixed_policy": {"name": "no_bet", "no_bet": True},
            "selection_conformal": {},
            "selection_conformal_artifacts_by_date": {
                "2026-07-30": artifact
            },
            "folds": [{
                "evaluation_date": "2026-07-30",
                "selection_conformal": artifact,
                "selection_observations_appended_after_decision": appended,
                "selected_policy": {"name": "no_bet", "no_bet": True},
                "leakage_guard": {
                    "selection_conformal_trained_through": "2026-07-29"
                },
                "bankroll": {
                    "tickets": 0,
                    "stake_yen": 0,
                    "selection_conformal": artifact,
                },
            }],
            v15.V12_PROSPECTIVE_OUTPUT_KEY: {
                "promotion_gate": {"base_roi_pass": True},
                "promotion_eligible": True,
            },
            "deployment_configuration": {},
        }

    monkeypatch.setattr(v15, "walk_forward_evaluate_v12", fake_v12)
    result = v15.walk_forward_evaluate_v15(
        races, daily_budget_yen=10_000, min_calibration_days=5
    )

    fold = result["folds"][-1]
    assert fold["closing_envelope_conformal"]["ready"] is True
    assert fold["closing_envelope_conformal"]["training_observations"] == 3600
    assert fold["bankroll"]["tickets"] == 0
    assert fold["bankroll"]["stake_yen"] == 0
    assert fold["closing_envelope_races_appended_after_decision"] == 6
    assert fold["selected_policy"]["no_bet"] is True
