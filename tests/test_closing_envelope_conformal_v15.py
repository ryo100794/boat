from __future__ import annotations

from copy import deepcopy
from itertools import permutations

import pytest

from boatrace_ai.listwise.closing_envelope_conformal_v15 import (
    METHOD,
    apply_closing_envelope_haircut_v15,
    artifact_fingerprint_v15,
    evaluate_closing_envelope_holdout_v15,
    extract_closing_ratio_observations_v15,
    fit_closing_envelope_conformal_v15,
)


COMBINATIONS = tuple(
    "".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(day: int, rno: int, ratio: float) -> dict:
    predicted = {
        combination: 10.0 + index / 10
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_date": f"2026-07-{day:02d}",
        "race_id": f"202607{day:02d}01{rno:02d}",
        "predicted_closing_odds": predicted,
        "closing_odds": {
            combination: odds * ratio
            for combination, odds in predicted.items()
        },
        "actual_combination": "123",
        "actual_payout_yen": 999_999,
        "purchase_candidates": ["123"],
    }


def _training_races() -> list[dict]:
    return [
        _race(day, rno, ratio=0.40 + day / 100)
        for day in range(20, 25)
        for rno in range(1, 7)
    ]


def test_fit_uses_every_combination_and_is_selection_free() -> None:
    races = _training_races()
    artifact = fit_closing_envelope_conformal_v15(
        races, evaluation_date="2026-07-25"
    )

    assert artifact["ready"] is True
    assert artifact["method"] == METHOD
    assert artifact["selection_free"] is True
    assert artifact["training_days"] == 5
    assert artifact["training_races"] == 30
    assert artifact["training_observations"] == 30 * 120
    assert artifact["daily_observations"] == {
        f"2026-07-{day:02d}": 6 * 120 for day in range(20, 25)
    }
    assert artifact["haircut"] == pytest.approx(0.60)

    changed = deepcopy(races)
    for race in changed:
        race["purchase_candidates"] = list(COMBINATIONS)
        race["actual_combination"] = "654"
        race["actual_payout_yen"] = 1
    changed_artifact = fit_closing_envelope_conformal_v15(
        changed, evaluation_date="2026-07-25"
    )
    assert artifact_fingerprint_v15(artifact) == artifact_fingerprint_v15(
        changed_artifact
    )


def test_strict_prior_rejects_same_and_future_days() -> None:
    for day in (25, 26):
        with pytest.raises(ValueError, match="strict-prior"):
            fit_closing_envelope_conformal_v15(
                [_race(day, 1, 0.8)], evaluation_date="2026-07-25",
                minimum_training_days=1, minimum_training_races=1,
            )


def test_input_order_and_mapping_order_are_reproducible() -> None:
    races = _training_races()
    reversed_races = []
    for race in reversed(races):
        changed = deepcopy(race)
        changed["predicted_closing_odds"] = dict(
            reversed(list(changed["predicted_closing_odds"].items()))
        )
        changed["closing_odds"] = dict(
            reversed(list(changed["closing_odds"].items()))
        )
        reversed_races.append(changed)

    first = fit_closing_envelope_conformal_v15(
        races, evaluation_date="2026-07-25"
    )
    second = fit_closing_envelope_conformal_v15(
        reversed_races, evaluation_date="2026-07-25"
    )
    assert first == second
    assert artifact_fingerprint_v15(first) == artifact_fingerprint_v15(second)


def test_missing_or_invalid_combination_rejects_whole_race_and_is_audited() -> None:
    complete = _race(20, 1, 0.8)
    missing = _race(20, 2, 0.8)
    missing["closing_odds"].pop("123")
    invalid = _race(20, 3, 0.8)
    invalid["predicted_closing_odds"]["456"] = float("nan")

    extracted = extract_closing_ratio_observations_v15(
        [invalid, complete, missing], evaluation_date="2026-07-21"
    )
    audit = extracted["audit"]

    assert len(extracted["observations"]) == 120
    assert audit["input_races"] == 3
    assert audit["accepted_races"] == 1
    assert audit["rejected_races"] == 2
    assert audit["rejected_observation_slots"] == 240
    assert audit["complete"] is False
    assert audit["rejection_reasons"] == {
        "incomplete_actual_closing_odds": 1,
        "invalid_predicted_closing_odds": 1,
    }
    by_id = {row["race_id"]: row for row in audit["races"]}
    assert by_id[missing["race_id"]]["actual"]["missing_values"] == 1
    assert by_id[invalid["race_id"]]["predicted"]["invalid_values"] == 1


@pytest.mark.parametrize(
    ("days", "races_per_day", "minimum_observations"),
    ((4, 8, 1), (5, 5, 1), (5, 6, 3601)),
)
def test_minimum_day_race_and_observation_gates(
    days: int, races_per_day: int, minimum_observations: int
) -> None:
    races = [
        _race(20 + day, rno, 0.8)
        for day in range(days)
        for rno in range(1, races_per_day + 1)
    ]
    artifact = fit_closing_envelope_conformal_v15(
        races,
        evaluation_date="2026-07-30",
        minimum_training_observations=minimum_observations,
    )
    assert artifact["ready"] is False
    assert artifact["haircut"] is None
    assert artifact["reason"] == "insufficient_strict_prior_complete_closing_data"


def test_strict_holdout_coverage_uses_all_120_and_ignores_result_payout() -> None:
    artifact = fit_closing_envelope_conformal_v15(
        _training_races(), evaluation_date="2026-07-25"
    )
    holdout = _race(25, 1, ratio=0.70)
    for combination in COMBINATIONS[:24]:
        holdout["closing_odds"][combination] = (
            holdout["predicted_closing_odds"][combination] * 0.50
        )

    first = evaluate_closing_envelope_holdout_v15(
        [holdout], artifact=artifact, evaluation_date="2026-07-25"
    )
    changed = deepcopy(holdout)
    changed["actual_combination"] = "654"
    changed["actual_payout_yen"] = 1
    changed["purchase_candidates"] = list(COMBINATIONS)
    second = evaluate_closing_envelope_holdout_v15(
        [changed], artifact=artifact, evaluation_date="2026-07-25"
    )

    assert first == second
    assert first["evaluated_observations"] == 120
    assert first["covered_observations"] == 96
    assert first["coverage"] == pytest.approx(0.80)
    assert first["target_coverage"] == pytest.approx(0.80)
    assert first["coverage_pass"] is True
    assert first["complete"] is True
    assert first["result_used_for_decision"] is False
    assert first["payout_used_for_decision"] is False
    assert first["actual_closing_odds_role"] == (
        "evaluation_only_after_purchase_decision"
    )


def test_strict_holdout_rejects_incomplete_race_atomically() -> None:
    artifact = fit_closing_envelope_conformal_v15(
        _training_races(), evaluation_date="2026-07-25"
    )
    holdout = _race(25, 1, ratio=0.90)
    holdout["closing_odds"].pop(COMBINATIONS[-1])

    result = evaluate_closing_envelope_holdout_v15(
        [holdout], artifact=artifact, evaluation_date="2026-07-25"
    )

    assert result["input_races"] == 1
    assert result["accepted_races"] == 0
    assert result["rejected_races"] == 1
    assert result["evaluated_observations"] == 0
    assert result["missing_observations"] == 120
    assert result["coverage"] is None
    assert result["coverage_pass"] is False
    assert result["complete"] is False
    assert result["rejection_reasons"] == {
        "incomplete_actual_closing_odds": 1
    }


def test_haircut_applies_to_scalar_and_mapping_without_mutation() -> None:
    artifact = fit_closing_envelope_conformal_v15(
        _training_races(), evaluation_date="2026-07-25"
    )
    forecast = {"2-1-3": 30.0, "1-2-3": 20.0}
    original = dict(forecast)

    assert apply_closing_envelope_haircut_v15(20.0, artifact) == pytest.approx(
        12.0
    )
    assert apply_closing_envelope_haircut_v15(forecast, artifact) == {
        "1-2-3": pytest.approx(12.0),
        "2-1-3": pytest.approx(18.0),
    }
    assert forecast == original


def test_incomplete_data_cannot_be_hidden_by_minimum_count_override() -> None:
    race = _race(20, 1, 0.8)
    race["closing_odds"].pop("123")
    artifact = fit_closing_envelope_conformal_v15(
        [race],
        evaluation_date="2026-07-21",
        minimum_training_days=1,
        minimum_training_races=1,
        minimum_training_observations=1,
    )

    assert artifact["ready"] is False
    assert artifact["training_races"] == 0
    assert artifact["training_observations"] == 0
    assert artifact["missing_audit"]["rejected_races"] == 1
