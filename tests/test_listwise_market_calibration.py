from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.listwise.market_calibration import (
    artifact_drop_feature_groups,
    artifact_classifier_probabilities_batch,
    artifact_model_probabilities,
    blend_probabilities,
    filter_clean_market_days,
    fixed_benchmark_population,
    fit_deployment_configuration,
    iter_artifact_feature_rows,
    normalized_market_probabilities,
    probability_metrics,
    prepare_policy_matrix,
    policy_calibration_eligible,
    predefined_ticket_diagnostics,
    registered_evaluation_dates,
    summarize_registered_policy_daily,
    load_scored_cache,
    score_real_odds_races,
    select_calibrator,
    select_policy,
    simulate_policy,
    snapshot_age_seconds,
    write_scored_cache,
    walk_forward_evaluate,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _distribution(primary: str, probability: float) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: probability if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _race(race_date: str, rno: int, *, winner: str = "1-2-3") -> dict:
    market = _distribution(winner, 0.20)
    model = _distribution(winner, 0.35)
    odds = {combination: 1.0 / probability / 1.25 for combination, probability in market.items()}
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": winner,
        "actual_payout_yen": 500,
        "model_probabilities": model,
        "market_probabilities": market,
        "odds": odds,
        "snapshot_id": rno,
    }


def test_market_probabilities_remove_overround() -> None:
    probabilities = normalized_market_probabilities({"1-2-3": 2.0, "2-1-3": 4.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["1-2-3"] == pytest.approx(2.0 / 3.0)


def test_probability_metrics_include_winner_marginals() -> None:
    metrics = probability_metrics(
        [_race("2026-07-18", 1)],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
    )

    assert metrics["model_winner_log_loss"] is not None
    assert metrics["market_winner_log_loss"] is not None
    assert metrics["calibrated_winner_log_loss"] == pytest.approx(
        metrics["model_winner_log_loss"]
    )
    assert metrics["model_winner_top1_accuracy"] == 1.0
    assert metrics["market_winner_top1_accuracy"] == 1.0
    assert metrics["calibrated_winner_top1_accuracy"] == 1.0


def test_classifier_market_probabilities_batch_matches_single_race_scoring() -> None:
    class Hasher:
        def __init__(self) -> None:
            self.calls = 0

        def transform(self, rows):
            self.calls += 1
            return sparse.csr_matrix(
                [[float(dict(row)["score"])] for row in rows],
                dtype=np.float64,
            )

    class Classifier:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, matrix):
            positive = np.clip(matrix.toarray()[:, 0], 0.01, 0.99)
            return np.column_stack((1.0 - positive, positive))

    feature_races = [
        [
            {"features": {"score": 0.1 * lane}}
            for lane in range(1, 7)
        ]
        for _race in range(3)
    ]
    artifact = {
        "model": None,
        "model_kind": "lightgbm",
        "hasher": Hasher(),
        "classifier": Classifier(),
    }

    batch = artifact_classifier_probabilities_batch(artifact, feature_races)
    singles = [
        artifact_model_probabilities(artifact, feature_rows)
        for feature_rows in feature_races
    ]

    assert batch == singles
    assert all(sum(row.values()) == pytest.approx(1.0) for row in batch)
    assert artifact["hasher"].calls == 4


def test_geometric_blend_has_exact_endpoints() -> None:
    model = {"a": 0.8, "b": 0.2}
    market = {"a": 0.3, "b": 0.7}
    assert blend_probabilities(
        model, market, model_weight=0.0, temperature=1.0
    ) == pytest.approx(market)
    assert blend_probabilities(
        model, market, model_weight=1.0, temperature=1.0
    ) == pytest.approx(model)


def test_calibrator_is_selected_without_bankroll_outcomes() -> None:
    races = [_race("2026-07-18", index) for index in range(1, 13)]
    selected, candidates = select_calibrator(races)
    assert len(candidates) == 15
    assert selected["model_weight"] == 1.0
    assert selected["temperature"] == 0.75


def test_policy_falls_back_to_no_bet_when_every_candidate_loses() -> None:
    races = []
    for index in range(1, 21):
        race = _race("2026-07-18", index)
        race["actual_combination"] = "6-5-4"
        race["actual_payout_yen"] = 1_000
        races.append(race)
    selected, _ = select_policy(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=[
            {"name": "no_bet", "no_bet": True},
            {
                "name": "loser",
                "ev_threshold": 1.0,
                "max_odds": None,
                "max_tickets_per_race": 1,
                "min_model_market_ratio": 1.0,
            },
        ],
    )
    assert selected == {"name": "no_bet", "no_bet": True}


def test_vectorized_policy_candidates_match_reference_simulation() -> None:
    races = [_race("2026-07-18", index) for index in range(1, 7)]
    calibrator = {"model_weight": 0.75, "temperature": 1.0}
    policy = {
        "name": "equivalence",
        "ev_threshold": 1.05,
        "max_odds": 40.0,
        "max_tickets_per_race": 3,
        "min_model_market_ratio": 1.0,
        "staking_mode": "kelly_025",
        "max_estimated_ev": 1.10,
    }

    reference = simulate_policy(
        races,
        calibrator=calibrator,
        policy=policy,
        daily_budget_yen=10_000,
    )
    vectorized = simulate_policy(
        races,
        calibrator=calibrator,
        policy=policy,
        daily_budget_yen=10_000,
        prepared_policy_matrix=prepare_policy_matrix(races, calibrator),
    )

    assert vectorized == reference


def test_walk_forward_uses_only_strictly_earlier_dates_for_selection() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21")
        for rno in range(1, 13)
    ]
    result = walk_forward_evaluate(races, min_calibration_days=2)
    assert result["evaluation_days"] == 2
    assert result["evaluation_races"] == 24
    assert result["winner_log_loss"] is not None
    assert result["winner_top1_accuracy"] == 1.0
    assert [row["evaluation_date"] for row in result["folds"]] == [
        "2026-07-20",
        "2026-07-21",
    ]
    assert result["folds"][0]["calibration_dates"] == ["2026-07-18", "2026-07-19"]
    assert result["folds"][1]["calibration_dates"] == [
        "2026-07-18",
        "2026-07-19",
        "2026-07-20",
    ]
    assert result["flat_policy_walk_forward"]["evaluation_days"] == 2
    assert result["folds"][0]["selected_flat_policy"]["no_bet"] is True
    deployment = result["deployment_configuration"]
    assert deployment["role"] == "next_day_refit_not_evaluation"
    assert deployment["trained_through_date"] == "2026-07-21"
    assert deployment["training_races"] == 48
    assert deployment["calibrator_strategy"] == "grid"
    assert result["promotion_gate"]["sample_size_pass"] is False
    assert deployment["walk_forward_gate"]["evaluation_days"] == 2
    assert deployment["walk_forward_gate"]["days_pass"] is False
    assert deployment["walk_forward_gate"]["pass"] is False
    assert deployment["selected_policy"] == {"name": "no_bet", "no_bet": True}
    assert deployment["candidate_policy"]
    assert deployment["operational_status"] == "shadow_only_insufficient_evidence"
    assert result["promotion_eligible"] is False


def test_closing_return_strategy_uses_prequential_price_teacher() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
            "2026-07-21",
        )
        for rno in range(1, 13)
    ]

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_closing_return",
    )

    assert result["model"] == "odds_path_closing_return_v3"
    assert result["evaluation_days"] == 2
    for fold in result["folds"]:
        operational_model = fold["operational_model"]
        assert operational_model["return_price_basis"] == "forecast_closing"
        assert operational_model["return_multiplier_mode"] == (
            "historical_forecast_closing_to_payout_bucket"
        )
    deployment = result["deployment_configuration"]
    assert deployment["role"] == "next_day_refit_not_evaluation"
    assert deployment["trained_through_date"] == "2026-07-21"
    assert deployment["training_races"] == 48
    assert deployment["calibrator_strategy"] == "odds_path_closing_return"
    assert deployment["operational_model"]["return_price_basis"] == "forecast_closing"


def test_walk_forward_reports_clean_evaluation_day_waiting_state() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-20", "2026-07-21")
        for rno in range(1, 13)
    ]

    result = walk_forward_evaluate(races, min_calibration_days=2)

    assert result["status"] == "waiting_for_clean_evaluation_day"
    assert result["available_days"] == 2
    assert result["required_additional_days"] == 1
    assert result["evaluation_races"] == 0
    assert result["registered_ev_band_walk_forward"]["evaluation_days"] == 0
    assert result["promotion_eligible"] is False
    assert result["promotion_gate"]["no_lookahead_pass"] is True
    assert all(
        not value
        for key, value in result["promotion_gate"].items()
        if key.endswith("_pass") and key != "no_lookahead_pass"
    )


def test_newton_deployment_refits_all_completed_dates() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-20", "2026-07-21", "2026-07-22")
        for rno in range(1, 5)
    ]
    for index, race in enumerate(races):
        if index % 2:
            race["actual_combination"] = "6-5-4"

    deployment = fit_deployment_configuration(
        races,
        daily_budget_yen=10_000,
        calibrator_strategy="newton_residual",
    )

    assert deployment["trained_through_date"] == "2026-07-22"
    assert deployment["training_races"] == 12
    assert deployment["calibrator"]["iterations"] <= 50
    assert deployment["calibrator"]["training_races"] == 12


def test_scored_cache_requires_exact_contract(tmp_path) -> None:
    path = tmp_path / "scores.joblib"
    contract = {
        "version": 1,
        "model_sha256": "a" * 64,
        "trained_through": ("race", "2026-05-09", "24", 12),
        "from_date": "2026-07-18",
        "through_date": "2026-07-21",
    }
    races = [_race("2026-07-18", 1)]
    dataset = {"target_complete_races": 1, "eligible_real_odds_races": 1}
    write_scored_cache(
        path,
        contract=contract,
        races=races,
        dataset=dataset,
    )
    assert load_scored_cache(path, contract=contract) == (races, dataset)
    assert load_scored_cache(
        path,
        contract={**contract, "through_date": "2026-07-22"},
    ) is None


def test_policy_calibration_requires_repeatable_winning_days() -> None:
    result = {
        "race_days": 3,
        "winning_days": 1,
        "tickets": 30,
        "stake_yen": 3_000,
        "return_yen": 5_000,
        "profit_yen": 2_000,
        "roi": 5 / 3,
        "max_drawdown_yen": 1_000,
    }
    assert not policy_calibration_eligible(
        result,
        minimum_tickets=20,
        minimum_stake_yen=2_000,
    )
    assert policy_calibration_eligible(
        {**result, "winning_days": 2},
        minimum_tickets=20,
        minimum_stake_yen=2_000,
    )


def test_snapshot_age_is_measured_against_t5_boundary() -> None:
    snapshot = {
        "captured_at": "2026-07-21T11:54:20+09:00",
        "odds_deadline_at": "2026-07-21T11:55:00+09:00",
    }
    assert snapshot_age_seconds(snapshot) == 40.0
    assert snapshot_age_seconds({"captured_at": "bad"}) is None


def test_predefined_ticket_diagnostics_are_separate_from_policy_selection() -> None:
    result = predefined_ticket_diagnostics([_race("2026-07-20", 1)])
    strategies = result["strategies"]
    assert result["uses_only_evaluation_folds"] is True
    assert result["daily_budget_applied"] is False
    assert strategies["top5_flat"]["tickets"] == 5
    assert strategies["top5_flat"]["return_yen"] == 500
    assert strategies["top5_odds_gte_5"]["tickets"] == 4
    assert strategies["top5_odds_gte_5"]["return_yen"] == 0
    assert strategies["top5_ev_gte_1"]["tickets"] == 1
    assert strategies["top5_ev_gte_1"]["roi"] == 5.0


def test_market_scoring_uses_artifact_feature_exclusions(monkeypatch) -> None:
    observed = {}

    def fake_rows(conn, *, include_races, drop_feature_groups, feature_schema_version):
        observed.update(
            conn=conn,
            include_races=include_races,
            drop_feature_groups=drop_feature_groups,
            feature_schema_version=feature_schema_version,
        )
        return iter(())

    monkeypatch.setattr(
        "boatrace_ai.listwise.market_calibration.iter_race_feature_rows",
        fake_rows,
    )
    assert list(
        iter_artifact_feature_rows(
            "connection",
            target_ids={"race-1"},
            artifact={"drop_feature_groups": ["base_pastlog"]},
        )
    ) == []
    assert observed == {
        "conn": "connection",
        "include_races": {"race-1"},
        "drop_feature_groups": ("base_pastlog",),
        "feature_schema_version": None,
    }
    assert artifact_drop_feature_groups({}) == ()


@pytest.mark.parametrize("model_kind", ["linear", "mlp", "lightgbm"])
def test_real_odds_scoring_accepts_classifier_artifact(
    monkeypatch, model_kind: str
) -> None:
    monkeypatch.setattr(
        "boatrace_ai.listwise.market_calibration.load_complete_race_ids",
        lambda _conn: [],
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.market_calibration._load_trifecta_payouts",
        lambda _conn: {},
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.market_calibration.iter_artifact_feature_rows",
        lambda *_args, **_kwargs: iter(()),
    )

    races, dataset = score_real_odds_races(
        "connection",
        artifact={
            "classifier": object(),
            "hasher": object(),
            "model_kind": model_kind,
            "trained_through": ("race-id", "2026-07-17", "24", 12),
        },
        from_date="2026-07-18",
        through_date="2026-07-24",
    )

    assert races == []
    assert dataset["target_complete_races"] == 0
    assert dataset["eligible_real_odds_races"] == 0


def test_clean_day_gate_excludes_partial_t5_and_missing_payout_days() -> None:
    races = [
        {"race_date": "2026-07-20", "race_id": "a"},
        {"race_date": "2026-07-20", "race_id": "b"},
        {"race_date": "2026-07-21", "race_id": "c"},
        {"race_date": "2026-07-22", "race_id": "d"},
        {"race_date": "2026-07-22", "race_id": "e"},
    ]
    targets = {
        "2026-07-20": {"complete_race_count": 2, "payout_race_count": 2},
        "2026-07-21": {"complete_race_count": 2, "payout_race_count": 2},
        "2026-07-22": {"complete_race_count": 2, "payout_race_count": 1},
    }

    clean, gate = filter_clean_market_days(
        races,
        day_targets=targets,
        minimum_day_coverage=1.0,
    )

    assert [row["race_id"] for row in clean] == ["a", "b"]
    assert gate["clean_dates"] == ["2026-07-20"]
    assert gate["clean_days"] == 1
    assert gate["excluded_days"] == 2
    assert gate["days"][1]["coverage"] == 0.5
    assert gate["days"][2]["payout_complete"] is False


def test_partial_t5_days_calibrate_only_clean_evaluation_day() -> None:
    races = [
        *[_race("2026-07-20", rno) for rno in range(1, 3)],
        *[_race("2026-07-21", rno) for rno in range(1, 4)],
        *[_race("2026-07-22", rno) for rno in range(1, 5)],
    ]
    targets = {
        race_date: {"complete_race_count": 4, "payout_race_count": 4}
        for race_date in ("2026-07-20", "2026-07-21", "2026-07-22")
    }

    clean, gate = filter_clean_market_days(
        races,
        day_targets=targets,
        minimum_day_coverage=1.0,
    )
    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=gate["clean_dates"],
    )

    assert len(clean) == 4
    assert gate["clean_dates"] == ["2026-07-22"]
    assert result["evaluation_candidate_dates"] == ["2026-07-22"]
    assert result["evaluation_days"] == 1
    assert result["evaluation_races"] == 4
    assert result["available_races"] == 9
    assert result["folds"][0]["calibration_dates"] == [
        "2026-07-20",
        "2026-07-21",
    ]
    assert result["folds"][0]["calibration_races"] == 5
    assert result["folds"][0]["evaluation_date"] == "2026-07-22"
    assert result["deployment_configuration"]["training_races"] == 9
    assert max(result["folds"][0]["calibration_dates"]) < "2026-07-22"
def test_fixed_benchmark_population_is_provisional_until_seven_days() -> None:


    targets = {
        f"2026-07-{day:02d}": {
            "complete_race_count": 100,
            "payout_race_count": 100,
        }
        for day in range(22, 29)
    }
    races = [
        {"race_date": f"2026-07-{day:02d}"}
        for day in range(22, 29)
        for _ in range(10)
    ]

    provisional = fixed_benchmark_population(
        races,
        day_targets=targets,
        evaluation_dates=[f"2026-07-{day:02d}" for day in range(22, 26)],
    )
    final = fixed_benchmark_population(
        races,
        day_targets=targets,
        evaluation_dates=[f"2026-07-{day:02d}" for day in range(22, 29)],
    )

    assert provisional["benchmark_status"] == "provisional"
    assert provisional["benchmark_days"] == 4
    assert provisional["benchmark_population_races"] == 400
    assert provisional["benchmark_odds_eligible_races"] == 40
    assert final["benchmark_status"] == "final"
    assert final["benchmark_days"] == 7
    assert final["benchmark_population_races"] == 700




def test_provisional_evaluation_starts_with_first_four_clean_days() -> None:
    assert registered_evaluation_dates(
        ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-25"]
    ) == ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-25"]
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        registered_evaluation_dates(["2026-07-24"], valid_from="bad")


def test_clean_day_gate_validates_coverage_threshold() -> None:
    with pytest.raises(ValueError, match="minimum_day_coverage"):
        filter_clean_market_days(
            [],
            day_targets={},
            minimum_day_coverage=0.0,
        )


def test_empty_registered_policy_summary_waits_for_unseen_day() -> None:
    result = summarize_registered_policy_daily([], evaluated_races=0)

    assert result["status"] == "waiting_for_first_unseen_day"
    assert result["evaluation_days"] == 0
    assert result["evaluated_races"] == 0


def test_registered_ev_band_uses_only_days_after_hypothesis_registration() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-23",
            "2026-07-24",
            "2026-07-25",
            "2026-07-26",
        )
        for rno in range(1, 4)
    ]

    result = walk_forward_evaluate(races, min_calibration_days=2)
    registered = result["registered_ev_band_walk_forward"]

    assert registered["registered_after"] == "2026-07-25"
    assert registered["status"] == "evaluating"
    assert registered["evaluation_days"] == 1
    assert registered["evaluated_races"] == 3
    assert [row["race_date"] for row in registered["daily"]] == ["2026-07-26"]
    assert result["folds"][0]["registered_ev_band_bankroll"] is None
    assert result["folds"][1]["registered_ev_band_bankroll"] is not None


def test_prospective_normalized_ev_uses_only_unseen_days_after_registration() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-24",
            "2026-07-25",
            "2026-07-26",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        )
        for rno in range(1, 4)
    ]

    result = walk_forward_evaluate(races, min_calibration_days=2)
    prospective = result["prospective_normalized_ev_walk_forward"]

    assert prospective["registered_after"] == "2026-07-27"
    assert prospective["status"] == "evaluating"
    assert prospective["evaluation_days"] == 2
    assert prospective["evaluated_races"] == 6
    assert [row["race_date"] for row in prospective["daily"]] == [
        "2026-07-28",
        "2026-07-29",
    ]
    folds = {row["evaluation_date"]: row for row in result["folds"]}
    assert folds["2026-07-27"]["prospective_normalized_ev_bankroll"] is None
    assert folds["2026-07-28"]["prospective_normalized_ev_bankroll"] is not None
    top5 = result["prospective_top5_narrow_ev_walk_forward"]
    assert top5["registered_after"] == "2026-07-28"
    assert top5["evaluation_days"] == 1
    assert top5["evaluated_races"] == 3
    assert [row["race_date"] for row in top5["daily"]] == ["2026-07-29"]
    assert folds["2026-07-28"]["prospective_top5_narrow_ev_bankroll"] is None
    assert folds["2026-07-29"]["prospective_top5_narrow_ev_bankroll"] is not None
