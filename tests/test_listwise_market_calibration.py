from __future__ import annotations

from datetime import date, timedelta
from itertools import permutations

import numpy as np
import boatrace_ai.listwise.market_calibration as market_calibration
import pytest
from scipy import sparse

from boatrace_ai.listwise.market_calibration import (
    _v6_selection_key,
    artifact_drop_feature_groups,
    artifact_classifier_probabilities_batch,
    artifact_model_probabilities,
    blend_scored_model_probabilities,
    blend_probabilities,
    build_parser,
    evaluation_dates_for_role,
    filter_clean_market_days,
    fixed_benchmark_population,
    fit_deployment_configuration,
    geometric_blend_model_probabilities,
    iter_artifact_feature_rows,
    normalized_market_probabilities,
    probability_metrics,
    prepare_policy_matrix,
    policy_calibration_eligible,
    predefined_ticket_diagnostics,
    registered_evaluation_dates,
    research_holdout_coverage_gate,
    summarize_registered_policy_daily,
    load_scored_cache,
    odds_path_model_name,
    score_real_odds_races,
    select_calibrator,
    select_policy,
    select_policy_v17,
    select_return_shrinkage_prequential,
    simulate_policy,
    v17_policy_ranking_key,
    snapshot_age_seconds,
    write_scored_cache,
    walk_forward_evaluate,
    validate_fixed_model_blend,
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


def test_v4_market_policy_adds_chronological_adaptive_metrics_compatibly() -> None:
    races = [_race("2026-07-18", index) for index in range(1, 4)]
    result = simulate_policy(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={
            "name": "v4-compatible-policy",
            "ev_threshold": 1.0,
            "max_estimated_ev": None,
            "max_odds": None,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.0,
            "staking_mode": "kelly_025",
        },
        daily_budget_yen=10_000,
        include_chronological=True,
    )

    legacy = result["daily"][0]
    chronological = legacy["chronological_bankroll"]
    aggregate = result["chronological_bankroll"]

    assert result["tickets"] == legacy["tickets"]
    assert result["stake_yen"] == legacy["stake_yen"]
    assert result["return_yen"] == legacy["return_yen"]
    assert chronological["allocation_method"].startswith(
        "chronological_adaptive_"
    )
    assert chronological["profit_reinvestment"] is True
    assert chronological["stake_granularity_yen"] == 100
    assert chronological["real_betting_enabled"] is False
    assert aggregate["daily"] == [chronological]
    assert aggregate["stake_yen"] == chronological["stake_yen"]
    assert aggregate["return_yen"] == chronological["return_yen"]
    decisions = [
        row for row in chronological["ledger"] if row["event"] == "decision"
    ]
    assert [row["race_id"] for row in decisions] == [
        race["race_id"] for race in races
    ]
    assert decisions[1]["outstanding_stake_yen"] >= decisions[0]["stake_yen"]


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


def test_min_raw_ev_rejects_return_multiplier_only_edges() -> None:
    race = _race("2026-07-18", 1)
    race["_policy_calibrated_probabilities"] = dict(
        race["market_probabilities"]
    )
    race["historical_return_multipliers"] = {
        combination: 2.0 for combination in COMBINATIONS
    }
    policy = {
        "name": "adjusted-only-edge",
        "ev_threshold": 1.5,
        "max_estimated_ev": None,
        "max_odds": None,
        "max_tickets_per_race": 1,
        "min_model_market_ratio": 1.0,
        "staking_mode": "kelly_100",
    }

    adjusted_only = simulate_policy(
        [race], calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy=policy, daily_budget_yen=10_000,
    )
    raw_guarded = simulate_policy(
        [race], calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={**policy, "min_raw_ev": 1.0}, daily_budget_yen=10_000,
    )
    vectorized = simulate_policy(
        [race], calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={**policy, "min_raw_ev": 1.0}, daily_budget_yen=10_000,
        prepared_policy_matrix=prepare_policy_matrix(
            [race], {"model_weight": 1.0, "temperature": 1.0}
        ),
    )

    assert adjusted_only["tickets"] == 1
    assert raw_guarded["tickets"] == 0
    assert vectorized == raw_guarded


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


def test_observed_closing_return_strategy_uses_prior_closing_teachers() -> None:
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
    for race in races:
        race["closing_odds"] = {
            combination: odds * 0.9 for combination, odds in race["odds"].items()
        }
        race["closing_odds_changed"] = True

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_observed_closing_return",
    )

    assert result["model"] == "odds_path_observed_closing_return_v4"
    assert result["evaluation_days"] == 2
    for fold in result["folds"]:
        operational_model = fold["operational_model"]
        chronological = fold["bankroll"]["chronological_bankroll"]
        assert operational_model["return_price_basis"] == "observed_closing"
        assert operational_model["return_multiplier_mode"] == (
            "historical_observed_closing_to_payout_bucket"
        )
        assert chronological["race_days"] == 1
        assert chronological["profit_reinvestment"] is True
        assert chronological["stake_granularity_yen"] == 100
        assert chronological["real_betting_enabled"] is False
    prospective = result[
        "prospective_observed_closing_return_v4_walk_forward"
    ]
    assert prospective["status"] == "waiting_for_first_unseen_day"
    assert prospective["registered_after"] == "2026-07-29"
    assert prospective["evaluation_days"] == 0
    assert result["promotion_gate"]["prospective_architecture_pass"] is False


def test_observed_closing_v4_counts_only_dates_after_registration() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        )
        for rno in range(1, 13)
    ]
    for race in races:
        race["closing_odds"] = dict(race["odds"])
        race["closing_odds_changed"] = True

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_observed_closing_return",
    )
    prospective = result[
        "prospective_observed_closing_return_v4_walk_forward"
    ]

    assert prospective["status"] == "evaluating"
    assert prospective["evaluation_days"] == 2
    assert prospective["evaluated_races"] == 24
    assert [row["race_date"] for row in prospective["daily"]] == [
        "2026-07-30",
        "2026-07-31",
    ]
    assert result["promotion_gate"]["prospective_architecture_pass"] is False
    assert result["deployment_configuration"]["walk_forward_gate"][
        "prospective_architecture_pass"
    ] is False


def test_non_v4_strategy_does_not_publish_v4_prospective_metrics() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        )
        for rno in range(1, 13)
    ]

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_probability",
    )

    assert "prospective_observed_closing_return_v4_walk_forward" not in result


def test_hit_shrunk_strategy_applies_conservative_prior_in_every_fold() -> None:
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
    for race in races:
        race["closing_odds"] = {
            combination: odds * 0.9 for combination, odds in race["odds"].items()
        }
        race["closing_odds_changed"] = True

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_hit_shrunk_return",
    )

    assert result["model"] == "odds_path_hit_shrunk_closing_return_v5"
    for fold in result["folds"]:
        operational_model = fold["operational_model"]
        assert operational_model["return_hit_prior"] == 20.0
        assert operational_model["return_multiplier_bounds"] == [0.5, 1.5]
        assert max(
            row["return_multiplier"]
            for row in operational_model["performance_priors"]["buckets"].values()
        ) <= 1.5


def test_v6_uses_conservative_fallback_when_inner_history_is_short() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-18", "2026-07-19", "2026-07-20")
        for rno in range(1, 5)
    ]

    selection = select_return_shrinkage_prequential(
        races,
        daily_budget_yen=10_000,
    )

    assert selection["status"] == "conservative_fallback"
    assert selection["fallback_reason"] == "insufficient_history_days"
    assert selection["selected"] == {
        "return_hit_prior": 20.0,
        "min_return_multiplier": 0.75,
        "max_return_multiplier": 1.25,
    }


def test_v6_selection_rule_prioritizes_largest_hit_excluded_roi() -> None:
    robust = {
        "roi_without_largest_hit": 1.01,
        "profitable_day_fraction": 0.4,
        "median_profit_per_day_yen": -100.0,
        "tickets": 20,
        "return_hit_prior": 20.0,
        "min_return_multiplier": 0.9,
        "max_return_multiplier": 1.1,
    }
    unstable_high_profit = {
        **robust,
        "roi_without_largest_hit": 0.99,
        "profitable_day_fraction": 1.0,
        "median_profit_per_day_yen": 10_000.0,
        "tickets": 200,
    }

    assert _v6_selection_key(robust) > _v6_selection_key(unstable_high_profit)


def test_v6_inner_prequential_grid_selects_robust_candidate(
    monkeypatch,
) -> None:
    import boatrace_ai.listwise.market_calibration as module
    import boatrace_ai.listwise.market_residual as residual

    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
            "2026-07-21",
        )
        for rno in range(1, 3)
    ]
    fit_calls = []

    def fake_fit(training, **_kwargs):
        fit_calls.append(sorted({race["race_date"] for race in training}))
        return {"performance_priors": {"candidate_prior": 0.0}}

    def fake_priors(_training, *, return_hit_prior, **_kwargs):
        return {"candidate_prior": float(return_hit_prior)}

    def fake_attach(rows, model):
        prior = float(model["performance_priors"]["candidate_prior"])
        return [{**race, "candidate_prior": prior} for race in rows]

    def fake_simulate(rows, **_kwargs):
        prior = rows[0]["candidate_prior"]
        returned = 700 if prior == 20.0 else 1_000
        largest = 100 if prior == 20.0 else 900
        return {
            "daily": [
                {
                    "tickets": 5,
                    "stake_yen": 500,
                    "return_yen": returned,
                    "profit_yen": returned - 500,
                    "largest_hit_return_yen": largest,
                }
            ]
        }

    monkeypatch.setattr(module, "V6_RETURN_HIT_PRIORS", (0.0, 20.0))
    monkeypatch.setattr(
        module,
        "V6_RETURN_MULTIPLIER_BOUNDS",
        ((0.75, 1.25),),
    )
    monkeypatch.setattr(module, "fit_odds_path_model", fake_fit)
    monkeypatch.setattr(module, "fit_performance_priors", fake_priors)
    monkeypatch.setattr(module, "attach_odds_path_model", fake_attach)
    monkeypatch.setattr(
        module,
        "prequential_closing_odds_policy_inputs",
        lambda _rows: {},
    )
    monkeypatch.setattr(
        module,
        "apply_prequential_closing_odds_policy_inputs",
        lambda rows, _inputs: rows,
    )
    monkeypatch.setattr(module, "simulate_policy", fake_simulate)
    monkeypatch.setattr(
        residual,
        "select_regularization_prequential",
        lambda _rows: {
            "final_calibrator": {"model_weight": 1.0, "temperature": 1.0}
        },
    )

    selection = select_return_shrinkage_prequential(
        races,
        daily_budget_yen=10_000,
    )

    assert selection["status"] == "selected"
    assert selection["selected"] == {
        "return_hit_prior": 20.0,
        "min_return_multiplier": 0.75,
        "max_return_multiplier": 1.25,
    }
    assert selection["inner_evaluation_dates"] == [
        "2026-07-20",
        "2026-07-21",
    ]
    assert fit_calls == [
        ["2026-07-18", "2026-07-19"],
        ["2026-07-18", "2026-07-19", "2026-07-20"],
    ]
    assert len(selection["candidates"]) == 2


def test_v6_outer_folds_never_pass_holdout_to_inner_selection(
    monkeypatch,
) -> None:
    import boatrace_ai.listwise.market_calibration as module

    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
            "2026-07-21",
        )
        for rno in range(1, 5)
    ]
    for race in races:
        race["closing_odds"] = dict(race["odds"])
        race["closing_odds_changed"] = True
    seen_dates = []

    def fake_selection(training_races, *, daily_budget_yen):
        assert daily_budget_yen == 10_000
        seen_dates.append(sorted({race["race_date"] for race in training_races}))
        return {
            "version": 1,
            "status": "conservative_fallback",
            "selected": {
                "return_hit_prior": 20.0,
                "min_return_multiplier": 0.75,
                "max_return_multiplier": 1.25,
            },
        }

    monkeypatch.setattr(
        module,
        "select_return_shrinkage_prequential",
        fake_selection,
    )

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="odds_path_prequential_shrinkage_return",
    )

    assert result["model"] == "odds_path_prequential_shrinkage_return_v6"
    assert seen_dates[0] == ["2026-07-18", "2026-07-19"]
    assert seen_dates[1] == ["2026-07-18", "2026-07-19", "2026-07-20"]
    assert seen_dates[2] == [
        "2026-07-18",
        "2026-07-19",
        "2026-07-20",
        "2026-07-21",
    ]
    assert all(
        fold["operational_model"]["model_type"]
        == "odds_path_prequential_shrinkage_return_v6"
        for fold in result["folds"]
    )
    deployment = result["deployment_configuration"]
    assert deployment["calibrator_strategy"] == (
        "odds_path_prequential_shrinkage_return"
    )
    assert deployment["operational_model"]["adaptive_return_selection"][
        "status"
    ] == "conservative_fallback"
    prospective = result[
        "prospective_prequential_shrinkage_return_v6_walk_forward"
    ]
    assert prospective["registered_after"] == "2026-07-29"
    assert prospective["status"] == "waiting_for_first_unseen_day"
    assert prospective["evaluation_days"] == 0
    assert result["promotion_gate"]["prospective_architecture_pass"] is False
    assert deployment["walk_forward_gate"][
        "prospective_architecture_pass"
    ] is False


def test_cli_accepts_v6_calibrator_strategy() -> None:
    args = build_parser().parse_args(
        [
            "--from-date",
            "2026-07-18",
            "--calibrator-strategy",
            "odds_path_prequential_shrinkage_return",
        ]
    )

    assert args.calibrator_strategy == "odds_path_prequential_shrinkage_return"


def test_cli_accepts_v7_calibrator_strategy() -> None:
    args = build_parser().parse_args(
        [
            "--from-date",
            "2026-07-18",
            "--calibrator-strategy",
            "odds_path_crossfit_conservative_ev",
        ]
    )

    assert args.calibrator_strategy == "odds_path_crossfit_conservative_ev"


def test_cli_accepts_v8_market_offset_strategy() -> None:
    args = build_parser().parse_args(
        [
            "--from-date",
            "2026-07-18",
            "--calibrator-strategy",
            "odds_path_market_offset_crossfit_conservative_ev",
        ]
    )

    assert args.calibrator_strategy == (
        "odds_path_market_offset_crossfit_conservative_ev"
    )


def test_walk_forward_dispatches_v8_market_offset_strategy(monkeypatch) -> None:
    import boatrace_ai.listwise.odds_path_conservative_v8 as v8

    expected = {"model": v8.MODEL_NAME, "dispatch": "v8"}
    seen = {}

    def fake_walk_forward(races, **kwargs):
        seen["races"] = races
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(v8, "walk_forward_evaluate_v8", fake_walk_forward)
    races = [_race("2026-07-30", 1)]

    result = walk_forward_evaluate(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=3,
        calibrator_strategy=(
            "odds_path_market_offset_crossfit_conservative_ev"
        ),
        evaluation_dates=("2026-07-30",),
    )

    assert result is expected
    assert seen == {
        "races": races,
        "daily_budget_yen": 10_000,
        "min_calibration_days": 3,
        "evaluation_dates": ("2026-07-30",),
    }


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


def test_fixed_model_geometric_blend_preserves_endpoints_and_normalizes() -> None:
    candidate = {"1-2-3": 0.8, "2-1-3": 0.2}
    baseline = {"1-2-3": 0.2, "2-1-3": 0.8}

    assert geometric_blend_model_probabilities(
        candidate, baseline, candidate_weight=0.0
    ) == baseline
    assert geometric_blend_model_probabilities(
        candidate, baseline, candidate_weight=1.0
    ) == candidate
    midpoint = geometric_blend_model_probabilities(
        candidate, baseline, candidate_weight=0.5
    )
    assert midpoint == pytest.approx({"1-2-3": 0.5, "2-1-3": 0.5})
    assert sum(midpoint.values()) == pytest.approx(1.0)


def test_fixed_model_blend_requires_paired_cli_arguments() -> None:
    assert validate_fixed_model_blend(None, None) is None
    assert validate_fixed_model_blend("baseline.joblib", 0.25) == 0.25
    with pytest.raises(ValueError, match="provided together"):
        validate_fixed_model_blend("baseline.joblib", None)
    with pytest.raises(ValueError, match="provided together"):
        validate_fixed_model_blend(None, 0.5)
    with pytest.raises(ValueError, match="finite and in"):
        validate_fixed_model_blend("baseline.joblib", float("nan"))


def test_fixed_model_scored_blend_validates_full_race_identity() -> None:
    candidate = _race("2026-07-18", 1)
    baseline = _race("2026-07-18", 1)
    baseline["model_probabilities"] = _distribution("2-1-3", 0.40)
    dataset = {"target_complete_races": 1, "eligible_real_odds_races": 1}

    blended = blend_scored_model_probabilities(
        [candidate],
        [baseline],
        candidate_weight=0.5,
        candidate_dataset=dataset,
        baseline_dataset=dict(dataset),
    )
    assert blended[0]["race_id"] == candidate["race_id"]
    assert sum(blended[0]["model_probabilities"].values()) == pytest.approx(1.0)
    assert candidate["model_probabilities"] != blended[0]["model_probabilities"]

    mismatched_odds = {**baseline, "odds": dict(baseline["odds"])}
    mismatched_odds["odds"]["1-2-3"] += 0.1
    with pytest.raises(ValueError, match="race data differ.*odds"):
        blend_scored_model_probabilities(
            [candidate], [mismatched_odds], candidate_weight=0.5
        )
    with pytest.raises(ValueError, match="race_id sets differ"):
        blend_scored_model_probabilities(
            [candidate], [{**baseline, "race_id": "other"}], candidate_weight=0.5
        )
    with pytest.raises(ValueError, match="scored datasets differ"):
        blend_scored_model_probabilities(
            [candidate],
            [baseline],
            candidate_weight=0.5,
            candidate_dataset=dataset,
            baseline_dataset={**dataset, "eligible_real_odds_races": 0},
        )


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


def test_trend_point_odds_safety_sweep_is_explicit_and_diagnostic_only() -> None:
    races = [
        _race("2026-07-20", 1),
        _race("2026-07-21", 1),
        _race("2026-07-22", 1),
    ]

    disabled = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=["2026-07-22"],
    )
    enabled = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=["2026-07-22"],
        trend_point_odds_safety_sweep=True,
    )

    assert disabled["trend_point_odds_safety_sweep"] is None
    sweep = enabled["trend_point_odds_safety_sweep"]
    assert sweep["status"] == "diagnostic_only_not_promotion_evidence"
    assert sweep["factors"] == [1.0, 1.05, 1.10, 1.15, 1.20]
    assert [row["odds_safety_factor"] for row in sweep["rows"]] == sweep[
        "factors"
    ]
    assert all("daily" not in row["retrospective"] for row in sweep["rows"])


def test_trend_empirical_ledger_applies_minimum_race_number_before_candidates() -> None:
    races = [_race("2026-07-22", 4), _race("2026-07-22", 5)]
    for race in races:
        race["_policy_calibrated_probabilities"] = dict(
            race["model_probabilities"]
        )

    prepared = market_calibration._trend_empirical_policy_races(
        races,
        odds_safety_factor=1.0,
        minimum_race_number=5,
    )

    assert set(prepared[0]["estimated_final_odds"].values()) == {1.0}
    assert max(prepared[1]["estimated_final_odds"].values()) > 1.0


def test_trend_empirical_ledger_applies_maximum_race_number_before_candidates() -> None:
    races = [_race("2026-07-22", 8), _race("2026-07-22", 9)]
    for race in races:
        race["_policy_calibrated_probabilities"] = dict(
            race["model_probabilities"]
        )

    prepared = market_calibration._trend_empirical_policy_races(
        races,
        odds_safety_factor=1.0,
        minimum_race_number=5,
        maximum_race_number=8,
    )

    assert max(prepared[0]["estimated_final_odds"].values()) > 1.0
    assert set(prepared[1]["estimated_final_odds"].values()) == {1.0}


def test_trend_point_required_ticket_count_is_explicit() -> None:
    races = [
        _race("2026-07-20", 1),
        _race("2026-07-21", 1),
        _race("2026-07-22", 1),
    ]

    result = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=["2026-07-22"],
        trend_point_registered_after="2026-07-21",
        trend_point_odds_safety_factor=1.10,
        trend_point_required_ticket_count=2,
        trend_point_require_reversed_place_pair=True,
        trend_point_maximum_forecast_odds=100.0,
    )

    assert result["trend_point_market_offset_kelly_diagnostic"]["policy"][
        "required_ticket_count"
    ] == 2
    assert result["trend_point_market_offset_kelly_diagnostic"]["policy"][
        "odds_safety_factor"
    ] == 1.10
    assert result["trend_point_market_offset_kelly_walk_forward"]["policy"][
        "required_ticket_count"
    ] == 2
    assert result["trend_point_market_offset_kelly_diagnostic"]["policy"][
        "require_reversed_place_pair"
    ] is True
    assert result["trend_point_market_offset_kelly_walk_forward"]["policy"][
        "maximum_forecast_odds"
    ] == 100.0
    assert result["trend_point_market_offset_kelly_walk_forward"][
        "registered_odds_safety_factor"
    ] == 1.10
    control = result["trend_point_market_only_control_diagnostic"]
    prospective_control = result[
        "trend_point_market_only_control_walk_forward"
    ]
    assert control["calibration"]["mode"] == (
        "raw_market_probability_control"
    )
    assert control["promotion_eligible"] is False
    assert control["policy"]["odds_safety_factor"] == 1.10
    assert control["policy"]["required_ticket_count"] == 2
    assert prospective_control["registered_after"] == "2026-07-21"
    assert prospective_control["promotion_eligible"] is False
    empirical = result["trend_point_empirical_lcb_walk_forward"]
    assert empirical["status"] == "unsupported_requested_ticket_constraints"
    assert empirical["promotion_gate"]["requested_policy_supported"] is False
    assert empirical["promotion_eligible"] is False
    assert result["trend_point_reversed_place_pair_diagnostic"]["policy"][
        "require_reversed_place_pair"
    ] is True
    assert (
        result["trend_point_reversed_place_pair_diagnostic"][
            "promotion_eligible"
        ]
        is False
    )
    with pytest.raises(ValueError, match="trend_point_required_ticket_count"):
        walk_forward_evaluate(
            races,
            trend_point_required_ticket_count=0,
        )
    with pytest.raises(ValueError, match="requires.*ticket_count=2"):
        walk_forward_evaluate(
            races,
            trend_point_require_reversed_place_pair=True,
        )
    with pytest.raises(ValueError, match="maximum_forecast_odds"):
        walk_forward_evaluate(
            races,
            trend_point_maximum_forecast_odds=1.0,
        )
    with pytest.raises(ValueError, match="odds_safety_factor"):
        walk_forward_evaluate(
            races,
            trend_point_odds_safety_factor=0.99,
        )


def test_prequential_conditional_order_is_explicit_and_audited() -> None:
    races = [
        _race("2026-07-20", 1),
        _race("2026-07-21", 1),
        _race("2026-07-22", 1),
    ]

    disabled = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=["2026-07-22"],
    )
    enabled = walk_forward_evaluate(
        races,
        min_calibration_days=2,
        evaluation_dates=["2026-07-22"],
        prequential_conditional_order=True,
    )

    assert disabled["prequential_conditional_order"] is None
    report = enabled["prequential_conditional_order"]
    assert report["status"] == "waiting"
    assert report["minimum_prior_days"] == 4
    assert report["transformed_races"] == 0
    with pytest.raises(ValueError, match="must be a boolean"):
        walk_forward_evaluate(
            races,
            prequential_conditional_order="yes",
        )
    with pytest.raises(ValueError, match="routed calibrator strategy"):
        walk_forward_evaluate(
            races,
            calibrator_strategy="odds_path_market_offset_discrete_log_ev_v9",
            prequential_conditional_order=True,
        )


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


def test_reused_holdout_research_uses_all_clean_dates() -> None:
    clean_dates = ["2025-07-26", "2026-07-24", "2026-08-04"]

    assert evaluation_dates_for_role(clean_dates) == [
        "2026-07-24",
        "2026-08-04",
    ]
    assert evaluation_dates_for_role(
        clean_dates,
        research_only_reused_holdout=True,
    ) == clean_dates


def test_research_holdout_coverage_requires_300_actual_clean_days() -> None:
    clean_dates = [
        (date(2025, 1, 1) + timedelta(days=offset)).isoformat()
        for offset in range(300)
    ]
    failing = research_holdout_coverage_gate(
        from_date="2025-01-01",
        through_date="2025-12-31",
        clean_dates=clean_dates[:-1],
    )
    passing = research_holdout_coverage_gate(
        from_date="2025-01-01",
        through_date="2025-12-31",
        clean_dates=clean_dates,
    )

    assert failing["requested_calendar_days"] == 365
    assert failing["source"] == "coverage_gate.clean_dates"
    assert failing["clean_days"] == 299
    assert failing["pass"] is False
    assert passing["clean_days"] == 300
    assert passing["pass"] is True


def test_clean_day_gate_validates_coverage_threshold() -> None:
    with pytest.raises(ValueError, match="minimum_day_coverage"):
        filter_clean_market_days(
            [],
            day_targets={},
            minimum_day_coverage=0.0,
        )


def test_trend_point_formal_gate_requires_exact_full_coverage() -> None:
    result = {
        "trend_point_market_offset_kelly_walk_forward": {
            "evaluation_dates": ["2026-08-04"],
            "evaluated_races": 2,
            "data_quality": {
                "operational_data_errors": 0,
                "lookahead_violations": 0,
            },
            "promotion_gate": {"pass": True},
            "promotion_eligible": True,
        }
    }
    coverage = {
        "minimum_day_coverage": 1.0,
        "formal_evaluation_dates": ["2026-08-04"],
        "days": [{
            "race_date": "2026-08-04",
            "eligible_t5_races": 2,
            "coverage": 1.0,
            "payout_complete": True,
            "clean": True,
        }],
    }

    market_calibration.apply_trend_point_formal_coverage_gate(
        result, coverage_gate=coverage, registered_after="2026-08-03"
    )
    candidate = result["trend_point_market_offset_kelly_walk_forward"]
    assert candidate["data_quality"]["pass"] is True
    assert candidate["promotion_gate"]["complete_market_data_pass"] is True
    assert candidate["promotion_eligible"] is True

    candidate["evaluated_races"] = 1
    candidate["promotion_gate"]["pass"] = True
    market_calibration.apply_trend_point_formal_coverage_gate(
        result, coverage_gate=coverage, registered_after="2026-08-03"
    )
    assert candidate["data_quality"]["race_set_complete"] is False
    assert candidate["promotion_gate"]["operational_data_errors_zero"] is False
    assert candidate["promotion_eligible"] is False


def test_empty_registered_policy_summary_waits_for_unseen_day() -> None:
    result = summarize_registered_policy_daily([], evaluated_races=0)

    assert result["status"] == "waiting_for_first_unseen_day"
    assert result["evaluation_days"] == 0
    assert result["evaluated_races"] == 0
    assert result["winning_days"] == 0
    assert result["profitable_day_fraction"] is None
    assert result["daily_cluster_bootstrap_roi_lower_95"] is None
    assert result["probability_roi_above_one"] is None


def test_registered_policy_summary_exposes_daily_robustness() -> None:
    daily = [
        {
            "race_date": "2026-07-26",
            "tickets": 2,
            "hit_tickets": 1,
            "races_bet": 1,
            "hit_races": 1,
            "stake_yen": 200,
            "return_yen": 300,
            "profit_yen": 100,
            "largest_hit_return_yen": 300,
            "hit_return_square_sum_yen2": 90_000,
        },
        {
            "race_date": "2026-07-27",
            "tickets": 1,
            "hit_tickets": 0,
            "races_bet": 1,
            "hit_races": 0,
            "stake_yen": 100,
            "return_yen": 0,
            "profit_yen": -100,
            "largest_hit_return_yen": 0,
            "hit_return_square_sum_yen2": 0,
        },
    ]

    result = summarize_registered_policy_daily(daily, evaluated_races=4)

    assert result["winning_days"] == 1
    assert result["profitable_day_fraction"] == 0.5
    assert result["stake_yen"] == 300
    assert result["return_yen"] == 300
    assert result["roi_without_largest_hit"] == 0.0
    assert result["daily_cluster_bootstrap_roi_lower_95"] == 0.0
    assert 0.0 < result["probability_roi_above_one"] < 1.0


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
    assert registered["comparison_role"].endswith("chronological_shadow")
    assert registered["allocation_time_basis"] == (
        "decision_time_order_with_settlement_only_reinvestment"
    )
    assert registered["status"] == "evaluating"
    assert registered["evaluation_days"] == 1
    assert registered["evaluated_races"] == 3
    assert [row["race_date"] for row in registered["daily"]] == ["2026-07-26"]
    assert registered["daily"][0]["information_boundary"][
        "settlement_joined_after_allocation"
    ] is True
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
    diagnostic = result["top5_narrow_retrospective_diagnostic"]
    assert diagnostic["status"] == "diagnostic_only_not_promotion_evidence"
    assert diagnostic["promotion_evidence"] is False
    assert diagnostic["evaluation_days"] == result["evaluation_days"]
    assert diagnostic["evaluated_races"] == result["evaluated_races"]
    assert diagnostic["evaluation_days"] > top5["evaluation_days"]
    assert all(
        fold["top5_narrow_retrospective_bankroll"] is not None
        for fold in folds.values()
    )


def test_cli_accepts_and_dispatches_v9_discrete_strategy(monkeypatch) -> None:
    import boatrace_ai.listwise.odds_path_discrete_v9 as v9

    args = build_parser().parse_args([
        "--from-date",
        "2026-07-22",
        "--calibrator-strategy",
        "odds_path_market_offset_discrete_log_ev_v9",
    ])
    assert args.calibrator_strategy == v9.STRATEGY_NAME

    expected = {"model": v9.MODEL_NAME, "dispatch": "v9"}
    seen = {}

    def fake_walk_forward(races, **kwargs):
        seen["races"] = races
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(v9, "walk_forward_evaluate_v9", fake_walk_forward)
    races = [{"race_id": "v9-race"}]
    result = walk_forward_evaluate(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=7,
        calibrator_strategy=v9.STRATEGY_NAME,
        evaluation_dates=("2026-07-29",),
    )

    assert result == expected
    assert seen == {
        "races": races,
        "daily_budget_yen": 10_000,
        "min_calibration_days": 7,
        "evaluation_dates": ("2026-07-29",),
    }


def test_cli_accepts_and_dispatches_v10_selection_conformal_strategy(
    monkeypatch,
) -> None:
    import boatrace_ai.listwise.odds_path_selection_conformal_v10 as v10

    args = build_parser().parse_args([
        "--from-date",
        "2026-07-22",
        "--calibrator-strategy",
        "odds_path_market_offset_selection_conformal_discrete_ev_v10",
    ])
    assert args.calibrator_strategy == v10.STRATEGY_NAME
    assert odds_path_model_name(args.calibrator_strategy) == v10.MODEL_NAME

    expected = {"model": v10.MODEL_NAME, "dispatch": "v10"}
    seen = {}

    def fake_walk_forward(races, **kwargs):
        seen["races"] = races
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(v10, "walk_forward_evaluate_v10", fake_walk_forward)
    races = [{"race_id": "v10-race"}]
    result = walk_forward_evaluate(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=7,
        calibrator_strategy=v10.STRATEGY_NAME,
        evaluation_dates=("2026-07-30",),
    )

    assert result == expected
    assert seen == {
        "races": races,
        "daily_budget_yen": 10_000,
        "min_calibration_days": 7,
        "evaluation_dates": ("2026-07-30",),
    }


@pytest.mark.parametrize(
    ("metric", "better", "worse"),
    [
        ("daily_cluster_bootstrap_roi_lower_95", 1.2, 1.1),
        ("roi_without_largest_hit", 1.2, 1.1),
        ("profitable_day_fraction", 0.7, 0.6),
        ("effective_hit_count", 8.0, 7.0),
        ("largest_hit_return_share", 0.2, 0.3),
        ("normalized_drawdown", 0.2, 0.3),
        ("roi", 1.2, 1.1),
    ],
)
def test_v17_policy_ranking_uses_fixed_robust_lexicographic_order(
    metric: str, better: float, worse: float
) -> None:
    base = {
        "daily_cluster_bootstrap_roi_lower_95": 1.1,
        "roi_without_largest_hit": 1.1,
        "profitable_day_fraction": 0.6,
        "effective_hit_count": 7.0,
        "largest_hit_return_share": 0.3,
        "normalized_drawdown": 0.3,
        "roi": 1.1,
    }
    preferred = {**base, metric: better, "policy": {"name": "z-policy"}}
    rejected = {**base, metric: worse, "policy": {"name": "a-policy"}}

    assert min([rejected, preferred], key=v17_policy_ranking_key) is preferred


def test_v17_policy_ranking_tie_breaks_by_policy_name_and_none_is_worst() -> None:
    metrics = {
        "daily_cluster_bootstrap_roi_lower_95": 1.1,
        "roi_without_largest_hit": 1.1,
        "profitable_day_fraction": 0.6,
        "effective_hit_count": 7.0,
        "largest_hit_return_share": 0.3,
        "normalized_drawdown": 0.3,
        "roi": 1.1,
    }
    a = {**metrics, "policy": {"name": "a-policy"}}
    z = {**metrics, "policy": {"name": "z-policy"}}
    missing = {**metrics, "daily_cluster_bootstrap_roi_lower_95": None,
               "policy": {"name": "missing"}}

    assert min([z, missing, a], key=v17_policy_ranking_key) is a


def test_v17_inner_policy_grid_never_calls_chronological_allocator(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("inner grid must not build chronological ledgers")

    monkeypatch.setattr(
        market_calibration, "simulate_chronological_bankroll_day", fail_if_called
    )
    races = [_race("2026-07-18", rno) for rno in range(1, 13)]
    selected, rows = select_policy_v17(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=[
            {"name": "no_bet", "no_bet": True},
            {
                "name": "candidate",
                "ev_threshold": 1.0,
                "max_estimated_ev": None,
                "max_odds": None,
                "max_tickets_per_race": 1,
                "min_model_market_ratio": 1.0,
                "staking_mode": "kelly_025",
            },
        ],
    )

    assert selected["name"] in {"no_bet", "candidate"}
    assert all("chronological_bankroll" not in row for row in rows)


def test_v17_is_strict_prior_result_invariant_and_chronological_primary(
    monkeypatch,
) -> None:
    policies = [
        {"name": "no_bet", "no_bet": True},
        {
            "name": "robust-a",
            "ev_threshold": 1.0,
            "max_estimated_ev": None,
            "max_odds": None,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.0,
            "staking_mode": "kelly_025",
        },
        {
            "name": "robust-b",
            "ev_threshold": 1.05,
            "max_estimated_ev": None,
            "max_odds": 40.0,
            "max_tickets_per_race": 2,
            "min_model_market_ratio": 1.0,
            "staking_mode": "kelly_025",
        },
    ]
    monkeypatch.setattr(market_calibration, "default_policy_grid", lambda: policies)
    dates = ["2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21"]
    races = [_race(day, rno) for day in dates for rno in range(1, 13)]
    for race in races:
        race["closing_odds"] = dict(race["odds"])
        race["closing_odds_changed"] = True
    contaminated = [dict(race) for race in races]
    for race in contaminated:
        if race["race_date"] == dates[-1]:
            race["actual_combination"] = "6-5-4"
            race["actual_payout_yen"] = 99_990

    kwargs = {
        "min_calibration_days": 2,
        "calibrator_strategy": market_calibration.V17_STRATEGY_NAME,
        "evaluation_dates": [dates[-1]],
    }
    clean = walk_forward_evaluate(races, **kwargs)
    changed = walk_forward_evaluate(contaminated, **kwargs)

    clean_fold = clean["folds"][0]
    changed_fold = changed["folds"][0]
    assert clean_fold["calibration_dates"] == dates[:-1]
    assert all(day < dates[-1] for day in clean_fold["calibration_dates"])
    assert clean_fold["selected_policy"] == changed_fold["selected_policy"]
    assert clean_fold["operational_model"]["return_price_basis"] == "observed_closing"
    assert clean["model"] == market_calibration.V17_MODEL_NAME
    assert clean["comparison_role"] == market_calibration.V17_COMPARISON_ROLE
    assert clean["real_betting_enabled"] is False
    assert clean["deployment_configuration"]["deployment_mode"] == "shadow_only"
    assert clean["deployment_configuration"]["real_betting_enabled"] is False
    assert clean["promotion_gate"]["primary_bankroll"] == "chronological_bankroll"
    assert clean["deployment_configuration"]["walk_forward_gate"][
        "primary_bankroll"
    ] == "chronological_bankroll"
    chronological = clean["chronological_bankroll"]
    assert chronological["primary_promotion_bankroll"] is True
    assert chronological["daily_stake_limit_fraction"] == 1.0
    assert chronological["real_betting_enabled"] is False
    assert chronological["daily"][0]["gross_stake_yen"] <= (
        chronological["daily"][0]["initial_gross_stake_allowance_yen"]
        + max(0, chronological["daily"][0]["realized_cumulative_profit_yen"])
    )


def test_v17_parser_and_model_identity_are_exact() -> None:
    strategy = market_calibration.V17_STRATEGY_NAME
    args = build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", strategy,
    ])
    assert args.calibrator_strategy == strategy
    assert odds_path_model_name(strategy) == strategy
