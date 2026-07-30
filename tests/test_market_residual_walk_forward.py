from boatrace_ai.listwise import market_calibration, market_residual


def _race(race_date: str, actual: str) -> dict:
    return {
        "race_id": f"{race_date}-{actual}",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "actual_combination": actual,
        "actual_payout_yen": 1_000,
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.55, "1-3-2": 0.45},
        "odds": {"1-2-3": 2.0, "1-3-2": 2.5},
    }


def test_walk_forward_executes_newton_residual_branch(monkeypatch) -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-21", "1-3-2"),
        _race("2026-07-22", "1-2-3"),
    ]
    selected = {
        "final_calibrator": {
            "model_weight": 0.1,
            "temperature": 0.9,
            "model_coefficient": 1.0 / 9.0,
            "market_coefficient": 1.0,
        },
        "candidates": [],
    }
    monkeypatch.setattr(
        market_residual,
        "select_regularization_prequential",
        lambda calibration_races: selected,
    )
    monkeypatch.setattr(
        market_calibration,
        "select_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )
    monkeypatch.setattr(
        market_calibration,
        "select_flat_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )

    def simulated(races, **kwargs):
        race_date = races[0]["race_date"]
        daily = [
            {
                "race_date": race_date,
                "tickets": 0,
                "hits": 0,
                "hit_tickets": 0,
                "stake_yen": 0,
                "return_yen": 0,
                "profit_yen": 0,
            }
        ]
        return {
            "evaluated_races": len(races),
            "race_days": 1,
            "evaluation_days": 1,
            "tickets": 0,
            "hit_tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "max_drawdown_yen": 0,
            "winning_days": 0,
            "daily": daily,
            "chronological_bankroll": {"daily": list(daily)},
        }

    monkeypatch.setattr(market_calibration, "simulate_policy", simulated)
    monkeypatch.setattr(market_calibration, "simulate_flat_policy", simulated)
    monkeypatch.setattr(
        market_calibration,
        "predefined_ticket_diagnostics",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        market_calibration,
        "summarize_policy_candidates",
        lambda rows: {},
    )
    monkeypatch.setattr(
        market_calibration,
        "summarize_flat_candidates",
        lambda rows: {},
    )

    result = market_calibration.walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="newton_residual",
    )

    assert result["calibrator_strategy"] == "newton_residual"
    assert result["evaluation_days"] == 1
    assert result["evaluation_races"] == 1
    assert result["folds"][0]["calibrator_selection"] is selected
    assert result["folds"][0]["calibrator"]["model_weight"] == 0.1


def test_v20_routes_dual_heads_without_outer_holdout_selection(monkeypatch) -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-21", "1-3-2"),
        _race("2026-07-22", "1-2-3"),
    ]
    probability_calibrator = {"model_weight": 1.0, "temperature": 1.0}
    purchase_calibrator = {"model_weight": 0.0, "temperature": 1.0}
    routed = {
        "selection_race_dates": [],
        "policy": [],
        "chronological": [],
        "probability_metrics": [],
        "market_comparison": [],
    }

    def fit_dual(prior_races):
        dates = [race["race_date"] for race in prior_races]
        routed["selection_race_dates"].append(dates)
        return {
            "architecture": "strict_prior_dual_calibrator_heads_v20",
            "selection_data": (
                "strict_prior_training_and_inner_prequential_folds_only"
            ),
            "outer_holdout_used": False,
            "training_dates": sorted(set(dates)),
            "trained_through_date": max(dates),
            "probability_head": {
                "role": "probability_reporting_and_promotion_calibration",
                "calibrator_strategy": market_calibration.V19_STRATEGY_NAME,
                "raw_nonregression_enforced": True,
                "calibrator": probability_calibrator,
                "selection": {"final_calibrator": probability_calibrator},
            },
            "purchase_head": {
                "role": "purchase_policy_and_chronological_bankroll",
                "calibrator_strategy": market_calibration.V18_STRATEGY_NAME,
                "raw_nonregression_enforced": False,
                "policy_strategy": market_calibration.V18_STRATEGY_NAME,
                "calibrator": purchase_calibrator,
                "selection": {"final_calibrator": purchase_calibrator},
            },
        }

    monkeypatch.setattr(
        market_calibration, "fit_v20_dual_head_calibrators", fit_dual
    )
    monkeypatch.setattr(
        market_calibration, "attach_observed_closing_return_prices", lambda rows: rows
    )
    monkeypatch.setattr(
        market_calibration, "fit_odds_path_model", lambda *args, **kwargs: {"v": 20}
    )
    monkeypatch.setattr(
        market_calibration, "attach_odds_path_model", lambda rows, model: rows
    )

    def select_policy(*args, calibrator, **kwargs):
        routed["policy"].append(calibrator)
        return {"name": "no_bet", "no_bet": True}, []

    monkeypatch.setattr(market_calibration, "select_policy_v18", select_policy)
    monkeypatch.setattr(
        market_calibration,
        "select_flat_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )

    original_probability_metrics = market_calibration.probability_metrics
    original_paired_market_differences = market_calibration.paired_market_differences

    def probability_metrics(rows, *, calibrator):
        routed["probability_metrics"].append(calibrator)
        return original_probability_metrics(rows, calibrator=calibrator)

    def paired_market_differences(rows, *, calibrator):
        routed["market_comparison"].append(calibrator)
        return original_paired_market_differences(rows, calibrator=calibrator)

    monkeypatch.setattr(market_calibration, "probability_metrics", probability_metrics)
    monkeypatch.setattr(
        market_calibration, "paired_market_differences", paired_market_differences
    )

    original_simulate_policy = market_calibration.simulate_policy

    def simulate_policy(rows, *, calibrator, include_chronological=False, **kwargs):
        if include_chronological:
            routed["chronological"].append(calibrator)
        return original_simulate_policy(
            rows,
            calibrator=calibrator,
            include_chronological=include_chronological,
            **kwargs,
        )

    monkeypatch.setattr(market_calibration, "simulate_policy", simulate_policy)

    result = market_calibration.walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy=market_calibration.V20_STRATEGY_NAME,
    )

    assert routed["selection_race_dates"][0] == ["2026-07-20", "2026-07-21"]
    # A later all-completed-data call is the explicitly labeled next-day refit.
    assert routed["selection_race_dates"][-1] == [
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
    ]
    assert routed["policy"] and all(
        calibrator == purchase_calibrator for calibrator in routed["policy"]
    )
    assert routed["chronological"] == [purchase_calibrator]
    assert routed["probability_metrics"] == [probability_calibrator]
    assert routed["market_comparison"] == [probability_calibrator]
    fold = result["folds"][0]
    assert fold["probability_metrics_head"] == "probability_head"
    assert fold["chronological_bankroll_head"] == "purchase_head"
    assert result["dual_head_architecture"]["outer_holdout_used"] is False
    assert result["deployment_mode"] == "evaluation_only"
    assert result["promotion_eligible"] is False


def test_v21_routes_three_heads_without_outer_holdout_selection(monkeypatch) -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-21", "1-3-2"),
        _race("2026-07-22", "1-2-3"),
    ]
    probability_calibrator = {"model_weight": 1.0, "temperature": 1.0}
    ranking_calibrator = {"model_weight": 0.5, "temperature": 1.0}
    purchase_calibrator = {"model_weight": 0.0, "temperature": 1.0}
    routed = {
        "selection_race_dates": [],
        "policy": [],
        "chronological": [],
        "probability_metrics": [],
        "market_comparison": [],
    }

    def fit_triple(prior_races):
        dates = [race["race_date"] for race in prior_races]
        routed["selection_race_dates"].append(dates)
        v18_selection = {"final_calibrator": purchase_calibrator}
        return {
            "architecture": "strict_prior_triple_calibrator_heads_v21",
            "selection_data": (
                "strict_prior_training_and_inner_prequential_folds_only"
            ),
            "outer_holdout_used": False,
            "training_dates": sorted(set(dates)),
            "trained_through_date": max(dates),
            "probability_head": {
                "role": "winner_and_trifecta_logloss",
                "calibrator_strategy": market_calibration.V19_STRATEGY_NAME,
                "raw_nonregression_enforced": True,
                "calibrator": probability_calibrator,
                "selection": {"final_calibrator": probability_calibrator},
            },
            "ranking_head": {
                "role": "trifecta_top5_ranking",
                "calibrator_strategy": market_calibration.V18_STRATEGY_NAME,
                "raw_nonregression_enforced": False,
                "calibrator": ranking_calibrator,
                "selection": v18_selection,
            },
            "purchase_head": {
                "role": "purchase_policy_and_chronological_bankroll",
                "calibrator_strategy": market_calibration.V18_STRATEGY_NAME,
                "raw_nonregression_enforced": False,
                "policy_strategy": market_calibration.V18_STRATEGY_NAME,
                "calibrator": purchase_calibrator,
                "selection": v18_selection,
            },
            "ranking_purchase_share_v18_selection": True,
        }

    monkeypatch.setattr(
        market_calibration, "fit_v21_triple_head_calibrators", fit_triple
    )
    monkeypatch.setattr(
        market_calibration, "attach_observed_closing_return_prices", lambda rows: rows
    )
    monkeypatch.setattr(
        market_calibration, "fit_odds_path_model", lambda *args, **kwargs: {"v": 21}
    )
    monkeypatch.setattr(
        market_calibration, "attach_odds_path_model", lambda rows, model: rows
    )

    def select_policy(*args, calibrator, **kwargs):
        routed["policy"].append(calibrator)
        return {"name": "no_bet", "no_bet": True}, []

    monkeypatch.setattr(market_calibration, "select_policy_v18", select_policy)
    monkeypatch.setattr(
        market_calibration,
        "select_flat_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )

    original_split_metrics = market_calibration.split_head_probability_metrics
    original_split_differences = (
        market_calibration.split_head_paired_market_differences
    )

    def split_metrics(
        rows, *, probability_calibrator, ranking_calibrator
    ):
        routed["probability_metrics"].append(
            (probability_calibrator, ranking_calibrator)
        )
        return original_split_metrics(
            rows,
            probability_calibrator=probability_calibrator,
            ranking_calibrator=ranking_calibrator,
        )

    def split_differences(
        rows, *, probability_calibrator, ranking_calibrator
    ):
        routed["market_comparison"].append(
            (probability_calibrator, ranking_calibrator)
        )
        return original_split_differences(
            rows,
            probability_calibrator=probability_calibrator,
            ranking_calibrator=ranking_calibrator,
        )

    monkeypatch.setattr(
        market_calibration, "split_head_probability_metrics", split_metrics
    )
    monkeypatch.setattr(
        market_calibration,
        "split_head_paired_market_differences",
        split_differences,
    )

    original_simulate_policy = market_calibration.simulate_policy

    def simulate_policy(rows, *, calibrator, include_chronological=False, **kwargs):
        if include_chronological:
            routed["chronological"].append(calibrator)
        return original_simulate_policy(
            rows,
            calibrator=calibrator,
            include_chronological=include_chronological,
            **kwargs,
        )

    monkeypatch.setattr(market_calibration, "simulate_policy", simulate_policy)

    result = market_calibration.walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy=market_calibration.V21_STRATEGY_NAME,
    )

    assert routed["selection_race_dates"][0] == ["2026-07-20", "2026-07-21"]
    assert routed["selection_race_dates"][-1] == [
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
    ]
    assert routed["policy"] and all(
        calibrator == purchase_calibrator for calibrator in routed["policy"]
    )
    assert routed["chronological"] == [purchase_calibrator]
    expected_split = [(probability_calibrator, ranking_calibrator)]
    assert routed["probability_metrics"] == expected_split
    assert routed["market_comparison"] == expected_split
    fold = result["folds"][0]
    assert fold["probability_metrics_head"] == "probability_head"
    assert fold["trifecta_top5_head"] == "ranking_head"
    assert fold["market_logloss_comparison_head"] == "probability_head"
    assert fold["market_top5_comparison_head"] == "ranking_head"
    assert fold["chronological_bankroll_head"] == "purchase_head"
    assert result["triple_head_architecture"]["outer_holdout_used"] is False
    assert result["market_comparison"]["logloss_difference_source"] == (
        "probability_head"
    )
    assert result["market_comparison"]["top5_difference_source"] == "ranking_head"
    assert result["deployment_mode"] == "evaluation_only"
    assert result["promotion_eligible"] is False


def test_walk_forward_supports_preregistered_single_day_provisional_track(
    monkeypatch,
) -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-21", "1-3-2"),
    ]
    fixed = {
        "validation_design": "preregistered single-day calibration",
        "selected_regularization": 1.0,
        "final_calibrator": {
            "model_weight": 0.1,
            "temperature": 0.9,
            "model_coefficient": 1.0 / 9.0,
            "market_coefficient": 1.0,
        },
        "candidates": [],
    }
    monkeypatch.setattr(market_residual, "fit_fixed_regularization", lambda races: fixed)
    monkeypatch.setattr(
        market_calibration,
        "select_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )
    monkeypatch.setattr(
        market_calibration,
        "select_flat_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )

    result = market_calibration.walk_forward_evaluate(
        races,
        min_calibration_days=1,
        calibrator_strategy="newton_residual",
    )

    selection = result["folds"][0]["calibrator_selection"]
    assert result["evaluation_days"] == 1
    assert selection is fixed
