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
            "daily": [
                {
                    "race_date": race_date,
                    "tickets": 0,
                    "hits": 0,
                    "hit_tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                }
            ],
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
