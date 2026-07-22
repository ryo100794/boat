from boatrace_ai.listwise import market_policy_diagnostics


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


def test_forward_policy_diagnostics_keeps_days_separate(monkeypatch) -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-21", "1-3-2"),
    ]
    calibrator = {
        "model_weight": 0.1,
        "temperature": 1.0,
        "model_coefficient": 0.1,
        "market_coefficient": 0.9,
    }
    monkeypatch.setattr(
        market_policy_diagnostics,
        "fit_log_pool_newton",
        lambda training, regularization: calibrator,
    )
    monkeypatch.setattr(
        market_policy_diagnostics,
        "select_policy",
        lambda training, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )
    monkeypatch.setattr(
        market_policy_diagnostics,
        "select_flat_policy",
        lambda training, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )

    def result(rows, **kwargs):
        return {
            "tickets": 0,
            "hit_tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "daily": [],
        }

    monkeypatch.setattr(market_policy_diagnostics, "simulate_policy", result)
    monkeypatch.setattr(market_policy_diagnostics, "simulate_flat_policy", result)

    report = market_policy_diagnostics.forward_policy_diagnostics(
        races, regularization=1.0
    )

    assert report["dates"] == ["2026-07-20", "2026-07-21"]
    assert report["folds"][0]["training_dates"] == ["2026-07-20"]
    assert report["folds"][0]["evaluation_date"] == "2026-07-21"
    assert report["adaptive"]["evaluation_races"] == 1
