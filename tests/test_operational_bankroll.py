from __future__ import annotations

import pytest

from boatrace_ai.historical_model import FEATURE_SET, make_pipeline
from boatrace_ai.operational_bankroll import (
    MODEL_NAME,
    operational_adaptive_bankroll,
    operational_policy,
    race_keys_from_meta,
)


def test_operational_pipeline_matches_deployed_profile() -> None:
    pipeline = make_pipeline()
    classifier = pipeline.named_steps["classifier"]

    assert list(pipeline.named_steps) == [
        "vectorizer",
        "sparse_index_32_a",
        "scaler",
        "sparse_index_32_b",
        "classifier",
    ]
    assert classifier.C == 0.20
    assert classifier.class_weight is None
    assert classifier.solver == "liblinear"
    assert pipeline.named_steps["scaler"].__class__.__name__ == "MaxAbsScaler"


def test_race_keys_from_meta_deduplicates_and_sorts() -> None:
    rows = [
        {"race_id": "2026-01-02-02-01", "race_date": "2026-01-02", "jcd": "02", "rno": 1},
        {"race_id": "2026-01-01-03-02", "race_date": "2026-01-01", "jcd": "03", "rno": 2},
        {"race_id": "2026-01-01-03-02", "race_date": "2026-01-01", "jcd": "03", "rno": 2},
    ]

    assert race_keys_from_meta(rows) == [
        ("2026-01-01-03-02", "2026-01-01", "03", 2),
        ("2026-01-02-02-01", "2026-01-02", "02", 1),
    ]


def test_operational_policy_identifies_model_and_same_allocation() -> None:
    policy = operational_policy(
        daily_budget_yen=10_000,
        ev_threshold=1.20,
        payout_prior_weight=30.0,
        fractional_kelly=0.25,
        max_daily_exposure_fraction=0.60,
        min_daily_exposure_fraction=0.40,
        race_cap_fraction=0.10,
        ticket_cap_fraction=0.03,
        max_daily_tickets=30,
        allocation_mode="normalized_kelly",
        stake_granularity_yen=100,
        min_stake_yen=100,
    )

    assert policy["model"] == MODEL_NAME
    assert policy["feature_set"] == FEATURE_SET
    assert policy["allocation_mode"] == "normalized_kelly"
    assert policy["max_daily_tickets"] == 30
    assert policy["stake_granularity_yen"] == 100


def test_operational_backtest_runs_end_to_end_on_time_folds(
    tmp_path,
    monkeypatch,
) -> None:
    from boatrace_ai import operational_bankroll as module

    features = []
    labels = []
    meta = []
    payouts = {}
    for day in range(1, 5):
        race_date = f"2026-01-{day:02d}"
        for rno in range(1, 3):
            race_id = f"{race_date}-01-{rno:02d}"
            payouts[race_id] = {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": "01",
                "rno": rno,
                "combination": "1-2-3",
                "payout_yen": 10_000,
                "popularity": 1,
            }
            for lane in range(1, 7):
                features.append({"lane": f"L{lane}"})
                labels.append(1 if lane == 1 else 0)
                meta.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "jcd": "01",
                        "rno": rno,
                        "lane": lane,
                        "rank": lane,
                    }
                )

    payout_model = {
        f"{first}-{second}-{third}": {
            "estimated_odds": 100.0,
            "estimated_payout_yen": 10_000.0,
            "history_count": 10.0,
        }
        for first in range(1, 7)
        for second in range(1, 7)
        for third in range(1, 7)
        if len({first, second, third}) == 3
    }
    def load_no_odds_examples(conn, *, include_odds, include_research):
        assert include_odds is False
        assert include_research is False
        return features, labels, meta

    monkeypatch.setattr(module, "load_training_examples", load_no_odds_examples)
    monkeypatch.setattr(module, "_load_trifecta_payouts", lambda conn: payouts)
    monkeypatch.setattr(
        module,
        "_build_payout_model",
        lambda payouts, train_races, prior_weight: payout_model,
    )

    output = tmp_path / "operational.json"
    checkpoint = tmp_path / "operational.checkpoint.json"
    write_checkpoint = module._write_json_atomic

    def interrupt_after_first_fold(path, payload):
        write_checkpoint(path, payload)
        if path == checkpoint and payload.get("next_fold") == 2:
            raise RuntimeError("simulated worker interruption")

    monkeypatch.setattr(module, "_write_json_atomic", interrupt_after_first_fold)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        operational_adaptive_bankroll(
            object(),
            output_path=output,
            checkpoint_path=checkpoint,
            folds=2,
            min_train_races=2,
            ev_threshold=1.0,
        )
    assert checkpoint.exists()
    assert not output.exists()

    monkeypatch.setattr(module, "_write_json_atomic", write_checkpoint)
    result = operational_adaptive_bankroll(
        object(),
        output_path=output,
        checkpoint_path=checkpoint,
        resume=True,
        folds=2,
        min_train_races=2,
        ev_threshold=1.0,
    )

    assert output.exists()
    assert not checkpoint.exists()
    assert result["model"] == MODEL_NAME
    assert result["comparison_role"] == "operational_model_same_policy_backtest"
    assert len(result["folds"]) == 2
    assert result["evaluated_races"] == 6
    assert result["tickets"] > 0
    assert result["stake_yen"] > 0

    adaptive_output = tmp_path / "operational_adaptive_no_bet.json"
    adaptive = operational_adaptive_bankroll(
        object(),
        output_path=adaptive_output,
        folds=2,
        min_train_races=2,
        ev_threshold=1.0,
        adaptive_no_bet=True,
        calibration_fraction=0.5,
    )

    assert adaptive_output.exists()
    assert adaptive["policy"]["adaptive_no_bet"] is True
    assert adaptive["folds"][0]["calibration_days"] == 1
    assert adaptive["folds"][0]["evaluation_days"] == 1
    assert adaptive["folds"][0]["calibration_policy_results"]
    assert "selected_candidate_policy" in adaptive["folds"][0]
