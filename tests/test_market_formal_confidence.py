from __future__ import annotations

from itertools import permutations

from boatrace_ai.listwise.market_calibration import (
    MARKET_EVALUATION_VERSION,
    MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    SCORED_CACHE_VERSION,
    market_comparison_confidence,
    scored_cache_contract,
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


def _race(race_date: str, rno: int) -> dict:
    actual = "1-2-3"
    market = _distribution(actual, 0.20)
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": actual,
        "actual_payout_yen": 500,
        "model_probabilities": _distribution(actual, 0.35),
        "market_probabilities": market,
        "odds": {
            combination: 1.0 / probability / 1.25
            for combination, probability in market.items()
        },
        "snapshot_id": rno,
    }


def test_market_confidence_requires_race_and_day_cluster_evidence() -> None:
    labels = [f"day-{index // 4}" for index in range(20)]
    passing = market_comparison_confidence(
        [-0.1] * 20,
        [0.0] * 20,
        cluster_labels=labels,
    )
    assert passing["confidence_pass"] is True
    assert passing["race_level_confidence_pass"] is True
    assert passing["day_cluster_confidence_pass"] is True
    assert passing["evaluation_days"] == 5
    assert passing["log_loss_difference_calibrated_minus_market"]["ci95_upper"] < 0
    assert passing[
        "day_cluster_log_loss_difference_calibrated_minus_market"
    ]["ci95_upper"] < 0
    assert passing["top5_hit_difference_calibrated_minus_market"]["ci95_lower"] == 0

    failing = market_comparison_confidence(
        [0.1] * 20,
        [0.0] * 20,
        cluster_labels=labels,
    )
    assert failing["confidence_pass"] is False


def test_walk_forward_reports_paired_market_confidence() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-18", "2026-07-19", "2026-07-20")
        for rno in range(1, 7)
    ]

    result = walk_forward_evaluate(races, min_calibration_days=2)

    comparison = result["market_comparison"]
    assert comparison["evaluation_races"] == 6
    assert comparison["log_loss_difference_calibrated_minus_market"][
        "observations"
    ] == 6
    assert result["folds"][0]["market_comparison"][
        "log_loss_mean_difference"
    ] < 0
    assert comparison["race_level_confidence_pass"] is True
    assert comparison["day_cluster_confidence_pass"] is False
    assert comparison["evaluation_days"] == 1
    assert result["promotion_gate"]["market_confidence_pass"] is False


def test_scored_cache_version_is_decoupled_from_evaluation_output(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fixed-model")

    contract = scored_cache_contract(
        model_path=model_path,
        artifact={},
        from_date="2026-07-18",
        through_date="2026-07-22",
        max_snapshot_age_seconds=60.0,
        odds_signature={"snapshot_count": 10},
    )

    assert MARKET_EVALUATION_VERSION == 16
    assert MARKET_MAX_SNAPSHOT_AGE_SECONDS == 65.0
    assert contract["version"] == SCORED_CACHE_VERSION == 10
