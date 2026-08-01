from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from boatrace_ai.runtime.v21_prospective_evidence import (
    COMBINATIONS,
    V21ProspectiveEvidenceConfig,
    aggregate_v21_prospective_evidence,
    collect_v21_prospective_evidence,
    write_v21_prospective_evidence_atomic,
)


MODEL_KEY = "v21_daily"
MODEL_HASH = "a" * 64
STRATEGY = "v21_triple_head_t300"
UTC = timezone.utc


def probability_vector(actual: str, mass: float = 0.5) -> dict[str, float]:
    remainder = (1.0 - mass) / 119
    return {
        combination: mass if combination == actual else remainder
        for combination in COMBINATIONS
    }


def source_prices(actual: str) -> dict[str, float]:
    top = [combination for combination in COMBINATIONS if combination != actual][:5]
    return {
        combination: 1_000.0 if combination == actual else 2.0 if combination in top else 100.0
        for combination in COMBINATIONS
    }


def evidence_rows(days: int = 1, races_per_day: int = 2) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {
        "races": [],
        "decisions": [],
        "settlements": [],
        "source_odds": [],
        "payouts": [],
    }
    base = datetime(2026, 7, 31, 1, 0, tzinfo=UTC)
    actuals = ["1-2-3", "2-1-3", "3-1-2", "4-1-2"]
    decision_id = 0
    for day_index in range(days):
        race_date = (base + timedelta(days=day_index)).date().isoformat()
        for race_index in range(races_per_day):
            decision_id += 1
            race_id = f"{race_date}-01-{race_index + 1:02d}"
            target = base + timedelta(days=day_index, minutes=race_index * 10)
            captured = target - timedelta(seconds=2)
            actual = actuals[(decision_id - 1) % len(actuals)]
            probabilities = probability_vector(actual)
            diagnostics = {
                "v21_triple_head": {
                    "source_snapshot_id": decision_id,
                    "ranking_probabilities": probabilities,
                    "decision_features": "t300_or_earlier",
                    "outer_result_used": False,
                    "outer_payout_used": False,
                    "real_betting_enabled": False,
                }
            }
            result["races"].append(
                {"race_id": race_id, "race_date": race_date, "lane_count": 6}
            )
            result["decisions"].append(
                {
                    "decision_id": decision_id,
                    "race_date": race_date,
                    "race_id": race_id,
                    "model_key": MODEL_KEY,
                    "model_hash": MODEL_HASH,
                    "strategy_name": STRATEGY,
                    "decision_at": target + timedelta(seconds=4),
                    "decision_completed_at": target + timedelta(seconds=4),
                    "target_t300_at": target,
                    "source_snapshot_id": decision_id,
                    "source_captured_at": captured,
                    "probabilities": probabilities,
                    "selected_candidates": [
                        {"combination": actual, "stake_yen": 100}
                    ],
                    "diagnostics": diagnostics,
                    "total_stake_yen": 100,
                }
            )
            result["settlements"].append(
                {
                    "decision_id": decision_id,
                    "result_status": "final",
                    "actual_combination": actual,
                    "payout_yen_per_100": 300,
                    "stake_yen": 100,
                    "return_yen": 300,
                    "profit_yen": 200,
                }
            )
            result["payouts"].append(
                {"race_id": race_id, "combination": actual, "payout_yen": 300}
            )
            for combination, odds in source_prices(actual).items():
                result["source_odds"].append(
                    {
                        "snapshot_id": decision_id,
                        "race_id": race_id,
                        "captured_at": captured,
                        "combination": combination,
                        "odds": odds,
                    }
                )
    return result


def config(**overrides) -> V21ProspectiveEvidenceConfig:
    values = {
        "start_date": "2026-07-31",
        "through_date": "2026-08-31",
        "model_key": MODEL_KEY,
        "expected_model_hash": MODEL_HASH,
        "bootstrap_samples": 200,
        "minimum_clean_days": 1,
        "minimum_races": 2,
        "minimum_tickets": 2,
        "minimum_effective_hits": 1.9,
    }
    values.update(overrides)
    return V21ProspectiveEvidenceConfig(**values)


def aggregate(rows: dict[str, list[dict]], **config_overrides) -> dict:
    return aggregate_v21_prospective_evidence(
        config=config(**config_overrides), **rows
    )


def test_clean_day_aggregates_bankroll_market_metrics_and_passes_gate() -> None:
    result = aggregate(evidence_rows())

    assert result["excluded_days"] == []
    assert result["model_identity"]["fixed"] is True
    assert result["daily"] == [
        {
            "race_date": "2026-07-31",
            "races": 2,
            "tickets": 2,
            "hit_tickets": 2,
            "expected_hit_tickets": 1.0,
            "expected_no_hit_probability": 0.25,
            "stake_yen": 200,
            "return_yen": 600,
            "profit_yen": 400,
            "roi": 3.0,
            "coverage": {
                "six_boat_races": 2,
                "model_decisions": 2,
                "valid_decision_boundaries": 2,
                "valid_settlements": 2,
            },
        }
    ]
    assert result["bankroll"]["roi"] == 3.0
    assert result["bankroll"]["roi_without_largest_hit"] == 1.5
    assert result["bankroll"]["daily_cluster_bootstrap_roi_lower_95"] == 3.0
    assert result["bankroll"]["profitable_day_fraction"] == 1.0
    assert result["bankroll"]["effective_hit_count"] == 2.0
    assert result["purchase_probability_calibration"] == {
        "selected_races": 2,
        "observed_hits": 2,
        "expected_hits": 1.0,
        "observed_to_expected_hit_ratio": 2.0,
        "expected_no_hit_probability": 0.25,
        "standardized_hit_residual": pytest.approx(2**0.5),
        "probability_at_most_observed_hits": 1.0,
        "method": "exact_poisson_binomial_lower_tail_over_disjoint_race_selections",
    }
    assert result["market"]["model_trifecta_log_loss"] < result["market"]["market_trifecta_log_loss"]
    assert result["market"]["model_trifecta_top5"] == 1.0
    assert result["market"]["market_trifecta_top5"] == 0.0
    assert result["market"]["market_noninferiority_confidence"] == 1.0
    assert result["market"]["top5_improvement_confidence"] == 1.0
    assert result["promotion_gate"]["pass"] is True


def test_missing_settlement_excludes_whole_day_with_coverage_reason() -> None:
    rows = evidence_rows()
    rows["settlements"].pop()

    result = aggregate(rows)

    assert result["daily"] == []
    assert result["bankroll"]["races"] == 0
    excluded = result["excluded_days"][0]
    assert "settlement_coverage_mismatch" in excluded["reasons"]
    assert excluded["coverage"] == {
        "six_boat_races": 2,
        "model_decisions": 2,
        "valid_decision_boundaries": 2,
        "valid_settlements": 1,
    }
    assert result["promotion_gate"]["pass"] is False


@pytest.mark.parametrize("unsafe", ["late_source", "delay_90", "real_betting", "outer_result"])
def test_t300_latency_and_shadow_information_boundary_fail_closed(unsafe: str) -> None:
    rows = evidence_rows()
    decision = rows["decisions"][0]
    if unsafe == "late_source":
        late = decision["target_t300_at"] + timedelta(microseconds=1)
        decision["source_captured_at"] = late
        for row in rows["source_odds"]:
            if row["snapshot_id"] == decision["source_snapshot_id"]:
                row["captured_at"] = late
    elif unsafe == "delay_90":
        decision["decision_completed_at"] = decision["target_t300_at"] + timedelta(seconds=90)
    elif unsafe == "real_betting":
        decision["diagnostics"]["v21_triple_head"]["real_betting_enabled"] = True
    else:
        decision["diagnostics"]["v21_triple_head"]["outer_result_used"] = True

    result = aggregate(rows)

    assert result["daily"] == []
    assert "decision_boundary_invalid" in result["excluded_days"][0]["reasons"]
    assert result["promotion_gate"]["pass"] is False


def test_model_identity_change_excludes_changed_day_and_blocks_promotion() -> None:
    rows = evidence_rows(days=2)
    for decision in rows["decisions"]:
        if decision["race_date"] == "2026-08-01":
            decision["model_hash"] = "b" * 64

    result = aggregate(rows, expected_model_hash=None)

    assert [row["race_date"] for row in result["daily"]] == ["2026-07-31"]
    assert result["excluded_days"][0]["race_date"] == "2026-08-01"
    assert result["model_identity"]["fixed"] is False
    assert result["promotion_gate"]["checks"]["identity_fixed"] is False
    assert result["promotion_gate"]["pass"] is False


def test_outer_outcomes_change_only_evaluation_not_decision_boundary() -> None:
    source = evidence_rows()
    losing = copy.deepcopy(source)
    for index, settlement in enumerate(losing["settlements"]):
        selected = losing["decisions"][index]["selected_candidates"][0]["combination"]
        actual = next(combination for combination in COMBINATIONS if combination != selected)
        settlement.update(
            actual_combination=actual,
            payout_yen_per_100=300,
            return_yen=0,
            profit_yen=-100,
        )
        losing["payouts"][index].update(combination=actual, payout_yen=300)

    winning_result = aggregate(source)
    losing_result = aggregate(losing)

    assert winning_result["model_identity"] == losing_result["model_identity"]
    assert winning_result["information_boundary"] == losing_result["information_boundary"]
    assert winning_result["daily"][0]["coverage"] == losing_result["daily"][0]["coverage"]
    assert winning_result["bankroll"]["return_yen"] == 600
    assert losing_result["bankroll"]["return_yen"] == 0
    assert losing_result["information_boundary"] == {
        "decision_inputs": "stored_decision_and_source_snapshot_at_t300_or_earlier",
        "outcomes_used_only_for": "settlement_and_evaluation",
        "outer_result_used_as_decision_feature": False,
        "outer_payout_used_as_decision_feature": False,
        "real_betting_enabled": False,
    }


def test_overconfident_selected_probabilities_block_promotion() -> None:
    rows = evidence_rows(races_per_day=20)
    for index, decision in enumerate(rows["decisions"]):
        selected = decision["selected_candidates"][0]["combination"]
        probabilities = probability_vector(selected, mass=0.9)
        decision["probabilities"] = probabilities
        decision["diagnostics"]["v21_triple_head"][
            "ranking_probabilities"
        ] = probabilities
        if index < 2:
            rows["settlements"][index].update(
                payout_yen_per_100=3000,
                return_yen=3000,
                profit_yen=2900,
            )
            rows["payouts"][index]["payout_yen"] = 3000
            continue
        actual = next(item for item in COMBINATIONS if item != selected)
        rows["settlements"][index].update(
            actual_combination=actual,
            return_yen=0,
            profit_yen=-100,
        )
        rows["payouts"][index].update(combination=actual)

    result = aggregate(rows, minimum_races=20, minimum_tickets=20)

    calibration = result["purchase_probability_calibration"]
    assert calibration["observed_hits"] == 2
    assert calibration["expected_hits"] == pytest.approx(18.0)
    assert calibration["probability_at_most_observed_hits"] < 0.05
    assert result["promotion_gate"]["checks"][
        "selected_probability_not_overconfident"
    ] is False
    assert result["promotion_gate"]["pass"] is False


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakePostgresqlConnection:
    dialect = "postgresql"

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, query, parameters):
        normalized = " ".join(query.split())
        self.calls.append((normalized, parameters))
        if "FROM races r" in normalized:
            return FakeCursor(self.rows["races"])
        if "FROM intraday_t300_shadow_settlements" in normalized:
            return FakeCursor(self.rows["settlements"])
        if "JOIN odds_snapshots" in normalized:
            return FakeCursor(self.rows["source_odds"])
        if "FROM payouts p" in normalized:
            return FakeCursor(self.rows["payouts"])
        if "FROM intraday_t300_shadow_decisions" in normalized:
            return FakeCursor(self.rows["decisions"])
        raise AssertionError(normalized)


def test_postgresql_entrypoint_reads_required_sources_and_atomically_writes_json(
    tmp_path: Path,
) -> None:
    conn = FakePostgresqlConnection(evidence_rows())
    output = tmp_path / "nested" / "v21-evidence.json"

    result = collect_v21_prospective_evidence(
        conn, config=config(), output_path=output
    )

    assert result["promotion_gate"]["pass"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not list(output.parent.glob(".*.tmp"))
    sql = " ".join(query for query, _ in conn.calls)
    assert "intraday_t300_shadow_decisions" in sql
    assert "intraday_t300_shadow_settlements" in sql
    assert "odds_snapshots" in sql and "odds_trifecta" in sql
    assert "payouts" in sql and "races" in sql


def test_atomic_writer_replaces_existing_json(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    output.write_text('{"old": true}', encoding="utf-8")

    write_v21_prospective_evidence_atomic(output, {"new": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"new": True}


def test_db_entrypoint_rejects_non_postgresql_connection() -> None:
    with pytest.raises(ValueError, match="requires PostgreSQL"):
        collect_v21_prospective_evidence(
            object(), config=config()
        )
