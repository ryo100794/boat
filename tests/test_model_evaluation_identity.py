from __future__ import annotations

import json
from pathlib import Path

from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _backtest_summary,
    _bankroll_summary,
    _database_evaluation_artifacts,
    _distinct_evaluation_results,
    _remote_backtest_report_summaries,
    _remote_bankroll_report_summaries,
)
from boatrace_ai.web.model_evaluation_identity import (
    compatible_evaluation_results,
    evaluation_cohort_payload,
    evaluation_identity,
)


def cohort(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evaluation_from": "2025-08-01",
        "evaluation_through": "2026-07-31",
        "evaluation_race_set_sha256": "race-a",
        "protocol_sha256": "protocol-a",
        "policy_sha256": "policy-a",
        "decision_odds_mode": "T-5",
        "decision_minutes_before_deadline": 5,
        "policy": {
            "daily_budget_yen": 10_000,
            "allocation_mode": "learned_allocation",
            "profit_reinvestment": True,
        },
        "fold_definition": {"kind": "walk_forward", "folds": 5},
    }
    value.update(overrides)
    return value


def test_cohort_hash_canonicalizes_every_comparison_axis() -> None:
    baseline = evaluation_identity(
        cohort(), path="alpha.json", label="alpha", result_kind="prediction"
    )
    reordered = cohort(
        fold_definition={"folds": 5, "kind": "walk_forward"},
        policy={
            "profit_reinvestment": True,
            "allocation_mode": "learned_allocation",
            "daily_budget_yen": 10_000,
        },
    )
    assert evaluation_identity(
        reordered, path="alpha.json", label="renamed", result_kind="prediction"
    )["evaluation_cohort_id"] == baseline["evaluation_cohort_id"]

    mutations = (
        {"evaluation_race_set_sha256": "race-b"},
        {"protocol_sha256": "protocol-b"},
        {"policy_sha256": "policy-b"},
        {"decision_odds_mode": "closing"},
        {"decision_minutes_before_deadline": 1},
        {"policy": {"daily_budget_yen": 20_000, "allocation_mode": "learned_allocation", "profit_reinvestment": True}},
        {"policy": {"daily_budget_yen": 10_000, "allocation_mode": "kelly", "profit_reinvestment": True}},
        {"policy": {"daily_budget_yen": 10_000, "allocation_mode": "learned_allocation", "profit_reinvestment": False}},
        {"fold_definition": {"kind": "walk_forward", "folds": 4}},
    )
    for mutation in mutations:
        changed = cohort(**mutation)
        identity = evaluation_identity(
            changed, path="alpha.json", label="alpha", result_kind="prediction"
        )
        assert identity["evaluation_cohort_id"] != baseline["evaluation_cohort_id"]


def test_missing_immutable_cohort_identity_never_generates_merge_bundle() -> None:
    legacy = {
        "evaluation_from": "2026-07-01",
        "evaluation_through": "2026-07-31",
        "decision_odds_mode": "T-5",
    }
    identity = evaluation_identity(
        legacy, path="legacy.json", label="same-model", result_kind="prediction"
    )
    assert evaluation_cohort_payload(legacy) is None
    assert identity["evaluation_cohort_id"] is None
    assert identity["evaluation_bundle_id"] is None


def test_same_artifact_prediction_and_bankroll_share_only_bundle_and_cohort() -> None:
    data = {
        **cohort(),
        "generated_at": "2026-08-01T00:00:00Z",
        "evaluated_races": 100,
        "entry_log_loss": 1.2,
        "roi": 1.01,
        "stake_yen": 1000,
        "return_yen": 1010,
    }
    prediction = _backtest_summary(Path("run.json"), "alpha", data)
    bankroll = _bankroll_summary(Path("run.json"), "alpha", data)
    other = _bankroll_summary(
        Path("run.json"), "alpha", {**data, "protocol_sha256": "other"}
    )

    assert prediction["evaluation_run_id"] == bankroll["evaluation_run_id"]
    assert prediction["result_id"] != bankroll["result_id"]
    assert compatible_evaluation_results(prediction, bankroll) is True
    assert compatible_evaluation_results(prediction, other) is False


def test_job_and_attempt_form_run_id() -> None:
    identity = evaluation_identity(
        cohort(),
        path="result.json",
        label="alpha",
        result_kind="prediction",
        job={"db_job_id": 42, "attempt": 3},
    )
    assert identity["evaluation_run_id"] == "job_42_attempt_3"
    assert identity["db_job_id"] == 42
    assert identity["attempt"] == 3


def test_database_artifacts_keep_same_model_from_distinct_runs(tmp_path: Path) -> None:
    candidates = []
    for job_id, protocol in ((10, "protocol-a"), (11, "protocol-b")):
        path = tmp_path / f"job-{job_id}.json"
        path.write_text(
            json.dumps(
                {
                    **cohort(protocol_sha256=protocol),
                    "model": "same-model",
                    "generated_at": f"2026-08-01T00:00:{job_id}Z",
                    "evaluated_races": 100,
                    "entry_log_loss": 1.2,
                    "bankroll": {
                        "evaluated_races": 100,
                        "roi": 0.9,
                        "stake_yen": 1000,
                        "return_yen": 900,
                    },
                }
            ),
            encoding="utf-8",
        )
        candidates.append(
            {
                "model_key": "same-model",
                "result_path": str(path),
                "db_job_id": job_id,
                "attempt": 1,
                "roi": 0.9,
            }
        )

    backtests, bankroll, _ = _database_evaluation_artifacts(
        {"candidates": candidates}, tmp_path
    )

    assert len(backtests) == 2
    assert len(bankroll) == 2
    assert {row["evaluation_run_id"] for row in backtests} == {
        "job_10_attempt_1",
        "job_11_attempt_1",
    }
    assert len({row["evaluation_cohort_id"] for row in backtests}) == 2


def test_legacy_javascript_never_joins_by_scope_and_model_key() -> None:
    source = MODEL_REPORT_HTML.split("function explicitComparisonIdentity(row){", 1)[1].split(
        "function predictionRow", 1
    )[0]

    assert "evaluation_bundle_id" in source
    assert "evaluation_cohort_id" in source
    assert "candidate.identity===identity&&!candidate[type]" in source
    assert "evaluation_scope+model_key" not in source
    assert 'scope+":"+modelKey' not in source
    assert "row[type]=item" in source


def test_result_deduplication_only_removes_exact_result_id() -> None:
    rows = _distinct_evaluation_results(
        [
            {"name": "same-model", "result_id": "result-a", "evaluation_run_id": "run-a"},
            {"name": "same-model", "result_id": "result-a", "evaluation_run_id": "run-a"},
            {"name": "same-model", "result_id": "result-b", "evaluation_run_id": "run-b"},
            {"name": "same-model", "result_id": None, "evaluation_run_id": None},
            {"name": "same-model", "result_id": None, "evaluation_run_id": None},
        ]
    )
    assert [row.get("result_id") for row in rows] == [
        "result-a",
        "result-b",
        None,
        None,
    ]


def test_remote_summaries_propagate_job_and_cohort_identity() -> None:
    metrics = {
        **cohort(),
        "evaluated_races": 100,
        "entry_log_loss": 1.2,
        "roi": 0.9,
        "stake_yen": 1000,
        "return_yen": 900,
    }
    remote = {
        "generated_at": "2026-08-02T00:00:00Z",
        "jobs": [
            {
                "db_job_id": 71,
                "attempt": 2,
                "name": "standardized_365d_v2_alpha",
                "kind": "standardized_365d_v2_model",
                "output": "/workspace/boat/data/models/alpha.json",
                "result": {"file": "alpha.json", "metrics": metrics},
            }
        ],
    }

    prediction = _remote_backtest_report_summaries(remote)[0]
    bankroll = _remote_bankroll_report_summaries(remote)[0]
    assert prediction["evaluation_run_id"] == bankroll["evaluation_run_id"] == "job_71_attempt_2"
    assert prediction["evaluation_cohort_id"] == bankroll["evaluation_cohort_id"]
    assert prediction["evaluation_bundle_id"] == bankroll["evaluation_bundle_id"]
    assert prediction["result_id"] != bankroll["result_id"]
