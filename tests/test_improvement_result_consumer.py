from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from boatrace_ai import improvement_result_consumer as consumer


NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)


def _candidate(
    candidate_id: int,
    *,
    source_job_id: int | None = 3565,
    roi: float | None = 1.1,
    robust_roi: float | None = 1.05,
    lower_roi: float | None = 1.01,
    evaluated_races: int = 12_000,
    parameters: dict | None = None,
    metrics: dict | None = None,
    result_identity: dict | None = None,
    producer_decision: str = "reject_or_research_only",
    created_at: str = "2026-07-30T08:00:00Z",
) -> consumer.Candidate:
    base_parameters = {
        "source_job_id": source_job_id,
        "evaluation_days": 365,
        "daily_budget_yen": 10_000,
        "learning_rate": 0.02,
        "epochs": 2,
        "batch_races": 1000,
        "candidate_count": 24,
        "finalists": 6,
        "bootstrap_samples": 20_000,
        "research_only": True,
        "timeout_seconds": 43_200,
    }
    if parameters:
        base_parameters.update(parameters)
    base_metrics = {
        "evaluation_days": 365,
        "evaluated_races": evaluated_races,
        "comparison_role": "bankroll_policy_model",
        "roi": roi,
        "roi_without_largest_hit": robust_roi,
        "roi_ci95_lower": lower_roi,
        "effective_hit_count": 24.0,
        "profit_yen": 10_000,
        "max_drawdown_yen": 2_000,
    }
    if metrics:
        base_metrics.update(metrics)
    identity = {
        "evaluation_scope": "untouched_365d_holdout",
        "evaluation_from": "2025-07-25",
        "evaluation_through": "2026-07-24",
        "selection_protocol": "strict_outer_holdout_v2",
        "race_universe_sha256": "race-universe-a",
        "odds_coverage_definition": "official_trifecta_closing_complete",
        "selected_policy": {"minimum_ev": 1.2, "risk_fraction": 0.05},
    }
    if result_identity:
        identity.update(result_identity)
    return consumer.Candidate(
        candidate_id=candidate_id,
        job_id=7000 + candidate_id,
        model_key=f"policy-{candidate_id}",
        task_type="bankroll_policy_search",
        producer_decision=producer_decision,
        metrics=base_metrics,
        parameters=base_parameters,
        result_path=f"job-{7000 + candidate_id}.json",
        created_at=created_at,
        result_identity=identity,
    )


class _Cursor:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = list(rows or [])

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _ReviewConnection:
    def __init__(self, *, active_jobs=0, existing_review=None, existing_job_id=None):
        self.active_jobs = active_jobs
        self.existing_review = existing_review
        self.existing_job_id = existing_job_id
        self.updates = []
        self.review_insert = None

    def execute(self, statement, parameters=()):
        normalized = " ".join(statement.split())
        if normalized.startswith("SELECT COUNT(*) AS count FROM model_evaluation_jobs"):
            return _Cursor({"count": self.active_jobs})
        if normalized.startswith("SELECT winner_candidate_id"):
            return _Cursor(self.existing_review)
        if normalized.startswith("SELECT job_id FROM model_evaluation_jobs"):
            row = {"job_id": self.existing_job_id} if self.existing_job_id else None
            return _Cursor(row)
        if normalized.startswith("INSERT INTO model_improvement_candidate_reviews"):
            self.review_insert = parameters
            return _Cursor({"comparison_group_key": parameters[0]})
        if normalized.startswith("UPDATE model_improvement_candidates"):
            self.updates.append(parameters)
            return _Cursor()
        raise AssertionError(normalized)


def test_comparison_identity_ignores_model_axes_and_runtime_limits() -> None:
    first = _candidate(1)
    second = _candidate(
        2,
        parameters={
            "learning_rate": 0.0075,
            "coefficient_optimizer": "newton_cg",
            "candidate_count": 64,
            "timeout_seconds": 86_400,
            "seed": 99,
        },
    )
    assert consumer.comparison_group_key(first) == consumer.comparison_group_key(second)


@pytest.mark.parametrize(
    "changed",
    [
        _candidate(2, source_job_id=3566),
        _candidate(2, evaluated_races=11_999),
        _candidate(2, metrics={"comparison_role": "different_protocol"}),
        _candidate(2, parameters={"daily_budget_yen": 20_000}),
        _candidate(2, result_identity={"source_artifact_sha256": "different"}),
        _candidate(2, result_identity={"odds_coverage_definition": "t300_only"}),
    ],
)
def test_comparison_identity_separates_semantically_different_candidates(changed) -> None:
    assert consumer.comparison_group_key(_candidate(1)) != consumer.comparison_group_key(changed)


def test_equal_size_different_date_windows_are_never_comparable() -> None:
    first = _candidate(1)
    shifted = _candidate(
        2,
        result_identity={
            "evaluation_from": "2025-07-26",
            "evaluation_through": "2026-07-25",
        },
    )
    assert first.metrics["evaluated_races"] == shifted.metrics["evaluated_races"]
    assert consumer.comparison_group_key(first) != consumer.comparison_group_key(shifted)


def test_missing_actual_boundaries_isolates_each_candidate() -> None:
    first = replace(_candidate(1), result_identity={})
    second = replace(_candidate(2), result_identity={})
    first = replace(first, metrics={k: v for k, v in first.metrics.items() if "date" not in k})
    second = replace(second, metrics={k: v for k, v in second.metrics.items() if "date" not in k})
    assert consumer.comparison_group_key(first) != consumer.comparison_group_key(second)



def test_scope_without_temporal_boundary_still_isolates_candidates() -> None:
    first = replace(
        _candidate(1),
        result_identity={"evaluation_scope": "holdout"},
    )
    second = replace(
        _candidate(2),
        result_identity={"evaluation_scope": "holdout"},
    )
    assert consumer.comparison_group_key(first) != consumer.comparison_group_key(second)


def test_result_metadata_prefers_top_level_evaluation_window(tmp_path) -> None:
    result = tmp_path / "candidate.json"
    result.write_text(
        __import__("json").dumps(
            {
                "calibration": {
                    "evaluation_from": "2024-01-01",
                    "evaluation_through": "2024-01-31",
                },
                "evaluation_from": "2025-07-25",
                "evaluation_through": "2026-07-24",
            }
        ),
        encoding="utf-8",
    )
    metadata = consumer._result_metadata(str(result))
    assert metadata["evaluation_from"] == "2025-07-25"
    assert metadata["evaluation_through"] == "2026-07-24"


def test_rank_candidates_prefers_robust_return_before_raw_roi() -> None:
    fragile = _candidate(1, roi=1.8, robust_roi=0.7, lower_roi=0.6)
    robust = _candidate(2, roi=1.2, robust_roi=1.1, lower_roi=1.02)
    invalid = _candidate(3, roi=None)
    ranked, rejected = consumer.rank_candidates([fragile, invalid, robust])
    assert [item.candidate_id for item in ranked] == [2, 1]
    assert [item.candidate_id for item in rejected] == [3]


def test_formal_follow_up_audits_winner_and_nested_reselection() -> None:
    task_type, model_key, parameters, audit = consumer.formal_follow_up(
        _candidate(
            7,
            parameters={
                "learning_rate": 0.0075,
                "epochs": 3,
                "batch_races": 1500,
                "bootstrap_samples": 30_000,
                "ev_threshold": 1.35,
            },
        )
    )
    assert task_type == "bankroll_policy_nested_annual"
    assert ":winner-7:formal-365d:" in model_key
    assert parameters["source_job_id"] == 3565
    assert parameters["selection_days"] == parameters["outer_days"] == 365
    assert parameters["selection_bootstrap_samples"] == 30_000
    assert audit["winner_candidate_id"] == 7
    assert audit["winner_job_id"] == 7007
    assert audit["winner_result_path"] == "job-7007.json"
    assert audit["winning_policy_configuration"]["minimum_ev"] == 1.2
    assert audit["formal_method"] == "nested_annual_reselection"
    assert "outer holdout remains untouched" in audit["reselection_reason"]


def test_review_group_queues_one_formal_job_and_records_all_decisions() -> None:
    conn = _ReviewConnection()
    calls = []

    def enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return 8801

    result = consumer.review_group(
        conn,
        [_candidate(1, robust_roi=0.9), _candidate(2, robust_roi=1.1)],
        settle_seconds=0,
        now=NOW,
        enqueue=enqueue,
    )
    assert result["winner_candidate_id"] == 2
    assert result["downstream_job_id"] == 8801
    assert len(calls) == 1
    assert calls[0]["audit_parameters"]["winner_candidate_id"] == 2
    assert calls[0]["parent_job_id"] == 7002
    decisions = {parameters[-1]: parameters for parameters in conn.updates}
    assert decisions[2][0] == "winner_formal_follow_up_queued"
    assert decisions[2][4] == 8801
    assert decisions[1][0] == "not_selected"


def test_late_arriving_singleton_remains_pending_then_can_join() -> None:
    conn = _ReviewConnection()
    queued = []
    first = _candidate(1)

    pending = consumer.review_group(
        conn,
        [first],
        settle_seconds=0,
        now=NOW,
        enqueue=lambda *_args, **_kwargs: pytest.fail("must remain pending"),
    )
    assert pending["status"] == "pending"
    assert pending["reason"] == "waiting_for_comparable_candidate"
    assert conn.updates == []

    completed = consumer.review_group(
        conn,
        [first, _candidate(2, robust_roi=1.2)],
        settle_seconds=0,
        now=NOW,
        enqueue=lambda *_args, **kwargs: queued.append(kwargs) or 9001,
    )
    assert completed["status"] == "formal_follow_up_queued"
    assert len(queued) == 1


def test_group_waits_while_a_producer_sibling_is_running() -> None:
    conn = _ReviewConnection(active_jobs=1)
    result = consumer.review_group(
        conn,
        [_candidate(1), _candidate(2)],
        settle_seconds=0,
        now=NOW,
        enqueue=lambda *_args, **_kwargs: pytest.fail("must wait"),
    )
    assert result["status"] == "pending"
    assert result["reason"] == "waiting_for_1_producer_jobs"
    assert conn.updates == []


def test_group_waits_for_configured_settle_interval() -> None:
    result = consumer.review_group(
        _ReviewConnection(),
        [
            _candidate(1, created_at="2026-07-30T11:59:30Z"),
            _candidate(2, created_at="2026-07-30T11:59:45Z"),
        ],
        settle_seconds=60,
        now=NOW,
        enqueue=lambda *_args, **_kwargs: pytest.fail("must wait"),
    )
    assert result["status"] == "pending"
    assert result["reason"] == "settling_for_siblings:45s"


def test_review_group_rejects_mixed_semantics_before_writing() -> None:
    conn = _ReviewConnection()
    with pytest.raises(ValueError, match="not semantically comparable"):
        consumer.review_group(conn, [_candidate(1), _candidate(2, source_job_id=9999)])
    assert conn.updates == []


def test_existing_group_review_is_idempotent_and_does_not_enqueue() -> None:
    conn = _ReviewConnection(
        existing_review={
            "winner_candidate_id": 1,
            "downstream_job_id": 8100,
            "decision": "formal_follow_up_queued",
            "reason": "selected",
        }
    )
    result = consumer.review_group(
        conn,
        [_candidate(2), _candidate(3)],
        settle_seconds=0,
        now=NOW,
        enqueue=lambda *_args, **_kwargs: pytest.fail("must not enqueue"),
    )
    assert result["status"] == "already_reviewed"
    assert all(parameters[0] == "group_already_reviewed" for parameters in conn.updates)


class _LoadConnection:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, statement, parameters=()):
        self.statements.append((" ".join(statement.split()), parameters))
        return _Cursor(rows=self.rows)


def _row(candidate_id: int) -> dict:
    candidate = _candidate(candidate_id)
    return {
        "candidate_id": candidate.candidate_id,
        "job_id": candidate.job_id,
        "model_key": candidate.model_key,
        "task_type": candidate.task_type,
        "decision": candidate.producer_decision,
        "metrics": candidate.metrics,
        "parameters": candidate.parameters,
        "result_path": None,
        "created_at": candidate.created_at,
    }


def test_batch_size_cannot_split_a_supported_comparison_group() -> None:
    conn = _LoadConnection([_row(1), _row(2), _row(3)])
    loaded = consumer._load_supported_candidates(conn, batch_size=1)
    assert [item.candidate_id for item in loaded] == [1, 2, 3]
    assert "LIMIT" not in conn.statements[0][0]


class _LockConnection:
    def execute(self, statement, parameters=()):
        assert "pg_try_advisory_xact_lock" in statement
        return _Cursor({"acquired": True})


def test_unsupported_actionable_candidates_remain_pending_and_are_ticketed(monkeypatch) -> None:
    ticket_counts = []
    monkeypatch.setattr(consumer, "_load_supported_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(consumer, "_review_clearly_non_actionable", lambda *_a, **_k: 2)
    monkeypatch.setattr(consumer, "_unsupported_pending_count", lambda *_a, **_k: 7)
    monkeypatch.setattr(
        consumer,
        "_record_unsupported_ticket",
        lambda _conn, count: ticket_counts.append(count),
    )
    result = consumer.process_once(_LockConnection(), batch_size=5)
    assert result["unsupported_pending"] == 7
    assert result["clearly_non_actionable_reviewed"] == 2
    assert result["reviewed"] == 2
    assert ticket_counts == [7]


def test_non_actionable_bulk_review_is_limited_to_invalid_data_source() -> None:
    conn = _LoadConnection([{"candidate_id": 1}])
    assert consumer._review_clearly_non_actionable(conn, batch_size=20) == 1
    statement = conn.statements[0][0]
    assert "c.decision = 'invalid_data_source'" in statement
    assert "task_type <>" not in statement


def test_unsupported_identity_can_never_group_unrelated_jobs() -> None:
    candidate = replace(_candidate(1), task_type="genetic_island_search")
    other = replace(candidate, candidate_id=2, job_id=7002)
    assert consumer.comparison_group_key(candidate) != consumer.comparison_group_key(other)


def test_supervisor_declares_settle_interval() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-improvement-consumer.ini"
    ).read_text(encoding="utf-8")
    assert "boatrace_ai.improvement_result_consumer" in config
    assert "--settle-seconds 3600" in config


def test_enqueue_formal_job_persists_audit_and_uses_it_for_dedupe() -> None:
    class InsertConnection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, parameters=()):
            self.calls.append((" ".join(statement.split()), parameters))
            return _Cursor({"job_id": 9901})

    conn = InsertConnection()
    executable = {"source_job_id": 3565, "outer_days": 365}
    audit = {"winner_candidate_id": 7, "winner_result_path": "job-7007.json"}
    job_id = consumer.enqueue_formal_job(
        conn,
        task_type="bankroll_policy_nested_annual",
        model_key="winner:formal",
        parameters=executable,
        audit_parameters=audit,
        priority=95,
        max_attempts=3,
        parent_job_id=7007,
    )
    statement, parameters = conn.calls[0]
    assert job_id == 9901
    assert "improvement_audit" in statement
    assert __import__("json").loads(parameters[2]) == executable
    assert __import__("json").loads(parameters[3]) == audit
    assert parameters[4] == consumer.dedupe_key(
        "bankroll_policy_nested_annual", "winner:formal", executable, audit
    )
    changed = {**audit, "winner_candidate_id": 8}
    assert consumer.dedupe_key(
        "bankroll_policy_nested_annual", "winner:formal", executable, audit
    ) != consumer.dedupe_key(
        "bankroll_policy_nested_annual", "winner:formal", executable, changed
    )
