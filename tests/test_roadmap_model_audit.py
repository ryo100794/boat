from pathlib import Path

from boatrace_ai.web import dashboard
from boatrace_ai.web.roadmap_model_status import archive_oracle_audit_status


def _job(job_id: int, status: str, *, eligible: bool = False) -> dict:
    return {
        "job_id": job_id,
        "task_type": "archive_market_oracle",
        "model_key": f"archive-{job_id}",
        "status": status,
        "decision": "reject_or_research_only" if status == "completed" else None,
        "updated_at": "2026-08-05T08:00:00+09:00",
        "metrics": {
            "model": "archive_closing_market_oracle_v1",
            "nested_value_model": "nested_stacked_value_calibration_v43",
            "nested_value_promotion_eligible": eligible,
        },
    }


def test_archive_audit_waits_for_running_and_queued_results() -> None:
    audit = archive_oracle_audit_status(
        {
            "latest_archive_oracle": _job(12806, "queued"),
            "running_archive_oracle": _job(12805, "running"),
            "queued_archive_oracle": _job(12806, "queued"),
            "latest_completed_archive_oracle": _job(12804, "completed"),
        }
    )

    assert audit["status"] == "評価実行中"
    assert audit["audit_ready"] is False
    assert audit["promotion_status"] == "未承認"
    assert audit["running"]["job_id"] == 12805
    assert audit["queued"]["job_id"] == 12806
    assert audit["latest_completed"]["job_id"] == 12804
    assert len(audit["audit_snapshot_id"]) == 64
    assert audit["snapshot_generated_at"] == "2026-08-05T08:00:00+09:00"
    assert audit["target_job_id"] == 12805
    assert "heartbeat timestamps excluded" in audit["audit_snapshot_basis"]


def test_archive_audit_becomes_available_only_after_latest_result() -> None:
    completed = _job(12806, "completed", eligible=True)
    audit = archive_oracle_audit_status(
        {
            "latest_archive_oracle": completed,
            "running_archive_oracle": None,
            "queued_archive_oracle": None,
            "latest_completed_archive_oracle": completed,
        }
    )

    assert audit["status"] == "外部監査可能"
    assert audit["audit_ready"] is True
    assert audit["promotion_status"] == "昇格ゲート合格"


def test_archive_audit_does_not_fall_back_after_latest_failure() -> None:
    audit = archive_oracle_audit_status(
        {
            "latest_archive_oracle": _job(12806, "failed"),
            "running_archive_oracle": None,
            "queued_archive_oracle": None,
            "latest_completed_archive_oracle": _job(12804, "completed"),
        }
    )

    assert audit["status"] == "評価失敗"
    assert audit["audit_ready"] is False

def test_archive_audit_snapshot_is_stable_across_heartbeat_updates() -> None:
    running = _job(12805, "running")
    queue = {
        "latest_archive_oracle": running,
        "running_archive_oracle": running,
        "queued_archive_oracle": None,
        "latest_completed_archive_oracle": _job(12804, "completed"),
    }
    first = archive_oracle_audit_status(queue)
    running["updated_at"] = "2026-08-05T08:01:00+09:00"
    second = archive_oracle_audit_status(queue)

    assert first["audit_snapshot_id"] == second["audit_snapshot_id"]


def test_archive_audit_snapshot_changes_with_logical_result() -> None:
    completed = _job(12806, "completed")
    queue = {
        "latest_archive_oracle": completed,
        "running_archive_oracle": None,
        "queued_archive_oracle": None,
        "latest_completed_archive_oracle": completed,
    }
    first = archive_oracle_audit_status(queue)
    completed["decision"] = "promote"
    second = archive_oracle_audit_status(queue)

    assert first["audit_snapshot_id"] != second["audit_snapshot_id"]


def test_dashboard_and_roadmap_raw_html_share_server_audit_snapshot(
    monkeypatch,
) -> None:
    completed = _job(12807, "completed")
    completed["metrics"].update({
        "frozen_probability_model": "prediction-v44",
        "joint_scenario_model_hash": "j" * 64,
        "calibrator_hash": "c" * 64,
        "calibration_ledger_hash": "l" * 64,
        "model": "strict-prior-v45",
        "settlement_engine_hash": "p" * 64,
        "decision_reason": "WARMUP_CANDIDATE_DAYS",
        "source_revision": "abc123",
    })
    queue = {
        "latest_archive_oracle": completed,
        "running_archive_oracle": None,
        "queued_archive_oracle": None,
        "latest_completed_archive_oracle": completed,
    }
    monkeypatch.setattr(
        dashboard,
        "archive_oracle_queue_status",
        lambda _path, *, connector: queue,
    )
    monkeypatch.setattr(
        dashboard,
        "_roadmap_work_tickets",
        lambda _path: [{
            "ticket_key": "AUDIT-1",
            "status": "completed",
            "progress": 100,
            "acceptance_criteria": "strict prior",
        }],
    )
    monkeypatch.setattr(
        dashboard,
        "roadmap_milestones_status",
        lambda _path, _query: {
            "milestones": [{
                "id": "M1",
                "status": "完了",
                "progress": 100,
                "next": "sealed evaluation",
            }]
        },
    )
    monkeypatch.setattr(
        dashboard,
        "model_performance_audit_snapshot",
        lambda _status: ('<section id="serverAuditSection"></section>', []),
    )

    dashboard_html = dashboard.dashboard_operational_html(Path("/tmp/test.db"))
    model_html = dashboard.model_performance_html(Path("/tmp/test.db"))
    roadmap_html = dashboard.roadmap_operational_html(Path("/tmp/test.db"))
    snapshot = archive_oracle_audit_status(queue)["audit_snapshot_id"]

    for raw_html in (dashboard_html, model_html, roadmap_html):
        assert 'id="serverOperationalAudit"' in raw_html
        assert snapshot in raw_html
        assert "active_prediction_model_id" in raw_html
        assert "2026-08-05T08:00:00+09:00" in raw_html
        assert "abc123" in raw_html
        assert "prediction-v44" in raw_html
        assert "WARMUP_CANDIDATE_DAYS" in raw_html
    assert "AUDIT-1" in roadmap_html
    assert "sealed evaluation" in roadmap_html
    assert "読込中" not in roadmap_html
