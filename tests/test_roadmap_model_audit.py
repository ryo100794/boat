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
