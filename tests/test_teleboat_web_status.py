from __future__ import annotations

import json
import os
from pathlib import Path

from boatrace_ai.web_dashboard import (
    HTML,
    TELEBOAT_REPORT_HTML,
    _roadmap_improvements,
    _roadmap_milestones,
    teleboat_status,
)
from teleboat_agent.login_probe import write_probe_status


def test_probe_status_preserves_phases_without_secret_values(tmp_path: Path) -> None:
    status_path = tmp_path / "teleboat_probe_status.json"
    write_probe_status(
        status_path,
        "public",
        {"success": True, "public_page_ready": True, "wager_actions": 0},
    )
    write_probe_status(
        status_path,
        "login",
        {
            "success": True,
            "authenticated": True,
            "logout_confirmed": True,
            "wager_actions": 0,
        },
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert payload["public"]["success"] is True
    assert payload["login"]["logout_confirmed"] is True
    assert payload["policy"] == {
        "live_wager_enabled": False,
        "wager_actions": 0,
    }


def test_web_status_filters_credentials_and_completes_m8(tmp_path: Path) -> None:
    db_path = tmp_path / "boatrace.sqlite"
    status_path = tmp_path / "teleboat_probe_status.json"
    status_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-19T00:00:00+00:00",
                "latest_phase": "login",
                "public": {
                    "success": True,
                    "public_page_ready": True,
                    "member_number": "must-not-leak",
                },
                "login": {
                    "success": True,
                    "authenticated": True,
                    "logout_confirmed": True,
                    "wager_actions": 0,
                    "pin": "must-not-leak",
                    "auth_secret": "must-not-leak",
                },
            }
        ),
        encoding="utf-8",
    )
    secret_path = tmp_path / "login.json"
    secret_path.write_text("{}", encoding="utf-8")
    os.chmod(secret_path, 0o600)

    status = teleboat_status(db_path, secret_path=secret_path)
    serialized = json.dumps(status)

    assert status["connection_status"] == "ログイン・ログアウト確認済み"
    assert status["readiness"]["secret_configured"] is True
    assert status["readiness"]["secret_permission_valid"] is True
    assert status["readiness"]["execution_host"] == "local"
    assert status["readiness"]["browser_mode"] == "headless"
    assert status["live_wager_enabled"] is False
    assert "member_number" not in serialized
    assert "pin" not in serialized
    assert "auth_secret" not in serialized

    milestones = {
        row["id"]: row
        for row in _roadmap_milestones({}, [], {}, status)
    }
    improvements = {
        row["id"]: row
        for row in _roadmap_improvements({}, [], {}, status)
    }
    assert milestones["M8"]["status"] == "完了"
    assert improvements["M8-1"]["progress"] == 100


def test_dashboard_links_to_sanitized_teleboat_report() -> None:
    assert 'href="/reports/teleboat"' in HTML
    assert "/api/reports/teleboat-status" in TELEBOAT_REPORT_HTML
    assert "加入番号" in TELEBOAT_REPORT_HTML
    assert "password" not in TELEBOAT_REPORT_HTML.lower()
