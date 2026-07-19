from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from boatrace_ai.web_dashboard import (
    HTML,
    TELEBOAT_REPORT_HTML,
    TELEBOAT_SETUP_HTML,
    _configure_teleboat_login,
    _roadmap_improvements,
    _teleboat_setup_allowed,
    _teleboat_setup_page,
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


def test_teleboat_setup_is_localhost_only_and_masks_all_inputs() -> None:
    local = SimpleNamespace(
        client_address=("127.0.0.1", 54321),
        headers={"Host": "localhost:10001"},
    )
    remote = SimpleNamespace(
        client_address=("192.0.2.10", 54321),
        headers={"Host": "localhost:10001"},
    )
    proxied = SimpleNamespace(
        client_address=("127.0.0.1", 54321),
        headers={"Host": "dashboard.example.invalid"},
    )
    secure_proxy = SimpleNamespace(
        client_address=("127.0.0.1", 54321),
        headers={
            "Host": "dashboard.example.invalid",
            "X-Forwarded-Proto": "https",
        },
    )

    page = _teleboat_setup_page("one-time-token", "<invalid>")

    assert _teleboat_setup_allowed(local) is True
    assert _teleboat_setup_allowed(remote) is False
    assert _teleboat_setup_allowed(proxied) is False
    assert _teleboat_setup_allowed(secure_proxy) is True
    assert page.count("type=\"password\"") == 3
    assert "one-time-token" in page
    assert "&lt;invalid&gt;" in page
    assert "__FORM__" not in page
    assert "__ERROR__" not in page
    assert "type=\"password\"" in TELEBOAT_SETUP_HTML or "__FORM__" in TELEBOAT_SETUP_HTML


def test_web_setup_saves_owner_only_secret_and_reports_sanitized_probe(tmp_path: Path) -> None:
    class FakeResult:
        public_page_ready = True
        authenticated = True
        logout_confirmed = True
        wager_actions = 0

        def to_dict(self):
            return {
                "mode": "mobile",
                "browser": "chromium",
                "public_page_ready": True,
                "authenticated": True,
                "logout_confirmed": True,
                "wager_actions": 0,
                "attempts": 1,
                "final_location": "https://spweb.brtb.jp/",
                "elapsed_seconds": 0.1,
            }

    class FakeProbe:
        def login_probe(self, login_secrets):
            assert "12345678" not in repr(login_secrets)
            return FakeResult()

    secret_path = tmp_path / "private" / "login.json"
    status_path = tmp_path / "teleboat_probe_status.json"
    result = _configure_teleboat_login(
        {
            "member_number": "12345678",
            "pin": "5678",
            "auth_secret": "4321",
        },
        secret_path=secret_path,
        status_path=status_path,
        probe_factory=FakeProbe,
    )
    public_status = json.loads(status_path.read_text(encoding="utf-8"))
    serialized_result = json.dumps(result)
    serialized_status = json.dumps(public_status)

    assert result["success"] is True
    assert secret_path.stat().st_mode & 0o777 == 0o600
    assert "12345678" not in serialized_result
    assert "5678" not in serialized_result
    assert "4321" not in serialized_result
    assert "12345678" not in serialized_status
    assert public_status["login"]["logout_confirmed"] is True


def test_web_setup_classifies_rejected_credentials_without_leaking_them(tmp_path: Path) -> None:
    class RejectedResult:
        public_page_ready = True
        authenticated = False
        logout_confirmed = False
        wager_actions = 0

        def to_dict(self):
            return {
                "mode": "mobile",
                "browser": "chromium",
                "public_page_ready": True,
                "authenticated": False,
                "logout_confirmed": False,
                "wager_actions": 0,
                "attempts": 1,
                "final_location": "https://login.brtb.jp/auth/realms/boat/",
                "elapsed_seconds": 1.0,
            }

    class RejectedProbe:
        def login_probe(self, login_secrets):
            return RejectedResult()

    result = _configure_teleboat_login(
        {
            "member_number": "12345678",
            "pin": "5678",
            "auth_secret": "4321",
        },
        secret_path=tmp_path / "private" / "login.json",
        status_path=tmp_path / "status.json",
        probe_factory=RejectedProbe,
    )

    assert result["success"] is False
    assert result["error_code"] == "credentials_rejected"
    assert "12345678" not in json.dumps(result)
