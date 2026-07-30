from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import teleboat_opening_preview as preview
from teleboat_agent.journal import verify_journal
from teleboat_agent.login_secrets import LoginSecrets, save_login_secrets


JST = timezone(timedelta(hours=9))
TARGET = date(2026, 7, 31)
NOW = datetime(2026, 7, 31, 8, 30, tzinfo=JST)


def _secret(path: Path) -> None:
    save_login_secrets(
        path,
        LoginSecrets(
            mode="mobile",
            member_number="12345678",
            pin="2468",
            auth_secret="1357",
        ),
    )


def _candidate(**updates):
    value = {
        "race_id": "20260731-01-01",
        "race_date": "2026-07-31",
        "jcd": "01",
        "rno": 1,
        "entries": 6,
        "buy_until_at": "2026-07-31T09:00:00+09:00",
        "top_prediction": {"combination": "1-2-3", "probability": 0.1},
    }
    value.update(updates)
    return value


def _args(tmp_path: Path, secret: Path, **updates):
    values = {
        "date": TARGET,
        "dashboard_url": "https://dashboard.example/",
        "poll_seconds": 0.001,
        "timeout_seconds": 0.02,
        "output": tmp_path / "preview.json",
        "secret_path": secret,
        "journal_path": tmp_path / "votes.jsonl",
        "validate_only": False,
    }
    values.update(updates)
    return type("Args", (), values)()


def _verified_result():
    keys = (
        "authentication", "official_host_allowlist", "mode_selection",
        "ticket_inputs", "confirmation_identity", "confirmation_selection",
        "confirmation_ticket_count", "confirmation_stake_yen",
        "official_hidden_total", "unfinished_marker", "final_button_ready",
    )
    return {
        "status": "preview_verified", "tickets": 1, "stake_yen": 100,
        "final_button_clicked": False, "logout_confirmed": True,
        "verifications": {key: True for key in keys},
    }


def test_preview_uses_first_candidate_quantity_one_and_never_executes(tmp_path):
    secret = tmp_path / "login.json"
    _secret(secret)
    calls = []

    class Executor:
        def __init__(self, settings):
            assert settings.member_number == "12345678"

        def preview(self, request):
            calls.append(request)
            assert request.tickets[0].quantity == 1
            assert request.total_stake_yen == 100
            assert request.tickets[0].betting_number.value == "123"
            return _verified_result()

        def execute(self, request):
            pytest.fail("execute must never be called")

    payload = {"date": "2026-07-31", "candidates": [
        _candidate(), _candidate(rno=2, top_prediction={"combination": "2-3-4"})
    ]}
    args = _args(tmp_path, secret)
    code = preview.run(
        args, fetch_guide=lambda *_args, **_kwargs: payload,
        executor_factory=Executor, clock=lambda: NOW,
        sleeper=lambda _seconds: None,
    )
    assert code == preview.EXIT_OK
    assert len(calls) == 1
    saved = json.loads(args.output.read_text())
    assert saved["selection"]["race_number"] == 1
    assert saved["selection"]["quantity"] == 1
    assert saved["selection"]["stake_yen"] == 100
    assert saved["submission_attempted"] is False
    assert saved["official_confirmation"]["final_button_clicked"] is False
    assert saved["request_id"]
    verified = verify_journal(args.journal_path)
    assert verified["valid"] is True
    assert verified["events"] == {"opening_preview_verified": 1}
    journal_row = json.loads(args.journal_path.read_text())
    assert journal_row["request_id"] == saved["request_id"]
    assert journal_row["request_snapshot"]["tickets"][0]["quantity"] == 1
    assert journal_row["final_button_clicked"] is False
    assert journal_row["submission_attempted"] is False
    journal_text = args.journal_path.read_text()
    assert "12345678" not in journal_text
    assert "2468" not in journal_text
    assert "1357" not in journal_text


@pytest.mark.parametrize(("payload", "expected"), [
    ({"date": "2026-07-30", "candidates": [_candidate()]}, preview.EXIT_GUIDE_REJECTED),
    ({"date": "2026-07-31", "candidates": [_candidate(entries=5)]}, preview.EXIT_NO_CANDIDATE),
    ({"date": "2026-07-31", "candidates": [_candidate(top_prediction=None)]}, preview.EXIT_NO_CANDIDATE),
    ({"date": "2026-07-31", "candidates": [_candidate(buy_until_at=NOW.isoformat())]}, preview.EXIT_NO_CANDIDATE),
])
def test_date_deadline_and_candidate_fail_closed(tmp_path, payload, expected):
    secret = tmp_path / "login.json"
    _secret(secret)
    args = _args(tmp_path, secret)
    code = preview.run(
        args, fetch_guide=lambda *_args, **_kwargs: payload,
        executor_factory=lambda _settings: pytest.fail("browser must not start"),
        clock=lambda: NOW, sleeper=lambda _seconds: None,
    )
    assert code == expected
    assert json.loads(args.output.read_text())["status"] == "failed"
    assert verify_journal(args.journal_path)["events"] == {
        "opening_preview_failed": 1
    }


def test_invalid_credentials_fail_closed_without_browser(tmp_path):
    secret = tmp_path / "login.json"
    secret.write_text('{"mode":"mobile","member_number":"invalid"}', encoding="utf-8")
    secret.chmod(0o600)
    args = _args(tmp_path, secret)
    code = preview.run(
        args, executor_factory=lambda _settings: pytest.fail("browser must not start")
    )
    assert code == preview.EXIT_PREVIEW_FAILED
    saved = args.output.read_text()
    assert "member_number" not in saved
    assert "invalid" not in saved
    journal_text = args.journal_path.read_text()
    assert "member_number" not in journal_text
    assert "invalid" not in journal_text


def test_validate_only_has_no_browser_or_network_and_no_secrets_in_output(tmp_path):
    secret = tmp_path / "login.json"
    _secret(secret)
    args = _args(tmp_path, secret, validate_only=True)
    code = preview.run(
        args,
        fetch_guide=lambda *_args, **_kwargs: pytest.fail("network must not be used"),
        executor_factory=lambda _settings: pytest.fail("browser must not start"),
    )
    assert code == preview.EXIT_OK
    saved = args.output.read_text()
    assert "12345678" not in saved
    assert "2468" not in saved
    assert "1357" not in saved
    assert json.loads(saved)["status"] == "validated"
    assert verify_journal(args.journal_path)["events"] == {
        "opening_preview_configuration_validated": 1
    }
    journal_text = args.journal_path.read_text()
    assert "12345678" not in journal_text
    assert "2468" not in journal_text
    assert "1357" not in journal_text


def test_unverified_official_confirmation_fails_closed(tmp_path):
    secret = tmp_path / "login.json"
    _secret(secret)

    class Executor:
        def __init__(self, _settings):
            pass

        def preview(self, _request):
            result = _verified_result()
            result["verifications"]["unfinished_marker"] = False
            return result

        def execute(self, _request):
            pytest.fail("execute must never be called")

    args = _args(tmp_path, secret)
    code = preview.run(
        args,
        fetch_guide=lambda *_args, **_kwargs: {
            "date": "2026-07-31", "candidates": [_candidate()]
        },
        executor_factory=Executor, clock=lambda: NOW,
        sleeper=lambda _seconds: None,
    )
    assert code == preview.EXIT_PREVIEW_FAILED
    assert json.loads(args.output.read_text())["code"] == "official_preview_not_verified"


def test_target_date_candidate_is_not_used_on_previous_jst_day(tmp_path):
    secret = tmp_path / "login.json"
    _secret(secret)
    args = _args(tmp_path, secret)
    previous_night = datetime(2026, 7, 30, 23, 59, tzinfo=JST)
    code = preview.run(
        args,
        fetch_guide=lambda *_args, **_kwargs: {
            "date": "2026-07-31",
            "candidates": [_candidate()],
        },
        executor_factory=lambda _settings: pytest.fail("browser must not start"),
        clock=lambda: previous_night,
        sleeper=lambda _seconds: None,
    )
    assert code == preview.EXIT_NO_CANDIDATE
    assert verify_journal(args.journal_path)["events"] == {
        "opening_preview_failed": 1
    }


def test_elapsed_target_date_fails_closed(tmp_path):
    with pytest.raises(preview.PreviewFailure, match="guide_date_mismatch"):
        preview.select_candidate(
            {"date": "2026-07-31", "candidates": [_candidate()]},
            target_date=TARGET,
            now=datetime(2026, 8, 1, 0, 0, tzinfo=JST),
        )


def test_journal_write_failure_is_fail_closed(tmp_path):
    secret = tmp_path / "login.json"
    _secret(secret)
    journal_directory = tmp_path / "not-a-file"
    journal_directory.mkdir()
    args = _args(
        tmp_path,
        secret,
        validate_only=True,
        journal_path=journal_directory,
    )
    code = preview.run(args)
    assert code == preview.EXIT_PREVIEW_FAILED
    saved = json.loads(args.output.read_text())
    assert saved["code"] == "journal_write_failed"
    assert saved["submission_attempted"] is False
