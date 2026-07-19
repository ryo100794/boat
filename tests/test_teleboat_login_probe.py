from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from teleboat_agent.login_probe import (
    LoginProbeError,
    TeleboatLoginProbe,
    chromium_launch_options,
)
from teleboat_agent.login_secrets import (
    LoginSecrets,
    SecretFileError,
    load_login_secrets,
    save_login_secrets,
)


@pytest.mark.parametrize(
    ("mode", "auth_secret"),
    [("mobile", "1234"), ("pc", "abc123")],
)
def test_login_secrets_round_trip_with_owner_only_permissions(
    tmp_path: Path,
    mode: str,
    auth_secret: str,
) -> None:
    path = tmp_path / "private" / "login.json"
    secrets = LoginSecrets.parse(
        {
            "mode": mode,
            "member_number": "12345678",
            "pin": "5678",
            "auth_secret": auth_secret,
        }
    )

    save_login_secrets(path, secrets)
    loaded = load_login_secrets(path)

    assert loaded == secrets
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert "12345678" not in repr(loaded)
    assert "5678" not in repr(loaded)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "mobile", "member_number": "abc", "pin": "5678", "auth_secret": "1234"},
        {"mode": "mobile", "member_number": "12345678", "pin": "12", "auth_secret": "1234"},
        {"mode": "mobile", "member_number": "12345678", "pin": "5678", "auth_secret": "abc"},
        {"mode": "pc", "member_number": "12345678", "pin": "5678", "auth_secret": "1234"},
    ],
)
def test_login_secrets_reject_invalid_values(payload) -> None:
    with pytest.raises(SecretFileError):
        LoginSecrets.parse(payload)


def test_secret_loader_rejects_group_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "login.json"
    path.write_text(
        '{"mode":"mobile","member_number":"12345678","pin":"5678","auth_secret":"1234"}'
    )
    os.chmod(path, 0o640)

    with pytest.raises(SecretFileError, match="0600"):
        load_login_secrets(path)


class FakePage:
    def __init__(self) -> None:
        self.url = "https://spweb.brtb.jp/"


class FakeProbe(TeleboatLoginProbe):
    def __init__(self) -> None:
        super().__init__(browser="chromium")
        self.page = FakePage()
        self.submitted = False
        self.closed = False

    @contextmanager
    def _browser_page(self, mode: str):
        try:
            yield self.page
        finally:
            self.closed = True

    def _open_official_page(self, page, mode: str) -> None:
        page.url = self._url(mode)

    def _wait_for_login_form(self, page, mode: str) -> bool:
        return True

    def _submit_login_once(self, page, secrets: LoginSecrets) -> None:
        self.submitted = True

    def _wait_until_authenticated(self, page, mode: str) -> bool:
        return True

    def _logout(self, page, mode: str) -> bool:
        return True


def test_login_probe_performs_one_login_and_zero_wager_actions() -> None:
    probe = FakeProbe()
    secrets = LoginSecrets.parse(
        {
            "mode": "mobile",
            "member_number": "12345678",
            "pin": "5678",
            "auth_secret": "1234",
        }
    )

    result = probe.login_probe(secrets)

    assert probe.submitted is True
    assert result.authenticated is True
    assert result.logout_confirmed is True
    assert result.attempts == 1
    assert result.wager_actions == 0
    assert probe.closed is True


@pytest.mark.parametrize(
    "url",
    [
        "https://mb.brtb.jp/",
        "https://spweb.brtb.jp/",
        "https://login.brtb.jp/auth/realms/boat/",
    ],
)
def test_login_probe_accepts_official_mobile_hosts(url: str) -> None:
    TeleboatLoginProbe._assert_allowed_host(url, "mobile")


def test_mobile_probe_starts_at_migrated_agent_endpoint() -> None:
    assert TeleboatLoginProbe._url("mobile") == "https://mb.brtb.jp/"


def test_login_probe_rejects_nonofficial_redirect() -> None:
    with pytest.raises(LoginProbeError, match="allowlisted"):
        TeleboatLoginProbe._assert_allowed_host(
            "https://example.invalid/login",
            "mobile",
        )


def test_x86_64_uses_plain_local_headless_chromium() -> None:
    assert chromium_launch_options("x86_64") == {"headless": True}


def test_arm64_uses_headless_compatibility_flags() -> None:
    options = chromium_launch_options("aarch64")

    assert options["headless"] is True
    assert "--disable-gpu" in options["args"]


def test_mobile_logout_completion_url_is_confirmed() -> None:
    assert TeleboatLoginProbe._logout_location_confirmed(
        "https://mb.brtb.jp/tohyo-ap-smtohyo/PWTAUT/F_PWTAUT_Logout/pwtautlogout_displayBL.do",
        "mobile",
    )
    assert not TeleboatLoginProbe._logout_location_confirmed(
        "https://mb.brtb.jp/",
        "mobile",
    )
    assert not TeleboatLoginProbe._logout_location_confirmed(
        "https://example.invalid/F_PWTAUT_Logout/pwtautlogout_displayBL.do",
        "mobile",
    )
