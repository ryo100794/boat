from __future__ import annotations

from contextlib import contextmanager

import pytest

from teleboat_agent.account_balance import AccountBalanceError, TeleboatBalanceProbe, parse_available_balance, verify_available_balance
from teleboat_agent.login_secrets import LoginSecrets


@pytest.mark.parametrize(("text", "expected"), [("投票可能残高 10,000円", 10_000), ("購入可能金額：￥12,300 円", 12_300), ("購入限度額 0円", 0), ("口座残高 1,234円", 1_234)])
def test_parse_available_balance(text: str, expected: int) -> None:
    assert parse_available_balance(text) == expected


def test_balance_parser_fails_closed_and_checks_required_stake() -> None:
    with pytest.raises(AccountBalanceError, match="not found"):
        parse_available_balance("ログインしました")
    with pytest.raises(AccountBalanceError, match="insufficient"):
        verify_available_balance("投票可能残高 9,900円", required_yen=10_000)
    assert verify_available_balance("投票可能残高 10,000円", required_yen=10_000) == 10_000


class _Body:
    def inner_text(self) -> str:
        return "会員メニュー 投票可能残高 10,000円"


class _Page:
    url = "https://spweb.brtb.jp/"
    def locator(self, selector: str):
        assert selector == "body"
        return _Body()


class _Probe(TeleboatBalanceProbe):
    def __init__(self) -> None:
        super().__init__(timeout=5)
        self.logout_calls = 0
    @contextmanager
    def _browser_page(self, mode: str):
        yield _Page()
    def _open_official_page(self, page, mode: str) -> None:
        pass
    def _wait_for_login_form(self, page, mode: str) -> bool:
        return True
    def _submit_login_once(self, page, secrets: LoginSecrets) -> None:
        pass
    def _wait_until_authenticated(self, page, mode: str) -> bool:
        return True
    def _logout(self, page, mode: str) -> bool:
        self.logout_calls += 1
        return True


def test_balance_probe_reads_only_and_logs_out() -> None:
    secrets = LoginSecrets.parse({"mode": "mobile", "member_number": "12345678", "pin": "5678", "auth_secret": "1234"})
    probe = _Probe()
    result = probe.balance_probe(secrets)
    assert result.available_balance_yen == 10_000
    assert result.logout_confirmed is True
    assert result.wager_actions == 0
    assert probe.logout_calls == 1
