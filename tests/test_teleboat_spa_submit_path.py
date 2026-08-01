from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import teleboat_agent.browser as browser_module
from teleboat_agent.browser import ConfirmationSummary, PlaywrightVoteExecutor
from teleboat_agent.config import Settings
from teleboat_agent.models import VoteRequest


class _Amount:
    def __init__(self) -> None:
        self.value = ""

    def fill(self, value: str) -> None:
        self.value = value

    def input_value(self) -> str:
        return self.value


class _Button:
    def __init__(self, *, fail: bool = False) -> None:
        self.clicks = 0
        self.fail = fail

    def click(self) -> None:
        self.clicks += 1
        if self.fail:
            raise RuntimeError("simulated unknown official response")


class _Page:
    def __init__(self) -> None:
        self.waits = []

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)


class _Probe:
    logout_calls = 0

    def __init__(self, timeout: float) -> None:
        assert timeout == 30

    @contextmanager
    def _browser_page(self, mode: str):
        assert mode == "mobile"
        yield _Page()

    def _open_official_page(self, page, mode: str) -> None:
        pass

    def _wait_for_login_form(self, page, mode: str) -> bool:
        return True

    def _submit_login_once(self, page, secrets) -> None:
        pass

    def _wait_until_authenticated(self, page, mode: str) -> bool:
        return True

    def _read_balance(self, page, mode: str) -> int:
        return 10_000

    def _logout(self, page, mode: str) -> bool:
        type(self).logout_calls += 1
        return True


class _Executor(PlaywrightVoteExecutor):
    def __init__(self, settings: Settings, *, fail_submit: bool = False) -> None:
        super().__init__(settings)
        self.amount = _Amount()
        self.button = _Button(fail=fail_submit)

    def _open_vote_menu(self, page, probe, request) -> None:
        pass

    def _add_regular(self, page, request) -> None:
        pass

    def _verify_confirmation(self, page, request) -> ConfirmationSummary:
        return ConfirmationSummary(1, 100, True, True)

    def _final_amount_input(self, page):
        return self.amount

    def _final_button(self, page):
        return self.button

    @staticmethod
    def _submission_completed(page) -> bool:
        return True


def _settings() -> Settings:
    return Settings(
        application_token="test-token",
        live_vote_enabled=True,
        live_confirmation_secret="test-confirmation",
        member_number="12345678",
        pin="1234",
        authorization_number_of_mobile="5678",
        journal_path="/tmp/teleboat-submit-path.jsonl",
        max_tickets_per_request=1,
        max_total_stake_yen=100,
        batch_size=1,
    )


def _request() -> VoteRequest:
    return VoteRequest.parse(
        {
            "race": {"stadium_tel_code": "10", "number": 4},
            "bet_type": "trifecta",
            "method": "regular",
            "tickets": [{"number": "156", "quantity": 1}],
        },
        max_tickets=1,
        max_total_stake_yen=100,
    )


def test_submit_path_enters_one_100_yen_unit_and_verifies_completion(monkeypatch) -> None:
    _Probe.logout_calls = 0
    monkeypatch.setattr(browser_module, "TeleboatBalanceProbe", _Probe)
    executor = _Executor(_settings())

    result = executor.execute(_request())[0]

    assert executor.amount.value == "1"
    assert executor.button.clicks == 1
    assert result["status"] == "submitted_verified"
    assert result["stake_yen"] == 100
    assert result["final_amount_units"] == 1
    assert result["final_button_clicked"] is True
    assert result["submission_verified"] is True
    assert result["logout_confirmed"] is True
    assert _Probe.logout_calls == 1


def test_submit_path_marks_unknown_response_without_retry(monkeypatch) -> None:
    _Probe.logout_calls = 0
    monkeypatch.setattr(browser_module, "TeleboatBalanceProbe", _Probe)
    executor = _Executor(_settings(), fail_submit=True)

    result = executor.execute(_request())[0]

    assert executor.amount.value == "1"
    assert executor.button.clicks == 1
    assert result["status"] == "submission_unknown"
    assert result["submission_verified"] is False
    assert result["final_button_clicked"] is True
    assert result["logout_confirmed"] is True
    assert _Probe.logout_calls == 1
