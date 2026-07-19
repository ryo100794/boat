from __future__ import annotations

import json
from io import BytesIO

import pytest

from teleboat_agent.api import TeleboatApplication, VOTES_PATH
from teleboat_agent.browser import (
    SeleniumVoteExecutor,
    VoteExecutionError,
    verify_confirmation_text,
)
from teleboat_agent.config import Settings
from teleboat_agent.models import (
    BetMethod,
    BetType,
    BettingNumber,
    Stadium,
    Ticket,
    ValidationError,
    VoteRequest,
)
from teleboat_agent.service import (
    AuthorizationError,
    DuplicateRequestError,
    VoteTicketsService,
)


def payload() -> dict:
    return {
        "race": {"stadium_tel_code": 4, "number": 1},
        "odds": [
            {"number": 123, "quantity": 2},
            {"number": 654, "quantity": 3},
        ],
    }


def settings(**overrides) -> Settings:
    values = {
        "application_token": "test-api-token",
        "max_tickets_per_request": 30,
        "max_total_stake_yen": 10_000,
        "batch_size": 10,
        "journal_path": "/tmp/teleboat-agent-test-journal.jsonl",
    }
    values.update(overrides)
    result = Settings(**values)
    result.validate()
    return result


def test_original_value_objects_are_reimplemented() -> None:
    assert BettingNumber.parse(123).value == "123"
    assert Stadium.parse(4).formal_tel_code == "04"
    assert Ticket.parse({"number": 654, "quantity": 999}).simple_betting_code(1) == (
        "0131654999"
    )
    assert Ticket.parse({"number": 241, "quantity": 1}).simple_betting_code(8) == (
        "0831241001"
    )
    with pytest.raises(ValidationError):
        BettingNumber.parse(121)
    with pytest.raises(ValidationError):
        Stadium.parse(25)


def test_vote_request_rejects_duplicates_and_stake_overflow() -> None:
    duplicated = payload()
    duplicated["odds"][1]["number"] = 123
    with pytest.raises(ValidationError, match="duplicate"):
        VoteRequest.parse(
            duplicated,
            max_tickets=30,
            max_total_stake_yen=10_000,
        )
    with pytest.raises(ValidationError, match="total stake"):
        VoteRequest.parse(
            payload(),
            max_tickets=30,
            max_total_stake_yen=400,
        )


def test_service_is_dry_run_by_default_and_does_not_create_executor() -> None:
    def forbidden_factory(_settings):
        raise AssertionError("executor must not be created in dry-run mode")

    service = VoteTicketsService(settings(), executor_factory=forbidden_factory)
    result = service.call(payload())

    assert result["mode"] == "dry_run"
    assert result["total_stake_yen"] == 500
    assert result["batches"][0]["codes"] == ["0131123002", "0131654003"]


def test_live_vote_requires_two_explicit_gates_and_idempotency() -> None:
    executions = []

    class FakeExecutor:
        def execute(self, request):
            executions.append(request)
            return [{"batch": 1, "status": "submitted"}]

    service = VoteTicketsService(
        settings(
            live_vote_enabled=True,
            live_confirmation_secret="confirm-secret",
            member_number="member",
            pin="pin",
            authorization_number_of_mobile="mobile",
        ),
        executor_factory=lambda _settings: FakeExecutor(),
    )
    with pytest.raises(AuthorizationError):
        service.call(payload(), live_requested=True)
    result = service.call(
        payload(),
        live_requested=True,
        live_confirmation="confirm-secret",
        idempotency_key="request-1",
    )
    assert result["mode"] == "live"
    assert len(executions) == 1
    with pytest.raises(DuplicateRequestError):
        service.call(
            payload(),
            live_requested=True,
            live_confirmation="confirm-secret",
            idempotency_key="request-1",
        )


def call_wsgi(application, *, token: str, body: dict):
    encoded = json.dumps(body).encode()
    environ = {
        "PATH_INFO": VOTES_PATH,
        "REQUEST_METHOD": "POST",
        "CONTENT_LENGTH": str(len(encoded)),
        "CONTENT_TYPE": "application/json",
        "HTTP_AUTHORIZATION": f"Bearer {token}",
        "wsgi.input": BytesIO(encoded),
    }
    response = {}

    def start_response(status, headers):
        response["status"] = status
        response["headers"] = dict(headers)

    chunks = application(environ, start_response)
    response["body"] = json.loads(b"".join(chunks))
    return response


def test_wsgi_api_uses_bearer_token_and_returns_dry_run() -> None:
    application = TeleboatApplication(settings())
    rejected = call_wsgi(application, token="wrong", body=payload())
    accepted = call_wsgi(application, token="test-api-token", body=payload())

    assert rejected["status"].startswith("403")
    assert accepted["status"].startswith("200")
    assert accepted["body"]["mode"] == "dry_run"
    assert accepted["headers"]["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "overrides",
    [
        {"application_token": "*****"},
        {"base_url": "https://example.invalid/"},
        {"base_url": "https://mb.brtb.jp:444/"},
    ],
)
def test_settings_reject_insecure_authentication_destinations(overrides) -> None:
    with pytest.raises(RuntimeError):
        settings(**overrides)


def test_verified_teleboat_selectors_use_current_named_controls() -> None:
    assert SeleniumVoteExecutor.LOGIN_MEMBER_XPATH == '//input[@name="userId"]'
    assert SeleniumVoteExecutor.LOGIN_PIN_XPATH == '//input[@name="pwd"]'
    assert SeleniumVoteExecutor.LOGIN_MOBILE_XPATH == '//input[@name="pinNum"]'
    assert SeleniumVoteExecutor.SIMPLE_VOTE_XPATH == '//a[normalize-space()="簡易投票する"]'
    assert SeleniumVoteExecutor.STADIUM_XPATH == '//input[@name="jyoCode"]'
    assert SeleniumVoteExecutor.REVIEW_XPATH == '//*[@id="btnAddList1"]'


@pytest.mark.parametrize(
    ("bet_type", "number", "label"),
    [
        ("win", "1", "単勝"),
        ("place", "1", "複勝"),
        ("exacta", "12", "2連単"),
        ("quinella", "21", "2連複"),
        ("quinella_place", "21", "拡連複"),
        ("trifecta", "123", "3連単"),
        ("trio", "321", "3連複"),
    ],
)
def test_regular_request_supports_every_bet_type(
    bet_type: str,
    number: str,
    label: str,
) -> None:
    request = VoteRequest.parse(
        {
            "race": {"stadium_tel_code": 20, "number": 11},
            "bet_type": bet_type,
            "method": "regular",
            "tickets": [{"number": number, "quantity": 1}],
        },
        max_tickets=30,
        max_total_stake_yen=10_000,
    )

    assert request.bet_type.label == label
    assert request.method is BetMethod.REGULAR
    assert request.total_stake_yen == 100
    assert request.tickets[0].betting_number.value == (
        number if request.bet_type.ordered else "".join(sorted(number))
    )


@pytest.mark.parametrize(
    ("bet_type", "expected"),
    [("exacta", 6), ("quinella", 3), ("quinella_place", 3), ("trifecta", 6), ("trio", 1)],
)
def test_box_expansion_matches_official_ticket_count(
    bet_type: str,
    expected: int,
) -> None:
    request = VoteRequest.parse(
        {
            "race": {"stadium_tel_code": 20, "number": 11},
            "bet_type": bet_type,
            "method": "box",
            "selections": [1, 2, 3],
            "quantity": 1,
        },
        max_tickets=30,
        max_total_stake_yen=10_000,
    )

    assert request.expanded_ticket_count == expected
    assert request.total_stake_yen == expected * 100


@pytest.mark.parametrize(
    ("bet_type", "formation", "expected"),
    [
        ("exacta", [[1], [2, 3]], 2),
        ("quinella", [[1], [2, 3]], 2),
        ("quinella_place", [[1], [2, 3]], 2),
        ("trifecta", [[1], [2, 3], [2, 3, 4]], 4),
        ("trio", [[1], [2], [3, 4]], 2),
    ],
)
def test_formation_expansion_matches_official_ticket_count(
    bet_type: str,
    formation: list[list[int]],
    expected: int,
) -> None:
    request = VoteRequest.parse(
        {
            "race": {"stadium_tel_code": 20, "number": 11},
            "bet_type": bet_type,
            "method": "formation",
            "formation": formation,
            "quantity": 1,
        },
        max_tickets=30,
        max_total_stake_yen=10_000,
    )

    assert request.expanded_ticket_count == expected
    assert len({ticket.betting_number.value for ticket in request.tickets}) == expected


def test_invalid_method_combinations_and_expansion_limits_fail_closed() -> None:
    with pytest.raises(ValidationError, match="not available"):
        VoteRequest.parse(
            {
                "race": {"stadium_tel_code": 20, "number": 11},
                "bet_type": "win",
                "method": "box",
                "selections": [1, 2],
                "quantity": 1,
            },
            max_tickets=30,
            max_total_stake_yen=10_000,
        )
    with pytest.raises(ValidationError, match="expanded ticket count"):
        VoteRequest.parse(
            {
                "race": {"stadium_tel_code": 20, "number": 11},
                "bet_type": "trifecta",
                "method": "box",
                "selections": [1, 2, 3, 4],
                "quantity": 1,
            },
            max_tickets=20,
            max_total_stake_yen=10_000,
        )


def test_confirmation_verifier_checks_identity_totals_and_every_ticket() -> None:
    request = VoteRequest.parse(
        {
            "race": {"stadium_tel_code": 20, "number": 11},
            "bet_type": "exacta",
            "method": "box",
            "selections": [1, 2, 3],
            "quantity": 1,
        },
        max_tickets=30,
        max_total_stake_yen=10_000,
    )
    text = (
        "ベットリスト 本画面では投票未完了です。 "
        "若松 11R 2連単 ボックス 123 6ベット×100円 =600円 "
        "合計ベット数 6ベット 購入金額 600円 投票する"
    )

    summary = verify_confirmation_text(
        text,
        request=request,
        final_button_ready=True,
    )

    assert summary.tickets == 6
    assert summary.stake_yen == 600
    with pytest.raises(VoteExecutionError, match="stake mismatch"):
        verify_confirmation_text(
            text.replace("600円", "500円"),
            request=request,
            final_button_ready=True,
        )


def test_uncertain_submission_keeps_idempotency_reservation() -> None:
    class UncertainExecutor:
        def execute(self, request):
            raise VoteExecutionError(
                "result page unavailable",
                submission_may_have_occurred=True,
            )

    service = VoteTicketsService(
        settings(
            live_vote_enabled=True,
            live_confirmation_secret="confirm-secret",
            member_number="member",
            pin="pin",
            authorization_number_of_mobile="mobile",
        ),
        executor_factory=lambda _settings: UncertainExecutor(),
    )
    with pytest.raises(VoteExecutionError):
        service.call(
            payload(),
            live_requested=True,
            live_confirmation="confirm-secret",
            idempotency_key="uncertain-1",
        )
    with pytest.raises(DuplicateRequestError):
        service.call(
            payload(),
            live_requested=True,
            live_confirmation="confirm-secret",
            idempotency_key="uncertain-1",
        )
