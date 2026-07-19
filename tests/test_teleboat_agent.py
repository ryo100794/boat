from __future__ import annotations

import json
from io import BytesIO

import pytest

from teleboat_agent.api import TeleboatApplication, VOTES_PATH
from teleboat_agent.config import Settings
from teleboat_agent.models import (
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
