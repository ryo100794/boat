import json
from pathlib import Path

from teleboat_agent.browser import purchase_amount_units, verify_confirmation_text
from teleboat_agent.models import VoteRequest


FIXTURE = Path(__file__).parent / "fixtures" / "teleboat" / "spweb_spa_20260801.json"


def _request() -> VoteRequest:
    return VoteRequest.parse(
        {
            "race": {"stadium_tel_code": "10", "number": 3},
            "bet_type": "trifecta",
            "method": "regular",
            "tickets": [{"number": "135", "quantity": 1}],
        },
        max_tickets=1,
        max_total_stake_yen=100,
    )


def test_recorded_official_spa_response_matches_runtime_contract() -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["source"] == "official_authenticated_live_capture"
    assert fixture["source_host"] == "spweb.brtb.jp"
    assert fixture["path"] == "/bet"
    assert fixture["stages"]["menu"]["forms"] == 0
    assert fixture["stages"]["input"]["checked_ids"] == [
        "bet1-1", "bet2-3", "bet3-5"
    ]
    assert fixture["stages"]["input"]["amount_units"] == "1"
    assert fixture["stages"]["review"]["next_control"]["text"] == "次へ"
    assert fixture["stages"]["final_confirmation"]["submit_control"]["text"] == "投票"


def test_recorded_final_confirmation_verifies_one_100_yen_ticket() -> None:
    fixture = json.loads(FIXTURE.read_text())
    text = "\n".join(fixture["stages"]["final_confirmation"]["safe_text"])
    summary = verify_confirmation_text(
        text,
        request=_request(),
        final_button_ready=True,
    )
    assert summary.tickets == 1
    assert summary.stake_yen == 100
    assert summary.unfinished is True
    assert purchase_amount_units(summary.stake_yen) == 1


def test_recorded_fixture_contains_no_credentials_or_account_identifier() -> None:
    text = FIXTURE.read_text()
    for forbidden in ("userId", "pinNum", "authorizationNumber", "member_number"):
        assert forbidden not in text
