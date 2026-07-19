from __future__ import annotations

import hashlib
import json

import pytest
from pathlib import Path

from teleboat_agent.journal import VoteJournal, VoteJournalError, request_snapshot, verify_journal
from teleboat_agent.models import VoteRequest
from teleboat_agent.service import VoteTicketsService

from test_teleboat_agent import payload, settings


def test_vote_journal_is_owner_only_chained_jsonl_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit" / "votes.jsonl"
    journal = VoteJournal(path)

    first = journal.append(
        {
            "request_id": "request-1",
            "event": "live_authorized",
            "member_number": "00000000",
            "details": {"pin": "0000", "safe": "kept"},
        }
    )
    second = journal.append(
        {
            "request_id": "request-1",
            "event": "live_execution_completed",
            "status": "preview_verified",
        }
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.stat().st_mode & 0o777 == 0o600
    assert len(rows) == 2
    assert rows[0]["member_number"] == "[REDACTED]"
    assert rows[0]["details"]["pin"] == "[REDACTED]"
    assert rows[0]["details"]["safe"] == "kept"
    assert rows[1]["previous_hash"] == rows[0]["record_hash"]
    assert first["record_hash"] == rows[0]["record_hash"]
    assert second["record_hash"] == rows[1]["record_hash"]

    for index, row in enumerate(rows):
        record_hash = row.pop("record_hash")
        previous_hash = "" if index == 0 else rows[index - 1]["record_hash"]
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert record_hash == hashlib.sha256(
            (previous_hash + canonical).encode()
        ).hexdigest()
        row["record_hash"] = record_hash


def test_dry_run_writes_expanded_request_and_verification_journal(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "votes.jsonl"
    service = VoteTicketsService(
        settings(journal_path=str(journal_path)),
    )

    result = service.call(payload())

    row = json.loads(journal_path.read_text().strip())
    assert result["mode"] == "dry_run"
    assert row["event"] == "dry_run_validated"
    assert row["request_id"] == result["request_id"]
    assert row["request"]["expanded_ticket_count"] == 2
    assert row["request"]["total_stake_yen"] == 500
    assert row["verifications"] == {
        "payload_validation": True,
        "stake_limit": True,
        "ticket_expansion": True,
        "ticket_limit": True,
    }
    serialized = journal_path.read_text()
    assert "confirm-secret" not in serialized
    assert "test-api-token" not in serialized


def test_request_snapshot_contains_reconstructable_box_expansion() -> None:
    request = VoteRequest.parse(
        {
            "race": {"stadium_tel_code": 20, "number": 11},
            "bet_type": "trifecta",
            "method": "box",
            "selections": [1, 2, 3],
            "quantity": 1,
        },
        max_tickets=30,
        max_total_stake_yen=10_000,
    )

    snapshot = request_snapshot(request)

    assert snapshot["stadium_name"] == "若松"
    assert snapshot["source_positions"] == [[1, 2, 3]]
    assert snapshot["expanded_ticket_count"] == 6
    assert [row["number"] for row in snapshot["tickets"]] == [
        "123",
        "132",
        "213",
        "231",
        "312",
        "321",
    ]


def test_verify_journal_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "votes.jsonl"
    journal = VoteJournal(path)
    journal.append({"request_id": "request-1", "event": "dry_run_validated"})
    journal.append({"request_id": "request-1", "event": "live_authorized"})

    result = verify_journal(path)

    assert result["valid"] is True
    assert result["records"] == 2
    assert result["events"] == {"dry_run_validated": 1, "live_authorized": 1}

    path.write_text(path.read_text().replace("live_authorized", "tampered"))
    with pytest.raises(VoteJournalError, match="hash mismatch"):
        verify_journal(path)
