from __future__ import annotations

import hmac
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .browser import SeleniumVoteExecutor, VoteExecutionError, VoteExecutor
from .config import Settings
from .journal import (
    VoteJournal,
    VoteJournalError,
    idempotency_fingerprint,
    request_snapshot,
)
from .models import VoteRequest


class AuthorizationError(PermissionError):
    pass


class DuplicateRequestError(RuntimeError):
    pass


class IdempotencyStore:
    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._lock = threading.Lock()

    def reserve(self, key: str) -> None:
        with self._lock:
            if key in self._keys:
                raise DuplicateRequestError("idempotency key has already been used")
            self._keys.add(key)

    def release(self, key: str) -> None:
        with self._lock:
            self._keys.discard(key)


@dataclass
class VoteTicketsService:
    settings: Settings
    executor_factory: Callable[[Settings], VoteExecutor] = SeleniumVoteExecutor
    idempotency_store: IdempotencyStore | None = None
    journal: VoteJournal | None = None

    def __post_init__(self) -> None:
        if self.idempotency_store is None:
            self.idempotency_store = IdempotencyStore()
        if self.journal is None:
            self.journal = VoteJournal(self.settings.journal_path)

    def call(
        self,
        payload: Any,
        *,
        live_requested: bool = False,
        live_confirmation: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        request = VoteRequest.parse(
            payload,
            max_tickets=self.settings.max_tickets_per_request,
            max_total_stake_yen=self.settings.max_total_stake_yen,
        )
        request_id = uuid.uuid4().hex
        summary = self._summary(request_id, request)
        journal_base = {
            "request_id": request_id,
            "idempotency_key_sha256": idempotency_fingerprint(idempotency_key),
            "request": request_snapshot(request),
        }

        if not live_requested:
            self._journal(
                {
                    **journal_base,
                    "event": "dry_run_validated",
                    "mode": "dry_run",
                    "status": "validated",
                    "final_button_clicked": False,
                    "verifications": {
                        "payload_validation": True,
                        "ticket_expansion": True,
                        "ticket_limit": True,
                        "stake_limit": True,
                    },
                }
            )
            return {"success": True, "mode": "dry_run", **summary}

        try:
            self._authorize_live(live_confirmation, idempotency_key)
        except AuthorizationError:
            self._journal(
                {
                    **journal_base,
                    "event": "live_authorization_rejected",
                    "mode": "live",
                    "status": "rejected",
                    "final_button_clicked": False,
                }
            )
            raise

        assert idempotency_key is not None
        assert self.idempotency_store is not None
        self.idempotency_store.reserve(idempotency_key)
        try:
            self._journal(
                {
                    **journal_base,
                    "event": "live_authorized",
                    "mode": "live",
                    "status": "authorized",
                    "final_button_clicked": False,
                    "verifications": {
                        "payload_validation": True,
                        "ticket_expansion": True,
                        "ticket_limit": True,
                        "stake_limit": True,
                        "bearer_authentication": True,
                        "live_gate": True,
                        "confirmation_secret": True,
                        "idempotency_reservation": True,
                    },
                }
            )
        except VoteJournalError:
            self.idempotency_store.release(idempotency_key)
            raise

        try:
            submitted = self.executor_factory(self.settings).execute(request)
        except VoteExecutionError as exc:
            self._journal_best_effort(
                {
                    **journal_base,
                    "event": "live_execution_failed",
                    "mode": "live",
                    "status": "submission_state_unknown"
                    if exc.submission_may_have_occurred
                    else "pre_submission_failed",
                    "stage": exc.stage,
                    "error_type": type(exc).__name__,
                    "final_button_clicked": exc.submission_may_have_occurred,
                    "retry_allowed": not exc.submission_may_have_occurred,
                }
            )
            if not exc.submission_may_have_occurred:
                self.idempotency_store.release(idempotency_key)
            raise
        except Exception as exc:
            self._journal_best_effort(
                {
                    **journal_base,
                    "event": "live_execution_failed",
                    "mode": "live",
                    "status": "pre_submission_failed",
                    "stage": "executor",
                    "error_type": type(exc).__name__,
                    "final_button_clicked": False,
                    "retry_allowed": True,
                }
            )
            self.idempotency_store.release(idempotency_key)
            raise

        verified = bool(submitted) and all(
            row.get("status") in {"submitted", "submitted_verified"}
            for row in submitted
        )
        journal_ok = True
        try:
            self._journal(
                {
                    **journal_base,
                    "event": "live_execution_completed",
                    "mode": "live",
                    "status": "submitted_verified" if verified else "submission_state_unknown",
                    "final_button_clicked": any(
                        bool(row.get("final_button_clicked")) for row in submitted
                    ),
                    "retry_allowed": False,
                    "submitted": submitted,
                }
            )
        except VoteJournalError:
            journal_ok = False
            verified = False

        return {
            "success": verified,
            "mode": "live",
            "requires_manual_confirmation": not verified,
            "journal_status": "recorded" if journal_ok else "write_failed_after_execution",
            "submitted": submitted,
            **summary,
        }

    def _summary(self, request_id: str, request: VoteRequest) -> dict[str, object]:
        return {
            "request_id": request_id,
            "stadium_tel_code": request.stadium.formal_tel_code,
            "race_number": request.race_number,
            "bet_type": request.bet_type.value,
            "bet_type_label": request.bet_type.label,
            "method": request.method.value,
            "tickets": len(request.tickets),
            "total_stake_yen": request.total_stake_yen,
            "batches": [
                {
                    "batch": index,
                    "tickets": len(batch),
                    "stake_yen": sum(ticket.stake_yen for ticket in batch),
                    "selections": [
                        {
                            "number": ticket.betting_number.value,
                            "quantity": ticket.quantity,
                            "stake_yen": ticket.stake_yen,
                        }
                        for ticket in batch
                    ],
                    "codes": (
                        [
                            ticket.simple_betting_code(request.race_number)
                            for ticket in batch
                        ]
                        if request.bet_type.value == "trifecta"
                        else []
                    ),
                }
                for index, batch in enumerate(
                    request.batches(self.settings.batch_size),
                    start=1,
                )
            ],
        }

    def _journal(self, event: dict[str, Any]) -> None:
        assert self.journal is not None
        self.journal.append(event)

    def _journal_best_effort(self, event: dict[str, Any]) -> None:
        try:
            self._journal(event)
        except VoteJournalError:
            return

    def _authorize_live(
        self,
        live_confirmation: str | None,
        idempotency_key: str | None,
    ) -> None:
        if not self.settings.live_vote_enabled:
            raise AuthorizationError("live voting is disabled")
        expected = self.settings.live_confirmation_secret or ""
        supplied = live_confirmation or ""
        if not expected or not hmac.compare_digest(expected, supplied):
            raise AuthorizationError("live vote confirmation is invalid")
        if not idempotency_key or len(idempotency_key) > 128:
            raise AuthorizationError(
                "a non-empty Idempotency-Key of at most 128 characters is required"
            )
