from __future__ import annotations

import hmac
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .browser import SeleniumVoteExecutor, VoteExecutor
from .config import Settings
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

    def __post_init__(self) -> None:
        if self.idempotency_store is None:
            self.idempotency_store = IdempotencyStore()

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
        summary = {
            "request_id": request_id,
            "stadium_tel_code": request.stadium.formal_tel_code,
            "race_number": request.race_number,
            "tickets": len(request.tickets),
            "total_stake_yen": request.total_stake_yen,
            "batches": [
                {
                    "batch": index,
                    "tickets": len(batch),
                    "stake_yen": sum(ticket.stake_yen for ticket in batch),
                    "codes": [
                        ticket.simple_betting_code(request.race_number)
                        for ticket in batch
                    ],
                }
                for index, batch in enumerate(
                    request.batches(self.settings.batch_size),
                    start=1,
                )
            ],
        }
        if not live_requested:
            return {"success": True, "mode": "dry_run", **summary}
        self._authorize_live(live_confirmation, idempotency_key)
        assert idempotency_key is not None
        assert self.idempotency_store is not None
        self.idempotency_store.reserve(idempotency_key)
        try:
            submitted = self.executor_factory(self.settings).execute(request)
        except Exception:
            self.idempotency_store.release(idempotency_key)
            raise
        return {
            "success": True,
            "mode": "live",
            "submitted": submitted,
            **summary,
        }

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
