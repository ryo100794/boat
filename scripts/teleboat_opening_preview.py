#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from teleboat_agent.browser import PlaywrightVoteExecutor
from teleboat_agent.config import Settings
from teleboat_agent.login_secrets import load_login_secrets
from teleboat_agent.journal import VoteJournal, request_snapshot
from teleboat_agent.models import VoteRequest


JST = timezone(timedelta(hours=9))
DEFAULT_SECRET_PATH = Path(".secrets/teleboat-login.json")
DEFAULT_OUTPUT_PATH = Path("data/teleboat-opening-preview.json")
DEFAULT_JOURNAL_PATH = Path("data/teleboat_vote_journal.jsonl")

EXIT_OK = 0
EXIT_CONFIGURATION = 2
EXIT_NO_CANDIDATE = 3
EXIT_GUIDE_REJECTED = 4
EXIT_PREVIEW_FAILED = 5
GUIDE_USER_AGENT = "boatrace-opening-preview/1.0"


class PreviewFailure(RuntimeError):
    def __init__(self, code: str, exit_code: int):
        super().__init__(code)
        self.code = code
        self.exit_code = exit_code


def _default_target_date() -> date:
    return datetime.now(JST).date()


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _guide_url(base_url: str, target_date: date) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PreviewFailure("invalid_dashboard_url", EXIT_CONFIGURATION)
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/guide"):
        path = f"{path}/api/guide" if path else "/api/guide"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["date"] = target_date.isoformat()
    return urlunparse(parsed._replace(path=path, query=urlencode(query), fragment=""))


def _fetch_guide(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": GUIDE_USER_AGENT,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise PreviewFailure("invalid_guide_payload", EXIT_GUIDE_REJECTED)
    return payload


def _parse_jst_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def select_candidate(
    payload: dict[str, Any],
    *,
    target_date: date,
    now: datetime,
) -> dict[str, Any] | None:
    now_jst = now.astimezone(JST)
    if now_jst.date() < target_date:
        return None
    if now_jst.date() > target_date:
        raise PreviewFailure("guide_date_mismatch", EXIT_GUIDE_REJECTED)
    if payload.get("date") != target_date.isoformat():
        raise PreviewFailure("guide_date_mismatch", EXIT_GUIDE_REJECTED)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise PreviewFailure("invalid_candidate_list", EXIT_GUIDE_REJECTED)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        race_date = candidate.get("race_date", payload.get("date"))
        if race_date != target_date.isoformat():
            continue
        if candidate.get("entries") != 6:
            continue
        decision = candidate.get("t300_decision")
        if not isinstance(decision, dict):
            continue
        if (
            decision.get("model_key") != "v21_daily"
            or decision.get("decision_status") != "selected"
            or not decision.get("source_snapshot_id")
        ):
            continue
        target_t300 = _parse_jst_datetime(decision.get("target_t300_at"))
        deadline = _parse_jst_datetime(candidate.get("deadline_at"))
        if (
            target_t300 is None
            or deadline is None
            or now_jst < target_t300
            or now_jst >= deadline
        ):
            continue
        selected = decision.get("selected_candidates")
        if not isinstance(selected, list) or not selected:
            continue
        normalized = []
        for ticket in selected:
            if not isinstance(ticket, dict) or not ticket.get("combination"):
                normalized = []
                break
            try:
                stake_yen = int(ticket.get("stake_yen") or 0)
            except (TypeError, ValueError):
                normalized = []
                break
            if stake_yen <= 0 or stake_yen % 100:
                normalized = []
                break
            normalized.append({**ticket, "stake_yen": stake_yen})
        if not normalized:
            continue
        return {**candidate, "t300_selection": normalized}
    return None


def build_vote_request(candidate: dict[str, Any]) -> VoteRequest:
    selected = candidate["t300_selection"]
    tickets = [
        {
            "number": str(ticket["combination"]).replace("-", "").strip(),
            "quantity": int(ticket["stake_yen"]) // 100,
        }
        for ticket in selected
    ]
    return VoteRequest.parse(
        {
            "race": {
                "stadium_tel_code": candidate.get("jcd"),
                "number": candidate.get("rno"),
            },
            "bet_type": "trifecta",
            "method": "regular",
            "tickets": tickets,
        },
        max_tickets=120,
        max_total_stake_yen=10_000,
    )


def load_settings(secret_path: Path, *, journal_path: Path) -> Settings:
    secrets = load_login_secrets(secret_path)
    if secrets.mode != "mobile":
        raise PreviewFailure("mobile_credentials_required", EXIT_CONFIGURATION)
    settings = Settings(
        application_token="opening-preview-only",
        live_vote_enabled=False,
        member_number=secrets.member_number,
        pin=secrets.pin,
        authorization_number_of_mobile=secrets.auth_secret,
        journal_path=str(journal_path),
        max_tickets_per_request=120,
        max_total_stake_yen=10_000,
        batch_size=1,
    )
    settings.validate()
    return settings


def _safe_preview_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise PreviewFailure("invalid_preview_result", EXIT_PREVIEW_FAILED)
    verifications = result.get("verifications")
    required = (
        "authentication",
        "available_balance",
        "sufficient_balance",
        "official_host_allowlist",
        "mode_selection",
        "ticket_inputs",
        "confirmation_identity",
        "confirmation_selection",
        "confirmation_ticket_count",
        "confirmation_stake_yen",
        "official_hidden_total",
        "unfinished_marker",
        "final_button_ready",
    )
    tickets = int(result.get("tickets") or 0)
    stake_yen = int(result.get("stake_yen") or 0)
    available_balance_yen = int(result.get("available_balance_yen") or 0)
    valid = (
        result.get("status") == "preview_verified"
        and result.get("final_button_clicked") is False
        and result.get("logout_confirmed") is True
        and isinstance(verifications, dict)
        and all(verifications.get(key) is True for key in required)
        and tickets > 0
        and stake_yen > 0
        and available_balance_yen >= stake_yen
    )
    if not valid:
        raise PreviewFailure("official_preview_not_verified", EXIT_PREVIEW_FAILED)
    return {
        "status": "preview_verified",
        "tickets": tickets,
        "stake_yen": stake_yen,
        "available_balance_yen": available_balance_yen,
        "unfinished": True,
        "final_button_ready": True,
        "final_button_clicked": False,
        "logout_confirmed": bool(result.get("logout_confirmed")),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        temp_path.unlink(missing_ok=True)


def _audit_payload(
    *,
    request_id: str,
    target_date: date,
    status: str,
    code: str,
    request: VoteRequest | None = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "generated_at": datetime.now(JST).isoformat(),
        "target_date": target_date.isoformat(),
        "status": status,
        "code": code,
        "action": "preview_only",
        "submission_attempted": False,
    }
    if request is not None:
        payload["selection"] = {
            "stadium_tel_code": request.stadium.formal_tel_code,
            "stadium_name": request.stadium.name,
            "race_number": request.race_number,
            "bet_type": request.bet_type.value,
            "method": request.method.value,
            "combination": request.tickets[0].betting_number.display(request.bet_type),
            "quantity": request.tickets[0].quantity,
            "tickets": [
                {
                    "combination": ticket.betting_number.display(request.bet_type),
                    "quantity": ticket.quantity,
                    "stake_yen": ticket.stake_yen,
                }
                for ticket in request.tickets
            ],
            "stake_yen": request.total_stake_yen,
        }
    if preview is not None:
        payload["official_confirmation"] = preview
    return payload


def _journal_event(
    *,
    request_id: str,
    event: str,
    target_date: date,
    status: str,
    code: str,
    request: VoteRequest | None,
    verification: dict[str, Any],
) -> dict[str, Any]:
    snapshot = (
        request_snapshot(request)
        if request is not None
        else {
            "target_date": target_date.isoformat(),
            "action": "preview_only",
            "wager_request_constructed": False,
        }
    )
    return {
        "request_id": request_id,
        "event": event,
        "mode": "opening_preview",
        "status": status,
        "code": code,
        "target_date": target_date.isoformat(),
        "request_snapshot": snapshot,
        "final_button_clicked": False,
        "submission_attempted": False,
        "verifications": verification,
    }


def _record_and_write(
    *,
    journal: VoteJournal,
    output: Path,
    journal_event: dict[str, Any],
    output_payload: dict[str, Any],
) -> None:
    journal.append(journal_event)
    _atomic_write_json(output, output_payload)


def run(
    args: argparse.Namespace,
    *,
    fetch_guide: Callable[..., dict[str, Any]] = _fetch_guide,
    executor_factory: Callable[[Settings], Any] = PlaywrightVoteExecutor,
    clock: Callable[[], datetime] = lambda: datetime.now(JST),
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    request_id = str(uuid.uuid4())
    request: VoteRequest | None = None
    journal = VoteJournal(args.journal_path)
    try:
        settings = load_settings(args.secret_path, journal_path=args.journal_path)
        if args.validate_only:
            verification = {
                "configuration": True,
                "browser_started": False,
                "network_requested": False,
            }
            _record_and_write(
                journal=journal,
                output=args.output,
                journal_event=_journal_event(
                    request_id=request_id,
                    event="opening_preview_configuration_validated",
                    target_date=args.date,
                    status="validated",
                    code="configuration_valid",
                    request=None,
                    verification=verification,
                ),
                output_payload=_audit_payload(
                    request_id=request_id,
                    target_date=args.date,
                    status="validated",
                    code="configuration_valid",
                ),
            )
            return EXIT_OK

        guide_url = _guide_url(args.dashboard_url, args.date)
        deadline = time.monotonic() + args.timeout_seconds
        candidate = None
        guide_fetch_succeeded = False
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                payload = fetch_guide(guide_url, timeout=min(10.0, remaining))
                guide_fetch_succeeded = True
                candidate = select_candidate(
                    payload, target_date=args.date, now=clock()
                )
            except PreviewFailure:
                raise
            except Exception:
                candidate = None
            if candidate is not None:
                break
            sleeper(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))
        if candidate is None:
            if not guide_fetch_succeeded:
                raise PreviewFailure("guide_unreachable", EXIT_GUIDE_REJECTED)
            raise PreviewFailure("no_eligible_candidate", EXIT_NO_CANDIDATE)

        request = build_vote_request(candidate)
        result = executor_factory(settings).preview(request)
        safe_result = _safe_preview_result(result)
        if (
            safe_result["tickets"] != len(request.tickets)
            or safe_result["stake_yen"] != request.total_stake_yen
        ):
            raise PreviewFailure("official_preview_totals_mismatch", EXIT_PREVIEW_FAILED)
        verification = {
            "official_confirmation": True,
            "authentication": True,
            "available_balance": safe_result["available_balance_yen"] > 0,
            "sufficient_balance": (
                safe_result["available_balance_yen"] >= request.total_stake_yen
            ),
            "race_identity": True,
            "selection": True,
            "stake_yen": safe_result["stake_yen"] == request.total_stake_yen,
            "unfinished_marker": safe_result["unfinished"],
            "final_button_ready": safe_result["final_button_ready"],
            "logout_confirmed": safe_result["logout_confirmed"],
        }
        _record_and_write(
            journal=journal,
            output=args.output,
            journal_event=_journal_event(
                request_id=request_id,
                event="opening_preview_verified",
                target_date=args.date,
                status="success",
                code="official_preview_verified",
                request=request,
                verification=verification,
            ),
            output_payload=_audit_payload(
                request_id=request_id,
                target_date=args.date,
                status="success",
                code="official_preview_verified",
                request=request,
                preview=safe_result,
            ),
        )
        return EXIT_OK
    except PreviewFailure as exc:
        failure = exc
    except Exception:
        failure = PreviewFailure("preview_failed_closed", EXIT_PREVIEW_FAILED)

    waiting = failure.code == "no_eligible_candidate"
    status = "waiting" if waiting else "failed"
    event = "opening_preview_waiting" if waiting else "opening_preview_failed"
    try:
        _record_and_write(
            journal=journal,
            output=args.output,
            journal_event=_journal_event(
                request_id=request_id,
                event=event,
                target_date=args.date,
                status=status,
                code=failure.code,
                request=request,
                verification={
                    "official_confirmation": False,
                    "failure_code": failure.code,
                },
            ),
            output_payload=_audit_payload(
                request_id=request_id,
                target_date=args.date,
                status=status,
                code=failure.code,
                request=request,
            ),
        )
    except Exception:
        try:
            _atomic_write_json(
                args.output,
                _audit_payload(
                    request_id=request_id,
                    target_date=args.date,
                    status="failed",
                    code="journal_write_failed",
                    request=request,
                ),
            )
        except Exception:
            pass
        return EXIT_PREVIEW_FAILED
    return failure.exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely verify one Teleboat opening candidate without submitting a wager."
    )
    parser.add_argument("--date", type=_parse_date, default=_default_target_date())
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:10001")
    parser.add_argument("--poll-seconds", type=_positive_float, default=15.0)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=3600.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--journal-path", type=Path, default=DEFAULT_JOURNAL_PATH)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
