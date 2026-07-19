from __future__ import annotations

import argparse
import hmac
import json
from http import HTTPStatus
from typing import Any, Callable
from wsgiref.simple_server import make_server

from .browser import VoteExecutionError
from .config import Settings
from .models import ValidationError
from .service import (
    AuthorizationError,
    DuplicateRequestError,
    VoteTicketsService,
)

MAX_BODY_BYTES = 64 * 1024
VOTES_PATH = "/api/internal/v1/tickets/votes"


class TeleboatApplication:
    def __init__(
        self,
        settings: Settings,
        *,
        service_factory: Callable[[Settings], VoteTicketsService] = VoteTicketsService,
    ) -> None:
        self.settings = settings
        self.service = service_factory(settings)

    def __call__(self, environ: dict[str, Any], start_response):
        try:
            status, payload = self._dispatch(environ)
        except ValidationError as exc:
            status, payload = HTTPStatus.BAD_REQUEST, {"errors": [{"message": str(exc)}]}
        except AuthorizationError as exc:
            status, payload = HTTPStatus.FORBIDDEN, {"errors": [{"message": str(exc)}]}
        except DuplicateRequestError as exc:
            status, payload = HTTPStatus.CONFLICT, {"errors": [{"message": str(exc)}]}
        except VoteExecutionError:
            status, payload = (
                HTTPStatus.BAD_GATEWAY,
                {"errors": [{"message": "live vote execution failed"}]},
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            status, payload = (
                HTTPStatus.BAD_REQUEST,
                {"errors": [{"message": "request body must be valid UTF-8 JSON"}]},
            )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(
            f"{status.value} {status.phrase}",
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ],
        )
        return [body]

    def _dispatch(self, environ: dict[str, Any]):
        if environ.get("PATH_INFO") != VOTES_PATH:
            return HTTPStatus.NOT_FOUND, {"errors": [{"message": "not found"}]}
        if environ.get("REQUEST_METHOD") != "POST":
            return HTTPStatus.METHOD_NOT_ALLOWED, {
                "errors": [{"message": "method not allowed"}]
            }
        self._authenticate(environ.get("HTTP_AUTHORIZATION", ""))
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError as exc:
            raise ValidationError("invalid Content-Length") from exc
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            raise ValidationError("request body size is invalid")
        body = environ["wsgi.input"].read(content_length)
        payload = json.loads(body.decode("utf-8"))
        result = self.service.call(
            payload,
            live_requested=(
                environ.get("HTTP_X_TELEBOAT_LIVE", "").lower() == "true"
            ),
            live_confirmation=environ.get("HTTP_X_TELEBOAT_LIVE_CONFIRMATION"),
            idempotency_key=environ.get("HTTP_IDEMPOTENCY_KEY"),
        )
        return HTTPStatus.OK, result

    def _authenticate(self, authorization: str) -> None:
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        if not supplied or not hmac.compare_digest(
            supplied,
            self.settings.application_token,
        ):
            raise AuthorizationError("invalid bearer token")


def create_application(settings: Settings | None = None) -> TeleboatApplication:
    return TeleboatApplication(settings or Settings.from_env())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audited Python Teleboat agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args(argv)
    application = create_application()
    with make_server(args.host, args.port, application) as server:
        print(
            f"teleboat-agent listening on http://{args.host}:{args.port} "
            "(dry-run unless live voting is explicitly enabled)",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
