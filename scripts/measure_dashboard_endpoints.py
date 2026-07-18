#!/usr/bin/env python3
"""Measure first-load dashboard endpoints and print JSON results."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:10001"
DEFAULT_TIMEOUT_SECONDS = 30.0
ENDPOINTS = (
    "/",
    "/api/venues",
    "/api/day?lite=1",
    "/api/guide",
    "/api/summary",
    "/api/progress",
    "/api/accuracy",
    "/api/backtest",
    "/api/live-wipe",
)
DATED_ENDPOINTS = {
    "/api/venues",
    "/api/day",
    "/api/guide",
    "/api/progress",
    "/api/accuracy",
    "/api/live-wipe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure WebUI first-load dashboard endpoint latency."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--date", help="Optional race date to send as date=YYYY-MM-DD.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS:g}",
    )
    return parser.parse_args()


def endpoint_path(endpoint: str) -> str:
    return urlsplit(endpoint).path


def with_date(endpoint: str, date: str | None) -> str:
    if not date or endpoint_path(endpoint) not in DATED_ENDPOINTS:
        return endpoint

    parts = urlsplit(endpoint)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key == "date" for key, _ in query):
        query.append(("date", date))
    return urlunsplit(("", "", parts.path, urlencode(query), parts.fragment))


def build_url(base_url: str, endpoint: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def measure_endpoint(base_url: str, endpoint: str, timeout: float) -> dict[str, Any]:
    url = build_url(base_url, endpoint)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "url": url,
        "status": None,
        "bytes": 0,
        "elapsed_ms": None,
    }

    try:
        request = Request(url, headers={"User-Agent": "dashboard-endpoint-measurer/1.0"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            result["status"] = response.status
            result["bytes"] = len(body)
    except HTTPError as exc:
        body = exc.read()
        result["status"] = exc.code
        result["bytes"] = len(body)
        result["error"] = str(exc)
    except (TimeoutError, URLError, OSError) as exc:
        result["error"] = str(exc)
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)

    return result


def main() -> int:
    args = parse_args()
    endpoints = [with_date(endpoint, args.date) for endpoint in ENDPOINTS]
    results = [
        measure_endpoint(args.base_url, endpoint, args.timeout) for endpoint in endpoints
    ]
    payload = {
        "base_url": args.base_url,
        "date": args.date,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
