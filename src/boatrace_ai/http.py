from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .constants import USER_AGENT


class FetchError(RuntimeError):
    pass


def _requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in deployed env
        raise FetchError("requests is required: pip install -e .") from exc
    return requests


def default_headers() -> dict[str, str]:
    contact = os.environ.get("BOATRACE_AI_CONTACT")
    ua = USER_AGENT if not contact else f"{USER_AGENT} ({contact})"
    return {
        "User-Agent": ua,
        "Accept-Language": "ja,en;q=0.8",
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cache_busted_url(url: str) -> str:
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("_boatrace_nonce", uuid.uuid4().hex))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def fetch_bytes(
    url: str,
    *,
    timeout: float = 30.0,
    retries: int = 2,
    sleep_seconds: float = 0.0,
    cache_bust: bool = False,
) -> tuple[int, bytes]:
    requests = _requests()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            headers = default_headers()
            request_url = url
            if cache_bust:
                headers["Cache-Control"] = "no-cache, no-store, max-age=0"
                request_url = _cache_busted_url(url)
            response = requests.get(
                request_url,
                headers=headers,
                timeout=timeout,
            )
            return response.status_code, response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(max(0.5, sleep_seconds))
    raise FetchError(f"failed to fetch {url}: {last_error}")


def fetch_text(
    url: str,
    *,
    timeout: float = 30.0,
    retries: int = 2,
    sleep_seconds: float = 0.0,
    cache_bust: bool = False,
) -> tuple[int, str, bytes]:
    status_code, payload = fetch_bytes(
        url,
        timeout=timeout,
        retries=retries,
        sleep_seconds=sleep_seconds,
        cache_bust=cache_bust,
    )
    for encoding in ("utf-8", "cp932", "shift_jis"):
        try:
            return status_code, payload.decode(encoding), payload
        except UnicodeDecodeError:
            continue
    return status_code, payload.decode("utf-8", errors="replace"), payload


def save_payload(path: str | Path, payload: bytes) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "local_path": str(target),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }
