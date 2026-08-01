from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .login_probe import LoginProbeError, TeleboatLoginProbe
from .login_secrets import LoginSecrets, load_login_secrets


DEFAULT_SECRET_PATH = Path(".secrets/teleboat-login.json")
_BALANCE_PATTERNS = (
    re.compile(r"(?:投票可能(?:残高|金額|額)|購入可能(?:残高|金額|額)|購入残高|購入限度額)\s*[:：]?\s*[￥¥]?\s*([0-9][0-9,]*)\s*円"),
    re.compile(r"(?:口座)?残高\s*[:：]?\s*[￥¥]?\s*([0-9][0-9,]*)\s*円"),
)
_BALANCE_LABEL = re.compile(
    r"(?:投票可能(?:残高|金額|額)|購入可能(?:残高|金額|額)|購入残高|購入限度額|(?:口座)?残高)"
)
_YEN_AMOUNT = re.compile(r"[￥¥]?\s*([0-9][0-9,]*)\s*円")


class AccountBalanceError(LoginProbeError):
    pass


@dataclass(frozen=True)
class AccountBalanceResult:
    available_balance_yen: int
    authenticated: bool
    logout_confirmed: bool
    wager_actions: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_available_balance(text: str) -> int:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    compact = re.sub(r"\s+", " ", normalized).strip()
    for pattern in _BALANCE_PATTERNS:
        match = pattern.search(compact)
        if match:
            return int(match.group(1).replace(",", ""))
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    for index, line in enumerate(lines):
        if not _BALANCE_LABEL.search(line):
            continue
        # The official mobile DOM inserts a display-control row between the
        # balance label and value, so inspect only the next three rows.
        for candidate in lines[index:index + 4]:
            amount = _YEN_AMOUNT.search(candidate)
            if amount:
                return int(amount.group(1).replace(",", ""))
    context = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and re.search(r"残|限度|入金|投票|購入|円", line):
            context.append(re.sub(r"\b\d{6,}\b", "[REDACTED]", line))
    if not context:
        for raw in text.splitlines():
            line = " ".join(raw.split())
            if line:
                context.append(re.sub(r"\b\d{6,}\b", "[REDACTED]", line))
    suffix = f"; labels={context[:40]!r}" if context else ""
    raise AccountBalanceError(f"official available balance was not found{suffix}")


def verify_available_balance(text: str, *, required_yen: int) -> int:
    if required_yen < 0:
        raise ValueError("required_yen must be non-negative")
    available = parse_available_balance(text)
    if available < required_yen:
        raise AccountBalanceError("official available balance is insufficient")
    return available


class TeleboatBalanceProbe(TeleboatLoginProbe):
    def _read_balance(self, page, mode: str) -> int:
        text = page.locator("body").inner_text()
        try:
            return parse_available_balance(text)
        except AccountBalanceError as initial_error:
            top_links = page.get_by_text("トップへ", exact=True)
            if not self._any_visible(top_links):
                raise initial_error
            self._visible_from_locator(top_links).click()
            page.wait_for_timeout(500)
            self._assert_allowed_host(page.url, mode)
            return parse_available_balance(page.locator("body").inner_text())

    def balance_probe(self, secrets: LoginSecrets) -> AccountBalanceResult:
        authenticated = False
        logout_confirmed = False
        balance: int | None = None
        with self._browser_page(secrets.mode) as page:
            try:
                self._open_official_page(page, secrets.mode)
                if not self._wait_for_login_form(page, secrets.mode):
                    raise AccountBalanceError("official login form was not available")
                self._submit_login_once(page, secrets)
                authenticated = self._wait_until_authenticated(page, secrets.mode)
                if not authenticated:
                    raise AccountBalanceError("official login was not authenticated")
                self._assert_allowed_host(page.url, secrets.mode)
                balance = self._read_balance(page, secrets.mode)
            finally:
                if authenticated:
                    logout_confirmed = self._logout(page, secrets.mode)
        if balance is None:
            raise AccountBalanceError("official available balance was not obtained")
        return AccountBalanceResult(balance, True, logout_confirmed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Teleboat available-balance probe; never wagers.")
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = TeleboatBalanceProbe(timeout=args.timeout).balance_probe(load_login_secrets(args.secrets))
        print(json.dumps({"success": True, **result.to_dict()}, ensure_ascii=False, sort_keys=True))
        return 0 if result.logout_confirmed else 2
    except (AccountBalanceError, OSError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
