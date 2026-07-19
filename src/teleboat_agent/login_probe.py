from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .login_secrets import LoginSecrets, SecretFileError, load_login_secrets


MOBILE_URL = "https://spweb.brtb.jp/"
PC_URL = "https://ib.mbrace.or.jp/"
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Mobile Safari/537.36"
)
LOGIN_CONFIRMATION = "LOGIN_ONLY_NO_WAGER"
DEFAULT_STATUS_PATH = Path("data/teleboat_probe_status.json")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYWRIGHT_BROWSERS = PROJECT_ROOT / ".tools" / "ms-playwright"


class LoginProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeResult:
    mode: str
    browser: str
    public_page_ready: bool
    authenticated: bool
    logout_confirmed: bool
    wager_actions: int
    attempts: int
    final_location: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TeleboatLoginProbe:
    def __init__(self, *, browser: str = "chromium", timeout: float = 20.0):
        if browser != "chromium":
            raise ValueError("browser must be chromium")
        self.browser = browser
        self.timeout = max(5.0, float(timeout))

    def public_probe(self, mode: str) -> ProbeResult:
        started = time.perf_counter()
        with self._browser_page(mode) as page:
            self._open_official_page(page, mode)
            ready = self._wait_for_login_form(page, mode)
            self._assert_allowed_host(page.url, mode)
            return ProbeResult(
                mode=mode,
                browser=self.browser,
                public_page_ready=ready,
                authenticated=False,
                logout_confirmed=False,
                wager_actions=0,
                attempts=0,
                final_location=self._safe_location(page.url),
                elapsed_seconds=round(time.perf_counter() - started, 3),
            )

    def login_probe(self, secrets: LoginSecrets) -> ProbeResult:
        started = time.perf_counter()
        with self._browser_page(secrets.mode) as page:
            self._open_official_page(page, secrets.mode)
            ready = self._wait_for_login_form(page, secrets.mode)
            self._assert_allowed_host(page.url, secrets.mode)
            if not ready:
                raise LoginProbeError("official login form was not available")
            self._submit_login_once(page, secrets)
            authenticated = self._wait_until_authenticated(page, secrets.mode)
            self._assert_allowed_host(page.url, secrets.mode)
            logout_confirmed = (
                self._logout(page, secrets.mode) if authenticated else False
            )
            return ProbeResult(
                mode=secrets.mode,
                browser=self.browser,
                public_page_ready=ready,
                authenticated=authenticated,
                logout_confirmed=logout_confirmed,
                wager_actions=0,
                attempts=1,
                final_location=self._safe_location(page.url),
                elapsed_seconds=round(time.perf_counter() - started, 3),
            )

    @contextmanager
    def _browser_page(self, mode: str):
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise LoginProbeError(
                "playwright is required; install the teleboat optional dependency"
            ) from exc

        os.environ.setdefault(
            "PLAYWRIGHT_BROWSERS_PATH",
            str(PLAYWRIGHT_BROWSERS),
        )
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    ignore_default_args=["--enable-unsafe-swiftshader"],
                    args=[
                        "--disable-gpu",
                        "--disable-gpu-compositing",
                        "--disable-software-rasterizer",
                        "--use-gl=disabled",
                        "--disable-vulkan",
                    ],
                )

                context = browser.new_context(
                    user_agent=MOBILE_USER_AGENT if mode == "mobile" else None,
                    viewport={"width": 390, "height": 844}
                    if mode == "mobile"
                    else {"width": 1280, "height": 900},
                )
                page = context.new_page()
                page.set_default_timeout(self.timeout * 1000)
                try:
                    yield page
                finally:
                    try:
                        page.evaluate(
                            "window.localStorage.clear(); window.sessionStorage.clear();"
                        )
                    except PlaywrightError:
                        pass
                    context.clear_cookies()
                    context.close()
                    browser.close()
        except LoginProbeError:
            raise
        except PlaywrightError as exc:
            raise LoginProbeError("chromium browser operation failed") from exc

    def _open_official_page(self, page, mode: str) -> None:
        page.goto(
            self._url(mode),
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )
        self._assert_allowed_host(page.url, mode)

    def _wait_for_login_form(self, page, mode: str) -> bool:
        try:
            if mode == "pc":
                page.locator("#loginButton").wait_for(state="visible")
                return True
            if not self._mobile_fields_visible(page):
                self._visible_locator(page, ".btn-login").click()
            fields = page.locator("input.login-input")
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if fields.count() == 3 and all(
                    fields.nth(index).is_visible() for index in range(3)
                ):
                    return True
                page.wait_for_timeout(200)
            return False
        except Exception as exc:
            if exc.__class__.__module__.startswith("playwright."):
                return False
            raise

    def _submit_login_once(self, page, secrets: LoginSecrets) -> None:
        if secrets.mode == "pc":
            fields = [
                page.locator("#memberNo"),
                page.locator("#pin"),
                page.locator("#authPassword"),
            ]
            button = page.locator("#loginButton")
        else:
            locator = page.locator("input.login-input")
            fields = [locator.nth(index) for index in range(3)]
            button = self._visible_locator(page, ".btn-login")
        for field, value in zip(
            fields,
            (secrets.member_number, secrets.pin, secrets.auth_secret),
        ):
            field.fill(value)
        button.click()

    def _wait_until_authenticated(self, page, mode: str) -> bool:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if not self._login_form_present(page, mode):
                return True
            page.wait_for_timeout(200)
        return False

    def _logout(self, page, mode: str) -> bool:
        try:
            if mode == "mobile":
                self._visible_locator(page, ".menu-open").click()
            page.get_by_text("ログアウト", exact=True).click()
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self._login_form_present(page, mode):
                    return True
                page.wait_for_timeout(200)
            return False
        except Exception as exc:
            if exc.__class__.__module__.startswith("playwright."):
                return False
            raise

    @staticmethod
    def _mobile_fields_visible(page) -> bool:
        fields = page.locator("input.login-input")
        return fields.count() == 3 and all(
            fields.nth(index).is_visible() for index in range(3)
        )

    @classmethod
    def _login_form_present(cls, page, mode: str) -> bool:
        if mode == "pc":
            return page.locator("#loginButton").is_visible()
        return cls._mobile_fields_visible(page)

    @staticmethod
    def _visible_locator(page, selector: str):
        candidates = page.locator(selector)
        for index in range(candidates.count() - 1, -1, -1):
            candidate = candidates.nth(index)
            if candidate.is_visible():
                return candidate
        raise LoginProbeError(f"visible element was not found: {selector}")

    @staticmethod
    def _url(mode: str) -> str:
        if mode == "mobile":
            return MOBILE_URL
        if mode == "pc":
            return PC_URL
        raise ValueError("mode must be mobile or pc")

    @staticmethod
    def _assert_allowed_host(url: str, mode: str) -> None:
        expected = (
            {"spweb.brtb.jp", "login.brtb.jp"}
            if mode == "mobile"
            else {"ib.mbrace.or.jp"}
        )
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in expected:
            raise LoginProbeError("browser left the allowlisted official host")

    @staticmethod
    def _safe_location(url: str) -> str:
        parsed = urlsplit(url)
        return f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"


def write_probe_status(path: Path, phase: str, result: dict[str, object]) -> None:
    if phase not in {"public", "login"}:
        raise ValueError("phase must be public or login")
    path = path.expanduser().resolve()
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_phase": phase,
        "public": previous.get("public") if phase != "public" else result,
        "login": previous.get("login") if phase != "login" else result,
        "policy": {"live_wager_enabled": False, "wager_actions": 0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-attempt Teleboat login-only probe; never performs wager actions."
    )
    parser.add_argument("--mode", choices=("mobile", "pc"), default="mobile")
    parser.add_argument("--browser", choices=("chromium",), default="chromium")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--secrets", type=Path)
    parser.add_argument("--confirm-login-only")
    parser.add_argument("--output", type=Path, default=DEFAULT_STATUS_PATH)
    args = parser.parse_args(argv)

    probe = TeleboatLoginProbe(browser=args.browser, timeout=args.timeout)
    try:
        if args.public_only:
            result = probe.public_probe(args.mode)
        else:
            if args.confirm_login_only != LOGIN_CONFIRMATION:
                parser.error(
                    f"--confirm-login-only {LOGIN_CONFIRMATION} is required"
                )
            if args.secrets is None:
                parser.error("--secrets is required for a login-only probe")
            secrets = load_login_secrets(args.secrets)
            if secrets.mode != args.mode:
                parser.error("secret file mode does not match --mode")
            result = probe.login_probe(secrets)
    except (LoginProbeError, SecretFileError) as exc:
        failure = {
            "success": False,
            "error_type": type(exc).__name__,
            "wager_actions": 0,
        }
        write_probe_status(args.output, "public" if args.public_only else "login", failure)
        print(json.dumps(failure))
        return 2

    payload = result.to_dict()
    payload["success"] = bool(
        result.public_page_ready
        and (
            args.public_only
            or (result.authenticated and result.logout_confirmed)
        )
    )
    write_probe_status(args.output, "public" if args.public_only else "login", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
