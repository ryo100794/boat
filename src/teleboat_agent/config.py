from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    application_token: str = field(repr=False)
    live_vote_enabled: bool = False
    live_confirmation_secret: str | None = field(default=None, repr=False)
    member_number: str | None = field(default=None, repr=False)
    pin: str | None = field(default=None, repr=False)
    authorization_number_of_mobile: str | None = field(default=None, repr=False)
    base_url: str = "https://mb.brtb.jp/"
    max_tickets_per_request: int = 30
    max_total_stake_yen: int = 10_000
    batch_size: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("TELEBOAT_AGENT_API_APPLICATION_TOKEN", "").strip()
        if not token or token == "*****":
            raise RuntimeError("TELEBOAT_AGENT_API_APPLICATION_TOKEN must be set securely")
        settings = cls(
            application_token=token,
            live_vote_enabled=_bool_env("TELEBOAT_ENABLE_LIVE_VOTE"),
            live_confirmation_secret=(
                os.getenv("TELEBOAT_LIVE_CONFIRMATION_SECRET") or None
            ),
            member_number=os.getenv("TELEBOAT_MEMBER_NUMBER") or None,
            pin=os.getenv("TELEBOAT_PIN") or None,
            authorization_number_of_mobile=(
                os.getenv("TELEBOAT_AUTHORIZATION_NUMBER_OF_MOBILE") or None
            ),
            base_url=os.getenv("TELEBOAT_BASE_URL", "https://mb.brtb.jp/"),
            max_tickets_per_request=int(
                os.getenv("TELEBOAT_MAX_TICKETS_PER_REQUEST", "30")
            ),
            max_total_stake_yen=int(
                os.getenv("TELEBOAT_MAX_TOTAL_STAKE_YEN", "10000")
            ),
            batch_size=int(os.getenv("TELEBOAT_BATCH_SIZE", "10")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.application_token or self.application_token == "*****":
            raise RuntimeError("application token must be set securely")
        parsed_base_url = urlparse(self.base_url)
        valid_base_url = (
            parsed_base_url.scheme == "https"
            and parsed_base_url.hostname == "mb.brtb.jp"
            and parsed_base_url.port in (None, 443)
            and parsed_base_url.username is None
            and parsed_base_url.password is None
        )
        if not valid_base_url:
            raise RuntimeError("TELEBOAT_BASE_URL must be https://mb.brtb.jp/")
        if self.max_tickets_per_request <= 0:
            raise RuntimeError("TELEBOAT_MAX_TICKETS_PER_REQUEST must be positive")
        if self.max_total_stake_yen < 100:
            raise RuntimeError("TELEBOAT_MAX_TOTAL_STAKE_YEN must be at least 100")
        if not 1 <= self.batch_size <= 12:
            raise RuntimeError("TELEBOAT_BATCH_SIZE must be between 1 and 12")
        if self.live_vote_enabled:
            missing = [
                name
                for name, value in (
                    ("TELEBOAT_LIVE_CONFIRMATION_SECRET", self.live_confirmation_secret),
                    ("TELEBOAT_MEMBER_NUMBER", self.member_number),
                    ("TELEBOAT_PIN", self.pin),
                    (
                        "TELEBOAT_AUTHORIZATION_NUMBER_OF_MOBILE",
                        self.authorization_number_of_mobile,
                    ),
                )
                if not value
            ]
            if missing:
                raise RuntimeError(
                    "live voting is enabled but required settings are missing: "
                    + ", ".join(missing)
                )
