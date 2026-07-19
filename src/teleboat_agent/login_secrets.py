from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


LoginMode = Literal["mobile", "pc"]


class SecretFileError(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class LoginSecrets:
    mode: LoginMode
    member_number: str
    pin: str
    auth_secret: str

    @classmethod
    def parse(cls, payload: Any) -> "LoginSecrets":
        if not isinstance(payload, dict):
            raise SecretFileError("secret payload must be an object")
        mode = str(payload.get("mode") or "").strip().lower()
        member_number = str(payload.get("member_number") or "").strip()
        pin = str(payload.get("pin") or "").strip()
        auth_secret = str(payload.get("auth_secret") or "").strip()
        if mode not in {"mobile", "pc"}:
            raise SecretFileError("mode must be mobile or pc")
        if not member_number.isdigit() or not 6 <= len(member_number) <= 10:
            raise SecretFileError("member number must contain 6 to 10 digits")
        if not pin.isdigit() or not 4 <= len(pin) <= 6:
            raise SecretFileError("PIN must contain 4 to 6 digits")
        if mode == "mobile":
            if not auth_secret.isdigit() or not 4 <= len(auth_secret) <= 6:
                raise SecretFileError(
                    "mobile authorization number must contain 4 to 6 digits"
                )
        elif not auth_secret.isalnum() or not 6 <= len(auth_secret) <= 8:
            raise SecretFileError(
                "PC authorization password must contain 6 to 8 alphanumeric characters"
            )
        return cls(
            mode=mode,
            member_number=member_number,
            pin=pin,
            auth_secret=auth_secret,
        )

    def __repr__(self) -> str:
        return f"LoginSecrets(mode={self.mode!r}, credentials=<redacted>)"

    def as_payload(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "member_number": self.member_number,
            "pin": self.pin,
            "auth_secret": self.auth_secret,
        }


def save_login_secrets(path: Path, secrets: LoginSecrets) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(secrets.as_payload(), handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_login_secrets(path: Path) -> LoginSecrets:
    path = path.expanduser()
    if path.is_symlink():
        raise SecretFileError("secret file must not be a symbolic link")
    path = path.resolve()
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise SecretFileError(f"secret file does not exist: {path}") from exc
    permissions = stat.S_IMODE(metadata.st_mode)
    if permissions & 0o077:
        raise SecretFileError("secret file permissions must be 0600")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecretFileError("secret file is not valid UTF-8 JSON") from exc
    return LoginSecrets.parse(payload)
