#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from teleboat_agent.login_secrets import LoginSecrets, save_login_secrets


DEFAULT_PATH = Path(".secrets/teleboat-login.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store Teleboat login-only probe credentials without terminal echo."
    )
    parser.add_argument("--mode", choices=("mobile", "pc"), default="mobile")
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    label = "authorization number" if args.mode == "mobile" else "authorization password"
    payload = {
        "mode": args.mode,
        "member_number": getpass.getpass("Teleboat member number: "),
        "pin": getpass.getpass("Teleboat PIN: "),
        "auth_secret": getpass.getpass(f"Teleboat {label}: "),
    }
    secrets = LoginSecrets.parse(payload)
    save_login_secrets(args.output, secrets)
    print(f"Stored login-only probe secrets at {args.output} with mode 0600.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
