from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..listwise.market_calibration import MARKET_EVALUATION_VERSION


JST = timezone(timedelta(hours=9))


def completed_through_date(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return (current.astimezone(JST).date() - timedelta(days=1)).isoformat()


def evaluation_due(
    state: dict[str, Any],
    *,
    through_date: str,
    output_exists: bool,
    model_sha256: str | None = None,
    evaluation_version: int | None = None,
) -> bool:
    if not output_exists:
        return True
    if model_sha256 and state.get("model_sha256") != model_sha256:
        return True
    if (
        evaluation_version is not None
        and state.get("evaluation_version") != evaluation_version
    ):
        return True
    return str(state.get("completed_through_date") or "") < through_date


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_command(args: argparse.Namespace, *, through_date: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "boatrace_ai.listwise.market_calibration",
        "--db",
        args.db,
        "--model",
        args.model,
        "--output",
        args.output,
        "--from-date",
        args.from_date,
        "--through-date",
        through_date,
        "--daily-budget-yen",
        str(args.daily_budget_yen),
        "--min-calibration-days",
        str(args.min_calibration_days),
        "--max-snapshot-age-seconds",
        str(args.max_snapshot_age_seconds),
    ]


def run_once(args: argparse.Namespace, *, now: datetime | None = None) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    through_date = completed_through_date(now)
    state_path = Path(args.state)
    output_path = Path(args.output)
    model_path = Path(args.model)
    model_hash = file_sha256(model_path)
    previous = read_state(state_path)
    event: dict[str, Any] = {
        "generated_at": generated_at,
        "target_through_date": through_date,
        "from_date": args.from_date,
        "output": str(output_path),
        "model": str(model_path),
        "model_sha256": model_hash,
        "evaluation_version": MARKET_EVALUATION_VERSION,
    }
    if not evaluation_due(
        previous,
        through_date=through_date,
        output_exists=output_path.exists(),
        model_sha256=model_hash,
        evaluation_version=MARKET_EVALUATION_VERSION,
    ):
        event.update(
            {
                "status": "up_to_date",
                "completed_through_date": previous.get("completed_through_date"),
            }
        )
        write_state(state_path, event)
        return event

    completed = subprocess.run(
        build_command(args, through_date=through_date),
        text=True,
        capture_output=True,
        timeout=max(60, int(args.timeout)),
    )
    event["returncode"] = completed.returncode
    event["stdout_tail"] = completed.stdout.splitlines()[-40:]
    event["stderr_tail"] = completed.stderr.splitlines()[-40:]
    if completed.returncode == 0:
        event["status"] = "evaluated"
        event["completed_through_date"] = through_date
    else:
        event["status"] = "error"
        event["error"] = f"market calibration exited {completed.returncode}"
        if previous.get("completed_through_date"):
            event["completed_through_date"] = previous["completed_through_date"]
    write_state(state_path, event)
    return event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the market-calibrated shadow once per completed JST day."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/listwise_newton_cg_v1.joblib")
    parser.add_argument(
        "--output",
        default="data/models/listwise_market_calibrated_shadow.json",
    )
    parser.add_argument(
        "--state",
        default="data/models/listwise_market_calibrated_shadow_state.json",
    )
    parser.add_argument("--from-date", default="2026-07-18")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--min-calibration-days", type=int, default=2)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=3600.0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        try:
            event = run_once(args)
        except subprocess.TimeoutExpired as exc:
            event = {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "status": "error",
                "error": f"market calibration timeout after {exc.timeout}s",
            }
            write_state(Path(args.state), event)
        except Exception as exc:
            event = {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_state(Path(args.state), event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if args.once:
            return 0 if event.get("status") != "error" else 1
        time.sleep(max(60.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
