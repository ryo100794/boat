from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..db import connection, init_db, insert_prediction_rows
from ..feature_tuning import build_race_features
from ..features import latest_trifecta_odds_before_deadline
from ..legacy_model_aliases import load_model_bundle
from ..listwise.closing_odds import decision_odds
from ..listwise.closing_odds_momentum import attach_selected_closing_odds
from ..listwise.market_calibration import (
    artifact_model_probabilities,
    blend_probabilities,
    earlier_market_fields,
    file_sha256,
    normalized_market_probabilities,
    snapshot_age_seconds,
)
from ..listwise.market_promotion import MANIFEST_VERSION, REQUIRED_PASS_GATES
from ..listwise.live_shadow import historical_state, load_date_races
from .time_semantics import operational_race_date


JST = timezone(timedelta(hours=9))
FEATURE_SET = "promoted_market_t5_v1"
MAX_SNAPSHOT_AGE_SECONDS = 60.0


def load_active_manifest(path: str | Path, *, race_date: date) -> dict[str, Any] | None:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"active market manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("active market manifest is not an object")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("active market manifest version mismatch")
    if manifest.get("status") != "active":
        raise ValueError("active market manifest status is not active")
    if str(manifest.get("valid_from_date") or "") > race_date.isoformat():
        return None
    deployment = manifest.get("deployment_configuration") or {}
    if str(deployment.get("trained_through_date") or "") >= race_date.isoformat():
        raise ValueError("deployment configuration is not strictly prior to race date")
    for key in REQUIRED_PASS_GATES:
        if (manifest.get("promotion_gate") or {}).get(key) is not True:
            raise ValueError(f"active market manifest gate failed: {key}")

    source_path = Path(str(manifest.get("source_model_path") or ""))
    evaluation_path = Path(str(manifest.get("evaluation_path") or ""))
    if not source_path.is_file() or not evaluation_path.is_file():
        raise ValueError("active market manifest artifact is missing")
    if file_sha256(source_path) != manifest.get("source_model_sha256"):
        raise ValueError("active market source model hash mismatch")
    if file_sha256(evaluation_path) != manifest.get("evaluation_sha256"):
        raise ValueError("active market evaluation hash mismatch")
    return manifest


def build_promoted_prediction_rows(
    model_probabilities: dict[str, float],
    *,
    snapshot: dict[str, Any],
    deployment: dict[str, Any],
    earlier_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    odds = {key: float(value) for key, value in (snapshot.get("odds") or {}).items()}
    market = normalized_market_probabilities(odds)
    if len(model_probabilities) != 120 or len(market) != 120:
        raise ValueError("promoted market prediction requires 120 combinations")
    calibrator = deployment.get("calibrator") or {}
    probabilities = blend_probabilities(
        model_probabilities,
        market,
        model_weight=float(calibrator["model_weight"]),
        temperature=float(calibrator["temperature"]),
    )
    race = {
        "odds": odds,
        "market_probabilities": market,
        **(earlier_fields or {}),
    }
    closing_selection = deployment.get("closing_odds_selection")
    if isinstance(closing_selection, dict):
        race = attach_selected_closing_odds([race], closing_selection)[0]
    forecast_odds = decision_odds(race)
    policy = deployment.get("selected_policy") or {}
    rows = []
    for combination, probability in probabilities.items():
        rows.append(
            {
                "combination": combination,
                "probability": float(probability),
                "odds": odds[combination],
                "expected_value": float(probability)
                * float(forecast_odds[combination]),
                "estimated_final_odds": float(forecast_odds[combination]),
                "rank_basis": "promoted_market_probability",
                "feature_set": FEATURE_SET,
                "snapshot_id": snapshot.get("snapshot_id"),
                "odds_captured_at": snapshot.get("captured_at"),
                "odds_deadline_at": snapshot.get("odds_deadline_at"),
                "selected_policy": policy.get("name"),
            }
        )
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


class MarketPredictor:
    def __init__(self) -> None:
        self._manifest_key: tuple[str, str] | None = None
        self._artifact: dict[str, Any] | None = None
        self._date_key: tuple[str, str] | None = None
        self._model_probabilities: dict[str, dict[str, float]] = {}
        self._last_snapshot: dict[str, int] = {}

    def _prepare(
        self,
        conn,
        *,
        manifest: dict[str, Any],
        race_date: date,
    ) -> None:
        manifest_key = (
            str(manifest["source_model_path"]),
            str(manifest["source_model_sha256"]),
        )
        if manifest_key != self._manifest_key:
            self._artifact = load_model_bundle(manifest["source_model_path"])
            self._manifest_key = manifest_key
            self._date_key = None
            self._last_snapshot.clear()
        date_key = (race_date.isoformat(), manifest_key[1])
        if date_key == self._date_key:
            return
        assert self._artifact is not None
        state = historical_state(conn, race_date=race_date.isoformat())
        rows_by_race = load_date_races(conn, race_date=race_date.isoformat())
        probabilities = {}
        dropped = tuple(self._artifact.get("drop_feature_groups") or ())
        for race_id, race_rows in rows_by_race.items():
            feature_rows = build_race_features(
                race_rows,
                state,
                drop_feature_groups=dropped,
                feature_schema_version=self._artifact.get("feature_schema_version"),
            )
            probabilities[race_id] = artifact_model_probabilities(
                self._artifact, feature_rows
            )
        self._model_probabilities = probabilities
        self._date_key = date_key
        self._last_snapshot.clear()

    def predict(
        self,
        conn,
        *,
        manifest: dict[str, Any],
        race_date: date,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._prepare(conn, manifest=manifest, race_date=race_date)
        now_jst = (now or datetime.now(timezone.utc)).astimezone(JST)
        deployment = manifest["deployment_configuration"]
        predicted = skipped_before_t5 = skipped_stale = unchanged = failed = 0
        for race_id, model_probabilities in self._model_probabilities.items():
            try:
                snapshot = latest_trifecta_odds_before_deadline(
                    conn,
                    race_id,
                    min_combinations=120,
                    decision_lead_minutes=5,
                )
                if snapshot is None:
                    continue
                odds_deadline = datetime.fromisoformat(str(snapshot["odds_deadline_at"]))
                if odds_deadline.tzinfo is None:
                    odds_deadline = odds_deadline.replace(tzinfo=JST)
                if now_jst < odds_deadline.astimezone(JST):
                    skipped_before_t5 += 1
                    continue
                age = snapshot_age_seconds(snapshot)
                if age is None or age < 0.0 or age > MAX_SNAPSHOT_AGE_SECONDS:
                    skipped_stale += 1
                    continue
                snapshot_id = int(snapshot["snapshot_id"])
                if self._last_snapshot.get(race_id) == snapshot_id:
                    unchanged += 1
                    continue
                earlier, _reason = earlier_market_fields(
                    conn,
                    race_id,
                    current_snapshot=snapshot,
                    max_snapshot_age_seconds=MAX_SNAPSHOT_AGE_SECONDS,
                )
                rows = build_promoted_prediction_rows(
                    model_probabilities,
                    snapshot=snapshot,
                    deployment=deployment,
                    earlier_fields=earlier,
                )
                generated_at = datetime.now(timezone.utc).replace(
                    microsecond=0
                ).isoformat()
                insert_prediction_rows(
                    conn,
                    race_id,
                    generated_at,
                    f"market:{manifest['selected_candidate_id']}@{manifest['evaluation_sha256'][:12]}",
                    rows,
                )
                self._last_snapshot[race_id] = snapshot_id
                predicted += 1
            except Exception:
                failed += 1
        return {
            "predicted": predicted,
            "skipped_before_t5": skipped_before_t5,
            "skipped_stale": skipped_stale,
            "unchanged": unchanged,
            "failed": failed,
            "prepared_races": len(self._model_probabilities),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Store T-5 predictions from a verified promoted market model."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument(
        "--manifest", default="data/models/active_market_model.json"
    )
    parser.add_argument("--date")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-loops", type=int)
    args = parser.parse_args(argv)
    init_db(args.db)
    fixed_date = date.fromisoformat(args.date) if args.date else None
    predictor = MarketPredictor()
    loop = 0
    while True:
        race_date = operational_race_date(fixed_date)
        event: dict[str, Any] = {
            "loop": loop,
            "race_date": race_date.isoformat(),
            "manifest": args.manifest,
        }
        try:
            manifest = load_active_manifest(args.manifest, race_date=race_date)
            if manifest is None:
                event["status"] = "inactive"
            else:
                with connection(args.db) as conn:
                    event.update(
                        predictor.predict(
                            conn,
                            manifest=manifest,
                            race_date=race_date,
                        )
                    )
                event["status"] = "active"
                event["selected_candidate_id"] = manifest[
                    "selected_candidate_id"
                ]
        except Exception as exc:
            event["status"] = "error"
            event["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(max(10.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
