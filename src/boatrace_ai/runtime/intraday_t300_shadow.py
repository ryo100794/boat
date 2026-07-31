from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Protocol, Sequence

import joblib

from ..adaptive_allocation import allocate_adaptive_day

from ..db import connection
from ..discrete_log_allocation import allocate_discrete_log_day
from ..feature_tuning import build_race_features
from ..features import stored_jst_timestamp_sql
from ..listwise.closing_envelope_conformal_v15 import (
    METHOD as V15_CLOSING_ENVELOPE_METHOD,
    apply_closing_envelope_haircut_v15,
)
from ..listwise.closing_odds_t300_nonlinear_v12 import (
    MODEL_NAME as V12_CLOSING_MODEL_NAME,
    forecast_closing_odds_t300_nonlinear_v12,
)
from ..listwise.closing_odds_momentum import attach_selected_closing_odds
from ..listwise.edge_conditional_probability_lcb_v14 import (
    METHOD as V14_PROBABILITY_LCB_METHOD,
    REGISTERED_DIVERGENCE_LOWER,
    REGISTERED_DIVERGENCE_UPPER,
    probability_lower_bound_details_v14,
    t300_snapshot_consistency,
)
from ..listwise.live_shadow import historical_state, load_date_races
from ..listwise.market_calibration import (
    artifact_model_probabilities,
    blend_probabilities,
    earlier_market_fields,
    normalized_market_probabilities,
    normalize_odds_checkpoint,
    odds_path_fields,
)
from ..listwise.odds_path_operational import attach_odds_path_model
from ..listwise.odds_path_conservative_v7 import (
    MAX_DAILY_EXPOSURE_FRACTION,
    MAX_TICKETS_PER_RACE,
    RACE_CAP_FRACTION,
    SAFE_EV_THRESHOLD,
    STAKE_GRANULARITY_YEN,
    TICKET_CAP_FRACTION,
    _policy_candidate,
)
from ..listwise.odds_path_probability_v8 import attach_odds_path_probability_v8
from ..listwise.odds_path_selection_conformal_v10 import _zero_reason
from ..listwise.selection_conformal import selected_safe_ev_candidates
from ..listwise.strict_prior_t300_divergence_passthrough_v16 import (
    METHOD as V16_PASSTHROUGH_METHOD,
    MODEL_NAME as V16_PASSTHROUGH_MODEL,
)
from ..odds_quality import TRIFECTA_PARSER_VERSION, plausible_trifecta_odds
from .top5_narrow_policy import (
    POLICY_NAME as V23_POLICY_NAME,
    REGISTERED_AFTER as V23_REGISTERED_AFTER,
    STAKE_YEN as V23_STAKE_YEN,
    daily_capital_limits,
    select_top5_narrow_candidates,
)


JST = timezone(timedelta(hours=9))
STARTING_BANKROLL_YEN = 10_000
T300_OFFSET_SECONDS = 300
DECISION_BEFORE_START_SECONDS = 600
DEFAULT_MAX_CHECKPOINT_AGE_SECONDS = 90.0
DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS = 120.0
DEFAULT_MAX_DECISION_DELAY_SECONDS = 90.0
DEFAULT_INTERVAL_SECONDS = 5.0
SCHEMA_VERSION = 1
_SHARED_HISTORICAL_STATE: dict[tuple[int, str], Any] = {}
_SHARED_DATE_RACES: dict[tuple[int, str, object], dict[str, list[Any]]] = {}

POSTGRESQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS intraday_t300_shadow_decisions (
  decision_id BIGSERIAL PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  race_date DATE NOT NULL,
  race_id TEXT NOT NULL REFERENCES races(race_id) ON DELETE RESTRICT,
  model_key TEXT NOT NULL,
  model_hash CHAR(64) NOT NULL,
  strategy_name TEXT NOT NULL,
  decision_at TIMESTAMPTZ NOT NULL,
  decision_completed_at TIMESTAMPTZ NOT NULL,
  target_t300_at TIMESTAMPTZ NOT NULL,
  source_snapshot_id BIGINT REFERENCES odds_snapshots(snapshot_id) ON DELETE RESTRICT,
  source_captured_at TIMESTAMPTZ,
  checkpoint_age_before_target_seconds DOUBLE PRECISION,
  source_update_staleness_seconds DOUBLE PRECISION,
  bankroll_before_yen INTEGER NOT NULL,
  decision_status TEXT NOT NULL CHECK (decision_status IN ('selected', 'no_bet')),
  no_bet_reason TEXT,
  probabilities JSONB NOT NULL,
  probability_summary JSONB NOT NULL,
  closing_lower_odds JSONB NOT NULL,
  closing_lower_summary JSONB NOT NULL,
  selected_candidates JSONB NOT NULL,
  diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
  total_stake_yen INTEGER NOT NULL,
  decision_hash CHAR(64) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (race_id, model_key),
  CHECK (
    (decision_status = 'selected' AND no_bet_reason IS NULL AND total_stake_yen > 0)
    OR
    (decision_status = 'no_bet' AND no_bet_reason IS NOT NULL AND total_stake_yen = 0)
  )
);
ALTER TABLE intraday_t300_shadow_decisions
  ADD COLUMN IF NOT EXISTS decision_completed_at TIMESTAMPTZ;
ALTER TABLE intraday_t300_shadow_decisions
  ADD COLUMN IF NOT EXISTS diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_intraday_t300_shadow_decisions_day_model
  ON intraday_t300_shadow_decisions(race_date, model_key, target_t300_at);

CREATE TABLE IF NOT EXISTS intraday_t300_shadow_settlements (
  settlement_id BIGSERIAL PRIMARY KEY,
  decision_id BIGINT NOT NULL UNIQUE
    REFERENCES intraday_t300_shadow_decisions(decision_id) ON DELETE RESTRICT,
  settled_at TIMESTAMPTZ NOT NULL,
  result_status TEXT NOT NULL,
  actual_combination TEXT NOT NULL,
  payout_yen_per_100 INTEGER NOT NULL,
  stake_yen INTEGER NOT NULL,
  return_yen INTEGER NOT NULL,
  profit_yen INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION reject_intraday_t300_shadow_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;
DROP TRIGGER IF EXISTS intraday_t300_shadow_decisions_immutable
  ON intraday_t300_shadow_decisions;
CREATE TRIGGER intraday_t300_shadow_decisions_immutable
BEFORE UPDATE OR DELETE ON intraday_t300_shadow_decisions
FOR EACH ROW EXECUTE FUNCTION reject_intraday_t300_shadow_mutation();
DROP TRIGGER IF EXISTS intraday_t300_shadow_settlements_immutable
  ON intraday_t300_shadow_settlements;
CREATE TRIGGER intraday_t300_shadow_settlements_immutable
BEFORE UPDATE OR DELETE ON intraday_t300_shadow_settlements
FOR EACH ROW EXECUTE FUNCTION reject_intraday_t300_shadow_mutation();
"""


@dataclass(frozen=True)
class ModelIdentity:
    model_key: str
    model_hash: str
    strategy_name: str


@dataclass(frozen=True)
class RaceWindow:
    race_id: str
    race_date: str
    jcd: str
    rno: int
    start_at: datetime

    @property
    def target_t300_at(self) -> datetime:
        return self.start_at - timedelta(seconds=DECISION_BEFORE_START_SECONDS)

    @property
    def betting_deadline_at(self) -> datetime:
        return self.start_at - timedelta(seconds=T300_OFFSET_SECONDS)


@dataclass(frozen=True)
class T300Snapshot:
    snapshot_id: int
    captured_at: datetime
    source_update_time: str | None
    raw_json: Any
    odds: dict[str, float]


@dataclass(frozen=True)
class SnapshotCheck:
    checkpoint_age_before_target_seconds: float | None
    source_update_staleness_seconds: float | None
    reason: str | None


@dataclass(frozen=True)
class ShadowDecision:
    probabilities: dict[str, float]
    closing_lower_odds: dict[str, float]
    selected_candidates: tuple[dict[str, Any], ...]
    no_bet_reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_stake_yen(self) -> int:
        return sum(int(row["stake_yen"]) for row in self.selected_candidates)

    @property
    def status(self) -> str:
        return "selected" if self.total_stake_yen > 0 else "no_bet"


class ShadowModelAdapter(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def decide(
        self,
        conn: Any,
        race: RaceWindow,
        snapshot: T300Snapshot,
        *,
        bankroll_yen: int,
    ) -> ShadowDecision: ...


class ShadowStore(Protocol):
    conn: Any

    def ensure_schema(self) -> None: ...
    def due_races(self, *, race_date: str, now: datetime) -> Sequence[RaceWindow]: ...
    def decision_identity(self, *, race_id: str, model_key: str) -> ModelIdentity | None: ...
    def latest_complete_snapshot(self, race: RaceWindow) -> T300Snapshot | None: ...
    def bankroll_yen(self, *, race_date: str, model_key: str, starting_yen: int) -> int: ...
    def insert_decision(self, **values: Any) -> bool: ...
    def append_available_settlements(self, *, race_date: str, now: datetime) -> int: ...


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_datetime(value: Any, *, default_tz=JST) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=default_tz) if parsed.tzinfo is None else parsed


def _canonical_combination(value: Any) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) != 3 or len(set(digits)) != 3:
        raise ValueError(f"invalid trifecta combination: {value}")
    return "-".join(digits)


def _value_summary(values: Mapping[str, float], *, descending: bool) -> dict[str, Any]:
    numeric = {str(key): float(value) for key, value in values.items()}
    ordered = sorted(
        numeric.items(), key=lambda row: ((-row[1]) if descending else row[1], row[0])
    )
    array = list(numeric.values())
    summary: dict[str, Any] = {
        "count": len(array),
        "minimum": min(array) if array else None,
        "maximum": max(array) if array else None,
        "median": median(array) if array else None,
        "top5": [
            {"combination": combination, "value": value}
            for combination, value in ordered[:5]
        ],
    }
    if descending and array:
        summary["sum"] = sum(array)
        summary["entropy"] = -sum(
            value * math.log(value) for value in array if value > 0.0
        )
    return summary


def _no_bet(reason: str, *, diagnostics: Mapping[str, Any] | None = None) -> ShadowDecision:
    return ShadowDecision({}, {}, (), reason, dict(diagnostics or {}))


def validate_snapshot(
    race: RaceWindow,
    snapshot: T300Snapshot,
    *,
    max_checkpoint_age_seconds: float,
    max_source_update_staleness_seconds: float,
) -> SnapshotCheck:
    age = (
        race.target_t300_at
        - snapshot.captured_at.astimezone(race.target_t300_at.tzinfo or JST)
    ).total_seconds()
    point = normalize_odds_checkpoint(
        {
            "snapshot_id": snapshot.snapshot_id,
            "captured_at": snapshot.captured_at.isoformat(),
            "source_update_time": snapshot.source_update_time,
            "raw_json": snapshot.raw_json,
            "betting_deadline_at": race.betting_deadline_at.isoformat(),
            "odds": snapshot.odds,
        },
        target_offset_seconds=T300_OFFSET_SECONDS,
    )
    source_staleness = (
        float(point["source_update_staleness_seconds"])
        if point is not None and point.get("source_update_staleness_seconds") is not None
        else None
    )
    if len(snapshot.odds) != 120:
        reason = "incomplete_t300_snapshot"
    elif not plausible_trifecta_odds(snapshot.odds) or point is None or age < 0.0:
        reason = "inconsistent_t300_snapshot"
    elif age > max_checkpoint_age_seconds:
        reason = "stale_t300_checkpoint"
    elif source_staleness is None:
        reason = "missing_source_update_staleness"
    elif source_staleness > max_source_update_staleness_seconds:
        reason = "stale_t300_source_update"
    else:
        reason = None
    return SnapshotCheck(age, source_staleness, reason)


class PostgresShadowStore:
    def __init__(self, conn: Any) -> None:
        if getattr(conn, "dialect", None) != "postgresql":
            raise ValueError("intraday T300 shadow recorder requires PostgreSQL")
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.executescript(POSTGRESQL_SCHEMA)

    def due_races(self, *, race_date: str, now: datetime) -> Sequence[RaceWindow]:
        rows = self.conn.execute(
            """
            SELECT r.race_id, r.race_date, r.jcd, r.rno, r.deadline_at
            FROM races r
            WHERE r.race_date = ? AND r.deadline_at IS NOT NULL
              AND (SELECT COUNT(DISTINCT e.lane) FROM entries e
                   WHERE e.race_id = r.race_id) = 6
            ORDER BY r.deadline_at, r.jcd, r.rno, r.race_id
            """,
            (race_date,),
        ).fetchall()
        result = []
        for row in rows:
            race = RaceWindow(
                str(row["race_id"]), str(row["race_date"]), str(row["jcd"]),
                int(row["rno"]), _as_datetime(row["deadline_at"]),
            )
            if race.target_t300_at <= now.astimezone(race.start_at.tzinfo or JST):
                result.append(race)
        return result

    def decision_identity(self, *, race_id: str, model_key: str) -> ModelIdentity | None:
        row = self.conn.execute(
            """SELECT model_key, model_hash, strategy_name
               FROM intraday_t300_shadow_decisions
               WHERE race_id = ? AND model_key = ?""",
            (race_id, model_key),
        ).fetchone()
        return (
            ModelIdentity(str(row["model_key"]), str(row["model_hash"]), str(row["strategy_name"]))
            if row is not None else None
        )

    def latest_complete_snapshot(self, race: RaceWindow) -> T300Snapshot | None:
        captured_at = stored_jst_timestamp_sql(self.conn, "candidate.captured_at")
        rows = self.conn.execute(
            f"""
            SELECT os.snapshot_id, os.captured_at, os.source_update_time,
                   os.raw_json, ot.combination, ot.odds
            FROM odds_snapshots os
            JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
            WHERE os.snapshot_id = (
              SELECT candidate.snapshot_id
              FROM odds_snapshots candidate
              JOIN odds_trifecta value ON value.snapshot_id = candidate.snapshot_id
              WHERE candidate.race_id = ? AND candidate.bet_type = 'trifecta'
                AND candidate.parser_version = ? AND {captured_at} <= CAST(? AS timestamptz)
                AND value.odds IS NOT NULL AND value.odds > 0
              GROUP BY candidate.snapshot_id, candidate.captured_at
              HAVING COUNT(*) = 120 AND COUNT(DISTINCT value.combination) = 120
              ORDER BY {captured_at} DESC, candidate.snapshot_id DESC LIMIT 1
            )
            ORDER BY ot.combination
            """,
            (race.race_id, TRIFECTA_PARSER_VERSION, race.target_t300_at.isoformat()),
        ).fetchall()
        if len(rows) != 120:
            return None
        first = rows[0]
        return T300Snapshot(
            int(first["snapshot_id"]), _as_datetime(first["captured_at"]),
            str(first["source_update_time"]) if first["source_update_time"] not in (None, "") else None,
            first["raw_json"],
            {str(row["combination"]): float(row["odds"]) for row in rows},
        )

    def bankroll_yen(self, *, race_date: str, model_key: str, starting_yen: int) -> int:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(d.total_stake_yen), 0) AS stake_yen,
                   COALESCE(SUM(s.return_yen), 0) AS return_yen
            FROM intraday_t300_shadow_decisions d
            LEFT JOIN intraday_t300_shadow_settlements s ON s.decision_id = d.decision_id
            WHERE d.race_date = ? AND d.model_key = ?
            """,
            (race_date, model_key),
        ).fetchone()
        return max(0, int(starting_yen) - int(row["stake_yen"] or 0) + int(row["return_yen"] or 0))

    def insert_decision(
        self, *, race: RaceWindow, identity: ModelIdentity, decision_at: datetime,
        decision_completed_at: datetime,
        snapshot: T300Snapshot | None, snapshot_check: SnapshotCheck,
        bankroll_before_yen: int, decision: ShadowDecision,
    ) -> bool:
        probabilities = dict(sorted(decision.probabilities.items()))
        closing = dict(sorted(decision.closing_lower_odds.items()))
        selected = [dict(row) for row in decision.selected_candidates]
        payload = {
            "schema_version": SCHEMA_VERSION, "race_id": race.race_id,
            "model_key": identity.model_key, "model_hash": identity.model_hash,
            "target_t300_at": race.target_t300_at.isoformat(),
            "source_snapshot_id": snapshot.snapshot_id if snapshot else None,
            "checkpoint_age_before_target_seconds": snapshot_check.checkpoint_age_before_target_seconds,
            "source_update_staleness_seconds": snapshot_check.source_update_staleness_seconds,
            "bankroll_before_yen": bankroll_before_yen, "status": decision.status,
            "no_bet_reason": decision.no_bet_reason, "probabilities": probabilities,
            "closing_lower_odds": closing, "selected_candidates": selected,
            "diagnostics": decision.diagnostics,
            "total_stake_yen": decision.total_stake_yen,
        }
        cursor = self.conn.execute(
            """
            INSERT INTO intraday_t300_shadow_decisions(
              schema_version, race_date, race_id, model_key, model_hash, strategy_name,
              decision_at, decision_completed_at, target_t300_at, source_snapshot_id, source_captured_at,
              checkpoint_age_before_target_seconds, source_update_staleness_seconds,
              bankroll_before_yen, decision_status, no_bet_reason, probabilities,
              probability_summary, closing_lower_odds, closing_lower_summary,
              selected_candidates, diagnostics, total_stake_yen, decision_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb,
                      ?::jsonb, ?::jsonb, ?::jsonb, ?::jsonb, ?::jsonb, ?, ?)
            ON CONFLICT (race_id, model_key) DO NOTHING
            """,
            (
                SCHEMA_VERSION, race.race_date, race.race_id, identity.model_key,
                identity.model_hash, identity.strategy_name, decision_at.isoformat(),
                decision_completed_at.isoformat(), race.target_t300_at.isoformat(),
                snapshot.snapshot_id if snapshot else None,
                snapshot.captured_at.isoformat() if snapshot else None,
                snapshot_check.checkpoint_age_before_target_seconds,
                snapshot_check.source_update_staleness_seconds, bankroll_before_yen,
                decision.status, decision.no_bet_reason,
                json.dumps(probabilities, sort_keys=True),
                json.dumps(_value_summary(probabilities, descending=True), sort_keys=True),
                json.dumps(closing, sort_keys=True),
                json.dumps(_value_summary(closing, descending=False), sort_keys=True),
                json.dumps(selected, sort_keys=True),
                json.dumps(decision.diagnostics, sort_keys=True),
                decision.total_stake_yen,
                _payload_hash(payload),
            ),
        )
        if cursor.rowcount == 1:
            return True
        if self.decision_identity(race_id=race.race_id, model_key=identity.model_key) != identity:
            raise ValueError(f"model identity conflict for {race.race_id} {identity.model_key}")
        return False

    def append_available_settlements(self, *, race_date: str, now: datetime) -> int:
        rows = self.conn.execute(
            """
            SELECT d.decision_id, d.selected_candidates, d.total_stake_yen,
                   p.combination AS actual_combination, p.payout_yen,
                   COALESCE(rs.status, 'final') AS result_status
            FROM intraday_t300_shadow_decisions d
            JOIN payouts p ON p.race_id = d.race_id AND p.bet_type = '3連単'
                          AND p.payout_yen IS NOT NULL
            LEFT JOIN race_result_status rs ON rs.race_id = d.race_id
            LEFT JOIN intraday_t300_shadow_settlements s ON s.decision_id = d.decision_id
            WHERE d.race_date = ? AND s.decision_id IS NULL ORDER BY d.decision_id
            """,
            (race_date,),
        ).fetchall()
        inserted = 0
        for row in rows:
            candidates = row["selected_candidates"]
            if isinstance(candidates, str):
                candidates = json.loads(candidates)
            actual = _canonical_combination(row["actual_combination"])
            payout = int(row["payout_yen"])
            returned = sum(
                int(item["stake_yen"]) * payout // 100 for item in (candidates or [])
                if _canonical_combination(item["combination"]) == actual
            )
            stake = int(row["total_stake_yen"] or 0)
            cursor = self.conn.execute(
                """
                INSERT INTO intraday_t300_shadow_settlements(
                  decision_id, settled_at, result_status, actual_combination,
                  payout_yen_per_100, stake_yen, return_yen, profit_yen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (decision_id) DO NOTHING
                """,
                (int(row["decision_id"]), now.isoformat(), str(row["result_status"]),
                 actual, payout, stake, returned, returned - stake),
            )
            inserted += int(cursor.rowcount == 1)
        return inserted


class V12RoleModelAdapter:
    strategy_name = "v12_role_t300"

    def __init__(self, *, model_key: str, bundle_path: Path, base_model_path: Path) -> None:
        loaded = joblib.load(bundle_path)
        if not isinstance(loaded, Mapping):
            raise ValueError("V12 shadow bundle must be a mapping")
        deployment = loaded.get("deployment")
        self._bundle = dict(deployment if isinstance(deployment, Mapping) else loaded)
        self._base_artifact = joblib.load(base_model_path)
        digest = hashlib.sha256(
            (_file_sha256(bundle_path) + _file_sha256(base_model_path)).encode("ascii")
        ).hexdigest()
        self._identity = ModelIdentity(model_key, digest, self.strategy_name)
        self._state_by_date: dict[str, Any] = {}
        self._rows_by_date: dict[str, dict[str, list[Any]]] = {}

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _component(self, *names: str) -> Mapping[str, Any]:
        for name in names:
            value = self._bundle.get(name)
            if isinstance(value, Mapping):
                return value
        raise ValueError(f"V12 shadow bundle lacks component: {names[0]}")

    def prewarm(self, conn: Any, race_date: str) -> None:
        if (
            race_date in self._state_by_date
            and race_date in self._rows_by_date
        ):
            return
        state_key = (id(conn), race_date)
        state = _SHARED_HISTORICAL_STATE.get(state_key)
        if state is None:
            state = historical_state(conn, race_date=race_date)
            _SHARED_HISTORICAL_STATE[state_key] = state
        schema = self._base_artifact.get("feature_schema_version")
        rows_key = (id(conn), race_date, schema)
        rows = _SHARED_DATE_RACES.get(rows_key)
        if rows is None:
            rows = load_date_races(
                conn,
                race_date=race_date,
                feature_schema_version=schema,
            )
            _SHARED_DATE_RACES[rows_key] = rows
        self._state_by_date[race_date] = state
        self._rows_by_date[race_date] = rows

    def _base_probabilities(self, conn: Any, race: RaceWindow) -> dict[str, float]:
        if race.race_date not in self._state_by_date:
            self.prewarm(conn, race.race_date)
        rows = self._rows_by_date[race.race_date].get(race.race_id)
        if rows is None:
            refreshed = load_date_races(
                conn, race_date=race.race_date,
                feature_schema_version=self._base_artifact.get("feature_schema_version"),
            )
            self._rows_by_date[race.race_date].clear()
            self._rows_by_date[race.race_date].update(refreshed)
            rows = self._rows_by_date[race.race_date].get(race.race_id)
        if rows is None or len(rows) != 6:
            raise ValueError("six complete entry rows are required")
        feature_rows = build_race_features(
            rows, self._state_by_date[race.race_date],
            drop_feature_groups=self._base_artifact.get("drop_feature_groups") or (),
            feature_schema_version=self._base_artifact.get("feature_schema_version"),
        )
        return artifact_model_probabilities(self._base_artifact, feature_rows)

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        if bankroll_yen < STAKE_GRANULARITY_YEN:
            return _no_bet("simulated_bankroll_below_minimum_stake")
        probability_model = self._component("probability_model", "operational_model")
        probability_lcb = self._component("probability_lcb")
        closing_model = self._component(
            "closing_t300_v12_model", "closing_v12_model", "closing_model"
        )
        conformal = self._component("selection_conformal")
        for name, artifact in (("probability_lcb", probability_lcb),
                               ("closing_v12", closing_model),
                               ("selection_conformal", conformal)):
            trained = artifact.get("trained_through_date")
            if trained is not None and str(trained) >= race.race_date:
                raise ValueError(f"{name} is not strictly prior to race date")
        if not probability_lcb.get("ready"):
            return _no_bet("probability_lcb_not_ready")
        if not closing_model.get("ready"):
            return _no_bet("closing_v12_not_ready")
        if str(closing_model.get("model_name")) != V12_CLOSING_MODEL_NAME:
            raise ValueError("closing component is not V12 T300")
        if not conformal.get("ready"):
            return _no_bet("selection_conformal_not_ready")

        point = normalize_odds_checkpoint(
            {"snapshot_id": snapshot.snapshot_id, "captured_at": snapshot.captured_at.isoformat(),
             "source_update_time": snapshot.source_update_time, "raw_json": snapshot.raw_json,
             "betting_deadline_at": race.betting_deadline_at.isoformat(), "odds": snapshot.odds},
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return _no_bet("inconsistent_t300_snapshot")
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return _no_bet("inconsistent_probability_combination_set")
        model_race = {
            "race_id": race.race_id, "race_date": race.race_date,
            "jcd": race.jcd, "rno": race.rno,
            "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base, "market_probabilities": market,
            "odds": dict(snapshot.odds), "odds_checkpoints": {"300": point},
            "odds_path": [{"minutes_before_decision": 0.0,
                           "snapshot_id": snapshot.snapshot_id,
                           "captured_at": snapshot.captured_at.isoformat(),
                           "market_probabilities": market}],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_probability_v8([model_race], dict(probability_model))[0]
        probabilities = {str(key): float(value) for key, value in transformed["model_probabilities"].items()}
        if (len(probabilities) != 120 or set(probabilities) != set(snapshot.odds)
                or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)):
            return _no_bet("invalid_v12_probability_output")
        forecast = forecast_closing_odds_t300_nonlinear_v12(
            transformed, closing_model, prediction_date=race.race_date
        )
        if forecast.get("future_checkpoint_offsets_used"):
            raise ValueError("V12 forecast used a post-T300 checkpoint")
        if not forecast.get("ready"):
            return _no_bet(str(forecast.get("reason") or "closing_forecast_not_ready"))
        closing = {str(key): float(value) for key, value in (forecast.get("lower_final_odds") or {}).items()}
        if len(closing) != 120 or set(closing) != set(probabilities):
            return _no_bet("invalid_v12_closing_lower_output")

        raw_selected = selected_safe_ev_candidates(
            [transformed], closing_forecasts={race.race_id: closing},
            probability_lcb=dict(probability_lcb),
        )
        haircut = float(conformal["haircut"])
        candidates = []
        for item in raw_selected:
            guarded_odds = float(item["predicted_closing"]) * haircut
            probability = float(item["probability"])
            safe_ev = probability * guarded_odds
            if safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidate = _policy_candidate(
                transformed, combination=str(item["combination"]), probability=probability,
                estimated_odds=guarded_odds, safe_ev=safe_ev,
            )
            candidate.update({"predicted_closing": float(item["predicted_closing"]),
                              "selection_conformal_haircut": haircut,
                              "raw_safe_ev": float(item["raw_safe_ev"]),
                              "probability_lcb_detail": item["probability_lcb_detail"],
                              "odds_source": "v12_t300_lower_times_selection_conformal"})
            candidates.append(candidate)
        allocated = allocate_discrete_log_day(
            race.race_date, candidates, {race.race_id}, daily_budget_yen=bankroll_yen,
            max_daily_exposure_fraction=MAX_DAILY_EXPOSURE_FRACTION,
            race_cap_fraction=RACE_CAP_FRACTION, ticket_cap_fraction=TICKET_CAP_FRACTION,
            max_daily_tickets=None, stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN, max_tickets_per_race=MAX_TICKETS_PER_RACE,
        )
        selected = tuple(
            {key: value for key, value in row.items() if key not in {"hit", "return_yen"}}
            for row in allocated["selected_sample"]
        )
        reason = None if selected else _zero_reason(
            conformal_ready=True, total_races=1, raw_candidates=len(raw_selected),
            guarded_candidates=len(candidates),
            allocation_candidates=int(allocated["allocation_candidate_tickets"]),
        )
        return ShadowDecision(probabilities, closing, selected, reason)


class V14RegisteredBandModelAdapter(V12RoleModelAdapter):
    """Apply the V14 strict-prior LCB while retaining the deployable V12 closer."""

    strategy_name = "v14_registered_band_t300"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        if self._bundle.get("calibrator_strategy") != (
            "odds_path_role_integrated_registered_band_lcb_v14"
        ):
            raise ValueError("V14 deployment has an unexpected calibrator strategy")
        probability_lcb = self._bundle.get("probability_lcb")
        if not isinstance(probability_lcb, Mapping):
            raise ValueError("V14 deployment lacks probability_lcb")
        if probability_lcb.get("method") != V14_PROBABILITY_LCB_METHOD:
            raise ValueError("V14 deployment has an unexpected probability LCB")
        registered_band = probability_lcb.get("registered_divergence_band")
        if not isinstance(registered_band, Mapping) or (
            float(registered_band.get("lower_inclusive", math.nan))
            != REGISTERED_DIVERGENCE_LOWER
            or float(registered_band.get("upper_exclusive", math.nan))
            != REGISTERED_DIVERGENCE_UPPER
        ):
            raise ValueError("V14 deployment must use registered band [0.5,1.0)")
        if probability_lcb.get("uses_payout") is not False:
            raise ValueError("V14 probability LCB must not use payout")
        closing = self._component(
            "closing_t300_v12_model", "closing_v12_model", "closing_model"
        )
        if not closing.get("ready") or closing.get("model_name") != V12_CLOSING_MODEL_NAME:
            raise ValueError("V14 merged bundle lacks a live V12 closing estimator")
        estimator = (closing.get("point_model") or {}).get("estimator")
        if estimator is None:
            raise ValueError("V14 merged bundle lacks a fitted V12 closing estimator")

    def _registered_band_diagnostic(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot
    ) -> dict[str, Any]:
        probability_model = self._component("probability_model", "operational_model")
        probability_lcb = self._component("probability_lcb")
        point = normalize_odds_checkpoint(
            {"snapshot_id": snapshot.snapshot_id,
             "captured_at": snapshot.captured_at.isoformat(),
             "source_update_time": snapshot.source_update_time,
             "raw_json": snapshot.raw_json,
             "betting_deadline_at": race.betting_deadline_at.isoformat(),
             "odds": snapshot.odds},
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return {"status": "inconsistent_t300_snapshot"}
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return {"status": "inconsistent_probability_combination_set"}
        model_race = {
            "race_id": race.race_id, "race_date": race.race_date,
            "jcd": race.jcd, "rno": race.rno, "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base, "market_probabilities": market,
            "odds": dict(snapshot.odds), "odds_checkpoints": {"300": point},
            "odds_path": [{"minutes_before_decision": 0.0,
                           "snapshot_id": snapshot.snapshot_id,
                           "captured_at": snapshot.captured_at.isoformat(),
                           "market_probabilities": market}],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_probability_v8(
            [model_race], dict(probability_model)
        )[0]
        consistency = t300_snapshot_consistency(transformed)
        if not consistency["consistent"]:
            return {"status": "inconsistent_t300_snapshot",
                    "reason": consistency["reason"]}
        details = {
            combination: probability_lower_bound_details_v14(
                transformed, combination, probability_lcb
            )
            for combination in sorted(snapshot.odds)
        }
        registered = {
            combination: detail for combination, detail in details.items()
            if detail.get("in_registered_divergence_band")
        }
        return {
            "status": "recorded",
            "checkpoint": "t300",
            "source_snapshot_id": snapshot.snapshot_id,
            "registered_divergence_band": "[0.5,1.0)",
            "registered_combination_count": len(registered),
            "adjusted_probability_sum": sum(
                float(detail.get("probability") or 0.0)
                for detail in registered.values()
            ),
            "registered_combinations": registered,
            "uses_result": False,
            "uses_payout": False,
        }

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        if bankroll_yen < STAKE_GRANULARITY_YEN:
            return _no_bet("simulated_bankroll_below_minimum_stake")
        probability_model = self._component("probability_model", "operational_model")
        probability_lcb = self._component("probability_lcb")
        closing_model = self._component(
            "closing_t300_v12_model", "closing_v12_model", "closing_model"
        )
        conformal = self._component("selection_conformal")
        for name, artifact in (
            ("probability_lcb", probability_lcb),
            ("closing_v12", closing_model),
            ("selection_conformal", conformal),
        ):
            trained = artifact.get("trained_through_date")
            if trained is not None and str(trained) >= race.race_date:
                raise ValueError(f"{name} is not strictly prior to race date")
        if not probability_lcb.get("ready"):
            return _no_bet("probability_lcb_not_ready")
        if not closing_model.get("ready"):
            return _no_bet("closing_v12_not_ready")
        if not conformal.get("ready"):
            diagnostic = self._registered_band_diagnostic(conn, race, snapshot)
            return _no_bet(
                "selection_conformal_not_ready",
                diagnostics={"v14_registered_band": diagnostic},
            )
        point = normalize_odds_checkpoint(
            {"snapshot_id": snapshot.snapshot_id,
             "captured_at": snapshot.captured_at.isoformat(),
             "source_update_time": snapshot.source_update_time,
             "raw_json": snapshot.raw_json,
             "betting_deadline_at": race.betting_deadline_at.isoformat(),
             "odds": snapshot.odds},
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return _no_bet("inconsistent_t300_snapshot")
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return _no_bet("inconsistent_probability_combination_set")
        model_race = {
            "race_id": race.race_id, "race_date": race.race_date,
            "jcd": race.jcd, "rno": race.rno, "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base, "market_probabilities": market,
            "odds": dict(snapshot.odds), "odds_checkpoints": {"300": point},
            "odds_path": [{"minutes_before_decision": 0.0,
                           "snapshot_id": snapshot.snapshot_id,
                           "captured_at": snapshot.captured_at.isoformat(),
                           "market_probabilities": market}],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_probability_v8(
            [model_race], dict(probability_model)
        )[0]
        probabilities = {
            str(key): float(value)
            for key, value in transformed["model_probabilities"].items()
        }
        if (
            len(probabilities) != 120
            or set(probabilities) != set(snapshot.odds)
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)
        ):
            return _no_bet("invalid_v14_probability_output")
        forecast = forecast_closing_odds_t300_nonlinear_v12(
            transformed, closing_model, prediction_date=race.race_date
        )
        if forecast.get("future_checkpoint_offsets_used"):
            raise ValueError("V12 forecast used a post-T300 checkpoint")
        if not forecast.get("ready"):
            return _no_bet(str(forecast.get("reason") or "closing_forecast_not_ready"))
        closing = {
            str(key): float(value)
            for key, value in (forecast.get("lower_final_odds") or {}).items()
        }
        if len(closing) != 120 or set(closing) != set(probabilities):
            return _no_bet("invalid_v12_closing_lower_output")

        raw_candidates = []
        all_details = {}
        for combination in sorted(closing):
            detail = probability_lower_bound_details_v14(
                transformed, combination, probability_lcb
            )
            all_details[combination] = detail
            probability = float(detail.get("probability") or 0.0)
            raw_safe_ev = probability * closing[combination]
            if raw_safe_ev < SAFE_EV_THRESHOLD:
                continue
            raw_candidates.append({
                "combination": combination,
                "probability": probability,
                "predicted_closing": closing[combination],
                "raw_safe_ev": raw_safe_ev,
                "probability_lcb_detail": detail,
            })
        raw_candidates.sort(key=lambda row: (
            -float(row["raw_safe_ev"]),
            -float(row["probability"]),
            str(row["combination"]),
        ))
        raw_selected = raw_candidates[:MAX_TICKETS_PER_RACE]

        haircut = float(conformal["haircut"])
        candidates = []
        for item in raw_selected:
            guarded_odds = float(item["predicted_closing"]) * haircut
            probability = float(item["probability"])
            guarded_safe_ev = probability * guarded_odds
            if guarded_safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidate = {
                "race_id": race.race_id,
                "race_date": race.race_date,
                "jcd": race.jcd,
                "rno": race.rno,
                "combination": str(item["combination"]),
                "probability": probability,
                "estimated_odds": guarded_odds,
                "estimated_ev": guarded_safe_ev,
                "safe_ev": guarded_safe_ev,
                "real_odds_snapshot_id": snapshot.snapshot_id,
                "real_odds_captured_at": snapshot.captured_at.isoformat(),
                "real_odds_combinations": len(snapshot.odds),
                "predicted_closing": float(item["predicted_closing"]),
                "selection_conformal_haircut": haircut,
                "raw_safe_ev": float(item["raw_safe_ev"]),
                "probability_lcb_detail": item["probability_lcb_detail"],
                "odds_source": "v12_t300_lower_v14_lcb_times_selection_conformal",
            }
            candidates.append(candidate)
        allocated = allocate_discrete_log_day(
            race.race_date,
            candidates,
            {race.race_id},
            daily_budget_yen=bankroll_yen,
            max_daily_exposure_fraction=MAX_DAILY_EXPOSURE_FRACTION,
            race_cap_fraction=RACE_CAP_FRACTION,
            ticket_cap_fraction=TICKET_CAP_FRACTION,
            max_daily_tickets=None,
            stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN,
            max_tickets_per_race=MAX_TICKETS_PER_RACE,
        )
        selected = tuple(
            {key: value for key, value in row.items() if key not in {"hit", "return_yen"}}
            for row in allocated["selected_sample"]
        )
        reason = None if selected else _zero_reason(
            conformal_ready=True,
            total_races=1,
            raw_candidates=len(raw_selected),
            guarded_candidates=len(candidates),
            allocation_candidates=int(allocated["allocation_candidate_tickets"]),
        )
        diagnostics = {
            "v14_registered_band": {
                "status": "recorded",
                "checkpoint": "t300",
                "source_snapshot_id": snapshot.snapshot_id,
                "registered_divergence_band": "[0.5,1.0)",
                "registered_combination_count": sum(
                    bool(detail.get("in_registered_divergence_band"))
                    for detail in all_details.values()
                ),
                "raw_safe_ev_candidates": len(raw_candidates),
                "top2_candidates": len(raw_selected),
                "guarded_candidates": len(candidates),
                "uses_result": False,
                "uses_payout": False,
            }
        }
        return ShadowDecision(
            probabilities, closing, selected, reason, diagnostics
        )


class V16FixedBandModelAdapter(V12RoleModelAdapter):
    """Use V8 raw probability in the fixed T300 band with the V15 envelope."""

    strategy_name = "v16_fixed_band_t300"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        if self._bundle.get("calibrator_strategy") != (
            "odds_path_role_integrated_fixed_band_passthrough_v16"
        ):
            raise ValueError("V16 deployment has an unexpected calibrator strategy")
        probability = self._bundle.get("probability_lcb")
        if not isinstance(probability, Mapping):
            raise ValueError("V16 deployment lacks probability artifact")
        if (
            probability.get("model_name") != V16_PASSTHROUGH_MODEL
            or probability.get("artifact_method") != V16_PASSTHROUGH_METHOD
            or probability.get("fixed_filter") is not True
            or probability.get("raw_probability_passthrough") is not True
            or probability.get("uses_result") is not False
            or probability.get("uses_payout") is not False
            or float(probability.get("registered_divergence_lower_inclusive", math.nan))
            != REGISTERED_DIVERGENCE_LOWER
            or float(probability.get("registered_divergence_upper_exclusive", math.nan))
            != REGISTERED_DIVERGENCE_UPPER
        ):
            raise ValueError("V16 probability artifact is unsafe or inconsistent")
        policy = self._bundle.get("candidate_policy")
        if not isinstance(policy, Mapping) or (
            float(policy.get("registered_divergence_lower_inclusive", math.nan))
            != REGISTERED_DIVERGENCE_LOWER
            or float(policy.get("registered_divergence_upper_exclusive", math.nan))
            != REGISTERED_DIVERGENCE_UPPER
            or policy.get("raw_model_probability_inside_fixed_band") is not True
            or policy.get("real_betting_enabled") is not False
        ):
            raise ValueError("V16 candidate policy is unsafe or inconsistent")
        closing = self._component(
            "closing_t300_v12_model", "closing_v12_model", "closing_model"
        )
        if not closing.get("ready") or closing.get("model_name") != V12_CLOSING_MODEL_NAME:
            raise ValueError("V16 merged bundle lacks a live V12 closing estimator")
        if (closing.get("point_model") or {}).get("estimator") is None:
            raise ValueError("V16 merged bundle lacks a fitted V12 closing estimator")
        if self._bundle.get("real_betting_enabled") is not False:
            raise ValueError("V16 deployment must disable real betting")

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        if bankroll_yen < STAKE_GRANULARITY_YEN:
            return _no_bet("simulated_bankroll_below_minimum_stake")
        probability_model = self._bundle.get("operational_model")
        probability_artifact = self._bundle.get("probability_lcb")
        closing_model = self._bundle.get("closing_t300_v12_model")
        envelope = self._bundle.get("closing_envelope_conformal")
        if not all(
            isinstance(value, Mapping)
            for value in (probability_model, probability_artifact, closing_model)
        ):
            return _no_bet("missing_v16_runtime_component")
        if not isinstance(envelope, Mapping) or not envelope.get("ready"):
            return _no_bet("closing_envelope_not_ready")
        if (
            envelope.get("method") != V15_CLOSING_ENVELOPE_METHOD
            or envelope.get("selection_free") is not True
        ):
            return _no_bet("inconsistent_closing_envelope")
        for name, artifact in (
            ("probability_artifact", probability_artifact),
            ("closing_v12", closing_model),
            ("closing_envelope", envelope),
        ):
            trained = artifact.get("trained_through_date")
            if trained is not None and str(trained) >= race.race_date:
                raise ValueError(f"{name} is not strictly prior to race date")

        point = normalize_odds_checkpoint(
            {
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "source_update_time": snapshot.source_update_time,
                "raw_json": snapshot.raw_json,
                "betting_deadline_at": race.betting_deadline_at.isoformat(),
                "odds": snapshot.odds,
            },
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return _no_bet("inconsistent_t300_snapshot")
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return _no_bet("inconsistent_probability_combination_set")
        model_race = {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "jcd": race.jcd,
            "rno": race.rno,
            "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base,
            "market_probabilities": market,
            "odds": dict(snapshot.odds),
            "odds_checkpoints": {"300": point},
            "odds_path": [{
                "minutes_before_decision": 0.0,
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "market_probabilities": market,
            }],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_probability_v8(
            [model_race], dict(probability_model)
        )[0]
        consistency = t300_snapshot_consistency(transformed)
        if not consistency["consistent"]:
            return _no_bet("inconsistent_t300_snapshot")
        probabilities = {
            str(key): float(value)
            for key, value in transformed["model_probabilities"].items()
        }
        if (
            len(probabilities) != 120
            or set(probabilities) != set(snapshot.odds)
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)
        ):
            return _no_bet("invalid_v16_probability_output")

        forecast = forecast_closing_odds_t300_nonlinear_v12(
            transformed, dict(closing_model), prediction_date=race.race_date
        )
        if forecast.get("future_checkpoint_offsets_used"):
            raise ValueError("V16 forecast used a post-T300 checkpoint")
        if not forecast.get("ready"):
            return _no_bet(str(forecast.get("reason") or "closing_forecast_not_ready"))
        point_closing = {
            str(key): float(value)
            for key, value in (forecast.get("point_final_odds") or {}).items()
        }
        if len(point_closing) != 120 or set(point_closing) != set(probabilities):
            return _no_bet("invalid_v12_closing_point_output")
        try:
            closing = apply_closing_envelope_haircut_v15(
                point_closing, envelope
            )
        except (TypeError, ValueError, OverflowError):
            return _no_bet("inconsistent_closing_envelope")
        if not isinstance(closing, dict) or len(closing) != 120:
            return _no_bet("invalid_v16_closing_envelope_output")

        raw_candidates = []
        registered = 0
        for combination in sorted(closing):
            probability = probabilities[combination]
            market_probability = market[combination]
            if probability <= 0.0 or market_probability <= 0.0:
                continue
            divergence = math.log(probability / market_probability)
            if not (
                REGISTERED_DIVERGENCE_LOWER
                <= divergence
                < REGISTERED_DIVERGENCE_UPPER
            ):
                continue
            registered += 1
            safe_ev = probability * closing[combination]
            if safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidate = {
                "race_id": race.race_id,
                "race_date": race.race_date,
                "jcd": race.jcd,
                "rno": race.rno,
                "combination": combination,
                "probability": probability,
                "estimated_odds": closing[combination],
                "estimated_ev": safe_ev,
                "safe_ev": safe_ev,
                "real_odds_snapshot_id": snapshot.snapshot_id,
                "real_odds_captured_at": snapshot.captured_at.isoformat(),
                "real_odds_combinations": len(snapshot.odds),
                "predicted_closing": point_closing[combination],
                "closing_envelope_haircut": float(envelope["haircut"]),
                "t300_log_divergence": divergence,
                "probability_source": "v8_raw_probability",
                "odds_source": "v12_t300_point_times_v15_selection_free_envelope",
            }
            raw_candidates.append(candidate)
        raw_candidates.sort(key=lambda row: (
            -float(row["safe_ev"]),
            -float(row["probability"]),
            str(row["combination"]),
        ))
        candidates = raw_candidates[:MAX_TICKETS_PER_RACE]
        allocated = allocate_discrete_log_day(
            race.race_date,
            candidates,
            {race.race_id},
            daily_budget_yen=bankroll_yen,
            max_daily_exposure_fraction=MAX_DAILY_EXPOSURE_FRACTION,
            race_cap_fraction=RACE_CAP_FRACTION,
            ticket_cap_fraction=TICKET_CAP_FRACTION,
            max_daily_tickets=None,
            stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN,
            max_tickets_per_race=MAX_TICKETS_PER_RACE,
        )
        selected = tuple(
            {key: value for key, value in row.items() if key not in {"hit", "return_yen"}}
            for row in allocated["selected_sample"]
        )
        reason = None if selected else _zero_reason(
            conformal_ready=True,
            total_races=1,
            raw_candidates=len(raw_candidates),
            guarded_candidates=len(candidates),
            allocation_candidates=int(allocated["allocation_candidate_tickets"]),
        )
        diagnostics = {
            "v16_fixed_band": {
                "status": "recorded",
                "checkpoint": "t300",
                "source_snapshot_id": snapshot.snapshot_id,
                "registered_divergence_band": "[0.5,1.0)",
                "registered_combination_count": registered,
                "raw_safe_ev_candidates": len(raw_candidates),
                "guarded_candidates": len(candidates),
                "uses_result": False,
                "uses_payout": False,
                "real_betting_enabled": False,
            }
        }
        return ShadowDecision(probabilities, closing, selected, reason, diagnostics)


class V18ScheduleQuotaModelAdapter(V12RoleModelAdapter):
    """Run the fixed job-8191 V18 policy prospectively at T300."""

    strategy_name = "v18_schedule_quota_t300"
    expected_calibrator_strategy = "odds_path_observed_closing_return_schedule_quota_v18"
    allowed_deployment_modes = ("shadow_only",)
    artifact_label = "V18"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        if self._bundle.get("calibrator_strategy") != self.expected_calibrator_strategy:
            raise ValueError(
                f"{self.artifact_label} deployment has an unexpected calibrator strategy"
            )
        if (
            self._bundle.get("deployment_mode") not in self.allowed_deployment_modes
            or self._bundle.get("real_betting_enabled") is not False
            or float(self._bundle.get("daily_stake_limit_fraction", 0.0)) != 1.0
        ):
            raise ValueError(
                f"{self.artifact_label} deployment must remain 10000-yen shadow-only"
            )
        self._calibrator = self._component("calibrator")
        self._operational_model = self._component("operational_model")
        self._policy = self._component("candidate_policy")
        self._formal_selection = self._component("selected_policy")
        control = self._policy.get("v18_ticket_control")
        if (
            self._calibrator.get("converged") is not True
            or self._operational_model.get("model_type")
            != "odds_path_observed_closing_return_v4"
            or not isinstance(control, Mapping)
            or control.get("method")
            != "strict_prior_daily_ticket_lower_quantile"
            or int(control.get("learned_daily_ticket_limit") or 0) <= 0
            or int(control.get("stake_granularity_yen") or 0) != 100
            or control.get("result_or_payout_fields_used") is not False
            or self._formal_selection.get("no_bet") is not True
        ):
            raise ValueError("V18 fixed policy artifacts are unsafe or inconsistent")
        self._ticket_limit = int(control["learned_daily_ticket_limit"])

    def _calibrated_head_output(
        self,
        conn: Any,
        race: RaceWindow,
        snapshot: T300Snapshot,
        calibrator: Mapping[str, Any],
    ) -> dict[str, float]:
        point = normalize_odds_checkpoint(
            {
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "source_update_time": snapshot.source_update_time,
                "raw_json": snapshot.raw_json,
                "betting_deadline_at": race.betting_deadline_at.isoformat(),
                "odds": snapshot.odds,
            },
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return {}
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return {}
        model_race = {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "jcd": race.jcd,
            "rno": race.rno,
            "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base,
            "market_probabilities": market,
            "odds": dict(snapshot.odds),
            "odds_checkpoints": {"300": point},
            "odds_path": [{
                "minutes_before_decision": 0.0,
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "market_probabilities": market,
            }],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_model(
            [model_race], dict(self._operational_model)
        )[0]
        probabilities = blend_probabilities(
            transformed["model_probabilities"],
            market,
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        if (
            len(probabilities) != 120
            or set(probabilities) != set(snapshot.odds)
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)
        ):
            return {}
        return probabilities

    def _runtime_limits(
        self, conn: Any, race: RaceWindow, *, bankroll_yen: int
    ) -> dict[str, int]:
        schedule = conn.execute(
            """
            SELECT r.race_id, r.deadline_at
            FROM races r
            WHERE r.race_date = ? AND r.deadline_at IS NOT NULL
              AND (SELECT COUNT(DISTINCT e.lane) FROM entries e
                   WHERE e.race_id = r.race_id) = 6
            ORDER BY r.deadline_at, r.jcd, r.rno, r.race_id
            """,
            (race.race_date,),
        ).fetchall()
        schedule_ids = [str(row["race_id"]) for row in schedule]
        if race.race_id not in schedule_ids:
            raise ValueError("V18 race is missing from known daily schedule")
        elapsed = schedule_ids.index(race.race_id) + 1
        quota = self._ticket_limit * elapsed // len(schedule_ids)
        rows = conn.execute(
            """
            SELECT d.selected_candidates, d.total_stake_yen, s.profit_yen
            FROM intraday_t300_shadow_decisions d
            LEFT JOIN intraday_t300_shadow_settlements s
              ON s.decision_id = d.decision_id
            WHERE d.race_date = ? AND d.model_key = ?
            """,
            (race.race_date, self.identity.model_key),
        ).fetchall()
        used_tickets = gross_stake_yen = realized_profit_yen = 0
        for row in rows:
            selected = row["selected_candidates"] or []
            if isinstance(selected, str):
                selected = json.loads(selected)
            used_tickets += len(selected)
            gross_stake_yen += int(row["total_stake_yen"] or 0)
            if row["profit_yen"] is not None:
                realized_profit_yen += int(row["profit_yen"])
        gross_allowance_yen = STARTING_BANKROLL_YEN + max(0, realized_profit_yen)
        remaining_gross_yen = max(0, gross_allowance_yen - gross_stake_yen)
        return {
            "schedule_races_elapsed": elapsed,
            "schedule_races_total": len(schedule_ids),
            "cumulative_ticket_quota": quota,
            "used_tickets": used_tickets,
            "remaining_ticket_quota": max(0, quota - used_tickets),
            "gross_stake_yen": gross_stake_yen,
            "realized_cumulative_profit_yen": realized_profit_yen,
            "gross_stake_allowance_yen": gross_allowance_yen,
            "remaining_gross_stake_allowance_yen": remaining_gross_yen,
            "allocatable_bankroll_yen": min(bankroll_yen, remaining_gross_yen),
        }

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        trained = str(self._bundle.get("trained_through_date") or "")
        if trained and trained >= race.race_date:
            raise ValueError(
                f"{self.artifact_label} artifacts are not strictly prior to race date"
            )
        limits = self._runtime_limits(conn, race, bankroll_yen=bankroll_yen)
        if limits["remaining_ticket_quota"] <= 0:
            return _no_bet(
                "v18_schedule_ticket_quota_not_released",
                diagnostics={"v18_schedule_quota": limits},
            )
        allocatable = limits["allocatable_bankroll_yen"]
        if allocatable < STAKE_GRANULARITY_YEN:
            return _no_bet(
                "v18_daily_gross_stake_allowance_exhausted",
                diagnostics={"v18_schedule_quota": limits},
            )
        point = normalize_odds_checkpoint(
            {
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "source_update_time": snapshot.source_update_time,
                "raw_json": snapshot.raw_json,
                "betting_deadline_at": race.betting_deadline_at.isoformat(),
                "odds": snapshot.odds,
            },
            target_offset_seconds=T300_OFFSET_SECONDS,
        )
        if point is None:
            return _no_bet("inconsistent_t300_snapshot")
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return _no_bet("inconsistent_probability_combination_set")
        model_race = {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "jcd": race.jcd,
            "rno": race.rno,
            "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base,
            "market_probabilities": market,
            "odds": dict(snapshot.odds),
            "odds_checkpoints": {"300": point},
            "odds_path": [{
                "minutes_before_decision": 0.0,
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at.isoformat(),
                "market_probabilities": market,
            }],
            "odds_path_points": 1,
        }
        transformed = attach_odds_path_model(
            [model_race], dict(self._operational_model)
        )[0]
        probabilities = blend_probabilities(
            transformed["model_probabilities"],
            market,
            model_weight=float(self._calibrator["model_weight"]),
            temperature=float(self._calibrator["temperature"]),
        )
        if (
            len(probabilities) != 120
            or set(probabilities) != set(snapshot.odds)
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)
        ):
            return _no_bet("invalid_v18_probability_output")
        raw_candidates = []
        multipliers = transformed.get("historical_return_multipliers") or {}
        for combination in sorted(probabilities):
            probability = float(probabilities[combination])
            market_probability = float(market[combination])
            odds = float(snapshot.odds[combination])
            multiplier = float(multipliers.get(combination, 1.0))
            estimated_ev = probability * odds * multiplier
            ratio = probability / max(1e-12, market_probability)
            if estimated_ev < float(self._policy["ev_threshold"]):
                continue
            if self._policy.get("max_estimated_ev") is not None and estimated_ev > float(
                self._policy["max_estimated_ev"]
            ):
                continue
            if self._policy.get("max_odds") is not None and odds > float(self._policy["max_odds"]):
                continue
            if ratio < float(self._policy["min_model_market_ratio"]):
                continue
            raw_candidates.append({
                "race_id": race.race_id,
                "race_date": race.race_date,
                "jcd": race.jcd,
                "rno": race.rno,
                "combination": combination,
                "probability": probability,
                "market_probability": market_probability,
                "model_probability": float(transformed["model_probabilities"][combination]),
                "estimated_odds": odds,
                "estimated_ev": estimated_ev,
                "historical_return_multiplier": multiplier,
                "real_odds_snapshot_id": snapshot.snapshot_id,
                "real_odds_captured_at": snapshot.captured_at.isoformat(),
                "real_odds_combinations": len(snapshot.odds),
                "odds_source": "real_t300_job8191_v18",
            })
        raw_candidates.sort(
            key=lambda row: (-float(row["estimated_ev"]), -float(row["probability"]), str(row["combination"]))
        )
        candidates = raw_candidates[: int(self._policy["max_tickets_per_race"])]
        allocated = allocate_adaptive_day(
            race.race_date,
            candidates,
            {race.race_id},
            daily_budget_yen=allocatable,
            fractional_kelly=1.0,
            max_daily_exposure_fraction=0.30,
            min_daily_exposure_fraction=0.0,
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_daily_tickets=limits["remaining_ticket_quota"],
            allocation_mode="kelly_floor",
            stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN,
        )
        selected = tuple(
            {key: value for key, value in row.items() if key not in {"hit", "return_yen"}}
            for row in allocated["selected_sample"]
        )
        diagnostics = {
            "v18_schedule_quota": {
                **limits,
                "checkpoint": "t300",
                "source_snapshot_id": snapshot.snapshot_id,
                "learned_daily_ticket_limit": self._ticket_limit,
                "candidate_policy": str(self._policy.get("name") or ""),
                "formal_selected_policy": str(self._formal_selection.get("name") or "no_bet"),
                "raw_candidates": len(raw_candidates),
                "allocation_candidates": int(allocated["allocation_candidate_tickets"]),
                "decision_features": "t300_or_earlier",
                "settlement_fields_used_for_capital_only": True,
                "uses_result_as_model_feature": False,
                "uses_payout_as_model_feature": False,
                "real_betting_enabled": False,
            }
        }
        reason = None if selected else _zero_reason(
            conformal_ready=True,
            total_races=1,
            raw_candidates=len(raw_candidates),
            guarded_candidates=len(candidates),
            allocation_candidates=int(allocated["allocation_candidate_tickets"]),
        )
        return ShadowDecision(
            probabilities, dict(snapshot.odds), selected, reason, diagnostics
        )


class V20DualHeadModelAdapter(V18ScheduleQuotaModelAdapter):
    """Report V20 probabilities while retaining the V18 purchase pipeline."""

    strategy_name = "v20_dual_head_t300"
    expected_calibrator_strategy = (
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20"
    )
    allowed_deployment_modes = ("evaluation_only",)
    artifact_label = "V20"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        self._probability_calibrator = self._component("probability_calibrator")
        self._purchase_calibrator = self._component("purchase_calibrator")
        dual = self._component("dual_head_calibration")
        probability_head = dual.get("probability_head")
        purchase_head = dual.get("purchase_head")
        if (
            self._bundle.get("source_evaluation_job_id") != 8458
            or self._bundle.get("probability_metrics_head") != "probability_head"
            or self._bundle.get("chronological_bankroll_head") != "purchase_head"
            or self._bundle.get("outer_result_or_payout_used") is not False
            or dual.get("architecture") != "strict_prior_dual_calibrator_heads_v20"
            or dual.get("outer_holdout_used") is not False
            or not isinstance(probability_head, Mapping)
            or not isinstance(purchase_head, Mapping)
            or probability_head.get("role")
            != "probability_reporting_and_promotion_calibration"
            or purchase_head.get("role")
            != "purchase_policy_and_chronological_bankroll"
            or probability_head.get("calibrator") != self._probability_calibrator
            or purchase_head.get("calibrator") != self._purchase_calibrator
            or self._probability_calibrator.get("converged") is not True
            or self._purchase_calibrator.get("converged") is not True
        ):
            raise ValueError("V20 dual-head routing or provenance is inconsistent")
        # The inherited V18 decision path is the canonical purchase head.
        self._calibrator = self._purchase_calibrator

    def _probability_head_output(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot
    ) -> dict[str, float]:
        return self._calibrated_head_output(
            conn, race, snapshot, self._probability_calibrator
        )

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        purchase = super().decide(
            conn, race, snapshot, bankroll_yen=bankroll_yen
        )
        diagnostics = dict(purchase.diagnostics)
        dual_diagnostic: dict[str, Any] = {
            "status": "recorded" if purchase.probabilities else "purchase_head_no_output",
            "checkpoint": "t300",
            "source_snapshot_id": snapshot.snapshot_id,
            "source_evaluation_job_id": 8458,
            "probability_output_head": "probability_head",
            "candidate_selection_head": "purchase_head",
            "chronological_bankroll_head": "purchase_head",
            "probability_calibrator_sha256": _payload_hash(
                self._probability_calibrator
            ),
            "purchase_calibrator_sha256": _payload_hash(
                self._purchase_calibrator
            ),
            "purchase_probabilities_sha256": _payload_hash(
                purchase.probabilities
            ),
            "purchase_decisions_sha256": _payload_hash(
                purchase.selected_candidates
            ),
            "decision_features": "t300_or_earlier",
            "outer_result_used": False,
            "outer_payout_used": False,
            "settlement_fields_used_for_capital_only": True,
            "real_betting_enabled": False,
        }
        if not purchase.probabilities:
            diagnostics["v20_dual_head"] = dual_diagnostic
            return ShadowDecision(
                {},
                purchase.closing_lower_odds,
                purchase.selected_candidates,
                purchase.no_bet_reason,
                diagnostics,
            )
        probability_output = self._probability_head_output(conn, race, snapshot)
        if not probability_output:
            return _no_bet(
                "invalid_v20_probability_head_output",
                diagnostics={
                    **diagnostics,
                    "v20_dual_head": {
                        **dual_diagnostic,
                        "status": "invalid_probability_head_output",
                    },
                },
            )
        dual_diagnostic["probability_output_sha256"] = _payload_hash(
            probability_output
        )
        diagnostics["v20_dual_head"] = dual_diagnostic
        return ShadowDecision(
            probability_output,
            purchase.closing_lower_odds,
            purchase.selected_candidates,
            purchase.no_bet_reason,
            diagnostics,
        )


class V21TripleHeadModelAdapter(V18ScheduleQuotaModelAdapter):
    """Record V21 probability, ranking, and V18 purchase heads at T300."""

    strategy_name = "v21_triple_head_t300"
    expected_calibrator_strategy = (
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
    )
    allowed_deployment_modes = ("evaluation_only",)
    artifact_label = "V21"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        self._probability_calibrator = self._component("probability_calibrator")
        self._ranking_calibrator = self._component("ranking_calibrator")
        self._purchase_calibrator = self._component("purchase_calibrator")
        triple = self._component("triple_head_calibration")
        probability_head = triple.get("probability_head")
        ranking_head = triple.get("ranking_head")
        purchase_head = triple.get("purchase_head")
        if (
            self._bundle.get("source_evaluation_job_id") != 8666
            or self._bundle.get("winner_and_logloss_head") != "probability_head"
            or self._bundle.get("trifecta_top5_head") != "ranking_head"
            or self._bundle.get("market_logloss_comparison_head") != "probability_head"
            or self._bundle.get("market_top5_comparison_head") != "ranking_head"
            or self._bundle.get("chronological_bankroll_head") != "purchase_head"
            or self._bundle.get("outer_result_or_payout_used") is not False
            or triple.get("architecture")
            != "strict_prior_triple_calibrator_heads_v21"
            or triple.get("selection_data")
            != "strict_prior_training_and_inner_prequential_folds_only"
            or triple.get("outer_holdout_used") is not False
            or triple.get("ranking_purchase_share_v18_selection") is not True
            or not isinstance(probability_head, Mapping)
            or not isinstance(ranking_head, Mapping)
            or not isinstance(purchase_head, Mapping)
            or probability_head.get("role") != "winner_and_trifecta_logloss"
            or ranking_head.get("role") != "trifecta_top5_ranking"
            or purchase_head.get("role")
            != "purchase_policy_and_chronological_bankroll"
            or probability_head.get("calibrator") != self._probability_calibrator
            or ranking_head.get("calibrator") != self._ranking_calibrator
            or purchase_head.get("calibrator") != self._purchase_calibrator
            or self._ranking_calibrator != self._purchase_calibrator
            or any(
                calibrator.get("converged") is not True
                for calibrator in (
                    self._probability_calibrator,
                    self._ranking_calibrator,
                    self._purchase_calibrator,
                )
            )
        ):
            raise ValueError("V21 triple-head routing or provenance is inconsistent")
        self._calibrator = self._purchase_calibrator

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        trained = str(self._bundle.get("trained_through_date") or "")
        if trained and trained >= race.race_date:
            raise ValueError("V21 artifacts are not strictly prior to race date")
        probability_output = self._calibrated_head_output(
            conn, race, snapshot, self._probability_calibrator
        )
        ranking_output = self._calibrated_head_output(
            conn, race, snapshot, self._ranking_calibrator
        )
        if not probability_output or not ranking_output:
            return _no_bet("invalid_v21_probability_or_ranking_head_output")

        purchase = super().decide(
            conn, race, snapshot, bankroll_yen=bankroll_yen
        )
        ranking_top5 = [
            combination
            for combination, _ in sorted(
                ranking_output.items(), key=lambda item: (-item[1], item[0])
            )[:5]
        ]
        diagnostics = dict(purchase.diagnostics)
        diagnostics["v21_triple_head"] = {
            "status": "recorded",
            "checkpoint": "t300",
            "source_snapshot_id": snapshot.snapshot_id,
            "source_evaluation_job_id": 8666,
            "probability_output_head": "probability_head",
            "ranking_output_head": "ranking_head",
            "candidate_selection_head": "purchase_head",
            "chronological_bankroll_head": "purchase_head",
            "probability_calibrator_sha256": _payload_hash(
                self._probability_calibrator
            ),
            "ranking_calibrator_sha256": _payload_hash(self._ranking_calibrator),
            "purchase_calibrator_sha256": _payload_hash(self._purchase_calibrator),
            "probability_output_sha256": _payload_hash(probability_output),
            "ranking_output_sha256": _payload_hash(ranking_output),
            "purchase_probabilities_sha256": _payload_hash(
                purchase.probabilities
            ),
            "purchase_decisions_sha256": _payload_hash(
                purchase.selected_candidates
            ),
            "ranking_probabilities": ranking_output,
            "ranking_top5": ranking_top5,
            "decision_features": "t300_or_earlier",
            "outer_result_used": False,
            "outer_payout_used": False,
            "settlement_fields_used_for_capital_only": True,
            "real_betting_enabled": False,
        }
        return ShadowDecision(
            probability_output,
            purchase.closing_lower_odds,
            purchase.selected_candidates,
            purchase.no_bet_reason,
            diagnostics,
        )


class V23Top5NarrowModelAdapter(V21TripleHeadModelAdapter):
    """Evaluate the preregistered top-5/forecast-price policy in shadow mode."""

    strategy_name = "v23_top5_narrow_t300"
    artifact_label = "V23"

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        self._closing_selection = self._component("closing_odds_selection")
        if (
            self._closing_selection.get("selected") not in {"baseline", "momentum"}
            or not isinstance(self._closing_selection.get("baseline_model"), Mapping)
            or self._bundle.get("real_betting_enabled") is not False
        ):
            raise ValueError("V23 closing forecast or shadow-only provenance is invalid")

    @staticmethod
    def _blend_head(
        transformed: Mapping[str, Any],
        market: Mapping[str, float],
        calibrator: Mapping[str, Any],
    ) -> dict[str, float]:
        output = blend_probabilities(
            transformed["model_probabilities"],
            market,
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        if (
            len(output) != 120
            or set(output) != set(market)
            or not math.isclose(sum(output.values()), 1.0, abs_tol=1e-8)
        ):
            return {}
        return output

    def _v23_model_race(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot
    ) -> tuple[dict[str, Any], dict[str, float], str] | None:
        current_snapshot = {
            "snapshot_id": snapshot.snapshot_id,
            "captured_at": snapshot.captured_at.isoformat(),
            "source_update_time": snapshot.source_update_time,
            "raw_json": snapshot.raw_json,
            "betting_deadline_at": race.betting_deadline_at.isoformat(),
            "odds": dict(snapshot.odds),
        }
        point = normalize_odds_checkpoint(
            current_snapshot, target_offset_seconds=T300_OFFSET_SECONDS
        )
        if point is None:
            return None
        base = self._base_probabilities(conn, race)
        market = normalized_market_probabilities(snapshot.odds)
        if len(base) != 120 or set(base) != set(snapshot.odds) or set(market) != set(base):
            return None
        model_race: dict[str, Any] = {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "jcd": race.jcd,
            "rno": race.rno,
            "snapshot_id": snapshot.snapshot_id,
            "model_probabilities": base,
            "market_probabilities": market,
            "odds": dict(snapshot.odds),
            "odds_checkpoints": {"300": point},
        }
        earlier, earlier_reason = earlier_market_fields(
            conn,
            race.race_id,
            current_snapshot=current_snapshot,
            max_snapshot_age_seconds=DEFAULT_MAX_CHECKPOINT_AGE_SECONDS,
        )
        if earlier is not None:
            model_race.update(earlier)
        model_race.update(
            odds_path_fields(
                conn,
                race.race_id,
                current_snapshot=current_snapshot,
                max_snapshot_age_seconds=DEFAULT_MAX_CHECKPOINT_AGE_SECONDS,
            )
        )
        transformed = attach_odds_path_model(
            [model_race], dict(self._operational_model)
        )[0]
        return transformed, market, earlier_reason

    def _capital_limits(
        self, conn: Any, race: RaceWindow, *, bankroll_yen: int
    ) -> dict[str, int]:
        rows = conn.execute(
            """
            SELECT d.total_stake_yen, s.profit_yen
            FROM intraday_t300_shadow_decisions d
            LEFT JOIN intraday_t300_shadow_settlements s
              ON s.decision_id = d.decision_id
            WHERE d.race_date = ? AND d.model_key = ?
            """,
            (race.race_date, self.identity.model_key),
        ).fetchall()
        return daily_capital_limits(
            list(rows),
            bankroll_yen=bankroll_yen,
            starting_bankroll_yen=STARTING_BANKROLL_YEN,
        )

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        trained = str(self._bundle.get("trained_through_date") or "")
        if trained and trained >= race.race_date:
            raise ValueError("V23 artifacts are not strictly prior to race date")
        prepared = self._v23_model_race(conn, race, snapshot)
        if prepared is None:
            return _no_bet("invalid_v23_t300_market_features")
        transformed, market, earlier_reason = prepared
        probability_output = self._blend_head(
            transformed, market, self._probability_calibrator
        )
        ranking_output = self._blend_head(
            transformed, market, self._ranking_calibrator
        )
        if not probability_output or not ranking_output:
            return _no_bet("invalid_v23_probability_or_ranking_head_output")
        closing = attach_selected_closing_odds(
            [dict(transformed)], dict(self._closing_selection)
        )[0]
        forecast_odds = dict(closing.get("estimated_final_odds") or {})
        if len(forecast_odds) != 120 or set(forecast_odds) != set(ranking_output):
            return _no_bet("invalid_v23_closing_odds_forecast")
        limits = self._capital_limits(conn, race, bankroll_yen=bankroll_yen)
        selected = select_top5_narrow_candidates(
            ranking_output,
            forecast_odds,
            race_id=race.race_id,
            race_date=race.race_date,
            jcd=race.jcd,
            rno=race.rno,
            snapshot_id=snapshot.snapshot_id,
            captured_at=snapshot.captured_at.isoformat(),
            available_capital_yen=limits["allocatable_bankroll_yen"],
        )
        diagnostics = {
            "v23_top5_narrow": {
                **limits,
                "status": "selected" if selected else "no_bet",
                "policy_name": V23_POLICY_NAME,
                "registered_after": V23_REGISTERED_AFTER,
                "checkpoint": "t300",
                "source_snapshot_id": snapshot.snapshot_id,
                "source_evaluation_job_id": 8666,
                "ranking_top5": sorted(
                    ranking_output,
                    key=lambda combination: (-ranking_output[combination], combination),
                )[:5],
                "ranking_probabilities": ranking_output,
                "closing_odds_forecast_source": closing.get(
                    "closing_odds_forecast_source"
                ),
                "earlier_market_status": earlier_reason,
                "odds_path_points": int(transformed.get("odds_path_points") or 0),
                "decision_features": "t300_or_earlier",
                "outer_result_used": False,
                "outer_payout_used": False,
                "settlement_fields_used_for_capital_only": True,
                "real_betting_enabled": False,
            }
        }
        if limits["allocatable_bankroll_yen"] < V23_STAKE_YEN:
            reason = "v23_daily_capital_exhausted"
        else:
            reason = None if selected else "v23_no_top5_candidate_in_registered_ev_band"
        return ShadowDecision(
            probability_output, forecast_odds, selected, reason, diagnostics
        )


ADAPTER_FACTORIES: dict[str, Callable[[str, Path, Path], ShadowModelAdapter]] = {
    "v12_role_t300": lambda key, bundle, base: V12RoleModelAdapter(
        model_key=key, bundle_path=bundle, base_model_path=base
    ),
    "v14_registered_band_t300": lambda key, bundle, base: (
        V14RegisteredBandModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
    "v16_fixed_band_t300": lambda key, bundle, base: (
        V16FixedBandModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
    "v18_schedule_quota_t300": lambda key, bundle, base: (
        V18ScheduleQuotaModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
    "v20_dual_head_t300": lambda key, bundle, base: (
        V20DualHeadModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
    "v21_triple_head_t300": lambda key, bundle, base: (
        V21TripleHeadModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
    "v23_top5_narrow_t300": lambda key, bundle, base: (
        V23Top5NarrowModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        )
    ),
}


def register_adapter(strategy_name: str, factory: Callable[[str, Path, Path], ShadowModelAdapter]) -> None:
    if not strategy_name or strategy_name in ADAPTER_FACTORIES:
        raise ValueError(f"invalid or duplicate strategy adapter: {strategy_name}")
    ADAPTER_FACTORIES[strategy_name] = factory


def build_adapter(specification: str) -> ShadowModelAdapter:
    parts = specification.split(":", 3)
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError("model spec must be MODEL_KEY:STRATEGY:BUNDLE_JOBLIB:BASE_MODEL_JOBLIB")
    model_key, strategy, bundle, base = parts
    if strategy not in ADAPTER_FACTORIES:
        raise ValueError(f"unknown T300 shadow strategy: {strategy}")
    return ADAPTER_FACTORIES[strategy](model_key, Path(bundle), Path(base))


def resolve_race_date(now: datetime, configured_date: str | None) -> str:
    return (date.fromisoformat(configured_date).isoformat() if configured_date
            else now.astimezone(JST).date().isoformat())


def run_cycle(
    store: ShadowStore, adapters: Sequence[ShadowModelAdapter], *, now: datetime,
    configured_date: str | None = None,
    max_checkpoint_age_seconds: float = DEFAULT_MAX_CHECKPOINT_AGE_SECONDS,
    max_source_update_staleness_seconds: float = DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS,
    max_decision_delay_seconds: float = DEFAULT_MAX_DECISION_DELAY_SECONDS,
    starting_bankroll_yen: int = STARTING_BANKROLL_YEN,
) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if (
        max_checkpoint_age_seconds <= 0
        or max_source_update_staleness_seconds <= 0
        or max_decision_delay_seconds <= 0
    ):
        raise ValueError("snapshot staleness and decision delay limits must be positive")
    cycle_started = time.perf_counter()
    race_date = resolve_race_date(now, configured_date)
    store.ensure_schema()
    prewarm_timing: dict[str, float] = {}
    store_conn = getattr(store, "conn", None)
    if isinstance(store, PostgresShadowStore):
        for adapter in adapters:
            prewarm = getattr(adapter, "prewarm", None)
            if not callable(prewarm):
                continue
            started = time.perf_counter()
            prewarm(store_conn, race_date)
            prewarm_timing[adapter.identity.model_key] = round(
                time.perf_counter() - started, 6
            )
    settlements = store.append_available_settlements(race_date=race_date, now=now)
    inserted = duplicate = model_errors = no_bets = selected = deferred = 0
    due_races = store.due_races(race_date=race_date, now=now)
    pending_backlog_max_seconds = 0.0
    pending_decisions = 0
    model_decide_timing: dict[str, dict[str, float | int]] = {}
    for race in due_races:
        decision_delay_seconds = max(
            0.0,
            (now - race.target_t300_at.astimezone(now.tzinfo)).total_seconds(),
        )
        snapshot_loaded = False
        snapshot: T300Snapshot | None = None
        check = SnapshotCheck(None, None, "missing_complete_t300_snapshot")
        for adapter in adapters:
            identity = adapter.identity
            existing = store.decision_identity(race_id=race.race_id, model_key=identity.model_key)
            if existing is not None:
                if existing != identity:
                    raise ValueError(f"model identity conflict for {race.race_id} {identity.model_key}")
                duplicate += 1
                continue
            pending_decisions += 1
            pending_backlog_max_seconds = max(
                pending_backlog_max_seconds,
                (now - race.target_t300_at.astimezone(now.tzinfo)).total_seconds(),
            )
            if not snapshot_loaded:
                snapshot = store.latest_complete_snapshot(race)
                snapshot_loaded = True
                if snapshot is not None:
                    check = validate_snapshot(
                        race, snapshot,
                        max_checkpoint_age_seconds=max_checkpoint_age_seconds,
                        max_source_update_staleness_seconds=max_source_update_staleness_seconds,
                    )
            if (
                (snapshot is None or check.reason is not None)
                and decision_delay_seconds < max_decision_delay_seconds
            ):
                deferred += 1
                continue
            bankroll = store.bankroll_yen(
                race_date=race_date, model_key=identity.model_key, starting_yen=starting_bankroll_yen
            )
            if snapshot is None or check.reason is not None:
                decision = _no_bet(check.reason or "missing_complete_t300_snapshot")
            else:
                try:
                    decision_started = time.perf_counter()
                    decision = adapter.decide(store.conn, race, snapshot, bankroll_yen=bankroll)
                except Exception as exc:
                    model_errors += 1
                    decision = _no_bet(
                        f"model_error:{type(exc).__name__}",
                        diagnostics={
                            "model_error": {
                                "type": type(exc).__name__,
                                "message": str(exc)[:300],
                            }
                        },
                    )
                decision_elapsed = time.perf_counter() - decision_started
                timing = model_decide_timing.setdefault(
                    identity.model_key,
                    {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
                )
                timing["calls"] = int(timing["calls"]) + 1
                timing["total_seconds"] = float(timing["total_seconds"]) + decision_elapsed
                timing["max_seconds"] = max(float(timing["max_seconds"]), decision_elapsed)
            decision_completed_at = datetime.now(timezone.utc)
            created = store.insert_decision(
                race=race, identity=identity, decision_at=now, snapshot=snapshot,
                decision_completed_at=decision_completed_at,
                snapshot_check=check, bankroll_before_yen=bankroll, decision=decision,
            )
            inserted += int(created)
            duplicate += int(not created)
            no_bets += int(created and decision.status == "no_bet")
            selected += int(created and decision.status == "selected")
    settlements += store.append_available_settlements(race_date=race_date, now=now)
    cycle_elapsed = time.perf_counter() - cycle_started
    timing_diagnostics = {
        key: {
            "calls": int(value["calls"]),
            "total_seconds": round(float(value["total_seconds"]), 6),
            "max_seconds": round(float(value["max_seconds"]), 6),
        }
        for key, value in model_decide_timing.items()
    }
    timing_payload = {
        "cycle_elapsed_seconds": round(cycle_elapsed, 6),
        "due_races_scanned": len(due_races),
        "pending_decisions": pending_decisions,
        "initial_pending_backlog_max_seconds": round(
            pending_backlog_max_seconds, 6
        ),
        "model_decide": timing_diagnostics,
    }
    if prewarm_timing:
        timing_payload["model_prewarm_seconds"] = prewarm_timing
    return {"race_date": race_date, "observed_at": now.isoformat(),
            "models": [adapter.identity.model_key for adapter in adapters],
            "decisions_inserted": inserted, "selected_decisions": selected,
            "no_bet_decisions": no_bets, "existing_decisions": duplicate,
            "deferred_decisions": deferred,
            "model_errors": model_errors, "settlements_inserted": settlements,
            "timing": timing_payload,
            "real_betting": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append-only all-venue intraday T300 role-model shadow recorder."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--model-spec", action="append", required=True,
                        help="MODEL_KEY:STRATEGY:BUNDLE_JOBLIB:BASE_MODEL_JOBLIB")
    parser.add_argument("--date", help="Fixed JST date; omit for automatic rollover")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-checkpoint-age-seconds", type=float,
                        default=DEFAULT_MAX_CHECKPOINT_AGE_SECONDS)
    parser.add_argument("--max-source-update-staleness-seconds", type=float,
                        default=DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS)
    parser.add_argument("--max-decision-delay-seconds", type=float,
                        default=DEFAULT_MAX_DECISION_DELAY_SECONDS)
    parser.add_argument("--starting-bankroll-yen", type=int, default=STARTING_BANKROLL_YEN)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapters = [build_adapter(value) for value in args.model_spec]
    while True:
        now = datetime.now(timezone.utc)
        with connection(args.db) as conn:
            result = run_cycle(
                PostgresShadowStore(conn), adapters, now=now, configured_date=args.date,
                max_checkpoint_age_seconds=args.max_checkpoint_age_seconds,
                max_source_update_staleness_seconds=args.max_source_update_staleness_seconds,
                max_decision_delay_seconds=args.max_decision_delay_seconds,
                starting_bankroll_yen=args.starting_bankroll_yen,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
