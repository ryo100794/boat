from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from boatrace_ai.db import connection, init_db
from boatrace_ai.web.intraday_bankroll import (
    PRIMARY_MODEL_ID,
    _available_models,
    day_bankroll_simulation,
)


def _active_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    bundle = model_dir / "daily-shadow-bundles" / "v21.joblib"
    base = model_dir / "evaluation_queue" / "job-00002707.joblib"
    bundle.parent.mkdir()
    base.parent.mkdir()
    bundle.write_bytes(b"v21")
    base.write_bytes(b"base")
    active = model_dir.parent / "runtime" / "daily-shadow-models" / "active"
    active.mkdir(parents=True)
    (active / "state.json").write_text(
        json.dumps({
            "model_specs": {
                "v21": f"v21_daily:v21_triple_head_t300:{bundle}:{base}"
            },
            "real_betting_enabled": False,
        }),
        encoding="utf-8",
    )


def _shadow_schema(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE intraday_t300_shadow_decisions (
          decision_id INTEGER PRIMARY KEY,
          race_date TEXT NOT NULL,
          race_id TEXT NOT NULL,
          model_key TEXT NOT NULL,
          decision_at TEXT NOT NULL,
          target_t300_at TEXT NOT NULL,
          source_snapshot_id INTEGER,
          decision_status TEXT NOT NULL,
          no_bet_reason TEXT,
          selected_candidates TEXT NOT NULL,
          total_stake_yen INTEGER NOT NULL
        );
        CREATE TABLE intraday_t300_shadow_settlements (
          decision_id INTEGER PRIMARY KEY,
          stake_yen INTEGER NOT NULL,
          return_yen INTEGER NOT NULL,
          profit_yen INTEGER NOT NULL,
          actual_combination TEXT NOT NULL
        );
        """
    )


def test_active_v21_is_the_default_primary_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "data" / "models"
    _active_model(model_dir)

    models = _available_models(model_dir)

    assert models[0]["id"] == PRIMARY_MODEL_ID
    assert models[0]["runtime_kind"] == "t300_shadow"
    assert "V21 主系" in models[0]["label"]
    assert models[0]["real_betting_enabled"] is False


def test_v21_primary_uses_frozen_shadow_decision_and_settlement(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "data" / "boat.sqlite"
    db_path.parent.mkdir()
    model_dir = db_path.parent / "models"
    _active_model(model_dir)
    init_db(db_path)
    race_id = "2026-07-31-01-01"
    with connection(db_path) as conn:
        _shadow_schema(conn)
        conn.execute(
            """
            INSERT INTO races(
              race_id, race_date, jcd, venue_name, rno, deadline_at, status
            ) VALUES (?, '2026-07-31', '01', '桐生', 1,
                      '2026-07-31T10:00:00+09:00', 'final')
            """,
            (race_id,),
        )
        conn.execute(
            """
            INSERT INTO intraday_t300_shadow_decisions VALUES(
              1, '2026-07-31', ?, 'v21_daily',
              '2026-07-31T00:55:00+00:00', '2026-07-31T00:55:00+00:00',
              10, 'selected', NULL, ?, 200
            )
            """,
            (race_id, json.dumps([{"combination": "1-2-3", "stake_yen": 200}])),
        )
        conn.execute(
            """
            INSERT INTO intraday_t300_shadow_settlements
            VALUES(1, 200, 4000, 3800, '1-2-3')
            """
        )

        result = day_bankroll_simulation(
            conn,
            race_date="2026-07-31",
            model_dir=model_dir,
            now=datetime(2026, 7, 31, 2, 0, tzinfo=timezone.utc),
        )

    assert result["selected_model"] == PRIMARY_MODEL_ID
    assert result["stats"]["current_bankroll_yen"] == 13_800
    assert result["stats"]["roi"] == 20.0
    assert result["stats"]["prediction_races"] == 1
    assert result["stats"]["evaluated_races"] == 1
    assert result["series"][0]["odds_basis"] == "締切5分前V21終値予測"
    assert result["policy"]["real_betting_enabled"] is False


def test_dashboard_migrates_saved_v8_selection_once() -> None:
    source = Path("src/boatrace_ai/templates/dashboard.html").read_text(encoding="utf-8")
    assert 'primaryModelVersion = "v21-primary-20260731"' in source
    assert 'localStorage.removeItem("boat.bankModel")' in source
    assert 'new URLSearchParams(location.search).get("date")' in source
    assert '/^\\d{4}-\\d{2}-\\d{2}$/' in source
