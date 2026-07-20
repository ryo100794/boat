from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..fast_math import TRIFECTA_COMBINATIONS


JST = timezone(timedelta(hours=9))
STARTING_BANKROLL_YEN = 10_000
DECISION_MINUTES_BEFORE_START = 10
EV_THRESHOLD = 1.20
FRACTIONAL_KELLY = 0.25
RACE_CAP_FRACTION = 0.10
TICKET_CAP_FRACTION = 0.03
STAKE_UNIT_YEN = 100
MAX_LANE_MARKER_ODDS = 8
COMBINATIONS = tuple("-".join(map(str, item)) for item in TRIFECTA_COMBINATIONS)


def model_label(model_path: str) -> str:
    stem = Path(model_path).stem
    labels = {
        "win_model_no_odds_v8": "no_odds_v8 主系",
        "win_model_no_odds_v7": "no_odds_v7",
        "win_model_no_odds_v6": "no_odds_v6",
        "win_model_pastlog_v7_stream_hash": "pastlog_v7",
        "listwise_newton_cg_v1": "listwise Newton-CG",
    }
    return labels.get(stem, stem)


def day_bankroll_simulation(
    conn,
    *,
    race_date: str,
    model_path: str | None = None,
    now: datetime | None = None,
    starting_bankroll_yen: int = STARTING_BANKROLL_YEN,
) -> dict[str, Any]:
    now_jst = (now or datetime.now(timezone.utc)).astimezone(JST)
    models = _available_models(conn, race_date)
    selected = model_path if any(row["id"] == model_path for row in models) else None
    if selected is None and models:
        selected = models[0]["id"]

    policy = {
        "starting_bankroll_yen": starting_bankroll_yen,
        "bet_type": "3連単",
        "decision": "締切5分前（保存出走時刻の10分前）",
        "ev_threshold": EV_THRESHOLD,
        "fractional_kelly": FRACTIONAL_KELLY,
        "race_cap_fraction": RACE_CAP_FRACTION,
        "ticket_cap_fraction": TICKET_CAP_FRACTION,
        "stake_unit_yen": STAKE_UNIT_YEN,
        "profit_reinvestment": True,
        "ticket_units": "0以上",
    }
    if not selected:
        return {
            "available": False,
            "date": race_date,
            "generated_at": now_jst.isoformat(timespec="seconds"),
            "models": models,
            "selected_model": None,
            "policy": policy,
            "stats": _empty_stats(starting_bankroll_yen),
            "series": [],
            "warnings": ["当日予測を持つモデルがありません。"],
        }

    races = _race_rows(conn, race_date)
    results = _results_by_race(conn, race_date)
    payouts = _payouts_by_race(conn, race_date)
    bankroll = starting_bankroll_yen
    peak = bankroll
    max_drawdown = 0
    total_stake = 0
    total_return = 0
    tickets = 0
    hits = 0
    selected_races = 0
    evaluated_races = 0
    prediction_races = 0
    valid_odds_races = 0
    fallback_predictions = 0
    rejected_snapshots = 0
    series: list[dict[str, Any]] = []

    for race in races:
        start_at = _parse_time(race.get("deadline_at"), default_tz=JST)
        if start_at is None or start_at > now_jst:
            continue
        actual = results.get(str(race["race_id"]))
        payout = payouts.get(str(race["race_id"]))
        if not actual or not payout or actual["combination"] != payout["combination"]:
            continue
        evaluated_races += 1
        decision_at = start_at - timedelta(minutes=DECISION_MINUTES_BEFORE_START)
        predictions, fallback = _prediction_rows(
            conn,
            race_id=str(race["race_id"]),
            model_path=selected,
            cutoff=decision_at,
        )
        if predictions:
            prediction_races += 1
        if fallback:
            fallback_predictions += 1
        snapshot, rejected = _latest_valid_odds_snapshot(
            conn,
            race_id=str(race["race_id"]),
            cutoff=decision_at,
        )
        rejected_snapshots += rejected
        if snapshot:
            valid_odds_races += 1

        allocation = _allocate_race(
            bankroll,
            predictions,
            (snapshot or {}).get("odds") or {},
            actual_combination=actual["combination"],
            payout_yen=int(payout["payout_yen"]),
        )
        bankroll = bankroll - allocation["stake_yen"] + allocation["return_yen"]
        total_stake += allocation["stake_yen"]
        total_return += allocation["return_yen"]
        tickets += allocation["tickets"]
        hits += allocation["hit_tickets"]
        if allocation["tickets"]:
            selected_races += 1
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)
        cumulative_profit = bankroll - starting_bankroll_yen
        series.append(
            {
                "race_id": race["race_id"],
                "venue": race.get("venue_name"),
                "jcd": race.get("jcd"),
                "rno": race.get("rno"),
                "race_time_at": start_at.isoformat(timespec="seconds"),
                "decision_at": decision_at.isoformat(timespec="seconds"),
                "bankroll_yen": bankroll,
                "profit_yen": cumulative_profit,
                "cumulative_roi": total_return / total_stake if total_stake else None,
                "stake_yen": allocation["stake_yen"],
                "return_yen": allocation["return_yen"],
                "tickets": allocation["tickets"],
                "hit": allocation["hit_tickets"] > 0,
                "actual": actual["combination"],
                "odds_captured_at": (snapshot or {}).get("captured_at"),
                "prediction_basis": "post-race static fallback" if fallback else "締切5分前以前",
            }
        )

    stats = {
        "starting_bankroll_yen": starting_bankroll_yen,
        "current_bankroll_yen": bankroll,
        "profit_yen": bankroll - starting_bankroll_yen,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "roi": total_return / total_stake if total_stake else None,
        "evaluated_races": evaluated_races,
        "prediction_races": prediction_races,
        "valid_odds_races": valid_odds_races,
        "selected_races": selected_races,
        "tickets": tickets,
        "hit_tickets": hits,
        "ticket_hit_rate": hits / tickets if tickets else None,
        "max_drawdown_yen": max_drawdown,
        "fallback_prediction_races": fallback_predictions,
        "rejected_odds_snapshots": rejected_snapshots,
    }
    warnings = []
    if fallback_predictions:
        warnings.append(
            f"{fallback_predictions}Rは締切前予測未保存のため、結果を使わない静的モデル再生成値を使用。"
        )
    if rejected_snapshots:
        warnings.append(
            f"艇番見出し混入など品質不合格のオッズ{rejected_snapshots}件を除外。"
        )
    if valid_odds_races < evaluated_races:
        warnings.append(
            f"締切5分前の正常な120組オッズがない{evaluated_races - valid_odds_races}Rは購入0口。"
        )
    return {
        "available": True,
        "date": race_date,
        "generated_at": now_jst.isoformat(timespec="seconds"),
        "through_race_time_at": series[-1]["race_time_at"] if series else None,
        "models": models,
        "selected_model": selected,
        "selected_model_label": model_label(selected),
        "policy": policy,
        "stats": stats,
        "series": series,
        "warnings": warnings,
    }


def _empty_stats(starting_bankroll_yen: int) -> dict[str, Any]:
    return {
        "starting_bankroll_yen": starting_bankroll_yen,
        "current_bankroll_yen": starting_bankroll_yen,
        "profit_yen": 0,
        "stake_yen": 0,
        "return_yen": 0,
        "roi": None,
        "evaluated_races": 0,
        "prediction_races": 0,
        "valid_odds_races": 0,
        "selected_races": 0,
        "tickets": 0,
        "hit_tickets": 0,
        "ticket_hit_rate": None,
        "max_drawdown_yen": 0,
        "fallback_prediction_races": 0,
        "rejected_odds_snapshots": 0,
    }


def _available_models(conn, race_date: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        WITH recent_predictions AS MATERIALIZED (
          SELECT prediction_id, race_id, model_path, generated_at
          FROM predictions
          WHERE model_path IS NOT NULL
          ORDER BY prediction_id DESC
          LIMIT 200000
        )
        SELECT p.model_path, COUNT(DISTINCT p.race_id) AS races,
               MAX(p.generated_at) AS latest_generated_at
        FROM recent_predictions p
        JOIN races r ON r.race_id = p.race_id
        WHERE r.race_date = ?
        GROUP BY p.model_path
        ORDER BY MAX(p.generated_at) DESC, p.model_path
        """,
        (race_date,),
    ).fetchall()
    return [
        {
            "id": str(row["model_path"]),
            "label": model_label(str(row["model_path"])),
            "prediction_races": int(row["races"] or 0),
            "latest_generated_at": row["latest_generated_at"],
        }
        for row in rows
    ]


def _race_rows(conn, race_date: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT race_id, jcd, venue_name, rno, deadline_at, status
        FROM races
        WHERE race_date = ? AND deadline_at IS NOT NULL
        ORDER BY deadline_at, jcd, rno
        """,
        (race_date,),
    ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _results_by_race(conn, race_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT rr.race_id, rr.lane, rr.rank
        FROM race_results rr
        JOIN races r ON r.race_id = rr.race_id
        WHERE r.race_date = ? AND rr.rank BETWEEN 1 AND 3
        ORDER BY rr.race_id, rr.rank
        """,
        (race_date,),
    ).fetchall()
    lanes: dict[str, list[int]] = {}
    for row in rows:
        lanes.setdefault(str(row["race_id"]), []).append(int(row["lane"]))
    return {
        race_id: {"combination": "-".join(map(str, values))}
        for race_id, values in lanes.items()
        if len(values) == 3
    }


def _payouts_by_race(conn, race_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT p.race_id, p.combination, p.payout_yen
        FROM payouts p
        JOIN races r ON r.race_id = p.race_id
        WHERE r.race_date = ? AND p.bet_type = '3連単' AND p.payout_yen IS NOT NULL
        """,
        (race_date,),
    ).fetchall()
    return {
        str(row["race_id"]): {
            "combination": str(row["combination"]),
            "payout_yen": int(row["payout_yen"]),
        }
        for row in rows
    }


def _prediction_rows(
    conn,
    *,
    race_id: str,
    model_path: str,
    cutoff: datetime,
) -> tuple[dict[str, float], bool]:
    cutoff_utc = cutoff.astimezone(timezone.utc).isoformat(timespec="seconds")
    latest = conn.execute(
        """
        SELECT generated_at
        FROM predictions
        WHERE race_id = ? AND model_path = ? AND generated_at <= ?
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        (race_id, model_path, cutoff_utc),
    ).fetchone()
    fallback = False
    if latest is None:
        latest = conn.execute(
            """
            SELECT generated_at
            FROM predictions
            WHERE race_id = ? AND model_path = ?
            ORDER BY generated_at
            LIMIT 1
            """,
            (race_id, model_path),
        ).fetchone()
        fallback = latest is not None
    if latest is None:
        return {}, False
    rows = conn.execute(
        """
        SELECT combination, probability
        FROM predictions
        WHERE race_id = ? AND model_path = ? AND generated_at = ?
        """,
        (race_id, model_path, latest["generated_at"]),
    ).fetchall()
    return {
        str(row["combination"]): float(row["probability"])
        for row in rows
        if row["probability"] is not None
    }, fallback


def _latest_valid_odds_snapshot(
    conn,
    *,
    race_id: str,
    cutoff: datetime,
) -> tuple[dict[str, Any] | None, int]:
    cutoff_utc = cutoff.astimezone(timezone.utc).isoformat(timespec="seconds")
    rows = conn.execute(
        """
        SELECT os.snapshot_id, os.captured_at
        FROM odds_snapshots os
        WHERE os.race_id = ? AND os.bet_type = 'trifecta'
          AND os.captured_at <= ?
          AND os.raw_json LIKE '%"parser_version": "odds3t_dom_v2"%'
        ORDER BY os.captured_at DESC, os.snapshot_id DESC
        LIMIT 8
        """,
        (race_id, cutoff_utc),
    ).fetchall()
    rejected = 0
    expected = set(COMBINATIONS)
    for row in rows:
        odds_rows = conn.execute(
            """
            SELECT combination, odds
            FROM odds_trifecta
            WHERE snapshot_id = ? AND odds IS NOT NULL
            """,
            (row["snapshot_id"],),
        ).fetchall()
        odds = {
            str(item["combination"]): float(item["odds"])
            for item in odds_rows
            if item["odds"] is not None
        }
        if set(odds) != expected or not _plausible_odds(odds):
            rejected += 1
            continue
        return {
            "snapshot_id": int(row["snapshot_id"]),
            "captured_at": row["captured_at"],
            "odds": odds,
        }, rejected
    return None, rejected


def _plausible_odds(odds: dict[str, float]) -> bool:
    values = list(odds.values())
    return (
        len(values) == 120
        and all(math.isfinite(value) and value >= 1.0 for value in values)
        and sum(value in {1, 2, 3, 4, 5, 6} for value in values)
        <= MAX_LANE_MARKER_ODDS
    )


def _allocate_race(
    bankroll_yen: int,
    probabilities: dict[str, float],
    odds: dict[str, float],
    *,
    actual_combination: str,
    payout_yen: int,
) -> dict[str, int]:
    prepared = []
    for combination in COMBINATIONS:
        probability = float(probabilities.get(combination) or 0.0)
        estimated_odds = float(odds.get(combination) or 0.0)
        if probability <= 0.0 or estimated_odds <= 1.0:
            continue
        expected_value = probability * estimated_odds
        if expected_value < EV_THRESHOLD:
            continue
        kelly = (expected_value - 1.0) / (estimated_odds - 1.0)
        fraction = min(TICKET_CAP_FRACTION, FRACTIONAL_KELLY * kelly)
        if fraction > 0.0 and math.isfinite(fraction):
            prepared.append((combination, expected_value, fraction))
    total_fraction = sum(item[2] for item in prepared)
    scale = min(1.0, RACE_CAP_FRACTION / total_fraction) if total_fraction else 0.0
    stakes = []
    for combination, expected_value, fraction in sorted(
        prepared, key=lambda item: (item[1], item[2]), reverse=True
    ):
        stake = int(bankroll_yen * fraction * scale // STAKE_UNIT_YEN) * STAKE_UNIT_YEN
        if stake >= STAKE_UNIT_YEN:
            stakes.append((combination, stake))
    total_stake = sum(stake for _combination, stake in stakes)
    if total_stake > bankroll_yen:
        return {"stake_yen": 0, "return_yen": 0, "tickets": 0, "hit_tickets": 0}
    hit_stake = sum(stake for combination, stake in stakes if combination == actual_combination)
    return_yen = int(round(hit_stake * payout_yen / 100)) if hit_stake else 0
    return {
        "stake_yen": total_stake,
        "return_yen": return_yen,
        "tickets": len(stakes),
        "hit_tickets": 1 if hit_stake else 0,
    }


def _parse_time(value: Any, *, default_tz: timezone) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(default_tz)
