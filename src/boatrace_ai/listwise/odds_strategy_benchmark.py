from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..db import connection, init_db
from ..features import latest_trifecta_odds_before_deadline


BET_TYPE = "3連単"
STAKE_YEN = 100
STRATEGIES = {
    "A_top5_flat": "Top5を全点100円購入",
    "B_top5_odds_gte_5": "Top5のうち締切前オッズ5倍以上を100円購入",
    "C_top5_ev_gte_1": "Top5のうちPL確率×締切前オッズ1.0以上を100円購入",
}
TRIFECTA_COMBINATIONS = {
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    for third in range(1, 7)
    if len({first, second, third}) == 3
}


def evaluate_odds_strategies(
    conn,
    *,
    prediction: dict[str, Any],
    source_path: Path | None = None,
) -> dict[str, Any]:
    payouts = _load_official_payouts(conn)
    totals = {
        name: {
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "hit_races": set(),
            "hit_tickets": 0,
            "winning_pre_deadline_odds": [],
        }
        for name in STRATEGIES
    }
    evaluated_races: set[str] = set()
    evaluated_dates: set[str] = set()
    seen_races: set[str] = set()
    metric_rows: list[tuple[dict[int, float], tuple[int, ...], tuple[str, ...]]] = []
    skipped_no_real_odds = 0
    skipped_no_payout = 0
    skipped_invalid_prediction = 0
    skipped_duplicate_race = 0

    for race in prediction.get("predictions") or []:
        normalized = _normalize_prediction(race, fallback_date=prediction.get("race_date"))
        if normalized is None:
            skipped_invalid_prediction += 1
            continue
        race_id, race_date, top5, lane_probabilities, actual_order = normalized
        if race_id in seen_races:
            skipped_duplicate_race += 1
            continue
        seen_races.add(race_id)
        official_payouts = payouts.get(race_id)
        if not official_payouts:
            skipped_no_payout += 1
            continue

        odds_snapshot = latest_trifecta_odds_before_deadline(
            conn,
            race_id,
            min_combinations=120,
        )
        odds = _normal_odds(odds_snapshot)
        if odds is None:
            skipped_no_real_odds += 1
            continue

        evaluated_races.add(race_id)
        if race_date:
            evaluated_dates.add(race_date)
        metric_rows.append((lane_probabilities, actual_order, top5))

        selections = {
            "A_top5_flat": list(top5),
            "B_top5_odds_gte_5": [combo for combo in top5 if odds[combo] >= 5.0],
            "C_top5_ev_gte_1": [
                combo
                for combo in top5
                if _pl_probability(lane_probabilities, combo) * odds[combo] >= 1.0
            ],
        }
        for strategy, combinations in selections.items():
            _settle_race(
                totals[strategy],
                race_id=race_id,
                combinations=combinations,
                odds=odds,
                official_payouts=official_payouts,
            )

    prediction_metrics = _prediction_metrics(metric_rows)
    strategy_rows = {
        name: _summarize_strategy(
            values,
            evaluated_races=len(evaluated_races),
            skipped_no_real_odds=skipped_no_real_odds,
            description=STRATEGIES[name],
        )
        for name, values in totals.items()
    }
    evaluation_period = _evaluation_period(evaluated_dates)
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": prediction.get("source_model") or prediction.get("model"),
        "source_prediction_file": str(source_path) if source_path is not None else None,
        "bet_type": BET_TYPE,
        "comparison_role": "real_odds_ticket_selection_diagnostic_not_promotion",
        "daily_budget_applied": False,
        "capital_policy": "全対象レースへの100円固定購入比較。日次10,000円上限と利益再投資は未適用",
        "stake_per_ticket_yen": STAKE_YEN,
        "evaluated_races": len(evaluated_races),
        "skipped_no_real_odds": skipped_no_real_odds,
        "skipped_no_payout": skipped_no_payout,
        "skipped_invalid_prediction": skipped_invalid_prediction,
        "skipped_duplicate_race": skipped_duplicate_race,
        "prediction_metrics": prediction_metrics,
        "evaluation_period": evaluation_period,
        "short_evaluation_warning": evaluation_period["is_short"],
        "short_evaluation_note": (
            "実オッズを利用できる評価期間が短いため、ROIと損益は暫定値です。"
            "期間と対象レースを増やして再検証してください。"
        ),
        "selection_guard": (
            "購入判定は予測Top5・6艇確率・締切前120通り実オッズだけを使用し、"
            "actual_orderと公式払戻は決済・評価にだけ使用"
        ),
        "strategies": strategy_rows,
    }


def _load_official_payouts(conn) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT race_id, combination, payout_yen
        FROM payouts
        WHERE bet_type = ? AND payout_yen IS NOT NULL
        """,
        (BET_TYPE,),
    ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        combination = _normalize_combination(row["combination"])
        if combination is None:
            continue
        payout_yen = int(row["payout_yen"])
        if payout_yen <= 0:
            continue
        result.setdefault(str(row["race_id"]), {})[combination] = payout_yen
    return result


def _normalize_prediction(
    race: Any,
    *,
    fallback_date: Any,
) -> tuple[str, str, tuple[str, ...], dict[int, float], tuple[int, ...]] | None:
    if not isinstance(race, dict) or not race.get("race_id"):
        return None
    top5_values = race.get("trifecta_top5") or []
    top5: list[str] = []
    for value in top5_values:
        raw = value.get("combination") if isinstance(value, dict) else value
        combination = _normalize_combination(raw)
        if combination is None:
            return None
        top5.append(combination)
    if len(top5) != 5 or len(set(top5)) != 5:
        return None

    lane_probabilities: dict[int, float] = {}
    for row in race.get("lane_probabilities") or []:
        try:
            lane = int(row["lane"])
            probability = float(row["probability"])
        except (KeyError, TypeError, ValueError):
            return None
        if lane not in range(1, 7) or lane in lane_probabilities:
            return None
        if not math.isfinite(probability) or probability < 0.0:
            return None
        lane_probabilities[lane] = probability
    total = sum(lane_probabilities.values())
    if set(lane_probabilities) != set(range(1, 7)) or total <= 0.0:
        return None
    lane_probabilities = {lane: value / total for lane, value in lane_probabilities.items()}

    try:
        actual_order = tuple(int(value) for value in race.get("actual_order") or ())
    except (TypeError, ValueError):
        return None
    if len(actual_order) < 3 or any(lane not in range(1, 7) for lane in actual_order[:3]):
        return None
    if len(set(actual_order[:3])) != 3:
        return None
    race_date = str(race.get("race_date") or fallback_date or str(race["race_id"])[:10])
    return (
        str(race["race_id"]),
        race_date,
        tuple(top5),
        lane_probabilities,
        actual_order,
    )


def _normal_odds(snapshot: dict[str, Any] | None) -> dict[str, float] | None:
    if not snapshot or int(snapshot.get("odds_count") or 0) != 120:
        return None
    source = snapshot.get("odds")
    if not isinstance(source, dict) or set(source) != TRIFECTA_COMBINATIONS:
        return None
    odds: dict[str, float] = {}
    for combination, raw_value in source.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        odds[combination] = value
    return odds


def _normalize_combination(value: Any) -> str | None:
    parts = str(value).strip().split("-")
    if len(parts) != 3:
        return None
    try:
        lanes = tuple(int(part) for part in parts)
    except ValueError:
        return None
    combination = "-".join(str(lane) for lane in lanes)
    return combination if combination in TRIFECTA_COMBINATIONS else None


def _pl_probability(lane_probabilities: dict[int, float], combination: str) -> float:
    first, second, third = (int(value) for value in combination.split("-"))
    first_probability = lane_probabilities[first]
    second_denominator = 1.0 - first_probability
    third_denominator = second_denominator - lane_probabilities[second]
    if second_denominator <= 0.0 or third_denominator <= 0.0:
        return 0.0
    return (
        first_probability
        * lane_probabilities[second]
        / second_denominator
        * lane_probabilities[third]
        / third_denominator
    )


def _settle_race(
    totals: dict[str, Any],
    *,
    race_id: str,
    combinations: list[str],
    odds: dict[str, float],
    official_payouts: dict[str, int],
) -> None:
    totals["tickets"] += len(combinations)
    totals["stake_yen"] += len(combinations) * STAKE_YEN
    race_hit = False
    for combination in combinations:
        payout_yen = official_payouts.get(combination)
        if payout_yen is None:
            continue
        totals["return_yen"] += payout_yen
        totals["hit_tickets"] += 1
        totals["winning_pre_deadline_odds"].append(odds[combination])
        race_hit = True
    if race_hit:
        totals["hit_races"].add(race_id)


def _summarize_strategy(
    values: dict[str, Any],
    *,
    evaluated_races: int,
    skipped_no_real_odds: int,
    description: str,
) -> dict[str, Any]:
    tickets = int(values["tickets"])
    stake = int(values["stake_yen"])
    returned = int(values["return_yen"])
    hit_tickets = int(values["hit_tickets"])
    winning_odds = values["winning_pre_deadline_odds"]
    return {
        "description": description,
        "evaluated_races": evaluated_races,
        "skipped_no_real_odds": skipped_no_real_odds,
        "tickets": tickets,
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "hit_races": len(values["hit_races"]),
        "hit_tickets": hit_tickets,
        "race_hit_rate": len(values["hit_races"]) / evaluated_races if evaluated_races else None,
        "average_winning_pre_deadline_odds": (
            sum(winning_odds) / len(winning_odds) if winning_odds else None
        ),
        "break_even_average_odds": tickets / hit_tickets if hit_tickets else None,
        "break_even_definition": "購入券数 ÷ 的中券数（1券100円、払戻倍率も100円単位）",
    }


def _prediction_metrics(
    rows: list[tuple[dict[int, float], tuple[int, ...], tuple[str, ...]]],
) -> dict[str, Any]:
    if not rows:
        return {
            "evaluated_races": 0,
            "entry_log_loss": None,
            "winner_top1_accuracy": None,
            "trifecta_top5_hit_rate": None,
        }
    loss = 0.0
    winner_hits = 0
    top5_hits = 0
    for probabilities, actual_order, top5 in rows:
        winner = actual_order[0]
        predicted_winner = max(range(1, 7), key=lambda lane: probabilities[lane])
        winner_hits += int(predicted_winner == winner)
        top5_hits += int("-".join(str(lane) for lane in actual_order[:3]) in top5)
        for lane in range(1, 7):
            probability = min(1.0 - 1e-15, max(1e-15, probabilities[lane]))
            label = int(lane == winner)
            loss -= label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
    race_count = len(rows)
    return {
        "evaluated_races": race_count,
        "entry_log_loss": loss / (race_count * 6),
        "winner_top1_accuracy": winner_hits / race_count,
        "trifecta_top5_hit_rate": top5_hits / race_count,
    }


def _evaluation_period(dates: set[str]) -> dict[str, Any]:
    ordered = sorted(dates)
    if not ordered:
        return {"start_date": None, "end_date": None, "race_days": 0, "is_short": True}
    return {
        "start_date": ordered[0],
        "end_date": ordered[-1],
        "race_days": len(ordered),
        "is_short": len(ordered) < 30,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Listwise Top5 real-odds strategy benchmark.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    prediction = json.loads(args.predictions.read_text(encoding="utf-8"))
    init_db(args.db)
    with connection(args.db) as conn:
        result = evaluate_odds_strategies(
            conn,
            prediction=prediction,
            source_path=args.predictions,
        )
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "evaluated_races": result["evaluated_races"],
                "skipped_no_real_odds": result["skipped_no_real_odds"],
                "strategies": {
                    name: {
                        key: row[key]
                        for key in ("tickets", "stake_yen", "return_yen", "profit_yen", "roi", "hit_races")
                    }
                    for name, row in result["strategies"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
