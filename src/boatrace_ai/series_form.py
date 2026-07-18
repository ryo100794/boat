from __future__ import annotations

import json
from typing import Any


ACCIDENT_MARKS = {"F", "L", "S", "K", "妨", "失", "転", "落", "沈", "不", "欠"}


def parse_series_results(value: str | None) -> dict[str, Any]:
    text = str(value or "").strip()
    finishes = [int(ch) for ch in text if ch in "123456"]
    accident_count = sum(1 for ch in text if ch in ACCIDENT_MARKS)
    starts = len(finishes)
    if not starts:
        return {
            "series_results_raw": text,
            "series_starts": 0,
            "series_avg_finish": -1.0,
            "series_latest_finish": -1.0,
            "series_best_finish": -1.0,
            "series_worst_finish": -1.0,
            "series_win_rate": -1.0,
            "series_top2_rate": -1.0,
            "series_top3_rate": -1.0,
            "series_finish_trend": -1.0,
            "series_accident_count": accident_count,
            "series_has_f": int("F" in text),
            "series_has_l": int("L" in text),
            "series_has_s": int("S" in text),
            "series_has_accident": int(accident_count > 0),
            "series_has_results": 0,
        }
    split_at = max(1, starts // 2)
    first = finishes[:split_at]
    last = finishes[split_at:] or finishes[-1:]
    first_avg = sum(first) / len(first)
    last_avg = sum(last) / len(last)
    return {
        "series_results_raw": text,
        "series_starts": starts,
        "series_avg_finish": sum(finishes) / starts,
        "series_latest_finish": finishes[-1],
        "series_best_finish": min(finishes),
        "series_worst_finish": max(finishes),
        "series_win_rate": sum(1 for rank in finishes if rank == 1) / starts,
        "series_top2_rate": sum(1 for rank in finishes if rank <= 2) / starts,
        "series_top3_rate": sum(1 for rank in finishes if rank <= 3) / starts,
        "series_finish_trend": first_avg - last_avg,
        "series_accident_count": accident_count,
        "series_has_f": int("F" in text),
        "series_has_l": int("L" in text),
        "series_has_s": int("S" in text),
        "series_has_accident": int(accident_count > 0),
        "series_has_results": 1,
    }


def extract_fixed_series_fields(line: str, *, data_end: int) -> dict[str, Any]:
    early_raw = line[71:73].strip() if len(line) >= 72 else ""
    early_look_rno = int(early_raw) if early_raw.isdigit() else None
    series_text = line[data_end:71].strip() if len(line) >= 71 else line[data_end:].strip()
    parsed = parse_series_results(series_text)
    return {
        "series_results": parsed["series_results_raw"],
        "early_look_rno": early_look_rno,
        **{key: value for key, value in parsed.items() if key != "series_results_raw"},
    }


def entry_series_features(row: Any) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.__getitem__
    try:
        raw = json.loads(get("raw_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    parsed = parse_series_results(raw.get("series_results"))
    rno = _int_or_none(get("rno"))
    early = _int_or_none(raw.get("early_look_rno"))
    gap = early - rno if early is not None and rno is not None else -1
    return {
        key: value
        for key, value in parsed.items()
        if key != "series_results_raw"
    } | {
        "has_early_look": int(early is not None),
        "early_look_rno": float(early if early is not None else -1),
        "early_look_gap": float(gap),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
