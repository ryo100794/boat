from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_THRESHOLD_CANDIDATES = (1.05, 1.10, 1.15, 1.20, 1.25)


def calibration_policy_split(
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    selection_days: int,
) -> int | None:
    dates = sorted({str(row[1]) for row in race_keys})
    if selection_days < 1 or len(dates) <= selection_days:
        return None
    selection_start = dates[-selection_days]
    split = next(
        (index for index, row in enumerate(race_keys) if str(row[1]) >= selection_start),
        None,
    )
    if split is None or split <= 0 or split >= len(race_keys):
        return None
    return split


def flat_threshold_diagnostics(
    expected_returns: np.ndarray,
    race_keys: Sequence[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    combination_index: dict[str, int],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    values = np.asarray(expected_returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(race_keys):
        raise ValueError("expected returns and race keys must align")
    rows = []
    valid_races = [
        index
        for index, race_key in enumerate(race_keys)
        if str(race_key[0]) in payouts
        and str(payouts[str(race_key[0])].get("combination")) in combination_index
    ]
    for threshold in thresholds:
        selected = np.zeros_like(values, dtype=bool)
        if valid_races:
            selected[valid_races] = values[valid_races] >= float(threshold)
        tickets = int(selected.sum())
        hits = 0
        return_yen = 0
        for race_index, race_key in enumerate(race_keys):
            actual = payouts.get(str(race_key[0]))
            if actual is None:
                continue
            index = combination_index.get(str(actual["combination"]))
            if index is None or not selected[race_index, index]:
                continue
            hits += 1
            return_yen += int(actual["payout_yen"])
        stake_yen = tickets * 100
        rows.append(
            {
                "ev_threshold": float(threshold),
                "tickets": tickets,
                "hits": hits,
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "roi": return_yen / stake_yen if stake_yen else 0.0,
            }
        )
    return rows


def select_policy_threshold(
    diagnostics: Sequence[dict[str, Any]],
    *,
    fallback: float,
    minimum_tickets: int,
    minimum_roi: float,
    minimum_hits: int = 0,
    minimum_winning_days: int = 0,
) -> tuple[float, str]:
    eligible = [
        row
        for row in diagnostics
        if int(row["tickets"]) >= minimum_tickets
        and int(row.get("hits") or 0) >= minimum_hits
        and int(row.get("winning_days") or 0) >= minimum_winning_days
        and float(row["roi"]) >= minimum_roi
        and int(row["profit_yen"]) > 0
    ]
    if not eligible:
        return float(fallback), "fallback_fixed_threshold"
    selected = min(eligible, key=lambda row: float(row["ev_threshold"]))
    return float(selected["ev_threshold"]), "pre_evaluation_temporal_selection"
