from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .adaptive_allocation import allocate_adaptive_day, append_day_result, folds_by_full_day
from .bankroll_backtest import _build_payout_model, _candidate_tickets
from .hashed_feature_dataset import HashedRaceDataset
from .listwise_ranking_model import (
    FEATURE_SET,
    MODEL_NAME,
    evaluate_range,
    fit_scaler,
    train_listwise_model,
)


def full_day_fold_boundaries(
    race_keys: list[tuple[str, str, str, int]],
    *,
    folds: int,
    min_train_races: int,
) -> list[tuple[int, int, set[str]]]:
    specs = folds_by_full_day(race_keys, folds=folds, min_train_races=min_train_races)
    index_by_id = {race_id: index for index, (race_id, *_rest) in enumerate(race_keys)}
    boundaries = []
    for train_ids, test_ids, test_dates in specs:
        train_end = max(index_by_id[race_id] for race_id in train_ids) + 1
        test_indices = sorted(index_by_id[race_id] for race_id in test_ids)
        expected = list(range(test_indices[0], test_indices[-1] + 1))
        if test_indices != expected or test_indices[0] != train_end:
            raise ValueError("fold boundary is not a contiguous chronological range")
        boundaries.append((train_end, test_indices[-1] + 1, test_dates))
    return boundaries


def nested_select_candidate(
    dataset: HashedRaceDataset,
    *,
    outer_train_end: int,
    targets: Sequence[str],
    alphas: Sequence[float],
    learning_rate: float,
    epochs: int,
    batch_races: int,
    validation_fraction: float,
    min_validation_races: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_count = max(
        int(min_validation_races), int(round(outer_train_end * validation_fraction))
    )
    validation_count = min(validation_count, max(1, outer_train_end // 3))
    inner_train_end = outer_train_end - validation_count
    if inner_train_end <= 0:
        raise ValueError("outer training fold is too small for nested validation")
    scaler = fit_scaler(dataset, race_end=inner_train_end, batch_rows=batch_races * 6)
    candidates: list[dict[str, Any]] = []
    for target in targets:
        for alpha in alphas:
            model, history = train_listwise_model(
                dataset,
                train_race_end=inner_train_end,
                target=target,
                alpha=float(alpha),
                learning_rate=learning_rate,
                epochs=epochs,
                batch_races=batch_races,
                scaler=scaler,
            )
            metrics, _ = evaluate_range(
                dataset,
                model,
                race_start=inner_train_end,
                race_end=outer_train_end,
                batch_races=batch_races,
            )
            candidates.append({
                "target": target,
                "alpha": float(alpha),
                "inner_train_races": inner_train_end,
                "validation_races": validation_count,
                "training_history": history,
                **metrics,
            })
    selected = min(candidates, key=lambda row: (
        float(row["ranking_log_loss"]),
        float(row["entry_log_loss"]),
        -float(row["trifecta_top5_hit_rate"]),
    ))
    return selected, candidates


def default_policy(*, daily_budget_yen: int, ev_threshold: float) -> dict[str, Any]:
    return {
        "daily_budget_yen": int(daily_budget_yen),
        "bet_type": "3連単",
        "include_odds": False,
        "ev_threshold": float(ev_threshold),
        "payout_prior_weight": 30.0,
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly",
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "payout_estimator": "training-fold trifecta payout mean with global prior",
        "selection": "nested time split; no outer test-fold tuning",
    }


def evaluate_bankroll_fold(
    *,
    rows_by_race: dict[str, list[dict[str, Any]]],
    train_races: set[str],
    test_dates: set[str],
    payouts: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    totals: dict[str, float],
    daily_rows: list[dict[str, Any]],
    profit_state: tuple[int, int, int],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    payout_model = _build_payout_model(
        payouts,
        train_races=train_races,
        prior_weight=float(policy["payout_prior_weight"]),
    )
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    candidate_count = 0
    for race_id, rows in rows_by_race.items():
        actual = payouts.get(race_id)
        if len(rows) != 6 or actual is None:
            continue
        race_date = str(rows[0]["race_date"])
        evaluated_by_day[race_date].add(race_id)
        candidates = _candidate_tickets(
            rows,
            actual=actual,
            payout_model=payout_model,
            ev_threshold=float(policy["ev_threshold"]),
        )
        candidates_by_day[race_date].extend(candidates)
        candidate_count += len(candidates)
    cumulative_profit, peak_profit, max_drawdown = profit_state
    fold_stake = fold_return = fold_tickets = 0
    for race_date in sorted(test_dates):
        result = allocate_adaptive_day(
            race_date,
            candidates_by_day.get(race_date, []),
            evaluated_by_day.get(race_date, set()),
            daily_budget_yen=int(policy["daily_budget_yen"]),
            fractional_kelly=float(policy["fractional_kelly"]),
            max_daily_exposure_fraction=float(policy["max_daily_exposure_fraction"]),
            min_daily_exposure_fraction=float(policy["min_daily_exposure_fraction"]),
            race_cap_fraction=float(policy["race_cap_fraction"]),
            ticket_cap_fraction=float(policy["ticket_cap_fraction"]),
            max_daily_tickets=policy["max_daily_tickets"],
            allocation_mode=str(policy["allocation_mode"]),
            stake_granularity_yen=int(policy["stake_granularity_yen"]),
            min_stake_yen=int(policy["min_stake_yen"]),
        )
        fold_stake += int(result["stake_yen"])
        fold_return += int(result["return_yen"])
        fold_tickets += int(result["tickets"])
        cumulative_profit, peak_profit, max_drawdown = append_day_result(
            daily_rows,
            totals,
            result,
            cumulative_profit=cumulative_profit,
            peak_profit=peak_profit,
            max_drawdown=max_drawdown,
        )
    metrics = {
        "candidate_tickets": candidate_count,
        "selected_tickets": fold_tickets,
        "stake_yen": fold_stake,
        "return_yen": fold_return,
        "profit_yen": fold_return - fold_stake,
        "roi": fold_return / fold_stake if fold_stake else 0.0,
    }
    return metrics, (cumulative_profit, peak_profit, max_drawdown)
