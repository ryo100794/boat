from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from .four_head_nested_v22 import (
    LabeledRace,
    RacePrediction,
    fit_four_head_nested_v22,
    predict_race,
)
from .four_head_v22_bankroll import (
    RESULT_AVAILABLE_AT_PROVENANCE,
    V22BankrollSettlement,
)
from .four_head_v22_evaluation import (
    DEFAULT_PROJECTION_DIMENSIONS,
    DecisionAudit,
    V22EvaluationData,
    _build_outer_settlements,
    load_v22_evaluation_data,
)
from .learned_purchase_allocation_v33 import (
    AllocationConfig,
    AllocationDecision,
    DEFAULT_CONFIGS,
    LearnedAllocationArtifact,
    allocation_decision,
    fit_learned_allocation_head,
)

MODEL_KEY = "learned_purchase_allocation_head_v33"
INITIAL_BANKROLL_YEN = 10_000
STAKE_UNIT_YEN = 100
MAX_T5_AGE_SECONDS = 300.0
ODDS_STRESS_MULTIPLIER = 0.95
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260802
_MISSING = object()


@dataclass(frozen=True)
class _PendingBet:
    race_id: str
    result_available_at: datetime
    stakes_yen: tuple[int, ...]


@contextmanager
def _legacy_main_pickle_aliases():
    """Resolve python -m caches without leaving mutations in __main__."""
    main = sys.modules.get("__main__")
    if main is None:
        raise RuntimeError("current process has no __main__ module")
    aliases = {
        "V22EvaluationData": V22EvaluationData,
        "DecisionAudit": DecisionAudit,
    }
    previous = {name: getattr(main, name, _MISSING) for name in aliases}
    try:
        for name, value in aliases.items():
            setattr(main, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                delattr(main, name)
            else:
                setattr(main, name, value)


def load_v22_evaluation_cache_compat(
    cache_path: Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> V22EvaluationData:
    """Load and reuse legacy schema-v1 V20/V22 joblib caches."""
    with _legacy_main_pickle_aliases():
        envelope = joblib.load(Path(cache_path))
    if not isinstance(envelope, Mapping) or envelope.get("schema_version") != 1:
        raise ValueError("unsupported V22 evaluation cache schema")
    if expected_signature is not None and envelope.get("signature") != dict(
        expected_signature
    ):
        raise ValueError("V22 evaluation cache signature mismatch")
    data = envelope.get("data")
    if not isinstance(data, V22EvaluationData):
        raise ValueError("V22 evaluation cache contains an invalid data object")
    return data


def _timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ordered_races(
    races: Iterable[LabeledRace], *, label: str
) -> tuple[LabeledRace, ...]:
    rows = tuple(races)
    if not rows:
        raise ValueError(f"{label} races are required")
    previous: tuple[str, str] | None = None
    seen: set[str] = set()
    for race in rows:
        decision = race.decision
        key = (str(decision.race_date), str(decision.race_id))
        if previous is not None and key <= previous:
            raise ValueError(f"{label} races must be uniquely chronological")
        if decision.race_id in seen:
            raise ValueError(f"duplicate {label} race id: {decision.race_id}")
        previous = key
        seen.add(decision.race_id)
        odds = np.asarray(decision.current_odds, dtype=np.float64)
        if odds.shape != (120,):
            raise ValueError("formal V33-LPA evaluation requires 120 T-5 odds")
        if not np.isfinite(odds).all() or np.any(odds <= 1.0):
            raise ValueError("formal V33-LPA T-5 odds must be finite and above one")
    return rows


def _settlement_index(
    data: V22EvaluationData,
    settlements: Iterable[V22BankrollSettlement],
    races: Sequence[LabeledRace],
    *,
    max_snapshot_age_seconds: float,
) -> dict[str, V22BankrollSettlement]:
    if (
        not math.isfinite(max_snapshot_age_seconds)
        or not 0.0 <= max_snapshot_age_seconds <= MAX_T5_AGE_SECONDS
    ):
        raise ValueError("formal T-5 snapshot age must be between zero and 300 seconds")
    expected = {race.decision.race_id for race in races}
    audits = {audit.race_id: audit for audit in data.decision_audit}
    if set(audits) != expected:
        raise ValueError("V33-LPA decision audit universe is incomplete")
    indexed: dict[str, V22BankrollSettlement] = {}
    for row in settlements:
        if row.race_id in indexed:
            raise ValueError(f"duplicate settlement: {row.race_id}")
        indexed[row.race_id] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"settlement universe mismatch; missing={missing}, extra={extra}"
        )
    for race_id, row in indexed.items():
        audit = audits[race_id]
        target = _timestamp(row.decision_target_at, field="decision_target_at")
        captured = _timestamp(row.odds_captured_at, field="odds_captured_at")
        result_at = _timestamp(row.result_available_at, field="result_available_at")
        if (
            target != _timestamp(audit.target_at, field="audit.target_at")
            or captured != _timestamp(audit.captured_at, field="audit.captured_at")
        ):
            raise ValueError(f"settlement T-5 audit mismatch: {race_id}")
        age = (target - captured).total_seconds()
        if age < 0.0 or age > max_snapshot_age_seconds:
            raise ValueError(f"unsafe T-5 snapshot: {race_id}")
        if result_at <= target:
            raise ValueError(f"settlement is not after decision: {race_id}")
        if row.result_available_at_source != RESULT_AVAILABLE_AT_PROVENANCE:
            raise ValueError(f"unsafe result time provenance: {race_id}")
    return indexed


def _training_payouts(
    training: Sequence[LabeledRace],
    settlements: Mapping[str, V22BankrollSettlement],
) -> dict[str, int]:
    payouts: dict[str, int] = {}
    for race in training:
        race_id = race.decision.race_id
        row = settlements[race_id]
        payout = row.official_payout_yen
        if isinstance(payout, bool) or not isinstance(payout, int) or payout < 100:
            raise ValueError(f"invalid official training payout: {race_id}")
        if int(row.official_winner_index) != int(race.outcome.winner_index):
            raise ValueError(f"official training winner mismatch: {race_id}")
        payouts[race_id] = payout
    return payouts


def _split_training(
    training: Sequence[LabeledRace],
    *,
    base_training_fraction: float,
    minimum_base_training_dates: int,
    minimum_lpa_teacher_dates: int,
) -> tuple[tuple[LabeledRace, ...], tuple[LabeledRace, ...]]:
    if not 0.1 <= base_training_fraction <= 0.9:
        raise ValueError("base_training_fraction must be between 0.1 and 0.9")
    if minimum_base_training_dates < 1 or minimum_lpa_teacher_dates < 4:
        raise ValueError("invalid minimum date requirements")
    dates = sorted({race.decision.race_date for race in training})
    boundary = int(math.floor(len(dates) * base_training_fraction + 1e-12))
    if (
        boundary < minimum_base_training_dates
        or len(dates) - boundary < minimum_lpa_teacher_dates
    ):
        raise ValueError(
            "training period cannot provide disjoint base-head and LPA teacher dates"
        )
    base_dates = set(dates[:boundary])
    teacher_dates = set(dates[boundary:])
    base = tuple(race for race in training if race.decision.race_date in base_dates)
    teacher = tuple(
        race for race in training if race.decision.race_date in teacher_dates
    )
    if base[-1].decision.race_date >= teacher[0].decision.race_date:
        raise AssertionError("base/LPA teacher boundary is not chronological")
    return base, teacher


def _digest(value: Any) -> str:
    if hasattr(value, "summary"):
        payload = value.summary()
    elif is_dataclass(value):
        payload = asdict(value)
    else:
        payload = value
    return hashlib.sha256(
        json.dumps(
            payload, default=str, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _prediction_digest(predictions: Sequence[RacePrediction]) -> str:
    return _digest(
        [
            {
                "race_id": row.race_id,
                "race_date": row.race_date,
                "probabilities": row.probabilities,
                "ranking_scores": row.ranking_scores,
                "predicted_closing_odds": row.predicted_closing_odds,
            }
            for row in predictions
        ]
    )


def _validated_stakes(
    allocation: AllocationDecision,
    *,
    race_id: str,
    choices: int,
    available_yen: int,
    stake_unit_yen: int,
) -> tuple[int, ...]:
    if allocation.race_id != race_id:
        raise ValueError(f"allocation race mismatch: {race_id}")
    stakes = tuple(int(value) for value in allocation.stakes_yen)
    if len(stakes) != choices:
        raise ValueError(f"allocation choice count mismatch: {race_id}")
    if any(value < 0 or value % stake_unit_yen for value in stakes):
        raise ValueError(f"allocation contains invalid stake units: {race_id}")
    total = sum(stakes)
    if total != int(allocation.proposed_stake_yen) or total > available_yen:
        raise ValueError(f"allocation exceeds available bankroll: {race_id}")
    return stakes


def _official_return_at_settlement(
    pending: _PendingBet,
    settlement: V22BankrollSettlement,
    *,
    payout_multiplier: float,
    stake_unit_yen: int,
) -> tuple[int, int]:
    """The only outer path that reads official winner and payout."""
    winner = settlement.official_winner_index
    payout = settlement.official_payout_yen
    if isinstance(winner, bool) or not isinstance(winner, int):
        raise ValueError(f"invalid official winner: {pending.race_id}")
    if not 0 <= winner < len(pending.stakes_yen):
        raise ValueError(f"official winner is outside choice range: {pending.race_id}")
    if isinstance(payout, bool) or not isinstance(payout, int) or payout < stake_unit_yen:
        raise ValueError(f"invalid official payout: {pending.race_id}")
    stressed = int(math.floor(payout * payout_multiplier))
    return pending.stakes_yen[winner] // stake_unit_yen * stressed, winner


def _simulate_day(
    race_date: str,
    races: Sequence[LabeledRace],
    predictions: Mapping[str, RacePrediction],
    settlements: Mapping[str, V22BankrollSettlement],
    artifact: LearnedAllocationArtifact,
    *,
    payout_multiplier: float,
    initial_bankroll_yen: int,
    stake_unit_yen: int,
) -> dict[str, Any]:
    ordered = sorted(
        races,
        key=lambda race: (
            _timestamp(
                settlements[race.decision.race_id].decision_target_at,
                field="decision_target_at",
            ),
            race.decision.race_id,
        ),
    )
    cash = peak_cash = initial_bankroll_yen
    max_drawdown = total_stake = total_return = tickets = hit_tickets = 0
    pending: list[_PendingBet] = []
    hit_returns: list[int] = []
    ledger: list[dict[str, Any]] = []

    def settle_due(as_of: datetime) -> None:
        nonlocal cash, peak_cash, max_drawdown, total_return, hit_tickets
        due = sorted(
            (item for item in pending if item.result_available_at <= as_of),
            key=lambda item: (item.result_available_at, item.race_id),
        )
        if not due:
            return
        due_ids = {item.race_id for item in due}
        pending[:] = [item for item in pending if item.race_id not in due_ids]
        for item in due:
            returned, winner = _official_return_at_settlement(
                item,
                settlements[item.race_id],
                payout_multiplier=payout_multiplier,
                stake_unit_yen=stake_unit_yen,
            )
            cash += returned
            total_return += returned
            if returned > 0:
                hit_tickets += 1
                hit_returns.append(returned)
            peak_cash = max(peak_cash, cash)
            max_drawdown = max(max_drawdown, peak_cash - cash)
            ledger.append(
                {
                    "event": "settlement",
                    "race_id": item.race_id,
                    "at": item.result_available_at.isoformat(),
                    "winner_index": winner,
                    "return_yen": returned,
                    "cash_after_yen": cash,
                    "outstanding_stake_yen": sum(
                        sum(row.stakes_yen) for row in pending
                    ),
                }
            )

    for race in ordered:
        race_id = race.decision.race_id
        settlement = settlements[race_id]
        decision_at = _timestamp(
            settlement.decision_target_at, field="decision_target_at"
        )
        settle_due(decision_at)
        before = cash
        decision = allocation_decision(
            artifact,
            race.decision,
            predictions[race_id],
            available_bankroll_yen=cash,
            stake_unit_yen=stake_unit_yen,
        )
        stakes = _validated_stakes(
            decision,
            race_id=race_id,
            choices=len(race.decision.current_odds),
            available_yen=cash,
            stake_unit_yen=stake_unit_yen,
        )
        stake = sum(stakes)
        cash -= stake
        total_stake += stake
        tickets += sum(value > 0 for value in stakes)
        pending.append(
            _PendingBet(
                race_id,
                _timestamp(settlement.result_available_at, field="result_available_at"),
                stakes,
            )
        )
        max_drawdown = max(max_drawdown, peak_cash - cash)
        ledger.append(
            {
                "event": "decision",
                "race_id": race_id,
                "at": decision_at.isoformat(),
                "result_available_at": settlement.result_available_at,
                "cash_before_yen": before,
                "cash_after_yen": cash,
                "stake_yen": stake,
                "outstanding_stake_yen": sum(
                    sum(row.stakes_yen) for row in pending
                ),
                "selections": [
                    {"choice_index": index, "stake_yen": value}
                    for index, value in enumerate(stakes)
                    if value > 0
                ],
            }
        )
    settle_due(datetime.max.replace(tzinfo=timezone.utc))
    return {
        "race_date": race_date,
        "evaluated_races": len(ordered),
        "opening_bankroll_yen": initial_bankroll_yen,
        "closing_bankroll_yen": cash,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": cash - initial_bankroll_yen,
        "roi": total_return / total_stake if total_stake else None,
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "max_drawdown_yen": max_drawdown,
        "largest_hit_return_yen": max(hit_returns, default=0),
        "hit_return_square_sum_yen2": sum(value * value for value in hit_returns),
        "ledger": ledger,
    }


def _bootstrap_daily_roi(
    daily: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, float | int | None]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    stakes = np.asarray([row["stake_yen"] for row in daily], dtype=np.float64)
    returns = np.asarray([row["return_yen"] for row in daily], dtype=np.float64)
    if not np.any(stakes > 0.0):
        return {
            "samples": samples,
            "roi_lower_95": None,
            "roi_upper_95": None,
            "probability_roi_above_one": None,
        }
    rng = np.random.default_rng(seed)
    ratios: list[np.ndarray] = []
    remaining = samples
    while remaining:
        size = min(256, remaining)
        indices = rng.integers(0, len(daily), size=(size, len(daily)))
        sampled_stake = stakes[indices].sum(axis=1)
        sampled_return = returns[indices].sum(axis=1)
        valid = sampled_stake > 0.0
        ratios.append(sampled_return[valid] / sampled_stake[valid])
        remaining -= size
    values = np.concatenate(ratios) if ratios else np.empty(0)
    return {
        "samples": samples,
        "roi_lower_95": float(np.quantile(values, 0.05)) if len(values) else None,
        "roi_upper_95": float(np.quantile(values, 0.95)) if len(values) else None,
        "probability_roi_above_one": float(np.mean(values > 1.0))
        if len(values)
        else None,
    }


def _summarize(
    daily: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    stake = sum(int(row["stake_yen"]) for row in daily)
    returned = sum(int(row["return_yen"]) for row in daily)
    largest = max((int(row["largest_hit_return_yen"]) for row in daily), default=0)
    square_sum = sum(float(row["hit_return_square_sum_yen2"]) for row in daily)
    profitable_days = sum(int(row["profit_yen"] > 0) for row in daily)
    max_drawdown = max((int(row["max_drawdown_yen"]) for row in daily), default=0)
    return {
        "evaluation_days": len(daily),
        "evaluated_races": sum(int(row["evaluated_races"]) for row in daily),
        "active_days": sum(int(row["stake_yen"] > 0) for row in daily),
        "profitable_days": profitable_days,
        "profitable_day_fraction": profitable_days / len(daily) if daily else None,
        "tickets": sum(int(row["tickets"]) for row in daily),
        "hit_tickets": sum(int(row["hit_tickets"]) for row in daily),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": sum(int(row["profit_yen"]) for row in daily),
        "roi": returned / stake if stake else None,
        "max_drawdown_yen": max_drawdown,
        "max_drawdown_fraction_of_daily_bankroll": max_drawdown
        / INITIAL_BANKROLL_YEN,
        "largest_hit_return_yen": largest,
        "roi_without_largest_hit": (returned - largest) / stake if stake else None,
        "largest_hit_return_share": largest / returned if returned else None,
        "effective_hit_count": returned * returned / square_sum if square_sum else 0.0,
        "daily_bootstrap": _bootstrap_daily_roi(
            daily, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "daily": list(daily),
    }


def _promotion_gate(
    result: Mapping[str, Any], stress: Mapping[str, Any]
) -> dict[str, bool]:
    bootstrap = result["daily_bootstrap"]
    return {
        "minimum_30_evaluation_days": int(result["evaluation_days"]) >= 30,
        "minimum_1000_evaluated_races": int(result["evaluated_races"]) >= 1_000,
        "minimum_300_tickets": int(result["tickets"]) >= 300,
        "minimum_20_hit_tickets": int(result["hit_tickets"]) >= 20,
        "minimum_60_active_days": int(result["active_days"]) >= 60,
        "roi_above_one": float(result["roi"] or 0.0) > 1.0,
        "daily_bootstrap_lower_above_one": float(
            bootstrap["roi_lower_95"] or 0.0
        )
        > 1.0,
        "bootstrap_probability_roi_above_one_95pct": float(
            bootstrap["probability_roi_above_one"] or 0.0
        )
        >= 0.95,
        "roi_without_largest_hit_above_one": float(
            result["roi_without_largest_hit"] or 0.0
        )
        > 1.0,
        "maximum_drawdown_at_most_30pct": float(
            result["max_drawdown_fraction_of_daily_bankroll"]
        )
        <= 0.30,
        "largest_hit_return_share_at_most_20pct": float(
            result["largest_hit_return_share"] or 1.0
        )
        <= 0.20,
        "five_percent_odds_stress_roi_above_one": float(stress["roi"] or 0.0)
        > 1.0,
    }


def _run_outer(
    outer: Sequence[LabeledRace],
    predictions: Sequence[RacePrediction],
    settlements: Mapping[str, V22BankrollSettlement],
    artifact: LearnedAllocationArtifact,
    *,
    payout_multiplier: float,
    initial_bankroll_yen: int,
    stake_unit_yen: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_prediction = {row.race_id: row for row in predictions}
    if set(by_prediction) != {race.decision.race_id for race in outer}:
        raise ValueError("outer prediction universe mismatch")
    by_date: dict[str, list[LabeledRace]] = defaultdict(list)
    for race in outer:
        by_date[race.decision.race_date].append(race)
    daily = [
        _simulate_day(
            date,
            by_date[date],
            by_prediction,
            settlements,
            artifact,
            payout_multiplier=payout_multiplier,
            initial_bankroll_yen=initial_bankroll_yen,
            stake_unit_yen=stake_unit_yen,
        )
        for date in sorted(by_date)
    ]
    return _summarize(
        daily, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
    )


def evaluate_learned_purchase_allocation_v33(
    data: V22EvaluationData,
    settlements: Iterable[V22BankrollSettlement],
    *,
    base_training_fraction: float = 0.60,
    minimum_base_training_dates: int = 5,
    minimum_lpa_teacher_dates: int = 4,
    base_fit_kwargs: Mapping[str, Any] | None = None,
    allocation_configs: Iterable[AllocationConfig] = DEFAULT_CONFIGS,
    allocation_validation_fraction: float = 0.25,
    allocation_max_iterations: int = 200,
    initial_bankroll_yen: int = INITIAL_BANKROLL_YEN,
    stake_unit_yen: int = STAKE_UNIT_YEN,
    max_snapshot_age_seconds: float = MAX_T5_AGE_SECONDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    artifact_output: Path | None = None,
) -> dict[str, Any]:
    """Fit V33-LPA and run a production-equivalent, unseen T-5 backtest."""
    if initial_bankroll_yen != INITIAL_BANKROLL_YEN:
        raise ValueError("formal V33-LPA evaluation requires JPY10000 per day")
    if stake_unit_yen != STAKE_UNIT_YEN:
        raise ValueError("formal V33-LPA evaluation requires JPY100 units")
    training = _ordered_races(data.training_races, label="training")
    outer = _ordered_races(data.outer_races, label="outer")
    if training[-1].decision.race_date >= outer[0].decision.race_date:
        raise ValueError("outer period must be completely unseen and after training")
    indexed = _settlement_index(
        data,
        settlements,
        (*training, *outer),
        max_snapshot_age_seconds=max_snapshot_age_seconds,
    )
    official_training_payouts = _training_payouts(training, indexed)
    base_training, lpa_teacher = _split_training(
        training,
        base_training_fraction=base_training_fraction,
        minimum_base_training_dates=minimum_base_training_dates,
        minimum_lpa_teacher_dates=minimum_lpa_teacher_dates,
    )
    frozen_base = fit_four_head_nested_v22(
        base_training, **dict(base_fit_kwargs or {})
    )
    base_trained_through = base_training[-1].decision.race_date
    teacher_predictions = tuple(
        predict_race(frozen_base, race.decision) for race in lpa_teacher
    )
    lpa_artifact = fit_learned_allocation_head(
        lpa_teacher,
        teacher_predictions,
        official_training_payouts,
        base_predictions_trained_through_date=base_trained_through,
        configs=tuple(allocation_configs),
        validation_fraction=allocation_validation_fraction,
        max_iterations=allocation_max_iterations,
    )

    # Refit only base heads on all training. Outer labels/results stay inaccessible.
    final_base = fit_four_head_nested_v22(training, **dict(base_fit_kwargs or {}))
    outer_predictions = tuple(
        predict_race(final_base, race.decision) for race in outer
    )
    common = {
        "initial_bankroll_yen": initial_bankroll_yen,
        "stake_unit_yen": stake_unit_yen,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    formal = _run_outer(
        outer,
        outer_predictions,
        indexed,
        lpa_artifact,
        payout_multiplier=1.0,
        **common,
    )
    stress = _run_outer(
        outer,
        outer_predictions,
        indexed,
        lpa_artifact,
        payout_multiplier=ODDS_STRESS_MULTIPLIER,
        **common,
    )
    gates = _promotion_gate(formal, stress)
    artifact_bundle = {
        "schema_version": 1,
        "model_key": MODEL_KEY,
        "trained_through_date": training[-1].decision.race_date,
        "base_model": final_base,
        "allocation_model": lpa_artifact,
        "information_boundary": {
            "base_predictions_fixed_before_lpa_teacher": True,
            "outer_outcomes_used_for_fit_or_selection": False,
            "decision_odds": "complete_official_trifecta_snapshot_at_T-5",
        },
    }
    artifact_record: dict[str, Any] = {
        "ready": False,
        "path": None,
        "sha256": None,
    }
    if artifact_output is not None:
        destination = Path(artifact_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            joblib.dump(artifact_bundle, temporary, compress=3)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        with destination.open("rb") as artifact_file:
            artifact_record = {
                "ready": True,
                "path": str(destination),
                "sha256": hashlib.file_digest(artifact_file, "sha256").hexdigest(),
            }
    return {
        "model_key": MODEL_KEY,
        "evaluation_role": "formal_production_equivalent_t5_bankroll_outer_only",
        "training_split": {
            "base_head_from": base_training[0].decision.race_date,
            "base_head_through": base_training[-1].decision.race_date,
            "base_head_races": len(base_training),
            "lpa_teacher_from": lpa_teacher[0].decision.race_date,
            "lpa_teacher_through": lpa_teacher[-1].decision.race_date,
            "lpa_teacher_races": len(lpa_teacher),
            "final_base_head_from": training[0].decision.race_date,
            "final_base_head_through": training[-1].decision.race_date,
            "final_base_head_races": len(training),
        },
        "outer_period": {
            "from": outer[0].decision.race_date,
            "through": outer[-1].decision.race_date,
            "races": len(outer),
        },
        "evaluation_from": outer[0].decision.race_date,
        "evaluation_through": outer[-1].decision.race_date,
        "teacher": {
            "name": lpa_artifact.teacher,
            "official_payout_source": "DB official trifecta payout_yen",
            "official_payout_races": len(official_training_payouts),
            "closing_odds_used_as_payout_teacher": False,
        },
        "information_boundary": {
            "base_predictions_fixed_before_lpa_teacher": True,
            "final_base_refit_on_all_training": True,
            "outer_outcomes_used_for_fit_or_selection": False,
            "outer_official_payout_read_at_settlement_only": True,
            "decision_odds": "complete_official_trifecta_snapshot_at_T-5",
        },
        "frozen_base_artifact_sha256": _digest(frozen_base),
        "final_base_artifact_sha256": _digest(final_base),
        "allocation_artifact_sha256": _digest(lpa_artifact),
        "frozen_outer_prediction_sha256": _prediction_digest(outer_predictions),
        "allocation_artifact": lpa_artifact.summary(),
        "deployment_artifact": artifact_record,
        "bankroll": formal,
        **{key: value for key, value in formal.items() if key != "daily"},
        "daily": formal["daily"],
        "odds_stress_5pct": {
            key: value for key, value in stress.items() if key != "daily"
        },
        "promotion_gate": gates,
        "promotion_eligible": all(gates.values()),
        "policy": {
            "initial_bankroll_yen_per_day": initial_bankroll_yen,
            "stake_granularity_yen": stake_unit_yen,
            "zero_or_more_tickets_per_race": True,
            "profit_reinvestment": True,
            "pending_stakes_reduce_available_cash": True,
            "decisions_ordered_by": "decision_at_then_race_id",
            "returns_released_by": "result_available_at_then_race_id",
            "official_payout_role": "settlement_only",
            "stress_official_payout_multiplier": ODDS_STRESS_MULTIPLIER,
        },
        "real_betting_enabled": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe T-5 evaluation for V33 learned allocation"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--training-from", required=True)
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--outer-from", required=True)
    parser.add_argument("--outer-through", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--data-cache")
    parser.add_argument("--max-races-per-day", type=int)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=300.0)
    parser.add_argument(
        "--projection-dimensions", type=int, default=DEFAULT_PROJECTION_DIMENSIONS
    )
    parser.add_argument("--base-training-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-base-training-dates", type=int, default=5)
    parser.add_argument("--minimum-lpa-teacher-dates", type=int, default=4)
    parser.add_argument("--minimum-inner-training-dates", type=int, default=2)
    parser.add_argument("--minimum-purchase-training-dates", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--allocation-validation-fraction", type=float, default=0.25)
    parser.add_argument("--allocation-max-iterations", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = Path(args.source_model)
    source_artifact = joblib.load(source_path)
    with source_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    cache_signature = {
        "source_sha256": source_sha256,
        "training_from": args.training_from,
        "training_through": args.training_through,
        "outer_from": args.outer_from,
        "outer_through": args.outer_through,
        "max_snapshot_age_seconds": args.max_snapshot_age_seconds,
        "projection_dimensions": args.projection_dimensions,
        "max_races_per_day": args.max_races_per_day,
    }
    from ..db import connection

    with connection(args.db) as conn:
        cache_path = Path(args.data_cache) if args.data_cache else None
        if cache_path is not None and cache_path.is_file():
            data = load_v22_evaluation_cache_compat(
                cache_path, expected_signature=cache_signature
            )
        else:
            data = load_v22_evaluation_data(
                conn,
                source_artifact=source_artifact,
                training_from_date=args.training_from,
                training_through_date=args.training_through,
                outer_from_date=args.outer_from,
                outer_through_date=args.outer_through,
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
                projection_dimensions=args.projection_dimensions,
                max_races_per_day=args.max_races_per_day,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(f".{cache_path.name}.tmp")
                try:
                    joblib.dump(
                        {"schema_version": 1, "signature": cache_signature, "data": data},
                        temporary,
                        compress=3,
                    )
                    temporary.replace(cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
        all_races = (*data.training_races, *data.outer_races)
        settlements = _build_outer_settlements(
            conn, all_races, data.decision_audit
        )
        result = evaluate_learned_purchase_allocation_v33(
            data,
            settlements,
            base_training_fraction=args.base_training_fraction,
            minimum_base_training_dates=args.minimum_base_training_dates,
            minimum_lpa_teacher_dates=args.minimum_lpa_teacher_dates,
            base_fit_kwargs={
                "minimum_inner_training_dates": args.minimum_inner_training_dates,
                "minimum_purchase_training_dates": args.minimum_purchase_training_dates,
                "alpha": args.alpha,
            },
            allocation_validation_fraction=args.allocation_validation_fraction,
            allocation_max_iterations=args.allocation_max_iterations,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            bootstrap_samples=args.bootstrap_samples,
            artifact_output=Path(args.model_output),
        )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    return 0


__all__ = [
    "evaluate_learned_purchase_allocation_v33",
    "load_v22_evaluation_cache_compat",
]


if __name__ == "__main__":
    raise SystemExit(main())
