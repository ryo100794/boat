from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from boatrace_ai.listwise import market_calibration


CALIBRATOR = {"model_weight": 0.75, "temperature": 1.0}
METRIC_NAMES = (
    "winner_log_loss",
    "winner_top1_accuracy",
    "trifecta_log_loss",
    "trifecta_top5_hit_rate",
)


def _race(race_date: str, *, actual: str = "1-2-3") -> dict[str, Any]:
    probabilities = {"1-2-3": 0.50, "1-3-2": 0.30, "2-1-3": 0.20}
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "actual_combination": actual,
        "actual_payout_yen": 800,
        "model_probabilities": probabilities,
        "market_probabilities": dict(probabilities),
        "odds": {combination: 4.0 for combination in probabilities},
        "historical_return_multipliers": {},
        "snapshot_id": race_date,
    }


def _races(days: int) -> list[dict[str, Any]]:
    start = date(2026, 1, 1)
    return [_race((start + timedelta(days=index)).isoformat()) for index in range(days)]


def _empty_bankroll(races: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    return {
        "tickets": 0,
        "hit_tickets": 0,
        "stake_yen": 0,
        "return_yen": 0,
        "profit_yen": 0,
        "roi": None,
        "daily": [
            {
                "race_date": race_date,
                "tickets": 0,
                "hit_tickets": 0,
                "stake_yen": 0,
                "return_yen": 0,
                "profit_yen": 0,
            }
            for race_date in dates
        ],
    }


def _stub_unrelated_evaluation(monkeypatch) -> None:
    metrics = {
        f"{source}_{metric}": 0.0
        for source in ("model", "market", "calibrated")
        for metric in METRIC_NAMES
    }
    monkeypatch.setattr(
        market_calibration,
        "select_calibrator",
        lambda races: (dict(CALIBRATOR), []),
    )
    monkeypatch.setattr(
        market_calibration,
        "select_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )
    monkeypatch.setattr(
        market_calibration,
        "simulate_policy",
        lambda races, **kwargs: _empty_bankroll(races),
    )
    monkeypatch.setattr(
        market_calibration,
        "select_flat_policy",
        lambda *args, **kwargs: ({"name": "no_bet", "no_bet": True}, []),
    )
    monkeypatch.setattr(
        market_calibration,
        "simulate_flat_policy",
        lambda races, **kwargs: {
            **_empty_bankroll(races),
            "daily": [
                {**row, "hits": row.pop("hit_tickets")}
                for row in _empty_bankroll(races)["daily"]
            ],
        },
    )
    monkeypatch.setattr(
        market_calibration,
        "probability_metrics",
        lambda *args, **kwargs: dict(metrics),
    )
    monkeypatch.setattr(
        market_calibration,
        "paired_market_differences",
        lambda races, **kwargs: ([0.0] * len(races), [0.0] * len(races)),
    )
    monkeypatch.setattr(
        market_calibration,
        "market_comparison_confidence",
        lambda *args, **kwargs: {"confidence_pass": False},
    )
    monkeypatch.setattr(market_calibration, "edge_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        market_calibration, "summarize_edge_records", lambda records: {}
    )
    monkeypatch.setattr(
        market_calibration, "predefined_ticket_diagnostics", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        market_calibration, "summarize_policy_candidates", lambda rows: {}
    )
    monkeypatch.setattr(
        market_calibration, "summarize_flat_candidates", lambda rows: {}
    )
    monkeypatch.setattr(
        market_calibration,
        "fit_deployment_configuration",
        lambda *args, **kwargs: {
            "selected_policy": {"name": "no_bet", "no_bet": True}
        },
    )
    monkeypatch.setattr(
        market_calibration, "bankroll_reliability_metrics", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        market_calibration, "evaluate_closing_odds_quantiles", lambda races: {}
    )


@dataclass(frozen=True)
class _Artifact:
    training_dates: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return len(self.training_dates) >= 30

    @property
    def trained_through_date(self) -> str | None:
        return self.training_dates[-1] if self.training_dates else None

    @property
    def training_days(self) -> int:
        return len(self.training_dates)

    @property
    def tickets(self) -> int:
        return len(self.training_dates)

    @property
    def candidate_days(self) -> int:
        return len(self.training_dates)

    @property
    def ready_reasons(self) -> tuple[str, ...]:
        return () if self.ready else ("insufficient_training_days",)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "training_days": len(self.training_dates),
            "trained_through_date": self.trained_through_date,
        }


def _run_with_empirical_spies(monkeypatch, races):
    _stub_unrelated_evaluation(monkeypatch)
    events: list[dict[str, Any]] = []

    def fit(records):
        training_dates = tuple(sorted({str(row["race_date"]) for row in records}))
        events.append({"kind": "fit", "training_dates": training_dates})
        return _Artifact(training_dates)

    def simulate(holdout, calibrator, probability_blender, artifact, daily_budget_yen):
        evaluation_date = str(holdout[0]["race_date"])
        events.append(
            {
                "kind": "simulate",
                "evaluation_date": evaluation_date,
                "training_dates": artifact.training_dates,
                "actual": str(holdout[0]["actual_combination"]),
            }
        )
        bankroll = _empty_bankroll(holdout)
        return {
            **bankroll,
            "status": "ready" if artifact.ready else "calibration_not_ready",
            "calibration": artifact.as_dict(),
            "evaluation_days": 1,
            "evaluated_races": len(holdout),
            "eligible_days": 0,
            "no_bet_days": 1,
            "eligible_tickets": 0,
        }

    def teacher(holdout, calibrator, probability_blender):
        evaluation_date = str(holdout[0]["race_date"])
        events.append(
            {
                "kind": "append_teacher",
                "evaluation_date": evaluation_date,
                "actual": str(holdout[0]["actual_combination"]),
            }
        )
        return [
            {
                "race_date": evaluation_date,
                "raw_estimated_ev": 1.10,
                "gross_return_per_yen": (
                    8.0 if holdout[0]["actual_combination"] == "1-2-3" else 0.0
                ),
            }
        ]

    monkeypatch.setattr(
        market_calibration, "fit_empirical_ev_calibration", fit, raising=False
    )
    monkeypatch.setattr(
        market_calibration, "simulate_empirical_lcb_policy", simulate, raising=False
    )
    monkeypatch.setattr(
        market_calibration, "policy_edge_records", teacher, raising=False
    )
    result = market_calibration.walk_forward_evaluate(
        races,
        min_calibration_days=2,
        calibrator_strategy="grid",
    )
    return result, events


def test_empirical_track_fits_prior_evaluation_folds_then_appends_holdout_teacher(
    monkeypatch,
) -> None:
    result, events = _run_with_empirical_spies(monkeypatch, _races(5))

    assert [event["kind"] for event in events] == [
        "fit",
        "simulate",
        "append_teacher",
    ] * 3
    evaluation_dates = ["2026-01-03", "2026-01-04", "2026-01-05"]
    for fold_index, evaluation_date in enumerate(evaluation_dates):
        fit_event, simulation, append = events[fold_index * 3 : fold_index * 3 + 3]
        assert fit_event["training_dates"] == tuple(evaluation_dates[:fold_index])
        assert simulation["training_dates"] == fit_event["training_dates"]
        assert simulation["evaluation_date"] == evaluation_date
        assert append["evaluation_date"] == evaluation_date
        assert all(day < evaluation_date for day in fit_event["training_dates"])

    audits = result["empirical_lcb_walk_forward"]["folds"]
    assert [audit["evaluation_date"] for audit in audits] == evaluation_dates
    for audit in audits:
        trained_through = audit["trained_through_date"]
        assert trained_through is None or trained_through < audit["evaluation_date"]


def test_empirical_track_stays_not_ready_and_places_no_bets_before_30_training_days(
    monkeypatch,
) -> None:
    result, _events = _run_with_empirical_spies(monkeypatch, _races(12))

    track = result["empirical_lcb_walk_forward"]
    assert track["status"] == "calibration_not_ready"
    assert track["tickets"] == 0
    assert track["stake_yen"] == 0
    assert track["return_yen"] == 0
    assert track["profit_yen"] == 0
    assert all(fold["calibration_ready"] is False for fold in track["folds"])
    assert max(fold["training_days"] for fold in track["folds"]) < 30


def test_future_actual_result_cannot_change_earlier_empirical_fold_audit(
    monkeypatch,
) -> None:
    races = _races(6)
    baseline, _ = _run_with_empirical_spies(monkeypatch, deepcopy(races))
    changed_races = deepcopy(races)
    changed_races[-1]["actual_combination"] = "2-1-3"
    changed, _ = _run_with_empirical_spies(monkeypatch, changed_races)

    future_date = str(changed_races[-1]["race_date"])
    baseline_earlier = [
        fold
        for fold in baseline["empirical_lcb_walk_forward"]["folds"]
        if fold["evaluation_date"] < future_date
    ]
    changed_earlier = [
        fold
        for fold in changed["empirical_lcb_walk_forward"]["folds"]
        if fold["evaluation_date"] < future_date
    ]
    assert changed_earlier == baseline_earlier
