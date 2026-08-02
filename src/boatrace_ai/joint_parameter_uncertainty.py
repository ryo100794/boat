from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

from .joint_market_value import JointMarketScenario
from .joint_scenario_model import (
    ConditionalJointScenarioModel,
    JointScenarioObservation,
    fit_conditional_joint_scenario_model,
    generate_joint_market_scenarios,
)


@dataclass(frozen=True)
class JointParameterDraw:
    draw_index: int
    model: ConditionalJointScenarioModel
    sampled_days: tuple[str, ...]
    unique_days: int
    manifest_sha256: str


def _iso_date(value: object, name: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _manifest(
    *,
    draw_index: int,
    decision_date: str,
    sampled_days: Sequence[str],
    observations: Sequence[JointScenarioObservation],
    expected_outcomes: Sequence[str] | None,
    fit_options: Mapping[str, Any],
    scale_selection_seed: int,
) -> str:
    payload = {
        "draw_index": draw_index,
        "decision_date": decision_date,
        "sampled_days": list(sampled_days),
        "expected_outcomes": (
            list(expected_outcomes) if expected_outcomes is not None else None
        ),
        "fit_options": dict(fit_options),
        "scale_selection_seed": scale_selection_seed,
        "samples": [
            [
                row.race_id,
                row.race_date,
                row.resample_key,
                row.terminal_probability_prediction_sha256,
                row.terminal_probability_outcome_schema_sha256,
            ]
            for row in observations
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bootstrap_joint_parameter_models(
    observations: Sequence[JointScenarioObservation],
    *,
    decision_date: str,
    draws: int = 20,
    seed: int = 33039,
    expected_outcomes: Sequence[str] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> list[JointParameterDraw]:
    """Resample complete training days and refit one model per outer draw."""
    cutoff = _iso_date(decision_date, "decision_date")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if len(observations) < 3:
        raise ValueError("at least three observations are required")
    by_day: dict[str, list[JointScenarioObservation]] = {}
    for row in observations:
        race_date = _iso_date(row.race_date, "race_date")
        if race_date >= cutoff:
            raise ValueError("parameter draws require strictly prior observations")
        by_day.setdefault(race_date, []).append(row)
    days = tuple(sorted(by_day))
    if len(days) < 2:
        raise ValueError("at least two strictly prior training days are required")
    options = dict(fit_options or {})
    forbidden = {"expected_outcomes", "scale_selection_seed"} & set(options)
    if forbidden:
        raise ValueError(
            "fit_options cannot override controlled fields: "
            + ", ".join(sorted(forbidden))
        )
    rng = random.Random(seed)
    result = []
    for draw_index in range(draws):
        sampled_days = tuple(rng.choice(days) for _ in days)
        sampled_rows = []
        for block_index, sampled_day in enumerate(sampled_days):
            for row in by_day[sampled_day]:
                sampled_rows.append(replace(
                    row,
                    resample_key=(
                        f"day-bootstrap-{draw_index:03d}-{block_index:03d}"
                    ),
                ))
        model = fit_conditional_joint_scenario_model(
            sampled_rows,
            expected_outcomes=expected_outcomes,
            scale_selection_seed=seed + draw_index,
            **options,
        )
        manifest = _manifest(
            draw_index=draw_index,
            decision_date=cutoff,
            sampled_days=sampled_days,
            observations=sampled_rows,
            expected_outcomes=expected_outcomes,
            fit_options=options,
            scale_selection_seed=seed + draw_index,
        )
        result.append(JointParameterDraw(
            draw_index=draw_index,
            model=model,
            sampled_days=sampled_days,
            unique_days=len(set(sampled_days)),
            manifest_sha256=manifest,
        ))
    return result


def generate_parameter_path_draws(
    parameter_draws: Sequence[JointParameterDraw],
    *,
    decision_probabilities: Mapping[str, float],
    decision_market_shares: Mapping[str, float],
    venue: str,
    decision_horizon_seconds: int,
    popularity_band: str,
    scenarios_per_draw: int = 256,
    seed: int = 33040,
) -> list[list[JointMarketScenario]]:
    """Generate inner future paths from each independently refitted draw."""
    if not parameter_draws:
        raise ValueError("parameter_draws must not be empty")
    if isinstance(scenarios_per_draw, bool) or not isinstance(
        scenarios_per_draw, int
    ) or scenarios_per_draw < 1:
        raise ValueError("scenarios_per_draw must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    outcomes = parameter_draws[0].model.outcomes
    if any(draw.model.outcomes != outcomes for draw in parameter_draws):
        raise ValueError("all parameter draws must use the same outcomes")
    result = []
    for draw in parameter_draws:
        generated = generate_joint_market_scenarios(
            draw.model,
            decision_probabilities=decision_probabilities,
            decision_market_shares=decision_market_shares,
            venue=venue,
            decision_horizon_seconds=decision_horizon_seconds,
            popularity_band=popularity_band,
            scenarios=scenarios_per_draw,
            seed=seed + draw.draw_index,
        )
        result.append([
            JointMarketScenario(
                probabilities=scenario.probabilities,
                market_state={
                    **scenario.market_state,
                    "parameter_draw_index": draw.draw_index,
                    "parameter_draw_manifest_sha256": draw.manifest_sha256,
                },
                weight=scenario.weight,
            )
            for scenario in generated
        ])
    return result


__all__ = [
    "JointParameterDraw",
    "bootstrap_joint_parameter_models",
    "generate_parameter_path_draws",
]
