from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from ..bankroll_bootstrap import bootstrap_daily_roi
from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .course_interaction_residual import (
    fit_structure_residual,
    structure_metrics,
    structure_probabilities,
)
from .empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .market_calibration import blend_probabilities
from .pruned_direct_context_v27 import (
    FEATURE_VARIANTS,
    _lane_context_matrix,
)


MODEL_NAME = "ticket_utility_meta_ranking_v31"
PROBABILITY_STRUCTURE = "shared_independent_core"
PROBABILITY_REGULARIZATION = 0.03
POLICY_CALIBRATION_DAYS = 30
STAKE_YEN = 100
TOP_K_CHOICES = (1, 3, 5)
LABEL_SCHEMES = (
    "winner",
    "gross_return_poisson_c50",
    "gross_return_poisson_c100",
)
TREE_PRESETS = (
    {"name": "compact", "num_leaves": 15, "max_depth": 5},
    {"name": "balanced", "num_leaves": 31, "max_depth": 7},
)
MIN_RACE_WEIGHT = 0.5
MAX_RACE_WEIGHT = 4.0
EPSILON = 1e-12
ACTIVE_CONTEXT_FEATURES = FEATURE_VARIANTS["independent_core"]
_BOOSTER_CACHE: dict[str, Any] = {}


def _lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("V31 requires the gbdt optional dependency") from exc
    return lightgbm


def _rank(values: np.ndarray, combinations: list[str]) -> np.ndarray:
    order = sorted(
        range(len(combinations)),
        key=lambda index: (-float(values[index]), combinations[index]),
    )
    result = np.empty(len(order), dtype=np.float32)
    for rank, index in enumerate(order, start=1):
        result[index] = float(rank)
    return result


def _entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def ticket_feature_matrix(
    race: Mapping[str, Any],
) -> tuple[list[str], np.ndarray]:
    market_source = race.get("market_probabilities")
    model_source = race.get("model_probabilities")
    odds_source = race.get("odds")
    if not all(
        isinstance(source, Mapping)
        for source in (market_source, model_source, odds_source)
    ):
        raise ValueError("V31 race requires model, market, and odds mappings")
    combinations = sorted(str(key) for key in market_source)
    if len(combinations) < 2:
        raise ValueError("V31 race requires at least two combinations")
    market = np.asarray(
        [max(EPSILON, float(market_source[key])) for key in combinations],
        dtype=np.float64,
    )
    model = np.asarray(
        [max(EPSILON, float(model_source.get(key, EPSILON))) for key in combinations],
        dtype=np.float64,
    )
    odds = np.asarray(
        [max(EPSILON, float(odds_source[key])) for key in combinations],
        dtype=np.float64,
    )
    market /= float(np.sum(market))
    model /= float(np.sum(model))
    count = len(combinations)
    market_rank = _rank(market, combinations) / count
    model_rank = _rank(model, combinations) / count
    lanes = np.asarray(
        [[int(value) for value in combination.split("-")] for combination in combinations],
        dtype=np.int8,
    )
    if lanes.shape != (count, 3):
        raise ValueError("V31 combinations must contain three lanes")

    columns = [
        np.log(model),
        np.log(market),
        np.log(odds),
        np.log(model / market),
        model * count,
        market * count,
        model_rank,
        market_rank,
        model_rank - market_rank,
        (lanes[:, 0] - 3.5) / 2.5,
        (lanes[:, 1] - 3.5) / 2.5,
        (lanes[:, 2] - 3.5) / 2.5,
    ]
    for stage in range(3):
        for lane in range(1, 7):
            columns.append((lanes[:, stage] == lane).astype(np.float64))

    lane_context = _lane_context_matrix(race, ACTIVE_CONTEXT_FEATURES)
    for stage in range(3):
        stage_rows = lane_context[lanes[:, stage] - 1]
        columns.extend(stage_rows[:, index] for index in range(stage_rows.shape[1]))

    try:
        jcd = int(str(race.get("jcd") or "0"))
    except ValueError:
        jcd = 0
    rno = int(race.get("rno") or 0)
    for value in range(1, 25):
        columns.append(np.full(count, float(jcd == value)))
    for value in range(1, 13):
        columns.append(np.full(count, float(rno == value)))

    market_sorted = np.sort(market)[::-1]
    model_sorted = np.sort(model)[::-1]
    race_globals = (
        _entropy(market),
        _entropy(model),
        float(market_sorted[0] - market_sorted[1]),
        float(model_sorted[0] - model_sorted[1]),
    )
    columns.extend(np.full(count, value) for value in race_globals)
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("V31 feature matrix contains non-finite values")
    return combinations, matrix


def _ranking_teacher_weights(
    races: list[dict[str, Any]], label_scheme: str
) -> np.ndarray:
    if label_scheme == "winner":
        return np.ones(len(races), dtype=np.float64)
    if label_scheme != "payout_weighted":
        raise ValueError(f"unknown V31 label scheme: {label_scheme}")
    payout_odds = np.asarray(
        [
            max(EPSILON, float(race["actual_payout_yen"]) / STAKE_YEN)
            for race in races
        ],
        dtype=np.float64,
    )
    median_odds = max(EPSILON, float(np.median(payout_odds)))
    raw_weights = np.sqrt(payout_odds / median_odds)
    clipped = np.clip(raw_weights, MIN_RACE_WEIGHT, MAX_RACE_WEIGHT)
    return clipped / float(np.mean(clipped))


def _gross_return_cap(label_scheme: str) -> float | None:
    prefix = "gross_return_poisson_c"
    if not label_scheme.startswith(prefix):
        return None
    try:
        cap = float(label_scheme.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"invalid V31 gross-return scheme: {label_scheme}") from exc
    if not math.isfinite(cap) or cap <= 1.0:
        raise ValueError(f"invalid V31 gross-return cap: {cap}")
    return cap


def fit_ticket_utility_ranker(
    races: list[dict[str, Any]],
    *,
    label_scheme: str,
    tree_preset: Mapping[str, Any],
    num_threads: int = 4,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one V31 race is required")
    gross_return_cap = _gross_return_cap(label_scheme)
    race_weights = (
        np.ones(len(races), dtype=np.float64)
        if gross_return_cap is not None
        else _ranking_teacher_weights(races, label_scheme)
    )
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[int] = []
    sample_weights: list[np.ndarray] = []
    for race, race_weight in zip(races, race_weights):
        combinations, matrix = ticket_feature_matrix(race)
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError("actual V31 combination is absent from market tickets")
        if gross_return_cap is None:
            relevance = np.zeros(len(combinations), dtype=np.int32)
            relevance[combinations.index(actual)] = 1
        else:
            relevance = np.zeros(len(combinations), dtype=np.float64)
            realized = float(race["actual_payout_yen"]) / STAKE_YEN
            relevance[combinations.index(actual)] = min(
                gross_return_cap, realized
            )
        matrices.append(matrix)
        labels.append(relevance)
        sample_weights.append(
            np.full(len(combinations), race_weight, dtype=np.float64)
        )
        groups.append(len(combinations))
    features = np.vstack(matrices)
    target = np.concatenate(labels)
    ticket_weights = np.concatenate(sample_weights)
    lightgbm = _lightgbm()
    common_parameters = dict(
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=int(tree_preset["num_leaves"]),
        max_depth=int(tree_preset["max_depth"]),
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=5.0,
        max_bin=127,
        random_state=20260731,
        n_jobs=max(1, int(num_threads)),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    if gross_return_cap is None:
        estimator = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1, 3, 7, 15],
            **common_parameters,
        )
        estimator.fit(
            features,
            target,
            group=groups,
            sample_weight=ticket_weights,
        )
        learner_objective = "lambdarank_winner"
    else:
        estimator = lightgbm.LGBMRegressor(
            objective="poisson",
            metric="poisson",
            max_delta_step=0.7,
            **common_parameters,
        )
        estimator.fit(features, target, sample_weight=ticket_weights)
        learner_objective = "poisson_expected_gross_return"
    model_text = estimator.booster_.model_to_string()
    return {
        "model": MODEL_NAME,
        "role": "ticket_utility_ranking_only",
        "label_scheme": label_scheme,
        "learner_objective": learner_objective,
        "gross_return_cap": gross_return_cap,
        "tree_preset": str(tree_preset["name"]),
        "teacher_weighting": label_scheme,
        "race_weight_minimum": float(np.min(race_weights)),
        "race_weight_maximum": float(np.max(race_weights)),
        "race_weight_mean": float(np.mean(race_weights)),
        "race_weight_normalized": True,
        "num_leaves": int(tree_preset["num_leaves"]),
        "max_depth": int(tree_preset["max_depth"]),
        "feature_dimension": int(features.shape[1]),
        "training_races": len(races),
        "training_tickets": int(features.shape[0]),
        "booster_model": model_text,
        "booster_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
    }


def _booster(artifact: Mapping[str, Any]) -> Any:
    digest = str(artifact["booster_sha256"])
    cached = _BOOSTER_CACHE.get(digest)
    if cached is None:
        model_text = str(artifact["booster_model"])
        if hashlib.sha256(model_text.encode()).hexdigest() != digest:
            raise ValueError("V31 booster digest mismatch")
        cached = _lightgbm().Booster(model_str=model_text)
        _BOOSTER_CACHE[digest] = cached
    return cached


def ticket_ranking(
    race: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[str]:
    combinations, matrix = ticket_feature_matrix(race)
    scores = np.asarray(_booster(artifact).predict(matrix), dtype=np.float64)
    if scores.shape != (len(combinations),) or not np.all(np.isfinite(scores)):
        raise ValueError("V31 booster returned invalid ranking scores")
    score_by_combination = dict(zip(combinations, scores))
    return sorted(
        combinations,
        key=lambda combination: (-float(score_by_combination[combination]), combination),
    )


def _ranking_provider(artifact: Mapping[str, Any]):
    def provider(
        race: Mapping[str, Any], _probabilities: Mapping[str, float]
    ) -> list[str]:
        return ticket_ranking(race, artifact)

    return provider


def ticket_ranking_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    totals = {
        top_k: {"hits": 0, "stake_yen": 0, "return_yen": 0}
        for top_k in TOP_K_CHOICES
    }
    daily: dict[int, dict[str, dict[str, int]]] = {
        top_k: defaultdict(lambda: {"stake_yen": 0, "return_yen": 0})
        for top_k in TOP_K_CHOICES
    }
    for race in races:
        ranked = ticket_ranking(race, artifact)
        actual = str(race["actual_combination"])
        payout = int(race["actual_payout_yen"])
        race_date = str(race["race_date"])
        for top_k in TOP_K_CHOICES:
            stake = top_k * STAKE_YEN
            hit = actual in ranked[:top_k]
            totals[top_k]["hits"] += int(hit)
            totals[top_k]["stake_yen"] += stake
            totals[top_k]["return_yen"] += payout if hit else 0
            daily[top_k][race_date]["stake_yen"] += stake
            daily[top_k][race_date]["return_yen"] += payout if hit else 0
    by_top_k: dict[str, Any] = {}
    for top_k in TOP_K_CHOICES:
        values = totals[top_k]
        stake = values["stake_yen"]
        returned = values["return_yen"]
        daily_rows = [
            {"race_date": day, **amounts}
            for day, amounts in sorted(daily[top_k].items())
        ]
        confidence = bootstrap_daily_roi(daily_rows, samples=2_000, seed=20260731)
        by_top_k[str(top_k)] = {
            "top_k": top_k,
            "evaluated_races": len(races),
            "hit_races": values["hits"],
            "hit_rate": values["hits"] / len(races) if races else None,
            "stake_yen": stake,
            "return_yen": returned,
            "profit_yen": returned - stake,
            "roi": returned / stake if stake else None,
            "roi_ci95_lower": confidence.get("roi_ci95_lower"),
            "probability_roi_above_one": confidence.get(
                "probability_roi_above_one"
            ),
        }
    return {"evaluated_races": len(races), "by_top_k": by_top_k}


def _replace_probability_head(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**race, "model_probabilities": structure_probabilities(race, artifact)}
        for race in races
    ]


def _candidate_score(candidate: Mapping[str, Any]) -> tuple[float, float, float, int]:
    metrics = candidate["selected_top_k_metrics"]
    return (
        float(metrics.get("roi_ci95_lower") or 0.0),
        float(metrics.get("roi") or 0.0),
        float(metrics.get("hit_rate") or 0.0),
        -int(candidate["top_k"]),
    )


def evaluate_temporal_ticket_utility_roles(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    policy_calibration_days: int = POLICY_CALIBRATION_DAYS,
    label_schemes: Iterable[str] = LABEL_SCHEMES,
    tree_presets: Iterable[Mapping[str, Any]] = TREE_PRESETS,
    probability_artifact: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    minimum_days = int(policy_calibration_days) + 10
    if len(dates) < minimum_days:
        return {
            "model": MODEL_NAME,
            "status": "insufficient_calibration_days",
            "calibration_days": len(dates),
            "required_calibration_days": minimum_days,
        }
    policy_dates = set(dates[-int(policy_calibration_days) :])
    ranking_dates = dates[: -int(policy_calibration_days)]
    ranking_date_set = set(ranking_dates)
    ranking_races = [
        race for race in calibration if str(race["race_date"]) in ranking_date_set
    ]
    policy_races = [
        race for race in calibration if str(race["race_date"]) in policy_dates
    ]
    split_index = max(1, min(len(ranking_dates) - 1, int(len(ranking_dates) * 0.8)))
    inner_fit_dates = set(ranking_dates[:split_index])
    inner_validation_dates = set(ranking_dates[split_index:])
    inner_fit = [
        race for race in ranking_races if str(race["race_date"]) in inner_fit_dates
    ]
    inner_validation = [
        race
        for race in ranking_races
        if str(race["race_date"]) in inner_validation_dates
    ]

    normalized_label_schemes = tuple(str(value) for value in label_schemes)
    normalized_tree_presets = tuple(dict(value) for value in tree_presets)
    candidates: list[dict[str, Any]] = []
    for label_scheme in normalized_label_schemes:
        for preset in normalized_tree_presets:
            artifact = fit_ticket_utility_ranker(
                inner_fit, label_scheme=label_scheme, tree_preset=preset
            )
            metrics = ticket_ranking_metrics(inner_validation, artifact)
            for top_k in TOP_K_CHOICES:
                candidates.append({
                    "label_scheme": label_scheme,
                    "tree_preset": str(preset["name"]),
                    "top_k": top_k,
                    "selected_top_k_metrics": metrics["by_top_k"][str(top_k)],
                })
    selected = max(candidates, key=_candidate_score)
    selected_preset = next(
        value
        for value in normalized_tree_presets
        if str(value["name"]) == str(selected["tree_preset"])
    )
    prior_ranking_artifact = fit_ticket_utility_ranker(
        ranking_races,
        label_scheme=str(selected["label_scheme"]),
        tree_preset=selected_preset,
    )
    prior_probability_artifact = fit_structure_residual(
        ranking_races,
        structure_variant=PROBABILITY_STRUCTURE,
        regularization=PROBABILITY_REGULARIZATION,
        max_iterations=200,
    )
    policy_probability_races = _replace_probability_head(
        policy_races, prior_probability_artifact
    )
    policy_records = policy_edge_records(
        policy_probability_races,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
        _ranking_provider(prior_ranking_artifact),
        max_rank=int(selected["top_k"]),
    )
    first_evaluation_date = min(str(race["race_date"]) for race in evaluation)
    empirical_artifact = fit_contextual_empirical_ev_calibration(
        policy_records,
        prediction_date=first_evaluation_date,
        bootstrap_samples=bootstrap_samples,
        min_days=int(policy_calibration_days),
        min_tickets=300,
        min_candidate_days=20,
        min_rank_days=15,
        min_rank_tickets=150,
        min_cell_days=10,
        min_cell_tickets=50,
    )

    final_probability_artifact = dict(probability_artifact or {})
    if not final_probability_artifact:
        final_probability_artifact = fit_structure_residual(
            calibration,
            structure_variant=PROBABILITY_STRUCTURE,
            regularization=PROBABILITY_REGULARIZATION,
            max_iterations=200,
        )
    final_ranking_artifact = fit_ticket_utility_ranker(
        calibration,
        label_scheme=str(selected["label_scheme"]),
        tree_preset=selected_preset,
    )
    evaluation_probability_races = _replace_probability_head(
        evaluation, final_probability_artifact
    )
    bankroll = simulate_empirical_lcb_policy(
        evaluation_probability_races,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
        empirical_artifact,
        daily_budget_yen,
        _ranking_provider(final_ranking_artifact),
        max_rank=int(selected["top_k"]),
    )
    ranking_metrics = ticket_ranking_metrics(evaluation, final_ranking_artifact)
    selected_top_k = str(selected["top_k"])
    ranking_metrics["selected_top_k"] = int(selected["top_k"])
    ranking_metrics["selected_top_k_metrics"] = ranking_metrics["by_top_k"][
        selected_top_k
    ]
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "Ticket-level LightGBM winner ranking and capped Poisson realized-"
            "gross-return heads are selected on an inner prior-day block. The selected "
            "rank cutoff limits both empirical-EV calibration and outer purchases; "
            "a separate proper probability head supplies EV."
        ),
        "ranking_training_from": ranking_dates[0],
        "ranking_training_through": ranking_dates[-1],
        "inner_fit_through": ranking_dates[split_index - 1],
        "inner_validation_from": ranking_dates[split_index],
        "policy_calibration_from": dates[-int(policy_calibration_days)],
        "policy_calibration_through": dates[-1],
        "evaluation_from": first_evaluation_date,
        "evaluation_through": max(str(race["race_date"]) for race in evaluation),
        "selected_candidate": selected,
        "candidates": candidates,
        "probability_artifact": final_probability_artifact,
        "probability_metrics": structure_metrics(
            evaluation, final_probability_artifact
        ),
        "prior_ranking_artifact": prior_ranking_artifact,
        "ranking_artifact": final_ranking_artifact,
        "ranking_metrics": ranking_metrics,
        "empirical_ev_calibration": empirical_artifact.as_dict(),
        "bankroll": bankroll,
        "promotion_eligible": empirical_bankroll_promotion_eligible(bankroll),
    }


__all__ = [
    "evaluate_temporal_ticket_utility_roles",
    "fit_ticket_utility_ranker",
    "ticket_feature_matrix",
    "ticket_ranking",
    "ticket_ranking_metrics",
]
