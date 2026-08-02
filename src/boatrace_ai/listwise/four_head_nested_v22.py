from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, PoissonRegressor, TweedieRegressor


MODEL_KEY = "four_head_nested_v22"
ARTIFACT_VERSION = 3


@dataclass(frozen=True)
class DecisionRace:
    """Information available when a purchase decision is made."""

    race_id: str
    race_date: str
    features: tuple[tuple[float, ...], ...]
    current_odds: tuple[float, ...]


@dataclass(frozen=True)
class RaceOutcome:
    """Settlement-only fields, deliberately separate from DecisionRace."""

    winner_index: int
    closing_odds: tuple[float, ...]
    ranking_order: tuple[int, ...]


@dataclass(frozen=True)
class LabeledRace:
    decision: DecisionRace
    outcome: RaceOutcome


@dataclass(frozen=True)
class LinearHead:
    name: str
    teacher: str
    coefficients: tuple[float, ...]
    intercept: float


@dataclass(frozen=True)
class InnerOOFFold:
    validation_date: str
    trained_through_date: str
    training_race_ids: tuple[str, ...]
    validation_race_ids: tuple[str, ...]


@dataclass(frozen=True)
class PurchaseOOFFold:
    validation_date: str
    trained_through_date: str
    training_base_oof_race_ids: tuple[str, ...]
    validation_race_ids: tuple[str, ...]


@dataclass(frozen=True)
class FourHeadArtifact:
    model_key: str
    artifact_version: int
    trained_through_date: str
    choice_count: int
    feature_count: int
    probability_head: LinearHead
    ranking_head: LinearHead
    closing_odds_head: LinearHead
    purchase_head: LinearHead
    purchase_threshold: float
    inner_oof_folds: tuple[InnerOOFFold, ...]
    inner_oof_race_ids: tuple[str, ...]
    inner_oof_prediction_sha256: str
    purchase_oof_folds: tuple[PurchaseOOFFold, ...]
    purchase_oof_race_ids: tuple[str, ...]
    purchase_oof_score_sha256: str
    purchase_threshold_input_sha256: str
    purchase_oof_score_sha256_by_date: tuple[tuple[str, str], ...]
    purchase_threshold_input_sha256_by_date: tuple[tuple[str, str], ...]
    training_race_ids: tuple[str, ...]
    purchase_payout_head: LinearHead | None = None
    purchase_calibration_head: LinearHead | None = None
    purchase_feature_map: str = "base_outputs_v1"
    purchase_probability_temperature: float = 1.0
    purchase_residual_scale: float = 1.0
    purchase_oof_market_log_loss: float | None = None
    purchase_oof_scaled_log_loss: float | None = None
    purchase_payout_residual_scale: float = 1.0
    purchase_oof_base_payout_log_mae: float | None = None
    purchase_oof_scaled_payout_log_mae: float | None = None
    information_boundary: str = "decision_features_and_current_odds_only"
    purchase_teacher_source: str = "strict_prior_base_head_oof_predictions"
    purchase_threshold_source: str = "learned_unit_return_break_even_zero"
    outer_outcomes_used: bool = False
    fixed_after_fit: bool = True


@dataclass(frozen=True)
class RacePrediction:
    race_id: str
    race_date: str
    probabilities: tuple[float, ...]
    ranking_scores: tuple[float, ...]
    predicted_closing_odds: tuple[float, ...]
    purchase_scores: tuple[float, ...]
    selected_indices: tuple[int, ...]


def _array(values: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError("features must have shape (choices >= 2, features >= 1)")
    if not np.isfinite(result).all():
        raise ValueError("features must be finite")
    return result


def _validate_races(races: Sequence[LabeledRace]) -> tuple[int, int]:
    if not races:
        raise ValueError("at least one labeled race is required")
    first = races[0].decision
    choices, feature_count = _array(first.features).shape
    seen: set[str] = set()
    previous: tuple[str, str] | None = None
    for race in races:
        decision, outcome = race.decision, race.outcome
        order_key = (decision.race_date, decision.race_id)
        if previous is not None and order_key <= previous:
            raise ValueError("races must be uniquely sorted by race_date and race_id")
        previous = order_key
        if decision.race_id in seen:
            raise ValueError("race_id must be unique")
        seen.add(decision.race_id)
        if _array(decision.features).shape != (choices, feature_count):
            raise ValueError("all races must share choice and feature dimensions")
        current = np.asarray(decision.current_odds, dtype=np.float64)
        closing = np.asarray(outcome.closing_odds, dtype=np.float64)
        if current.shape != (choices,) or closing.shape != (choices,):
            raise ValueError("odds must contain one value per choice")
        if (
            not np.isfinite(current).all()
            or not np.isfinite(closing).all()
            or np.any(current <= 1.0)
            or np.any(closing <= 1.0)
        ):
            raise ValueError("odds must be finite and greater than one")
        if not 0 <= int(outcome.winner_index) < choices:
            raise ValueError("winner_index is outside the choice range")
        ranking = tuple(int(value) for value in outcome.ranking_order)
        if len(ranking) != choices or set(ranking) != set(range(choices)):
            raise ValueError("ranking_order must be a permutation of choices")
        if ranking[0] != int(outcome.winner_index):
            raise ValueError("ranking_order must start with winner_index")
    return choices, feature_count


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not len(y):
        raise ValueError("invalid ridge training matrix")
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    if sample_weight is not None:
        weights = np.sqrt(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
        if weights.shape != y.shape or np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("sample weights must be finite and positive")
        design = design * weights[:, None]
        y = y * weights
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    solution = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    if not np.isfinite(solution).all():
        raise ValueError("ridge fit produced non-finite coefficients")
    return solution[1:], float(solution[0])


def _head(name: str, teacher: str, fitted: tuple[np.ndarray, float]) -> LinearHead:
    coefficients, intercept = fitted
    return LinearHead(
        name=name,
        teacher=teacher,
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
    )


def _scores(head: LinearHead, matrix: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(head.coefficients, dtype=np.float64)
    if matrix.shape[1] != len(coefficients):
        raise ValueError(f"{head.name} feature dimension mismatch")
    return np.asarray(matrix @ coefficients + head.intercept, dtype=np.float64)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    numerator = np.exp(np.clip(shifted, -700.0, 0.0))
    return numerator / max(float(numerator.sum()), 1e-15)


def _base_matrix(decision: DecisionRace) -> np.ndarray:
    features = _array(decision.features)
    odds = np.asarray(decision.current_odds, dtype=np.float64)
    if odds.shape != (len(features),) or np.any(odds <= 1.0):
        raise ValueError("decision current_odds are invalid")
    return np.column_stack((features, np.log(odds)))


def _fit_base_heads(
    races: Sequence[LabeledRace], *, alpha: float
) -> tuple[LinearHead, LinearHead, LinearHead]:
    matrices: list[np.ndarray] = []
    probability_targets: list[np.ndarray] = []
    ranking_targets: list[np.ndarray] = []
    closing_targets: list[np.ndarray] = []
    for race in races:
        choices = len(race.decision.current_odds)
        matrices.append(_base_matrix(race.decision))
        probability = np.zeros(choices, dtype=np.float64)
        probability[race.outcome.winner_index] = 1.0
        probability_targets.append(probability - 1.0 / choices)
        order = race.outcome.ranking_order
        relevance = np.empty(choices, dtype=np.float64)
        for rank, choice in enumerate(order):
            relevance[choice] = 1.0 / math.log2(rank + 2.0)
        ranking_targets.append(relevance - float(relevance.mean()))
        closing_targets.append(np.log(np.asarray(race.outcome.closing_odds)))
    matrix = np.vstack(matrices)
    choices = len(races[0].decision.current_odds)
    positive_weight = np.concatenate(probability_targets) > 0
    probability_weight = np.where(positive_weight, choices - 1.0, 1.0)
    return (
        _head(
            "probability_head",
            "winning_combination_multinomial_surrogate",
            _fit_ridge(
                matrix,
                np.concatenate(probability_targets),
                alpha=alpha,
                sample_weight=probability_weight,
            ),
        ),
        _head(
            "ranking_head",
            "discounted_full_order_relevance",
            _fit_ridge(matrix, np.concatenate(ranking_targets), alpha=alpha),
        ),
        _head(
            "closing_odds_head",
            "log_official_closing_odds",
            _fit_ridge(matrix, np.concatenate(closing_targets), alpha=alpha),
        ),
    )


def _base_outputs(
    decision: DecisionRace,
    probability_head: LinearHead,
    ranking_head: LinearHead,
    closing_head: LinearHead,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _base_matrix(decision)
    probability = _softmax(_scores(probability_head, matrix))
    ranking = _scores(ranking_head, matrix)
    closing = np.exp(np.clip(_scores(closing_head, matrix), math.log(1.01), math.log(1e6)))
    return probability, ranking, closing


def _purchase_matrix(
    decision: DecisionRace,
    probability: np.ndarray,
    ranking: np.ndarray,
    predicted_closing: np.ndarray,
    *,
    feature_map: str = "base_outputs_v1",
) -> np.ndarray:
    current = np.asarray(decision.current_odds, dtype=np.float64)
    market = 1.0 / current
    market /= market.sum()
    ranking_probability = _softmax(ranking)
    base = np.column_stack(
        (
            probability,
            ranking_probability,
            np.log(predicted_closing),
            probability * predicted_closing - 1.0,
            probability - market,
            np.log(current),
        )
    )
    if feature_map == "base_outputs_v1":
        return base
    if feature_map == "decision_context_v2":
        return np.column_stack((base, _array(decision.features)))
    if feature_map == "decision_context_interactions_v3":
        contextual = np.column_stack((base, _array(decision.features)))
        left, right = np.triu_indices(contextual.shape[1])
        return np.column_stack(
            (contextual, contextual[:, left] * contextual[:, right])
        )
    raise ValueError(f"unsupported purchase feature map: {feature_map}")


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _oof_payload(
    race: LabeledRace,
    probability: np.ndarray,
    ranking: np.ndarray,
    closing: np.ndarray,
) -> dict[str, Any]:
    return {
        "race_id": race.decision.race_id,
        "race_date": race.decision.race_date,
        "probability": probability.tolist(),
        "ranking": ranking.tolist(),
        "predicted_closing_odds": closing.tolist(),
    }


def _fit_multinomial_offset_purchase_head(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
) -> LinearHead:
    """Learn within-race probability residuals around the frozen base head."""
    if not matrices:
        raise ValueError("offset purchase teacher requires at least one race")
    dimension = int(matrices[0].shape[1])
    winners: list[int] = []
    offsets: list[np.ndarray] = []
    for matrix, returns in zip(matrices, realized_returns, strict=True):
        if matrix.ndim != 2 or matrix.shape[1] != dimension:
            raise ValueError("offset purchase matrices must share dimensions")
        winner_indices = np.flatnonzero(np.asarray(returns) >= 0.0)
        if len(winner_indices) != 1:
            raise ValueError("offset purchase teacher requires one winner per race")
        winners.append(int(winner_indices[0]))
        offsets.append(
            np.log(np.clip(np.asarray(matrix[:, 0], dtype=np.float64), 1e-12, 1.0))
        )

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros(dimension, dtype=np.float64)
        for matrix, offset, winner in zip(
            matrices, offsets, winners, strict=True
        ):
            probabilities = _softmax(offset + matrix @ coefficients)
            loss -= math.log(max(float(probabilities[winner]), 1e-15))
            errors = probabilities.copy()
            errors[winner] -= 1.0
            gradient += matrix.T @ errors
        scale = 1.0 / len(matrices)
        loss = loss * scale + 0.5 * float(alpha) * float(
            coefficients @ coefficients
        )
        gradient = gradient * scale + float(alpha) * coefficients
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
    )
    coefficients = np.asarray(fitted.x, dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("offset purchase fit produced non-finite coefficients")
    return _head(
        "purchase_multinomial_offset_head",
        "multinomial_probability_residual_from_strict_prior_base_head_oof",
        (coefficients, 0.0),
    )


def _market_probability_from_purchase_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] < 6:
        raise ValueError("purchase matrix lacks current-odds feature")
    current_odds = np.exp(np.asarray(matrix[:, 5], dtype=np.float64))
    if not np.isfinite(current_odds).all() or np.any(current_odds <= 1.0):
        raise ValueError("purchase matrix current odds are invalid")
    inverse = 1.0 / current_odds
    return inverse / float(inverse.sum())


def _fit_market_offset_purchase_head(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
) -> LinearHead:
    """Learn outcome residuals around the T-5 market-implied distribution."""
    if not matrices:
        raise ValueError("market-offset teacher requires at least one race")
    dimension = int(matrices[0].shape[1])
    winners: list[int] = []
    offsets: list[np.ndarray] = []
    for matrix, returns in zip(matrices, realized_returns, strict=True):
        if matrix.ndim != 2 or matrix.shape[1] != dimension:
            raise ValueError("market-offset matrices must share dimensions")
        winner_indices = np.flatnonzero(np.asarray(returns) >= 0.0)
        if len(winner_indices) != 1:
            raise ValueError("market-offset teacher requires one winner per race")
        winners.append(int(winner_indices[0]))
        offsets.append(
            np.log(np.clip(_market_probability_from_purchase_matrix(matrix), 1e-12, 1.0))
        )

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros(dimension, dtype=np.float64)
        for matrix, offset, winner in zip(
            matrices, offsets, winners, strict=True
        ):
            probabilities = _softmax(offset + matrix @ coefficients)
            loss -= math.log(max(float(probabilities[winner]), 1e-15))
            errors = probabilities.copy()
            errors[winner] -= 1.0
            gradient += matrix.T @ errors
        scale = 1.0 / len(matrices)
        loss = loss * scale + 0.5 * float(alpha) * float(
            coefficients @ coefficients
        )
        gradient = gradient * scale + float(alpha) * coefficients
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
    )
    coefficients = np.asarray(fitted.x, dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("market-offset fit produced non-finite coefficients")
    return _head(
        "purchase_t5_market_multinomial_offset_head",
        "multinomial_probability_residual_from_t5_market_strict_prior_oof",
        (coefficients, 0.0),
    )


def _fit_uncapped_payout_residual_head(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
) -> LinearHead:
    winner_rows: list[np.ndarray] = []
    log_residuals: list[float] = []
    for matrix, returns in zip(matrices, realized_returns, strict=True):
        winner_indices = np.flatnonzero(np.asarray(returns) >= 0.0)
        if len(winner_indices) != 1:
            raise ValueError("payout residual teacher requires one winner per race")
        winner = int(winner_indices[0])
        actual_gross = float(returns[winner]) + 1.0
        predicted_closing = math.exp(float(matrix[winner, 2]))
        if actual_gross <= 1.0 or predicted_closing <= 1.0:
            raise ValueError("payout residual teacher requires valid gross odds")
        winner_rows.append(np.asarray(matrix[winner], dtype=np.float64))
        log_residuals.append(math.log(actual_gross / predicted_closing))
    return _head(
        "purchase_uncapped_payout_residual_head",
        "uncapped_log_payout_residual_to_predicted_closing_strict_prior_oof",
        _fit_ridge(
            np.vstack(winner_rows),
            np.asarray(log_residuals, dtype=np.float64),
            alpha=alpha,
        ),
    )


def _fit_all_choice_closing_residual_head(
    matrices: Sequence[np.ndarray],
    conditional_gross_payouts: Sequence[np.ndarray],
    *,
    alpha: float,
) -> LinearHead:
    """Learn conditional return from every ticket's official closing odds."""
    if not matrices or len(matrices) != len(conditional_gross_payouts):
        raise ValueError("all-choice closing teacher requires aligned races")
    rows: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    dimension = int(matrices[0].shape[1])
    for matrix, gross_payouts in zip(
        matrices, conditional_gross_payouts, strict=True
    ):
        gross = np.asarray(gross_payouts, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != dimension:
            raise ValueError("all-choice closing matrices must share dimensions")
        if gross.shape != (len(matrix),) or np.any(gross <= 1.0):
            raise ValueError("all-choice closing odds must be finite and above one")
        if not np.isfinite(gross).all():
            raise ValueError("all-choice closing odds must be finite and above one")
        predicted_log_closing = np.asarray(matrix[:, 2], dtype=np.float64)
        rows.append(np.asarray(matrix, dtype=np.float64))
        residuals.append(np.log(gross) - predicted_log_closing)
    return _head(
        "purchase_all_choice_closing_residual_head",
        "all_choice_log_closing_odds_residual_from_strict_prior_base_head_oof",
        _fit_ridge(
            np.vstack(rows),
            np.concatenate(residuals),
            alpha=alpha,
        ),
    )


def _fit_pairwise_purchase_head(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
) -> LinearHead:
    differences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for matrix, returns in zip(matrices, realized_returns, strict=True):
        values = np.asarray(returns, dtype=np.float64)
        winners = np.flatnonzero(values >= 0.0)
        if len(winners) != 1:
            raise ValueError("pairwise purchase teacher requires one winner per race")
        winner = int(winners[0])
        losers = np.flatnonzero(values < 0.0)
        forward = np.asarray(matrix[winner] - matrix[losers], dtype=np.float64)
        race_weight = float(np.clip(values[winner] + 1.0, 1.0, 51.0))
        differences.extend((forward, -forward))
        labels.extend(
            (
                np.ones(len(forward), dtype=np.int8),
                np.zeros(len(forward), dtype=np.int8),
            )
        )
        weights.extend(
            (
                np.full(len(forward), race_weight, dtype=np.float64),
                np.full(len(forward), race_weight, dtype=np.float64),
            )
        )
    pair_matrix = np.vstack(differences)
    fitted = LogisticRegression(
        C=1.0 / max(float(alpha), 1e-9),
        fit_intercept=False,
        max_iter=500,
        tol=1e-8,
    ).fit(
        pair_matrix,
        np.concatenate(labels),
        sample_weight=np.concatenate(weights),
    )
    return _head(
        "purchase_pairwise_rank_head",
        "payout_weighted_winner_over_loser_pairwise_strict_prior_oof",
        (
            np.asarray(fitted.coef_[0], dtype=np.float64),
            0.0,
        ),
    )


def _fit_purchase_heads(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
    purchase_loss: str,
    conditional_gross_payouts: Sequence[np.ndarray] | None = None,
) -> tuple[LinearHead, LinearHead | None]:
    matrix = np.vstack(matrices)
    returns = np.concatenate(realized_returns)
    if purchase_loss in {
        "multinomial_market_offset_all_choice_closing",
        "multinomial_market_offset_oof_scaled_all_choice_closing",
        "multinomial_market_offset_oof_scaled_payout_closing",
        "multinomial_market_offset_oof_scaled_payout_tweedie",
        "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
        "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
    }:
        if conditional_gross_payouts is None:
            raise ValueError("market-offset closing teacher targets are required")
        return (
            _fit_market_offset_purchase_head(
                matrices, realized_returns, alpha=alpha
            ),
            _fit_all_choice_closing_residual_head(
                matrices, conditional_gross_payouts, alpha=alpha
            ),
        )
    if purchase_loss in {
        "multinomial_offset_all_choice_closing",
        "multinomial_offset_all_choice_closing_temperature",
    }:

        if conditional_gross_payouts is None:
            raise ValueError("all-choice closing teacher targets are required")
        return (
            _fit_multinomial_offset_purchase_head(
                matrices, realized_returns, alpha=alpha
            ),
            _fit_all_choice_closing_residual_head(
                matrices, conditional_gross_payouts, alpha=alpha
            ),
        )
    if purchase_loss == "multinomial_offset_uncapped_lognormal":
        return (
            _fit_multinomial_offset_purchase_head(
                matrices, realized_returns, alpha=alpha
            ),
            _fit_uncapped_payout_residual_head(
                matrices, realized_returns, alpha=alpha
            ),
        )
    if purchase_loss == "pairwise_contextual_rank_calibrated":
        return (
            _fit_pairwise_purchase_head(
                matrices,
                realized_returns,
                alpha=alpha,
            ),
            None,
        )
    if purchase_loss == "ridge_capped_net":
        return (
            _head(
                "purchase_head",
                "capped_realized_unit_return_from_strict_prior_base_head_oof_inputs",
                _fit_ridge(
                    matrix,
                    np.clip(returns, -1.0, 50.0),
                    alpha=alpha,
                ),
            ),
            None,
        )
    if purchase_loss == "poisson_capped_gross":
        target = np.clip(returns + 1.0, 0.0, 51.0)
        fitted = PoissonRegressor(
            alpha=alpha,
            fit_intercept=True,
            max_iter=500,
            tol=1e-8,
        ).fit(matrix, target)
        return (
            _head(
                "purchase_head",
                (
                    "poisson_expected_capped_gross_return_from_"
                    "strict_prior_base_head_oof_inputs"
                ),
                (
                    np.asarray(fitted.coef_, dtype=np.float64),
                    float(fitted.intercept_),
                ),
            ),
            None,
        )
    if purchase_loss == "tweedie_capped_gross":
        target = np.clip(returns + 1.0, 0.0, 51.0)
        fitted = TweedieRegressor(
            power=1.5,
            alpha=alpha,
            link="log",
            fit_intercept=True,
            max_iter=500,
            tol=1e-8,
        ).fit(matrix, target)
        return (
            _head(
                "purchase_head",
                (
                    "tweedie_expected_capped_gross_return_from_"
                    "strict_prior_base_head_oof_inputs"
                ),
                (
                    np.asarray(fitted.coef_, dtype=np.float64),
                    float(fitted.intercept_),
                ),
            ),
            None,
        )
    if purchase_loss in {
        "hurdle_logistic_lognormal",
        "hurdle_logistic_lognormal_calibrated",
        "hurdle_contextual_lognormal",
        "hurdle_contextual_interactions_lognormal",
    }:
        hit = returns >= 0.0
        fitted_hit = LogisticRegression(
            C=1.0 / max(float(alpha), 1e-9),
            fit_intercept=True,
            max_iter=500,
            tol=1e-8,
        ).fit(matrix, hit.astype(np.int8))
        hit_head = _head(
            "purchase_hit_head",
            "logistic_hit_probability_from_strict_prior_base_head_oof_inputs",
            (
                np.asarray(fitted_hit.coef_[0], dtype=np.float64),
                float(fitted_hit.intercept_[0]),
            ),
        )
        payout_head = _head(
            "purchase_payout_head",
            "log_capped_gross_return_conditional_on_hit_from_strict_prior_oof",
            _fit_ridge(
                matrix[hit],
                np.log(np.clip(returns[hit] + 1.0, 1.0, 51.0)),
                alpha=alpha,
            ),
        )
        return hit_head, payout_head
    raise ValueError(f"unsupported purchase_loss: {purchase_loss}")


def _fit_purchase_head(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
    purchase_loss: str,
) -> LinearHead:
    head, _payout_head = _fit_purchase_heads(
        matrices,
        realized_returns,
        alpha=alpha,
        purchase_loss=purchase_loss,
    )
    return head


def _offset_hit_probabilities(
    head: LinearHead,
    matrix: np.ndarray,
    *,
    temperature: float = 1.0,
    residual_scale: float = 1.0,
) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("purchase probability temperature must be positive")
    if not math.isfinite(residual_scale) or residual_scale < 0.0:
        raise ValueError("purchase residual scale must be non-negative")
    if "_from_t5_market_" in head.teacher:
        base_probability = _market_probability_from_purchase_matrix(matrix)
    else:
        base_probability = np.clip(matrix[:, 0], 1e-12, 1.0)
    logits = np.log(base_probability) + residual_scale * _scores(head, matrix)
    return _softmax(logits / temperature)


def _fit_multinomial_residual_scale(
    market_probabilities: Sequence[np.ndarray],
    residual_logits: Sequence[np.ndarray],
    winner_indices: Sequence[int],
    *,
    alpha: float,
) -> tuple[float, float, float]:
    """Learn how much strict-OOF context may move probability off market."""
    if (
        not market_probabilities
        or len(market_probabilities) != len(residual_logits)
        or len(market_probabilities) != len(winner_indices)
    ):
        raise ValueError("residual scale requires aligned OOF races")

    def log_loss(scale: float) -> float:
        losses = []
        for market, residual, winner in zip(
            market_probabilities, residual_logits, winner_indices, strict=True
        ):
            probabilities = _softmax(
                np.log(np.clip(market, 1e-15, 1.0)) + scale * residual
            )
            losses.append(-math.log(max(float(probabilities[int(winner)]), 1e-15)))
        return float(np.mean(losses))

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        scale = float(values[0])
        loss = 0.0
        gradient = 0.0
        for market, residual, winner in zip(
            market_probabilities, residual_logits, winner_indices, strict=True
        ):
            probabilities = _softmax(
                np.log(np.clip(market, 1e-15, 1.0)) + scale * residual
            )
            loss -= math.log(max(float(probabilities[int(winner)]), 1e-15))
            gradient += float(probabilities @ residual) - float(residual[int(winner)])
        count = len(market_probabilities)
        return (
            loss / count + 0.5 * float(alpha) * scale * scale,
            np.asarray([gradient / count + float(alpha) * scale]),
        )

    fitted = minimize(
        objective,
        np.asarray([0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, 2.0),),
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9},
    )
    scale = float(fitted.x[0])
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("purchase residual scale is invalid")
    return scale, log_loss(0.0), log_loss(scale)


def _fit_payout_residual_scale(
    base_log_payouts: Sequence[np.ndarray],
    residual_predictions: Sequence[np.ndarray],
    target_log_payouts: Sequence[np.ndarray],
    *,
    alpha: float,
) -> tuple[float, float, float]:
    """Shrink closing-payout movement using strict-OOF all-choice targets."""
    if (
        not base_log_payouts
        or len(base_log_payouts) != len(residual_predictions)
        or len(base_log_payouts) != len(target_log_payouts)
    ):
        raise ValueError("payout residual scale requires aligned OOF races")
    base = np.concatenate(base_log_payouts)
    residual = np.concatenate(residual_predictions)
    target = np.concatenate(target_log_payouts)
    if not (
        np.isfinite(base).all()
        and np.isfinite(residual).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("payout residual scale inputs must be finite")
    movement = target - base
    denominator = float(residual @ residual) + float(alpha) * len(residual)
    scale = (
        float(np.clip(float(residual @ movement) / denominator, 0.0, 2.0))
        if denominator > 0.0
        else 0.0
    )
    baseline_mae = float(np.mean(np.abs(base - target)))
    scaled_mae = float(np.mean(np.abs(base + scale * residual - target)))
    if scaled_mae > baseline_mae:
        scale = 0.0
        scaled_mae = baseline_mae
    return scale, baseline_mae, scaled_mae


def _fit_multinomial_temperature(
    probabilities: Sequence[np.ndarray],
    winner_indices: Sequence[int],
    *,
    alpha: float,
) -> float:
    """Calibrate probability sharpness on strict-prior purchase OOF races."""
    if not probabilities or len(probabilities) != len(winner_indices):
        raise ValueError("temperature calibration requires aligned OOF races")
    log_probabilities = [
        np.log(np.clip(np.asarray(values, dtype=np.float64), 1e-15, 1.0))
        for values in probabilities
    ]
    for values, winner in zip(log_probabilities, winner_indices, strict=True):
        if values.ndim != 1 or not 0 <= int(winner) < len(values):
            raise ValueError("temperature calibration winner is invalid")

    def objective(theta_values: np.ndarray) -> tuple[float, np.ndarray]:
        theta = float(theta_values[0])
        temperature = math.exp(theta)
        loss = 0.0
        gradient = 0.0
        for logits, winner in zip(
            log_probabilities, winner_indices, strict=True
        ):
            calibrated = _softmax(logits / temperature)
            loss -= math.log(max(float(calibrated[int(winner)]), 1e-15))
            gradient += (
                float(logits[int(winner)]) - float(calibrated @ logits)
            ) / temperature
        scale = 1.0 / len(log_probabilities)
        return (
            loss * scale + 0.5 * float(alpha) * theta * theta,
            np.asarray([gradient * scale + float(alpha) * theta]),
        )

    fitted = minimize(
        objective,
        np.zeros(1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-4.0, 4.0),),
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9},
    )
    temperature = math.exp(float(fitted.x[0]))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature calibration produced an invalid value")
    return temperature


def _purchase_net_scores(
    head: LinearHead,
    payout_head: LinearHead | None,
    matrix: np.ndarray,
    *,
    probability_temperature: float = 1.0,
    probability_residual_scale: float = 1.0,
    payout_residual_scale: float = 1.0,
) -> np.ndarray:
    linear = _scores(head, matrix)
    if head.teacher.startswith("multinomial_probability_residual"):
        if payout_head is None or not payout_head.teacher.startswith(
            (
                "uncapped_log_payout_residual",
                "all_choice_log_closing_odds_residual",
            )
        ):
            raise ValueError("offset purchase head requires payout residual head")
        hit_probability = _offset_hit_probabilities(
            head, matrix, temperature=probability_temperature,
            residual_scale=probability_residual_scale,
        )
        conditional_payout = np.exp(
            np.clip(
                matrix[:, 2] + payout_residual_scale * _scores(
                    payout_head, matrix
                ),
                math.log(1.01),
                math.log(1e6),
            )
        )
        return hit_probability * conditional_payout - 1.0
    if payout_head is not None:
        hit_probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        conditional_payout = np.exp(
            np.clip(_scores(payout_head, matrix), 0.0, math.log(51.0))
        )
        return hit_probability * conditional_payout - 1.0
    if "_expected_capped_gross" in head.teacher:
        return np.exp(np.clip(linear, -30.0, math.log(51.0))) - 1.0
    return linear


def _purchase_factor_calibration_matrix(
    head: LinearHead,
    payout_head: LinearHead | None,
    matrix: np.ndarray,
    *,
    probability_residual_scale: float,
    payout_residual_scale: float,
) -> np.ndarray:
    """Keep hit probability and conditional payout separate for EV learning."""
    if payout_head is None or not head.teacher.startswith(
        "multinomial_probability_residual"
    ):
        raise ValueError("factor calibration requires probability and payout heads")
    hit_probability = _offset_hit_probabilities(
        head,
        matrix,
        residual_scale=probability_residual_scale,
    )
    conditional_payout = np.exp(
        np.clip(
            matrix[:, 2]
            + payout_residual_scale * _scores(payout_head, matrix),
            math.log(1.01),
            math.log(1e6),
        )
    )
    return np.column_stack(
        (
            np.log(np.clip(hit_probability, 1e-15, 1.0)),
            np.log(conditional_payout),
        )
    )


def _purchase_context_factor_calibration_matrix(
    head: LinearHead,
    payout_head: LinearHead | None,
    matrix: np.ndarray,
    *,
    probability_residual_scale: float,
    payout_residual_scale: float,
) -> np.ndarray:
    """Add decision-time context to decomposed gross-return factors."""
    factors = _purchase_factor_calibration_matrix(
        head,
        payout_head,
        matrix,
        probability_residual_scale=probability_residual_scale,
        payout_residual_scale=payout_residual_scale,
    )
    market_probability = _market_probability_from_purchase_matrix(matrix)
    log_current_odds = np.asarray(matrix[:, 5], dtype=np.float64)
    residual_context = np.column_stack(
        (
            factors[:, 0] - np.log(np.clip(market_probability, 1e-15, 1.0)),
            factors[:, 1] - log_current_odds,
            np.asarray(matrix[:, 0] - matrix[:, 1], dtype=np.float64),
        )
    )
    decision_context = (
        np.asarray(matrix[:, 6:], dtype=np.float64)
        if matrix.shape[1] > 6
        else np.empty((len(matrix), 0), dtype=np.float64)
    )
    return np.column_stack((factors, residual_context, decision_context))


def fit_four_head_nested_v22(
    races: Iterable[LabeledRace],
    *,
    minimum_inner_training_dates: int = 2,
    minimum_purchase_training_dates: int = 2,
    alpha: float = 1e-3,
    purchase_loss: str = "ridge_capped_net",
) -> FourHeadArtifact:
    """Fit four heads with base-head OOF nested inside purchase-head OOF."""

    ordered = tuple(races)
    choices, feature_count = _validate_races(ordered)
    if minimum_inner_training_dates < 1:
        raise ValueError("minimum_inner_training_dates must be positive")
    if minimum_purchase_training_dates < 1:
        raise ValueError("minimum_purchase_training_dates must be positive")
    dates = sorted({race.decision.race_date for race in ordered})
    purchase_feature_map = {
        "hurdle_contextual_lognormal": "decision_context_v2",
        "hurdle_contextual_interactions_lognormal": (
            "decision_context_interactions_v3"
        ),
        "pairwise_contextual_rank_calibrated": "decision_context_v2",
        "multinomial_offset_uncapped_lognormal": "decision_context_v2",
        "multinomial_offset_all_choice_closing": "decision_context_v2",
        "multinomial_market_offset_all_choice_closing": (
            "decision_context_v2"
        ),
        "multinomial_market_offset_oof_scaled_all_choice_closing": (
            "decision_context_v2"
        ),
        "multinomial_market_offset_oof_scaled_payout_closing": (
            "decision_context_v2"
        ),
        "multinomial_market_offset_oof_scaled_payout_tweedie": (
            "decision_context_v2"
        ),
        "multinomial_market_offset_oof_scaled_payout_factor_tweedie": (
            "decision_context_v2"
        ),
        "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie": (
            "decision_context_v2"
        ),
        "multinomial_offset_all_choice_closing_temperature": (
            "decision_context_v2"
        ),
    }.get(purchase_loss, "base_outputs_v1")
    if len(dates) <= minimum_inner_training_dates:
        raise ValueError("not enough whole dates for strict-prior inner OOF")
    by_date = {
        date: tuple(race for race in ordered if race.decision.race_date == date)
        for date in dates
    }
    folds: list[InnerOOFFold] = []
    oof_payloads: list[dict[str, Any]] = []
    base_oof_by_date: dict[
        str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    for validation_index in range(minimum_inner_training_dates, len(dates)):
        validation_date = dates[validation_index]
        training_dates = dates[:validation_index]
        training = tuple(race for date in training_dates for race in by_date[date])
        validation = by_date[validation_date]
        probability_head, ranking_head, closing_head = _fit_base_heads(
            training, alpha=alpha
        )
        base_oof_by_date[validation_date] = []
        for race in validation:
            probability, ranking, closing = _base_outputs(
                race.decision, probability_head, ranking_head, closing_head
            )
            purchase_matrix = _purchase_matrix(
                race.decision,
                probability,
                ranking,
                closing,
                feature_map=purchase_feature_map,
            )
            realized = np.full(choices, -1.0, dtype=np.float64)
            realized[race.outcome.winner_index] = (
                float(race.outcome.closing_odds[race.outcome.winner_index]) - 1.0
            )
            base_oof_by_date[validation_date].append(
                (
                    race.decision.race_id,
                    purchase_matrix,
                    realized,
                    np.asarray(race.outcome.closing_odds, dtype=np.float64),
                )
            )
            oof_payloads.append(_oof_payload(race, probability, ranking, closing))
        folds.append(
            InnerOOFFold(
                validation_date=validation_date,
                trained_through_date=training_dates[-1],
                training_race_ids=tuple(race.decision.race_id for race in training),
                validation_race_ids=tuple(race.decision.race_id for race in validation),
            )
        )
    base_oof_dates = tuple(base_oof_by_date)
    if len(base_oof_dates) <= minimum_purchase_training_dates:
        raise ValueError("not enough base-head OOF dates for purchase-head OOF")
    purchase_folds: list[PurchaseOOFFold] = []
    purchase_oof_ids: list[str] = []
    purchase_score_payloads: list[dict[str, Any]] = []
    threshold_input_payloads: list[dict[str, Any]] = []
    score_sha_by_date: list[tuple[str, str]] = []
    threshold_sha_by_date: list[tuple[str, str]] = []
    purchase_oof_scores_for_calibration: list[float] = []
    purchase_oof_returns_for_calibration: list[float] = []
    purchase_oof_factor_features: list[np.ndarray] = []
    purchase_oof_context_factor_features: list[np.ndarray] = []
    purchase_oof_probabilities_for_temperature: list[np.ndarray] = []
    purchase_oof_winners_for_temperature: list[int] = []
    purchase_oof_market_probabilities_for_scale: list[np.ndarray] = []
    purchase_oof_residual_logits_for_scale: list[np.ndarray] = []
    purchase_oof_winners_for_scale: list[int] = []
    purchase_oof_base_log_payouts: list[np.ndarray] = []
    purchase_oof_payout_residuals: list[np.ndarray] = []
    purchase_oof_target_log_payouts: list[np.ndarray] = []
    purchase_oof_scale_records: list[
        tuple[str, str, LinearHead, LinearHead | None, np.ndarray, np.ndarray]
    ] = []
    for validation_index in range(
        minimum_purchase_training_dates, len(base_oof_dates)
    ):
        validation_date = base_oof_dates[validation_index]
        prior_dates = base_oof_dates[:validation_index]
        training_records = [
            record for date in prior_dates for record in base_oof_by_date[date]
        ]
        validation_records = base_oof_by_date[validation_date]
        fold_head, fold_payout_head = _fit_purchase_heads(
            [record[1] for record in training_records],
            [record[2] for record in training_records],
            alpha=alpha,
            purchase_loss=purchase_loss,
            conditional_gross_payouts=[record[3] for record in training_records],
        )
        date_score_payloads: list[dict[str, Any]] = []
        date_threshold_payloads: list[dict[str, Any]] = []
        for race_id, matrix, realized, _conditional_gross in validation_records:
            scores = _purchase_net_scores(
                fold_head, fold_payout_head, matrix
            )
            purchase_oof_scores_for_calibration.extend(scores.tolist())
            purchase_oof_returns_for_calibration.extend(realized.tolist())
            if purchase_loss in {
                "multinomial_market_offset_oof_scaled_all_choice_closing",
                "multinomial_market_offset_oof_scaled_payout_closing",
                "multinomial_market_offset_oof_scaled_payout_tweedie",
                "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
                "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
            }:
                winner_indices = np.flatnonzero(realized >= 0.0)
                if len(winner_indices) != 1:
                    raise ValueError("residual-scale OOF race requires one winner")
                purchase_oof_market_probabilities_for_scale.append(
                    _market_probability_from_purchase_matrix(matrix)
                )
                purchase_oof_residual_logits_for_scale.append(
                    _scores(fold_head, matrix)
                )
                purchase_oof_winners_for_scale.append(int(winner_indices[0]))
                if purchase_loss in {
                    "multinomial_market_offset_oof_scaled_payout_closing",
                    "multinomial_market_offset_oof_scaled_payout_tweedie",
                    "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
                    "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
                }:
                    if fold_payout_head is None:
                        raise ValueError("scaled payout OOF head is missing")
                    purchase_oof_base_log_payouts.append(matrix[:, 2].copy())
                    purchase_oof_payout_residuals.append(
                        _scores(fold_payout_head, matrix)
                    )
                    purchase_oof_target_log_payouts.append(
                        np.log(np.asarray(_conditional_gross, dtype=np.float64))
                    )
                purchase_oof_scale_records.append(
                    (
                        validation_date, race_id, fold_head, fold_payout_head,
                        matrix, realized,
                    )
                )
            if purchase_loss == "multinomial_offset_all_choice_closing_temperature":
                purchase_oof_probabilities_for_temperature.append(
                    _offset_hit_probabilities(fold_head, matrix)
                )
                winner_indices = np.flatnonzero(realized >= 0.0)
                if len(winner_indices) != 1:
                    raise ValueError("temperature OOF race requires one winner")
                purchase_oof_winners_for_temperature.append(
                    int(winner_indices[0])
                )
            score_payload = {
                "race_id": race_id,
                "race_date": validation_date,
                "scores": scores.tolist(),
            }
            threshold_payload = {
                **score_payload,
                "realized_unit_returns": realized.tolist(),
            }
            purchase_oof_ids.append(race_id)
            purchase_score_payloads.append(score_payload)
            threshold_input_payloads.append(threshold_payload)
            date_score_payloads.append(score_payload)
            date_threshold_payloads.append(threshold_payload)
        score_sha_by_date.append(
            (validation_date, _payload_sha256(date_score_payloads))
        )
        threshold_sha_by_date.append(
            (validation_date, _payload_sha256(date_threshold_payloads))
        )
        purchase_folds.append(
            PurchaseOOFFold(
                validation_date=validation_date,
                trained_through_date=prior_dates[-1],
                training_base_oof_race_ids=tuple(
                    record[0] for record in training_records
                ),
                validation_race_ids=tuple(
                    record[0] for record in validation_records
                ),
            )
        )
    purchase_probability_temperature = 1.0
    purchase_residual_scale = 1.0
    purchase_oof_market_log_loss = None
    purchase_oof_scaled_log_loss = None
    purchase_payout_residual_scale = 1.0
    purchase_oof_base_payout_log_mae = None
    purchase_oof_scaled_payout_log_mae = None
    if purchase_loss in {
        "multinomial_market_offset_oof_scaled_all_choice_closing",
        "multinomial_market_offset_oof_scaled_payout_closing",
        "multinomial_market_offset_oof_scaled_payout_tweedie",
        "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
        "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
    }:
        (
            purchase_residual_scale,
            purchase_oof_market_log_loss,
            purchase_oof_scaled_log_loss,
        ) = _fit_multinomial_residual_scale(
            purchase_oof_market_probabilities_for_scale,
            purchase_oof_residual_logits_for_scale,
            purchase_oof_winners_for_scale,
            alpha=alpha,
        )
        if purchase_loss in {
            "multinomial_market_offset_oof_scaled_payout_closing",
            "multinomial_market_offset_oof_scaled_payout_tweedie",
            "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
            "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
        }:
            (
                purchase_payout_residual_scale,
                purchase_oof_base_payout_log_mae,
                purchase_oof_scaled_payout_log_mae,
            ) = _fit_payout_residual_scale(
                purchase_oof_base_log_payouts,
                purchase_oof_payout_residuals,
                purchase_oof_target_log_payouts,
                alpha=alpha,
            )
        purchase_oof_scores_for_calibration.clear()
        purchase_oof_returns_for_calibration.clear()
        purchase_score_payloads.clear()
        threshold_input_payloads.clear()
        score_sha_by_date.clear()
        threshold_sha_by_date.clear()
        for validation_date in base_oof_dates:
            date_score_payloads = []
            date_threshold_payloads = []
            for (
                record_date, race_id, fold_head, fold_payout_head, matrix, realized
            ) in purchase_oof_scale_records:
                if record_date != validation_date:
                    continue
                scores = _purchase_net_scores(
                    fold_head,
                    fold_payout_head,
                    matrix,
                    probability_residual_scale=purchase_residual_scale,
                    payout_residual_scale=purchase_payout_residual_scale,
                )
                purchase_oof_scores_for_calibration.extend(scores.tolist())
                purchase_oof_returns_for_calibration.extend(realized.tolist())
                if purchase_loss in {
                    "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
                    "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
                }:
                    factor_builder = (
                        _purchase_context_factor_calibration_matrix
                        if purchase_loss.endswith("context_factor_tweedie")
                        else _purchase_factor_calibration_matrix
                    )
                    factor_target = (
                        purchase_oof_context_factor_features
                        if purchase_loss.endswith("context_factor_tweedie")
                        else purchase_oof_factor_features
                    )
                    factor_target.append(
                        factor_builder(
                            fold_head,
                            fold_payout_head,
                            matrix,
                            probability_residual_scale=purchase_residual_scale,
                            payout_residual_scale=purchase_payout_residual_scale,
                        )
                    )
                score_payload = {
                    "race_id": race_id,
                    "race_date": validation_date,
                    "scores": scores.tolist(),
                }
                threshold_payload = {
                    **score_payload,
                    "realized_unit_returns": realized.tolist(),
                }
                purchase_score_payloads.append(score_payload)
                threshold_input_payloads.append(threshold_payload)
                date_score_payloads.append(score_payload)
                date_threshold_payloads.append(threshold_payload)
            if date_score_payloads:
                score_sha_by_date.append(
                    (validation_date, _payload_sha256(date_score_payloads))
                )
                threshold_sha_by_date.append(
                    (validation_date, _payload_sha256(date_threshold_payloads))
                )
    if purchase_loss == "multinomial_offset_all_choice_closing_temperature":
        purchase_probability_temperature = _fit_multinomial_temperature(
            purchase_oof_probabilities_for_temperature,
            purchase_oof_winners_for_temperature,
            alpha=alpha,
        )
    purchase_calibration_head = None
    if purchase_loss in {
        "hurdle_logistic_lognormal_calibrated",
        "pairwise_contextual_rank_calibrated",
        "multinomial_market_offset_oof_scaled_payout_tweedie",
        "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
        "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
    }:
        if purchase_loss in {
            "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
            "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
        }:
            factor_features = (
                purchase_oof_context_factor_features
                if purchase_loss.endswith("context_factor_tweedie")
                else purchase_oof_factor_features
            )
            if not factor_features:
                raise ValueError("factor calibration requires strict OOF features")
            calibration_matrix = np.vstack(factor_features)
        else:
            calibration_matrix = np.asarray(
                purchase_oof_scores_for_calibration, dtype=np.float64
            ).reshape(-1, 1)
        calibration_target = np.clip(
            np.asarray(purchase_oof_returns_for_calibration, dtype=np.float64)
            + 1.0,
            0.0,
            51.0,
        )
        if purchase_loss in {
            "multinomial_market_offset_oof_scaled_payout_tweedie",
            "multinomial_market_offset_oof_scaled_payout_factor_tweedie",
            "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
        }:
            calibration_target = np.maximum(
                np.asarray(purchase_oof_returns_for_calibration, dtype=np.float64)
                + 1.0,
                0.0,
            )
            fitted_calibration = TweedieRegressor(
                power=1.5,
                alpha=alpha,
                link="log",
                fit_intercept=True,
                max_iter=1000,
                tol=1e-8,
            ).fit(calibration_matrix, calibration_target)
            if purchase_loss.endswith("context_factor_tweedie"):
                calibration_teacher = (
                    "tweedie_power_1_5_context_factor_calibration_of_"
                    "strict_purchase_oof_gross_return"
                )
            elif purchase_loss.endswith("factor_tweedie"):
                calibration_teacher = (
                    "tweedie_power_1_5_factor_calibration_of_"
                    "strict_purchase_oof_gross_return"
                )
            else:
                calibration_teacher = (
                    "tweedie_power_1_5_calibration_of_strict_purchase_oof_gross_return"
                )
        else:
            fitted_calibration = PoissonRegressor(
                alpha=alpha,
                fit_intercept=True,
                max_iter=500,
                tol=1e-8,
            ).fit(calibration_matrix, calibration_target)
            calibration_teacher = (
                "poisson_calibration_of_strict_purchase_head_oof_gross_return"
            )
        purchase_calibration_head = _head(
            "purchase_calibration_head",
            calibration_teacher,
            (
                np.asarray(fitted_calibration.coef_, dtype=np.float64),
                float(fitted_calibration.intercept_),
            ),
        )

    # The purchase head predicts net unit return. Zero is its semantic break-even
    # boundary, not a return-maximizing threshold selected on validation labels.
    threshold = 0.0

    all_base_oof_records = [
        record for date in base_oof_dates for record in base_oof_by_date[date]
    ]
    # Threshold selection is complete before the deployable purchase head sees all
    # base OOF rows. The final refit cannot feed back into threshold selection.
    purchase_head, purchase_payout_head = _fit_purchase_heads(
        [record[1] for record in all_base_oof_records],
        [record[2] for record in all_base_oof_records],
        alpha=alpha,
        purchase_loss=purchase_loss,
        conditional_gross_payouts=[record[3] for record in all_base_oof_records],
    )
    probability_head, ranking_head, closing_head = _fit_base_heads(
        ordered, alpha=alpha
    )
    oof_ids = tuple(payload["race_id"] for payload in oof_payloads)
    return FourHeadArtifact(
        model_key=MODEL_KEY,
        artifact_version=ARTIFACT_VERSION,
        trained_through_date=dates[-1],
        choice_count=choices,
        feature_count=feature_count,
        probability_head=probability_head,
        ranking_head=ranking_head,
        closing_odds_head=closing_head,
        purchase_head=purchase_head,
        purchase_payout_head=purchase_payout_head,
        purchase_calibration_head=purchase_calibration_head,
        purchase_feature_map=purchase_feature_map,
        purchase_probability_temperature=purchase_probability_temperature,
        purchase_residual_scale=purchase_residual_scale,
        purchase_oof_market_log_loss=purchase_oof_market_log_loss,
        purchase_oof_scaled_log_loss=purchase_oof_scaled_log_loss,
        purchase_payout_residual_scale=purchase_payout_residual_scale,
        purchase_oof_base_payout_log_mae=purchase_oof_base_payout_log_mae,
        purchase_oof_scaled_payout_log_mae=purchase_oof_scaled_payout_log_mae,
        purchase_threshold=threshold,
        inner_oof_folds=tuple(folds),
        inner_oof_race_ids=oof_ids,
        inner_oof_prediction_sha256=_payload_sha256(oof_payloads),
        purchase_oof_folds=tuple(purchase_folds),
        purchase_oof_race_ids=tuple(purchase_oof_ids),
        purchase_oof_score_sha256=_payload_sha256(purchase_score_payloads),
        purchase_threshold_input_sha256=_payload_sha256(
            threshold_input_payloads
        ),
        purchase_oof_score_sha256_by_date=tuple(score_sha_by_date),
        purchase_threshold_input_sha256_by_date=tuple(threshold_sha_by_date),
        training_race_ids=tuple(race.decision.race_id for race in ordered),
    )


def artifact_fingerprint(artifact: FourHeadArtifact) -> str:
    def head_payload(head: LinearHead) -> dict[str, Any]:
        return {
            "name": head.name,
            "teacher": head.teacher,
            "coefficients": head.coefficients,
            "intercept": head.intercept,
        }

    return _payload_sha256(
        {
            "model_key": artifact.model_key,
            "artifact_version": artifact.artifact_version,
            "trained_through_date": artifact.trained_through_date,
            "choice_count": artifact.choice_count,
            "feature_count": artifact.feature_count,
            "probability_head": head_payload(artifact.probability_head),
            "ranking_head": head_payload(artifact.ranking_head),
            "closing_odds_head": head_payload(artifact.closing_odds_head),
            "purchase_head": head_payload(artifact.purchase_head),
            "purchase_payout_head": (
                head_payload(artifact.purchase_payout_head)
                if artifact.purchase_payout_head is not None
                else None
            ),
            "purchase_calibration_head": (
                head_payload(artifact.purchase_calibration_head)
                if artifact.purchase_calibration_head is not None
                else None
            ),
            "purchase_feature_map": artifact.purchase_feature_map,
            "purchase_probability_temperature": getattr(
                artifact, "purchase_probability_temperature", 1.0
            ),
            "purchase_residual_scale": getattr(
                artifact, "purchase_residual_scale", 1.0
            ),
            "purchase_oof_market_log_loss": getattr(
                artifact, "purchase_oof_market_log_loss", None
            ),
            "purchase_oof_scaled_log_loss": getattr(
                artifact, "purchase_oof_scaled_log_loss", None
            ),
            "purchase_payout_residual_scale": getattr(
                artifact, "purchase_payout_residual_scale", 1.0
            ),
            "purchase_oof_base_payout_log_mae": getattr(
                artifact, "purchase_oof_base_payout_log_mae", None
            ),
            "purchase_oof_scaled_payout_log_mae": getattr(
                artifact, "purchase_oof_scaled_payout_log_mae", None
            ),
            "purchase_threshold": artifact.purchase_threshold,
            "inner_oof_folds": [fold.__dict__ for fold in artifact.inner_oof_folds],
            "inner_oof_race_ids": artifact.inner_oof_race_ids,
            "inner_oof_prediction_sha256": artifact.inner_oof_prediction_sha256,
            "purchase_oof_folds": [
                fold.__dict__ for fold in artifact.purchase_oof_folds
            ],
            "purchase_oof_race_ids": artifact.purchase_oof_race_ids,
            "purchase_oof_score_sha256": artifact.purchase_oof_score_sha256,
            "purchase_threshold_input_sha256": (
                artifact.purchase_threshold_input_sha256
            ),
            "purchase_oof_score_sha256_by_date": (
                artifact.purchase_oof_score_sha256_by_date
            ),
            "purchase_threshold_input_sha256_by_date": (
                artifact.purchase_threshold_input_sha256_by_date
            ),
            "training_race_ids": artifact.training_race_ids,
            "information_boundary": artifact.information_boundary,
            "purchase_teacher_source": artifact.purchase_teacher_source,
            "purchase_threshold_source": artifact.purchase_threshold_source,
            "outer_outcomes_used": artifact.outer_outcomes_used,
            "fixed_after_fit": artifact.fixed_after_fit,
        }
    )


def predict_race(
    artifact: FourHeadArtifact, decision: DecisionRace
) -> RacePrediction:
    if artifact.model_key != MODEL_KEY or artifact.artifact_version != ARTIFACT_VERSION:
        raise ValueError("not a V22 four-head artifact")
    matrix = _base_matrix(decision)
    if matrix.shape != (artifact.choice_count, artifact.feature_count + 1):
        raise ValueError("decision dimensions differ from the fixed artifact")
    probability, ranking, closing = _base_outputs(
        decision,
        artifact.probability_head,
        artifact.ranking_head,
        artifact.closing_odds_head,
    )
    purchase_matrix = _purchase_matrix(
        decision,
        probability,
        ranking,
        closing,
        feature_map=artifact.purchase_feature_map,
    )
    purchase = _purchase_net_scores(
        artifact.purchase_head,
        artifact.purchase_payout_head,
        purchase_matrix,
        probability_temperature=getattr(
            artifact, "purchase_probability_temperature", 1.0
        ),
        probability_residual_scale=getattr(
            artifact, "purchase_residual_scale", 1.0
        ),
        payout_residual_scale=getattr(
            artifact, "purchase_payout_residual_scale", 1.0
        ),
    )
    if artifact.purchase_calibration_head is not None:
        calibration_matrix = purchase.reshape(-1, 1)
        calibration_teacher = artifact.purchase_calibration_head.teacher
        if calibration_teacher.startswith(
            "tweedie_power_1_5_context_factor_calibration"
        ):
            calibration_matrix = _purchase_context_factor_calibration_matrix(
                artifact.purchase_head,
                artifact.purchase_payout_head,
                purchase_matrix,
                probability_residual_scale=getattr(
                    artifact, "purchase_residual_scale", 1.0
                ),
                payout_residual_scale=getattr(
                    artifact, "purchase_payout_residual_scale", 1.0
                ),
            )
        elif calibration_teacher.startswith(
            "tweedie_power_1_5_factor_calibration"
        ):
            calibration_matrix = _purchase_factor_calibration_matrix(
                artifact.purchase_head,
                artifact.purchase_payout_head,
                purchase_matrix,
                probability_residual_scale=getattr(
                    artifact, "purchase_residual_scale", 1.0
                ),
                payout_residual_scale=getattr(
                    artifact, "purchase_payout_residual_scale", 1.0
                ),
            )
        purchase = np.exp(
            np.clip(
                _scores(
                    artifact.purchase_calibration_head,
                    calibration_matrix,
                ),
                -30.0,
                (
                    math.log(1e6)
                    if "tweedie_power_1_5" in artifact.purchase_calibration_head.teacher
                    else math.log(51.0)
                ),
            )
        ) - 1.0
    selected = tuple(
        int(index)
        for index in np.flatnonzero(purchase >= artifact.purchase_threshold)
    )
    return RacePrediction(
        race_id=decision.race_id,
        race_date=decision.race_date,
        probabilities=tuple(float(value) for value in probability),
        ranking_scores=tuple(float(value) for value in ranking),
        predicted_closing_odds=tuple(float(value) for value in closing),
        purchase_scores=tuple(float(value) for value in purchase),
        selected_indices=selected,
    )


def predict_purchase_hit_probabilities(
    artifact: FourHeadArtifact,
    decision: DecisionRace,
) -> np.ndarray | None:
    """Return the learned purchase-head hit distribution when it is probabilistic."""
    if not artifact.purchase_head.teacher.startswith(
        "multinomial_probability_residual"
    ):
        return None
    probability, ranking, closing = _base_outputs(
        decision,
        artifact.probability_head,
        artifact.ranking_head,
        artifact.closing_odds_head,
    )
    matrix = _purchase_matrix(
        decision,
        probability,
        ranking,
        closing,
        feature_map=artifact.purchase_feature_map,
    )
    return _offset_hit_probabilities(
        artifact.purchase_head,
        matrix,
        temperature=getattr(artifact, "purchase_probability_temperature", 1.0),
        residual_scale=getattr(artifact, "purchase_residual_scale", 1.0),
    )


def predict_purchase_gross_payouts(
    artifact: FourHeadArtifact,
    decision: DecisionRace,
) -> np.ndarray | None:
    if artifact.purchase_payout_head is None or not artifact.purchase_head.teacher.startswith(
        "multinomial_probability_residual"
    ):
        return None
    probability, ranking, closing = _base_outputs(
        decision,
        artifact.probability_head,
        artifact.ranking_head,
        artifact.closing_odds_head,
    )
    matrix = _purchase_matrix(
        decision,
        probability,
        ranking,
        closing,
        feature_map=artifact.purchase_feature_map,
    )
    return np.exp(
        np.clip(
            matrix[:, 2]
            + getattr(artifact, "purchase_payout_residual_scale", 1.0)
            * _scores(artifact.purchase_payout_head, matrix),
            math.log(1.01),
            math.log(1e6),
        )
    )


def prediction_fingerprint(predictions: Sequence[RacePrediction]) -> str:
    return _payload_sha256([prediction.__dict__ for prediction in predictions])


def _purchase_value_diagnostics(
    races: Sequence[LabeledRace],
    predictions: Sequence[RacePrediction],
) -> dict[str, Any]:
    predicted_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    for race, prediction in zip(races, predictions, strict=True):
        predicted_rows.append(
            np.asarray(prediction.purchase_scores, dtype=np.float64)
        )
        observed = np.full(len(prediction.purchase_scores), -1.0)
        winner = int(race.outcome.winner_index)
        observed[winner] = min(
            float(race.outcome.closing_odds[winner]) - 1.0,
            50.0,
        )
        observed_rows.append(observed)
    predicted = np.concatenate(predicted_rows)
    observed = np.concatenate(observed_rows)
    positive = predicted >= 0.0
    correlation = None
    if float(np.std(predicted)) > 0.0 and float(np.std(observed)) > 0.0:
        correlation = float(np.corrcoef(predicted, observed)[0, 1])

    bins: list[dict[str, Any]] = []
    absolute_error_sum = 0.0
    for index, indices in enumerate(
        np.array_split(np.argsort(predicted, kind="stable"), 10), start=1
    ):
        predicted_mean = float(np.mean(predicted[indices]))
        observed_mean = float(np.mean(observed[indices]))
        absolute_error_sum += len(indices) * abs(predicted_mean - observed_mean)
        bins.append(
            {
                "quantile": index,
                "tickets": int(len(indices)),
                "predicted_net_unit_return": predicted_mean,
                "observed_capped_net_unit_return": observed_mean,
                "observed_capped_roi": observed_mean + 1.0,
            }
        )
    positive_count = int(np.sum(positive))
    return {
        "schema_version": 1,
        "teacher": "capped_realized_unit_return_max_50",
        "tickets": int(len(predicted)),
        "predicted_mean": float(np.mean(predicted)),
        "observed_mean": float(np.mean(observed)),
        "pearson_correlation": correlation,
        "calibration_mae": absolute_error_sum / len(predicted),
        "positive_predicted_tickets": positive_count,
        "positive_predicted_fraction": positive_count / len(predicted),
        "positive_predicted_mean": (
            float(np.mean(predicted[positive])) if positive_count else None
        ),
        "positive_observed_capped_roi": (
            float(np.mean(observed[positive] + 1.0)) if positive_count else None
        ),
        "calibration_deciles": bins,
    }


def evaluate_outer_outcomes(
    artifact: FourHeadArtifact, outer_races: Iterable[LabeledRace]
) -> dict[str, Any]:
    """Freeze predictions first, then use outer outcomes for metrics only."""

    races = tuple(outer_races)
    choices, feature_count = _validate_races(races)
    if (choices, feature_count) != (artifact.choice_count, artifact.feature_count):
        raise ValueError("outer race dimensions differ from the fixed artifact")
    training_ids = set(artifact.training_race_ids)
    if any(race.decision.race_id in training_ids for race in races):
        raise ValueError("outer evaluation races overlap artifact training races")
    if any(race.decision.race_date <= artifact.trained_through_date for race in races):
        raise ValueError("outer evaluation must be strictly after trained_through_date")

    # This phase has no access to RaceOutcome. Its hash is the evaluation boundary.
    predictions = tuple(predict_race(artifact, race.decision) for race in races)
    frozen_prediction_sha256 = prediction_fingerprint(predictions)

    log_loss_sum = 0.0
    top5_hits = 0
    closing_absolute_log_errors: list[float] = []
    tickets = hits = 0
    return_units = 0.0
    for race, prediction in zip(races, predictions, strict=True):
        outcome = race.outcome
        probability = max(prediction.probabilities[outcome.winner_index], 1e-15)
        log_loss_sum -= math.log(probability)
        ranking_order = np.argsort(-np.asarray(prediction.ranking_scores))
        top5_hits += int(outcome.winner_index in ranking_order[: min(5, choices)])
        closing_absolute_log_errors.extend(
            np.abs(
                np.log(np.asarray(prediction.predicted_closing_odds))
                - np.log(np.asarray(outcome.closing_odds))
            ).tolist()
        )
        tickets += len(prediction.selected_indices)
        if outcome.winner_index in prediction.selected_indices:
            hits += 1
            return_units += float(outcome.closing_odds[outcome.winner_index])
    return {
        "model_key": MODEL_KEY,
        "artifact_sha256": artifact_fingerprint(artifact),
        "frozen_prediction_sha256": frozen_prediction_sha256,
        "outer_outcomes_role": "evaluation_only_after_predictions_frozen",
        "outer_outcomes_used_for_fit_or_selection": False,
        "races": len(races),
        "probability_log_loss": log_loss_sum / len(races),
        "ranking_top5_hit_rate": top5_hits / len(races),
        "closing_odds_log_mae": float(np.mean(closing_absolute_log_errors)),
        "production_bankroll_evaluated": False,
        "purchase_value_diagnostics": _purchase_value_diagnostics(
            races, predictions
        ),
        "diagnostic_unit_stake": {
            "label": "equal_one_unit_per_selected_ticket_not_production_bankroll",
            "stake_units": tickets,
            "return_units": return_units,
            "hits": hits,
            "roi": return_units / tickets if tickets else None,
        },
    }
