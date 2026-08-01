from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
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


def _fit_purchase_heads(
    matrices: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    *,
    alpha: float,
    purchase_loss: str,
) -> tuple[LinearHead, LinearHead | None]:
    matrix = np.vstack(matrices)
    returns = np.concatenate(realized_returns)
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


def _purchase_net_scores(
    head: LinearHead,
    payout_head: LinearHead | None,
    matrix: np.ndarray,
) -> np.ndarray:
    linear = _scores(head, matrix)
    if payout_head is not None:
        hit_probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        conditional_payout = np.exp(
            np.clip(_scores(payout_head, matrix), 0.0, math.log(51.0))
        )
        return hit_probability * conditional_payout - 1.0
    if "_expected_capped_gross" in head.teacher:
        return np.exp(np.clip(linear, -30.0, math.log(51.0))) - 1.0
    return linear


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
    purchase_feature_map = (
        "decision_context_v2"
        if purchase_loss == "hurdle_contextual_lognormal"
        else "base_outputs_v1"
    )
    if len(dates) <= minimum_inner_training_dates:
        raise ValueError("not enough whole dates for strict-prior inner OOF")
    by_date = {
        date: tuple(race for race in ordered if race.decision.race_date == date)
        for date in dates
    }
    folds: list[InnerOOFFold] = []
    oof_payloads: list[dict[str, Any]] = []
    base_oof_by_date: dict[
        str, list[tuple[str, np.ndarray, np.ndarray]]
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
                (race.decision.race_id, purchase_matrix, realized)
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
        )
        date_score_payloads: list[dict[str, Any]] = []
        date_threshold_payloads: list[dict[str, Any]] = []
        for race_id, matrix, realized in validation_records:
            scores = _purchase_net_scores(
                fold_head, fold_payout_head, matrix
            )
            purchase_oof_scores_for_calibration.extend(scores.tolist())
            purchase_oof_returns_for_calibration.extend(realized.tolist())
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
    purchase_calibration_head = None
    if purchase_loss == "hurdle_logistic_lognormal_calibrated":
        calibration_matrix = np.asarray(
            purchase_oof_scores_for_calibration, dtype=np.float64
        ).reshape(-1, 1)
        calibration_target = np.clip(
            np.asarray(purchase_oof_returns_for_calibration, dtype=np.float64)
            + 1.0,
            0.0,
            51.0,
        )
        fitted_calibration = PoissonRegressor(
            alpha=alpha,
            fit_intercept=True,
            max_iter=500,
            tol=1e-8,
        ).fit(calibration_matrix, calibration_target)
        purchase_calibration_head = _head(
            "purchase_calibration_head",
            "poisson_calibration_of_strict_purchase_head_oof_gross_return",
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
    purchase = _purchase_net_scores(
        artifact.purchase_head,
        artifact.purchase_payout_head,
        _purchase_matrix(
            decision,
            probability,
            ranking,
            closing,
            feature_map=artifact.purchase_feature_map,
        ),
    )
    if artifact.purchase_calibration_head is not None:
        purchase = np.exp(
            np.clip(
                _scores(
                    artifact.purchase_calibration_head,
                    purchase.reshape(-1, 1),
                ),
                -30.0,
                math.log(51.0),
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
