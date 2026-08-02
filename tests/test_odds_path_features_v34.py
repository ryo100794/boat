from datetime import datetime, timedelta, timezone

from boatrace_ai.listwise.four_head_nested_v22 import (
    DecisionOddsPath,
    DecisionOddsPoint,
    DecisionRace,
    RacePrediction,
    T5_ODDS_PATH_SCHEMA,
)
from boatrace_ai.listwise.learned_purchase_allocation_v33 import (
    decision_feature_matrices,
)


def _point(offset: int) -> DecisionOddsPoint:
    t300 = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
    target = t300 - timedelta(seconds=offset - 300)
    return DecisionOddsPoint(
        offset,
        (target - timedelta(seconds=10)).isoformat(),
        target.isoformat(),
        (20.0,) * 120,
    )


def test_path_features_distinguish_missing_history_from_zero_slope() -> None:
    full = DecisionOddsPath(
        T5_ODDS_PATH_SCHEMA,
        tuple(_point(offset) for offset in (1800, 1200, 600, 420, 300)),
    )
    sparse = DecisionOddsPath(T5_ODDS_PATH_SCHEMA, (_point(300),))
    prediction = RacePrediction(
        race_id="r1",
        race_date="2026-08-01",
        probabilities=(1 / 120,) * 120,
        ranking_scores=tuple(float(120 - index) for index in range(120)),
        predicted_closing_odds=(20.0,) * 120,
        purchase_scores=(0.0,) * 120,
        selected_indices=(),
    )

    def matrices(path):
        decision = DecisionRace(
            "r1", "2026-08-01", ((1.0,),) * 120, (20.0,) * 120, path
        )
        return decision_feature_matrices(decision, prediction)

    full_ticket, full_race, _ = matrices(full)
    sparse_ticket, sparse_race, _ = matrices(sparse)

    assert full_ticket.shape == sparse_ticket.shape
    assert full_race.shape == sparse_race.shape
    assert tuple(full_ticket[0, -4:]) == (1.0, 1.0, 1.0, 1.0)
    assert tuple(sparse_ticket[0, -4:]) == (0.0, 0.0, 0.0, 0.0)
    assert full_race[-4:].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert sparse_race[-4:].tolist() == [0.0, 0.0, 0.0, 0.0]
