from boatrace_ai.runtime.intraday_t300_shadow import V23Top5NarrowModelAdapter
from boatrace_ai.runtime.stable_cell_shadow import StableCellTop5ModelAdapter
from boatrace_ai.runtime.stable_cell_shadow_policy import (
    POLICY_NAME,
    registration,
    select_stable_cell_candidates,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _probabilities() -> dict[str, float]:
    ranked = sorted(COMBINATIONS)[:5]
    values = {combination: 0.795 / 115 for combination in COMBINATIONS}
    for combination, probability in zip(ranked, (0.06, 0.055, 0.05, 0.02, 0.02)):
        values[combination] = probability
    return values


def test_selects_only_preregistered_low_odds_top5_cell() -> None:
    probabilities = _probabilities()
    ranked = sorted(COMBINATIONS)[:5]
    odds = {combination: 200.0 for combination in COMBINATIONS}
    odds[ranked[0]] = 17.0
    odds[ranked[1]] = 20.0
    odds[ranked[2]] = 50.0
    odds[ranked[3]] = 52.5

    selected = select_stable_cell_candidates(
        probabilities,
        odds,
        race_id="2026-08-02-01-01",
        race_date="2026-08-02",
        jcd="01",
        rno=1,
        snapshot_id=1,
        captured_at="2026-08-02T08:20:00+09:00",
        available_capital_yen=10_000,
    )

    assert [row["combination"] for row in selected] == [ranked[0]]
    assert selected[0]["estimated_ev"] == 1.02
    assert selected[0]["estimated_odds"] < 20.0
    assert selected[0]["policy_name"] == POLICY_NAME


def test_registration_discloses_development_look_and_disables_betting() -> None:
    value = registration()

    assert value["development_holdout_used_to_choose_policy"] is True
    assert value["promotion_evidence_start_date"] == "2026-08-02"
    assert value["real_betting_enabled"] is False


def test_stable_adapter_allows_only_registered_contextual_closing_model() -> None:
    assert V23Top5NarrowModelAdapter.allowed_closing_forecasts == frozenset({"baseline", "momentum"})
    assert StableCellTop5ModelAdapter.allowed_closing_forecasts == frozenset({"contextual"})
