from __future__ import annotations

from boatrace_ai.listwise.feature_search import _selected_row


def _candidate(
    name: str,
    *,
    ranking_loss: float,
    top5: float,
    top1: float = 0.55,
    entry_loss: float = 0.34,
) -> dict[str, float | str]:
    return {
        "feature_variant": name,
        "ranking_log_loss": ranking_loss,
        "entry_log_loss": entry_loss,
        "winner_top1_accuracy": top1,
        "trifecta_top5_hit_rate": top5,
    }


def test_selection_prefers_top5_within_ranking_loss_tolerance() -> None:
    best_loss = _candidate("best-loss", ranking_loss=1.0, top5=0.29)
    better_top5 = _candidate("better-top5", ranking_loss=1.0099, top5=0.31)

    assert _selected_row([best_loss, better_top5])["feature_variant"] == (
        "better-top5"
    )


def test_selection_rejects_top5_gain_outside_ranking_loss_tolerance() -> None:
    best_loss = _candidate("best-loss", ranking_loss=1.0, top5=0.29)
    outside = _candidate("outside", ranking_loss=1.0101, top5=0.40)

    assert _selected_row([best_loss, outside])["feature_variant"] == "best-loss"


def test_selection_uses_top1_then_entry_loss_as_tie_breakers() -> None:
    lower_top1 = _candidate(
        "lower-top1",
        ranking_loss=1.0,
        top5=0.31,
        top1=0.55,
        entry_loss=0.32,
    )
    higher_top1 = _candidate(
        "higher-top1",
        ranking_loss=1.004,
        top5=0.31,
        top1=0.56,
        entry_loss=0.35,
    )
    better_entry = _candidate(
        "better-entry",
        ranking_loss=1.003,
        top5=0.31,
        top1=0.56,
        entry_loss=0.33,
    )

    assert _selected_row([lower_top1, higher_top1, better_entry])[
        "feature_variant"
    ] == "better-entry"
