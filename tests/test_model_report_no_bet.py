from pathlib import Path

from boatrace_ai.web.dashboard import _bankroll_summary


def test_bankroll_summary_preserves_no_bet_selection_reason() -> None:
    summary = _bankroll_summary(
        Path("conditional.json"),
        "conditional",
        {
            "policy": {
                "no_bet": True,
                "no_bet_reason": "selection_gate_no_bet",
            },
            "policy_selection": {"source": "selection_gate_no_bet"},
            "evaluated_races": 100,
            "selected_tickets": 0,
            "stake_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
        },
    )

    assert summary["no_bet"] is True
    assert summary["no_bet_reason"] == "selection_gate_no_bet"
    assert summary["policy_selection_source"] == "selection_gate_no_bet"
    assert summary["roi"] == 0.0


def test_model_report_labels_no_bet_as_purchase_skipped() -> None:
    template = (
        Path(__file__).parents[1]
        / "src"
        / "boatrace_ai"
        / "templates"
        / "model_report.html"
    ).read_text(encoding="utf-8")

    assert 'if(b.no_bet) return {label:"購入見送り"' in template
    assert template.count("b.no_bet?") >= 2
