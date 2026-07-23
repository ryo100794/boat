from __future__ import annotations

from boatrace_ai.web.dashboard import (
    _daily_report_rows,
    _model_report_catalog,
    _report_model_key,
)


def test_daily_rows_normalize_dates_merge_duplicates_and_recompute_totals() -> None:
    rows = _daily_report_rows(
        [
            {
                "race_date": "2026-07-20",
                "evaluated_races": 2,
                "tickets": 1,
                "stake_yen": 100,
                "return_yen": 0,
                "profit_yen": -100,
                "budget_used_fraction": 0.01,
            },
            {
                "date": "20260719",
                "races": 3,
                "selected_tickets": 2,
                "selected_races": 1,
                "stake_yen": 200,
                "return_yen": 500,
                "hit_tickets": 1,
                "budget_used_fraction": 0.02,
            },
            {
                "day": "2026-07-19T12:00:00",
                "evaluated_races": 4,
                "tickets": 3,
                "races_bet": 2,
                "stake_yen": 300,
                "return_yen": 100,
                "profit_yen": -200,
                "budget_used_fraction": 0.03,
            },
            {"race_date": "not-a-date", "tickets": 99},
        ]
    )

    assert [row["date"] for row in rows] == ["2026-07-19", "2026-07-20"]
    first = rows[0]
    assert first["evaluated_races"] == 7
    assert first["tickets"] == 5
    assert first["races_bet"] == 3
    assert first["stake_yen"] == 500
    assert first["return_yen"] == 600
    assert first["profit_yen"] == 100
    assert first["cumulative_profit_yen"] == 100
    assert first["cumulative_stake_yen"] == 500
    assert first["cumulative_return_yen"] == 600
    assert first["cumulative_roi"] == 1.2
    assert first["cumulative_roi_delta"] == 0.2
    assert first["roi"] == 1.2
    assert first["budget_used_fraction"] == 0.05
    assert first["ticket_hit_rate"] == 0.2
    assert rows[1]["cumulative_profit_yen"] == 0
    assert rows[1]["cumulative_stake_yen"] == 600
    assert rows[1]["cumulative_return_yen"] == 600
    assert rows[1]["cumulative_roi"] == 1.0
    assert rows[1]["cumulative_roi_delta"] == 0.0
    assert _daily_report_rows(rows) == rows


def test_cumulative_roi_delta_stays_negative_during_losing_streak() -> None:
    rows = _daily_report_rows(
        [
            {"race_date": "2026-07-19", "stake_yen": 100, "return_yen": 0},
            {"race_date": "2026-07-20", "stake_yen": 200, "return_yen": 0},
            {"race_date": "2026-07-21", "stake_yen": 100, "return_yen": 50},
        ]
    )

    assert [row["cumulative_profit_yen"] for row in rows] == [-100, -300, -350]
    assert [row["cumulative_roi_delta"] for row in rows] == [-1.0, -1.0, -0.875]


def test_catalog_assigns_stable_keys_and_prefers_standard_365d_daily() -> None:
    legacy = {
        "name": "alpha",
        "file": "alpha.json",
        "evaluation_scope": "legacy:2026-07-01:2026-07-02:2",
    }
    standard = {
        "name": "standardized_365d_v2_alpha",
        "file": "standardized_365d_v2/alpha.json",
        "evaluation_scope": "standard_365d_v2",
    }
    no_daily = {
        "name": "beta",
        "file": "beta.json",
        "evaluation_scope": "legacy:unknown",
    }
    catalog, daily = _model_report_catalog(
        model_tracks=[],
        backtests=[standard, no_daily],
        bankroll=[legacy, standard],
        fold_metrics=[],
        evaluation_jobs=[],
        feature_diagnostics=[],
        sweeps=[],
        bankroll_daily={
            "alpha": [
                {
                    "race_date": "2026-07-22",
                    "stake_yen": 100,
                    "return_yen": 50,
                }
            ],
            "standardized_365d_v2_alpha": [
                {
                    "race_date": "2025-07-23",
                    "evaluated_races": 10,
                    "tickets": 4,
                    "stake_yen": 400,
                    "return_yen": 600,
                    "budget_used_fraction": 0.04,
                }
            ],
        },
    )

    assert _report_model_key("standardized_365d_v2_alpha.json") == "alpha"
    assert legacy["model_key"] == standard["model_key"] == "alpha"
    assert catalog[0]["model_key"] == "alpha"
    assert daily["alpha"]["evaluation_scope"] == "standard_365d_v2"
    assert daily["alpha"]["rows"][0] == {
        "date": "2025-07-23",
        "evaluated_races": 10,
        "tickets": 4,
        "races_bet": 0,
        "stake_yen": 400,
        "return_yen": 600,
        "profit_yen": 200,
        "cumulative_profit_yen": 200,
        "cumulative_stake_yen": 400,
        "cumulative_return_yen": 600,
        "cumulative_roi": 1.5,
        "cumulative_roi_delta": 0.5,
        "roi": 1.5,
        "budget_used_fraction": 0.04,
        "ticket_hit_rate": 0.0,
    }
    assert daily["beta"]["rows"] == []
    assert "artifact/job" in daily["beta"]["unavailable_reason"]


def test_catalog_links_model_track_aliases_to_daily_artifacts() -> None:
    track = {
        "id": "historical_main",
        "label": "過去ログ主系",
        "model_file": "win_model_no_odds_v8.joblib",
    }
    bankroll = {
        "name": "no_odds_v8_relative_weather_sparse32_scaled_logreg_C0.20_unweighted",
        "file": "standardized_365d_no_odds_v8_backtest.json",
        "evaluation_scope": "standard_365d",
    }

    catalog, daily = _model_report_catalog(
        model_tracks=[track],
        backtests=[],
        bankroll=[bankroll],
        fold_metrics=[],
        evaluation_jobs=[],
        feature_diagnostics=[],
        sweeps=[],
        bankroll_daily={
            bankroll["name"]: [
                {
                    "race_date": "2026-07-22",
                    "stake_yen": 100,
                    "return_yen": 200,
                }
            ]
        },
    )

    assert _report_model_key("standardized_365d_no_odds_v8_backtest.json") == "no_odds_v8"
    assert _report_model_key("calibrated_mlp_shadow_2fold.json") == "calibrated_mlp_shadow"
    assert track["model_key"] == bankroll["model_key"] == "no_odds_v8"
    assert [row["model_key"] for row in catalog] == ["no_odds_v8"]
    assert daily["no_odds_v8"]["rows"][0]["date"] == "2026-07-22"


def test_catalog_adds_model_key_to_every_source_row() -> None:
    groups = {
        "model_tracks": [{"id": "track-a", "label": "Track A"}],
        "backtests": [{"name": "test-a"}],
        "bankroll": [{"name": "bank-a"}],
        "fold_metrics": [{"model": "fold-a"}],
        "evaluation_jobs": [{"model_key": "job-a", "name": "Job A"}],
        "feature_diagnostics": [{"name": "feature-a"}],
        "sweeps": [{"variant": "sweep-a"}],
    }

    catalog, _ = _model_report_catalog(
        **groups,
        bankroll_daily={},
    )

    assert len(catalog) == len(groups)
    assert all(row.get("model_key") for rows in groups.values() for row in rows)
