from pathlib import Path

from boatrace_ai.web.dashboard import serve


def test_removed_duplicate_modules_stay_removed() -> None:
    removed = (
        "src/boatrace_ai/server.py",
        "src/boatrace_ai/web_dashboard.py",
        "src/boatrace_ai/realtime_collector.py",
        "src/boatrace_ai/realtime_predictor.py",
        "src/boatrace_ai/model_cycle.py",
        "src/boatrace_ai/model_extended.py",
        "src/boatrace_ai/ingestion/archive_core.py",
        "src/boatrace_ai/ingestion/archive_extended.py",
    )
    assert not [path for path in removed if Path(path).exists()]


def test_dashboard_has_one_server_implementation() -> None:
    assert callable(serve)
