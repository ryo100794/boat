from boatrace_ai.evaluation_queue import TASK_PROFILES


def test_archive_oracle_reservation_matches_observed_bounded_memory() -> None:
    profile = TASK_PROFILES["archive_market_oracle"]
    assert profile["memory_mb"] == 4096
    assert profile["max_parallel"] == 1
