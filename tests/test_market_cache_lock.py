from __future__ import annotations

import fcntl

import pytest

from boatrace_ai.listwise.market_calibration import scored_cache_build_lock


def test_scored_cache_build_lock_excludes_competing_writer(tmp_path) -> None:
    cache = tmp_path / "shared.races.joblib"

    with scored_cache_build_lock(cache) as lock_path:
        assert lock_path.is_file()
        with lock_path.open("a+b") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competitor.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )

    with lock_path.open("a+b") as competitor:
        fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competitor.fileno(), fcntl.LOCK_UN)
