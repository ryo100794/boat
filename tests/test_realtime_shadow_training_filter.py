import numpy as np
from scipy.sparse import csr_matrix

from boatrace_ai.modeling import _ensure_int32_sparse_indices, _make_pipeline
from boatrace_ai.db import connection, init_db
from boatrace_ai.features import load_training_examples


def test_shadow_training_requires_odds_history_and_complete_results(tmp_path) -> None:
    db_path = tmp_path / "shadow.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        for rno, race_id in enumerate(("ready", "no-odds", "incomplete"), start=1):
            conn.execute(
                "INSERT INTO races (race_id, race_date, jcd, venue_name, rno) "
                "VALUES (?, '2026-07-18', '01', '桐生', ?)",
                (race_id, rno),
            )
            for lane in range(1, 7):
                conn.execute(
                    "INSERT INTO entries (race_id, lane) VALUES (?, ?)",
                    (race_id, lane),
                )
                if race_id != "incomplete" or lane <= 3:
                    conn.execute(
                        "INSERT INTO race_results (race_id, lane, rank) VALUES (?, ?, ?)",
                        (race_id, lane, lane),
                    )
            if race_id != "no-odds":
                for snapshot in range(10):
                    cursor = conn.execute(
                        "INSERT INTO odds_snapshots "
                        "(race_id, bet_type, captured_at) VALUES (?, '3t', ?)",
                        (race_id, f"2026-07-18T10:{snapshot:02}:00+09:00"),
                    )
                    conn.execute(
                        "INSERT INTO odds_trifecta "
                        "(snapshot_id, race_id, combination, odds) VALUES (?, ?, '1-2-3', ?)",
                        (cursor.lastrowid, race_id, 10.0 + snapshot),
                    )

        features, labels, meta = load_training_examples(
            conn,
            from_date="2026-07-18",
            include_odds=True,
            min_odds_snapshots=10,
            complete_results_only=True,
        )

    assert len(features) == len(labels) == len(meta) == 6
    assert {row["race_id"] for row in meta} == {"ready"}
    assert all(feature["odds_snapshot_count"] == 10.0 for feature in features)


def test_shadow_pipeline_accepts_scipy_int64_sparse_indices() -> None:
    matrix = csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)

    converted = _ensure_int32_sparse_indices(matrix)

    assert converted.indices.dtype == np.int32
    assert converted.indptr.dtype == np.int32
    pipeline = _make_pipeline()
    pipeline.fit(
        [{"lane": 1}, {"lane": 2}, {"lane": 1}, {"lane": 2}],
        [1, 0, 1, 0],
    )
