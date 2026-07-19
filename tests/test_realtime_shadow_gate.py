import sqlite3

from boatrace_ai.runtime.model_cycle import dataset_counts


def test_shadow_gate_counts_only_races_with_enough_odds_snapshots() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE races (race_id TEXT PRIMARY KEY, race_date TEXT);
        CREATE TABLE entries (race_id TEXT, lane INTEGER);
        CREATE TABLE race_results (race_id TEXT, lane INTEGER, rank INTEGER);
        CREATE TABLE odds_snapshots (snapshot_id INTEGER PRIMARY KEY, race_id TEXT);
        """
    )
    for race_id, race_date, snapshots in (
        ("old", "2026-07-17", 12),
        ("ready", "2026-07-18", 10),
        ("short", "2026-07-18", 9),
        ("incomplete", "2026-07-18", 10),
    ):
        conn.execute("INSERT INTO races VALUES (?, ?)", (race_id, race_date))
        lanes = (1, 2, 3) if race_id == "incomplete" else (1, 2, 3, 4, 5, 6)
        for lane in lanes:
            conn.execute("INSERT INTO entries VALUES (?, ?)", (race_id, lane))
            conn.execute("INSERT INTO race_results VALUES (?, ?, ?)", (race_id, lane, lane))
        conn.executemany(
            "INSERT INTO odds_snapshots (race_id) VALUES (?)",
            [(race_id,)] * snapshots,
        )

    counts = dataset_counts(
        conn,
        from_date="2026-07-18",
        require_odds=True,
        min_odds_snapshots=10,
    )

    assert counts == {"examples": 6, "races": 1, "odds_result_races": 1}
