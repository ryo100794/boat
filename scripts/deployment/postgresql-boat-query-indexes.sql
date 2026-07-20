ALTER TABLE odds_snapshots
  ADD COLUMN IF NOT EXISTS parser_version text;

CREATE INDEX IF NOT EXISTS idx_odds_snapshot_parser_cutoff
  ON odds_snapshots (race_id, parser_version, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_beforeinfo_race_lane_captured
  ON beforeinfo (race_id, lane, captured_at DESC);

ANALYZE odds_snapshots;
ANALYZE beforeinfo;

CREATE INDEX IF NOT EXISTS idx_races_jcd_date_race
  ON races (jcd, race_date, race_id);

CREATE INDEX IF NOT EXISTS idx_entries_racer_race
  ON entries (racer_no, race_id);

CREATE INDEX IF NOT EXISTS idx_entries_motor_race
  ON entries (motor_no, race_id);

CREATE INDEX IF NOT EXISTS idx_entries_boat_race
  ON entries (boat_no, race_id);

CREATE INDEX IF NOT EXISTS idx_payouts_type_combo_race
  ON payouts (bet_type, combination, race_id);

ANALYZE races;
ANALYZE entries;
ANALYZE payouts;
