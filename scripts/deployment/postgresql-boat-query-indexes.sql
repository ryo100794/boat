ALTER TABLE odds_snapshots
  ADD COLUMN IF NOT EXISTS parser_version text;

CREATE INDEX IF NOT EXISTS idx_odds_snapshot_parser_cutoff
  ON odds_snapshots (race_id, parser_version, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_beforeinfo_race_lane_captured
  ON beforeinfo (race_id, lane, captured_at DESC);

ANALYZE odds_snapshots;
ANALYZE beforeinfo;
