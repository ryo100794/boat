from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from boatrace_ai.db import connection


QUERY = """
WITH per_race AS (
  SELECT
    r.race_date,
    r.race_id,
    CAST(r.deadline_at AS timestamptz) - interval '5 minutes' AS cutoff_at,
    MAX(CAST(os.captured_at AS timestamptz)) FILTER (
      WHERE CAST(os.captured_at AS timestamptz)
        <= CAST(r.deadline_at AS timestamptz) - interval '5 minutes'
    ) AS last_pre,
    COUNT(*) FILTER (
      WHERE CAST(os.captured_at AS timestamptz) BETWEEN
        CAST(r.deadline_at AS timestamptz) - interval '6 minutes'
        AND CAST(r.deadline_at AS timestamptz) - interval '5 minutes'
    ) AS final_minute_snapshots
  FROM races r
  LEFT JOIN odds_snapshots os ON os.race_id = r.race_id
  WHERE r.race_date BETWEEN ? AND ?
  GROUP BY r.race_date, r.race_id, r.deadline_at
)
SELECT
  race_date,
  COUNT(*) FILTER (WHERE last_pre IS NOT NULL) AS races_with_pre_cutoff_odds,
  ROUND(
    AVG(EXTRACT(epoch FROM cutoff_at - last_pre))
      FILTER (WHERE last_pre IS NOT NULL),
    1
  ) AS average_gap_seconds,
  ROUND((
    percentile_cont(0.5) WITHIN GROUP (
      ORDER BY EXTRACT(epoch FROM cutoff_at - last_pre)
    ) FILTER (WHERE last_pre IS NOT NULL)
  )::numeric, 1) AS median_gap_seconds,
  ROUND((
    percentile_cont(0.9) WITHIN GROUP (
      ORDER BY EXTRACT(epoch FROM cutoff_at - last_pre)
    ) FILTER (WHERE last_pre IS NOT NULL)
  )::numeric, 1) AS p90_gap_seconds,
  COUNT(*) FILTER (
    WHERE last_pre IS NOT NULL
      AND cutoff_at - last_pre <= interval '15 seconds'
  ) AS within_15_seconds,
  COUNT(*) FILTER (WHERE final_minute_snapshots > 0) AS with_final_minute_snapshot
FROM per_race
GROUP BY race_date
ORDER BY race_date
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit final pre-cutoff trifecta odds capture latency."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--through-date", type=date.fromisoformat, required=True)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    if args.days < 1 or args.days > 365:
        parser.error("--days must be between 1 and 365")
    from_date = args.through_date - timedelta(days=args.days - 1)
    with connection(args.db) as conn:
        rows = conn.execute(
            QUERY,
            (from_date.isoformat(), args.through_date.isoformat()),
        ).fetchall()
    print(
        json.dumps(
            [{key: row[key] for key in row.keys()} for row in rows],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
