from __future__ import annotations

import argparse
import json

from ..db import connection, init_db
from .feature_search import FeatureVariants
from .feature_search import build_parser as build_feature_search_parser
from .feature_search import search


COMBINED_FEATURE_VARIANTS: FeatureVariants = (
    (
        "drop_base_pastlog_research_correlates",
        ("base_pastlog", "research_correlates"),
    ),
    (
        "drop_base_pastlog_series_cached",
        ("base_pastlog", "series_cached"),
    ),
    (
        "drop_base_pastlog_series_relative",
        ("base_pastlog", "series_relative"),
    ),
    (
        "drop_base_pastlog_rolling_history",
        ("base_pastlog", "rolling_history"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_feature_search_parser()
    parser.description = "Fixed combined feature-group ablation search."
    parser.set_defaults(
        output="data/models/listwise_combined_feature_search_v1.json",
        cache_dir="data/models/listwise_combined_search_cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = search(conn, args=args, variants=COMBINED_FEATURE_VARIANTS)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"search_results", "daily"}
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
